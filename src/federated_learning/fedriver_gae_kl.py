# fedriver_gae_kl.py
"""
FedRIVER — Variant B: Variance Regularization + Fixed KL + FedComb
-------------------------------------------------------------------
Extends Variant A by adding a fixed-weight KL divergence term that
aligns each client's local latent distribution toward the global one.

Local objective:
    L = mean(node_errors)
      + λ · var(node_errors)                    ← variance penalty
      + μ · KL(p_local_latent ‖ p_global_latent) ← distribution alignment

KL is estimated as:
    KL(N(μ_local, σ_local) ‖ N(μ_global, σ_global))
Using the closed-form KL between two Gaussians.

Motivation:
  The KL term encourages clients to stay close to the global latent
  space, preventing drift on heterogeneous machines.  Fixed μ means
  the alignment strength is constant across all rounds.

Compare with Variant A (no KL) to see if distribution alignment helps.
Expected: slight improvement over A from reduced client drift, but
          the fixed μ may over-constrain diverse clients → see Variant C.

Run:
    python fedriver_gae_kl.py --data_dir ./dataset
"""

import argparse
import random
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import f1_score

from prep_data_lib import get_all_graphs, get_fl_partition
from models import GraphAutoencoder


# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────

SEED              = 42
NUM_ROUNDS        = 10
EPOCHS            = 50
LR                = 0.01
CLIENTS_PER_ROUND = 7
EXTREME_P         = 0.05
EXTREME_ALPHA     = 3.0
THRESHOLD_Q       = 0.95

LAMBDA_VAR        = 0.5     # variance regularisation weight
MU_KL             = 0.005   # KL divergence weight (fixed)

BETA_MOMENTUM     = 0.9
ETA               = 0.01
EPSILON           = 1e-8


# ─────────────────────────────────────────────
#  Reproducibility
# ─────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────
#  KL divergence between two Gaussians
# ─────────────────────────────────────────────

def gaussian_kl(mu1, std1, mu2, std2):
    """
    Closed-form KL(N(mu1,std1²) ‖ N(mu2,std2²)) averaged over nodes.
    All inputs: [N, latent_dim] tensors.
    """
    var1 = std1 ** 2 + 1e-8
    var2 = std2 ** 2 + 1e-8
    kl = (torch.log(std2 / (std1 + 1e-8))
          + (var1 + (mu1 - mu2) ** 2) / (2 * var2)
          - 0.5)
    return kl.mean()


# ─────────────────────────────────────────────
#  FedComb server
# ─────────────────────────────────────────────

class FedCombServer:
    def __init__(self, global_model, beta=0.9, eta=0.01, eps=1e-8):
        self.beta = beta
        self.eta  = eta
        self.eps  = eps
        ref = global_model.state_dict()
        self.momentum = {k: torch.zeros_like(v.float()) for k, v in ref.items()}
        self.v        = {k: torch.zeros_like(v.float()) for k, v in ref.items()}

    def aggregate(self, global_model, local_states):
        global_state = global_model.state_dict()
        delta = {
            k: torch.stack([s[k].float() - global_state[k].float()
                            for s in local_states]).mean(0)
            for k in global_state
        }
        new_state = {}
        for k in global_state:
            self.momentum[k] = self.beta * self.momentum[k] + (1 - self.beta) * delta[k]
            self.v[k]       += self.momentum[k] ** 2
            new_state[k]     = (global_state[k].float()
                                + self.eta * self.momentum[k]
                                / (torch.sqrt(self.v[k]) + self.eps))
        global_model.load_state_dict(new_state)


# ─────────────────────────────────────────────
#  Local training — var + KL loss
# ─────────────────────────────────────────────

def train_local(model, data, device, epochs, lr,
                lambda_var, mu_kl, global_latent_stats):
    """
    Local GAE training with variance reg + fixed KL alignment.

    global_latent_stats: (mu_g, std_g) both [N_train, latent_dim]
                         computed from the global model before the round.
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    data = data.to(device)

    mu_g, std_g = global_latent_stats  # reference distribution

    for _ in range(epochs):
        optimizer.zero_grad()
        x_rec, latent = model(data)   # latent: [N, latent_dim]

        # reconstruction error per node (train nodes)
        node_err = torch.mean(
            (x_rec[data.train_mask] - data.x[data.train_mask]) ** 2,
            dim=1
        )

        # local latent stats for train nodes
        z_train   = latent[data.train_mask]
        mu_local  = z_train.mean(dim=0, keepdim=True).expand_as(z_train)
        std_local = z_train.std(dim=0, keepdim=True).expand_as(z_train) + 1e-8

        kl_term = gaussian_kl(mu_local, std_local, mu_g, std_g)

        loss = (node_err.mean()
                + lambda_var * node_err.var()
                + mu_kl * kl_term)

        loss.backward()
        optimizer.step()


def get_global_latent_stats(model, data, device):
    """Compute (mu, std) of global model latent for train nodes."""
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        _, latent = model(data)
        z     = latent[data.train_mask]
        mu_g  = z.mean(dim=0, keepdim=True).expand_as(z)
        std_g = z.std(dim=0, keepdim=True).expand_as(z) + 1e-8
    return mu_g.detach(), std_g.detach()


# ─────────────────────────────────────────────
#  Threshold & evaluation
# ─────────────────────────────────────────────

def extreme_error(err, p=0.05, alpha=3.0):
    q = torch.quantile(err, 1 - p)
    return torch.where(err > q, err * alpha, err)


def compute_threshold(model, graphs, device, quantile=0.95):
    model.eval()
    all_errors = []
    with torch.no_grad():
        for data in graphs.values():
            data     = data.to(device)
            x_rec, _ = model(data)
            err = torch.mean((x_rec - data.x) ** 2, dim=1)
            err = extreme_error(err, EXTREME_P, EXTREME_ALPHA)
            all_errors.append(err[data.train_mask].cpu().numpy())
    return float(np.quantile(np.concatenate(all_errors), quantile))


def evaluate(model, graphs, threshold, device):
    model.eval()
    total_nodes = 0
    weighted_f1 = 0.0
    with torch.no_grad():
        for data in graphs.values():
            data     = data.to(device)
            x_rec, _ = model(data)
            err = torch.mean((x_rec - data.x) ** 2, dim=1)
            err = extreme_error(err, EXTREME_P, EXTREME_ALPHA)

            test_mask = ~data.train_mask
            y_pred = (err[test_mask].cpu().numpy()    > threshold).astype(int)
            y_true = (data.y[test_mask].cpu().numpy() != 0).astype(int)

            f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
            weighted_f1 += data.num_nodes * f1
            total_nodes += data.num_nodes
    return weighted_f1 / total_nodes


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main(data_dir: str):
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device           : {device}')
    print(f'Rounds           : {NUM_ROUNDS}')
    print(f'Local epochs     : {EPOCHS}')
    print(f'Clients per round: {CLIENTS_PER_ROUND}')
    print(f'Lambda (var reg) : {LAMBDA_VAR}')
    print(f'Mu (KL weight)   : {MU_KL}')
    print(f'FedComb β={BETA_MOMENTUM}  η={ETA}')

    print('\nLoading graphs...')
    graphs = get_all_graphs(data_dir, verbose=False)
    client_ids, _ = get_fl_partition(graphs)
    print(f'Total clients: {len(client_ids)}')

    global_model = GraphAutoencoder().to(device)
    server       = FedCombServer(global_model, beta=BETA_MOMENTUM,
                                 eta=ETA, eps=EPSILON)

    f1_per_round = []

    print('\nFedRIVER (Variant B — Var + Fixed KL + FedComb)...')
    for rnd in range(1, NUM_ROUNDS + 1):

        selected     = random.sample(client_ids, CLIENTS_PER_ROUND)
        local_states = []

        for cid in selected:
            local_model = copy.deepcopy(global_model)

            # compute global reference distribution for this client's graph
            g_stats = get_global_latent_stats(global_model, graphs[cid], device)

            train_local(local_model, graphs[cid], device, EPOCHS, LR,
                        LAMBDA_VAR, MU_KL, g_stats)
            local_states.append(local_model.state_dict())

        server.aggregate(global_model, local_states)

        threshold = compute_threshold(global_model, graphs, device, THRESHOLD_Q)
        f1        = evaluate(global_model, graphs, threshold, device)
        f1_per_round.append(f1)

        print(f'  Round {rnd:2d}/{NUM_ROUNDS}  '
              f'clients={selected}  '
              f'threshold={threshold:.6f}  '
              f'F1={f1:.4f}')

    best_f1 = max(f1_per_round)
    print(f'\n{"="*50}')
    print(f'FedRIVER Var+KL  —  Best weighted micro-F1 : {best_f1:.4f}')
    print(f'                    (round {f1_per_round.index(best_f1)+1})')
    print(f'{"="*50}')
    print(f'\nRound-wise F1: {[round(f,4) for f in f1_per_round]}')

    return best_f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./dataset')
    args = parser.parse_args()
    main(args.data_dir)
