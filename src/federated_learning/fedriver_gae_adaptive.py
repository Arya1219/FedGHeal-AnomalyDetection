# fedriver_gae_adaptive.py
"""
FedRIVER — Variant C: Adaptive λ·KL (Full Design / FedDual)
------------------------------------------------------------
The complete FedRIVER design: the KL penalty weight is adaptively
computed per client per round based on how much the client's local
performance deviates from the global performance.

Local objective:
    L = mean(node_errors)
      + σ(acc_local - acc_global) · KL(p_local ‖ p_global)

where:
  - σ(·) is the sigmoid function → output in (0, 1)
  - acc_local  = local F1 on train nodes using local model
  - acc_global = local F1 on train nodes using global model
  - KL(p_local ‖ p_global) = KL between local and global latent Gaussians

Intuition:
  If a client diverges from global (acc_local >> acc_global), the KL
  penalty is high → pulls it back.
  If the client is already well-aligned, the KL penalty is near zero
  → local adaptation is unconstrained.

This is the most principled variant but requires estimating per-round
client accuracy, adding overhead compared to Variant A.

Run:
    python fedriver_gae_adaptive.py --data_dir ./dataset
"""

import argparse
import random
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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
#  Utility
# ─────────────────────────────────────────────

def extreme_error(err, p=0.05, alpha=3.0):
    q = torch.quantile(err, 1 - p)
    return torch.where(err > q, err * alpha, err)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def gaussian_kl(mu1, std1, mu2, std2):
    """KL(N(mu1,std1²) ‖ N(mu2,std2²)), averaged over elements."""
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
#  Client accuracy helper
# ─────────────────────────────────────────────

def client_accuracy(model, data, threshold, device):
    """Micro-F1 on train nodes only (used to compute adaptive λ)."""
    model.eval()
    with torch.no_grad():
        data     = data.to(device)
        x_rec, _ = model(data)
        err = torch.mean((x_rec - data.x) ** 2, dim=1)
        err = extreme_error(err, EXTREME_P, EXTREME_ALPHA)

        train_err  = err[data.train_mask].cpu().numpy()
        train_lbls = data.y[data.train_mask].cpu().numpy()

        y_pred = (train_err    > threshold).astype(int)
        y_true = (train_lbls   != 0).astype(int)
        return f1_score(y_true, y_pred, average='micro', zero_division=0)


# ─────────────────────────────────────────────
#  Latent stats helper
# ─────────────────────────────────────────────

def latent_stats(model, data, device):
    """Mean and std of latent vectors for train nodes."""
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        _, latent = model(data)
        z   = latent[data.train_mask]
        mu  = z.mean(dim=0, keepdim=True).expand_as(z)
        std = z.std(dim=0, keepdim=True).expand_as(z) + 1e-8
    return mu.detach(), std.detach()


# ─────────────────────────────────────────────
#  Local training — adaptive KL
# ─────────────────────────────────────────────

def train_local(model, data, device, epochs, lr,
                lambda_kl_weight, global_latent_stats):
    """
    Local training with adaptive KL penalty.

    lambda_kl_weight: scalar computed as σ(acc_local - acc_global)
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    data = data.to(device)

    mu_g, std_g = global_latent_stats

    for _ in range(epochs):
        optimizer.zero_grad()
        x_rec, latent = model(data)

        node_err = torch.mean(
            (x_rec[data.train_mask] - data.x[data.train_mask]) ** 2,
            dim=1
        )

        z_train   = latent[data.train_mask]
        mu_local  = z_train.mean(dim=0, keepdim=True).expand_as(z_train)
        std_local = z_train.std(dim=0, keepdim=True).expand_as(z_train) + 1e-8

        kl_term = gaussian_kl(mu_local, std_local, mu_g, std_g)

        loss = node_err.mean() + lambda_kl_weight * kl_term
        loss.backward()
        optimizer.step()


# ─────────────────────────────────────────────
#  Threshold & evaluation
# ─────────────────────────────────────────────

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
    print(f'FedComb β={BETA_MOMENTUM}  η={ETA}')
    print(f'Adaptive λ = σ(acc_local - acc_global)')

    print('\nLoading graphs...')
    graphs = get_all_graphs(data_dir, verbose=False)
    client_ids, _ = get_fl_partition(graphs)
    print(f'Total clients: {len(client_ids)}')

    global_model = GraphAutoencoder().to(device)
    server       = FedCombServer(global_model, beta=BETA_MOMENTUM,
                                 eta=ETA, eps=EPSILON)

    f1_per_round = []

    print('\nFedRIVER (Variant C — Adaptive KL + FedComb)...')
    for rnd in range(1, NUM_ROUNDS + 1):

        selected     = random.sample(client_ids, CLIENTS_PER_ROUND)
        local_states = []

        # compute global threshold once per round for adaptive λ calculation
        global_threshold = compute_threshold(global_model, graphs, device, THRESHOLD_Q)

        for cid in selected:
            local_model = copy.deepcopy(global_model)

            # global model accuracy on this client
            acc_global = client_accuracy(global_model, graphs[cid],
                                         global_threshold, device)

            # local model accuracy after one quick pass (proxy for convergence gap)
            temp_model = copy.deepcopy(global_model)
            temp_model.train()
            opt_tmp = optim.Adam(temp_model.parameters(), lr=LR)
            data_tmp = graphs[cid].to(device)
            opt_tmp.zero_grad()
            x_rec_tmp, _ = temp_model(data_tmp)
            loss_tmp = nn.MSELoss()(
                x_rec_tmp[data_tmp.train_mask],
                data_tmp.x[data_tmp.train_mask]
            )
            loss_tmp.backward()
            opt_tmp.step()
            acc_local = client_accuracy(temp_model, graphs[cid],
                                        global_threshold, device)

            # adaptive λ
            lambda_kl = sigmoid(acc_local - acc_global)

            # get global latent stats for KL reference
            g_stats = latent_stats(global_model, graphs[cid], device)

            train_local(local_model, graphs[cid], device, EPOCHS, LR,
                        lambda_kl, g_stats)
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
    print(f'\n{"="*55}')
    print(f'FedRIVER Adaptive KL  —  Best weighted micro-F1 : {best_f1:.4f}')
    print(f'                         (round {f1_per_round.index(best_f1)+1})')
    print(f'{"="*55}')
    print(f'\nRound-wise F1: {[round(f,4) for f in f1_per_round]}')

    return best_f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./dataset')
    args = parser.parse_args()
    main(args.data_dir)
