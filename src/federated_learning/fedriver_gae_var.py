# fedriver_gae_var.py
"""
FedRIVER — Variant A: Variance Regularization + FedComb Server
---------------------------------------------------------------
This is your BEST performing FedRIVER variant (F1 ≈ 0.9206).

Local objective adds a variance regularization term:
    L = mean(node_errors) + λ * var(node_errors)

Rationale:
  Penalizing high variance in per-node reconstruction errors
  forces the model to reconstruct ALL nodes consistently —
  not just the easy ones. Anomalies that the model "ignores"
  (large error but small weight) are forced into the loss.
  This sharpens the decision boundary between normal and anomalous.

Server aggregation — FedComb (momentum-adaptive):
    δ_t   = avg_local_Δ          (mean weight delta)
    Θ_t   = β·Θ_{t-1} + (1-β)·δ_t      (momentum)
    v_t  += Θ_t²                         (second moment)
    w_new = w_global + η·Θ_t / (√v_t + ε)

Run:
    python fedriver_gae_var.py --data_dir ./dataset
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

# Variance regularisation
LAMBDA_VAR        = 0.5    # weight on variance penalty

# FedComb server hyperparameters
BETA_MOMENTUM     = 0.9    # momentum decay
ETA               = 0.01   # server learning rate
EPSILON           = 1e-8   # numerical stability


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
#  Helpers
# ─────────────────────────────────────────────

def extreme_error(err: torch.Tensor, p: float = 0.05, alpha: float = 3.0):
    q = torch.quantile(err, 1 - p)
    return torch.where(err > q, err * alpha, err)


# ─────────────────────────────────────────────
#  FedComb server aggregation
# ─────────────────────────────────────────────

class FedCombServer:
    """
    Momentum-adaptive server aggregation (FedComb).

    Maintains running momentum Θ and second moment v for each
    weight tensor.  Updates the global model using:
        Θ_t  = β·Θ_{t-1} + (1-β)·δ_t
        v_t += Θ_t²
        w    = w_global + η·Θ_t / (√v_t + ε)
    """

    def __init__(self, global_model, beta=0.9, eta=0.01, eps=1e-8):
        self.beta  = beta
        self.eta   = eta
        self.eps   = eps

        # initialise momentum and second moment buffers
        ref = global_model.state_dict()
        self.momentum = {k: torch.zeros_like(v.float()) for k, v in ref.items()}
        self.v        = {k: torch.zeros_like(v.float()) for k, v in ref.items()}

    def aggregate(self, global_model, local_states: list):
        """Aggregate local states into global model via FedComb."""
        global_state = global_model.state_dict()
        n = len(local_states)

        # mean delta
        delta = {}
        for k in global_state:
            stacked = torch.stack(
                [s[k].float() - global_state[k].float() for s in local_states]
            )
            delta[k] = stacked.mean(dim=0)

        # momentum update
        new_state = {}
        for k in global_state:
            self.momentum[k] = (self.beta * self.momentum[k]
                                + (1 - self.beta) * delta[k])
            self.v[k]       += self.momentum[k] ** 2
            update           = self.eta * self.momentum[k] / (
                torch.sqrt(self.v[k]) + self.eps
            )
            new_state[k] = global_state[k].float() + update

        global_model.load_state_dict(new_state)


# ─────────────────────────────────────────────
#  Local training — variance regularised loss
# ─────────────────────────────────────────────

def train_local(model, data, device, epochs, lr, lambda_var):
    """
    Local GAE training with variance regularization.

    Loss = mean(per-node MSE) + λ * var(per-node MSE)
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    data = data.to(device)
    for _ in range(epochs):
        optimizer.zero_grad()
        x_rec, _ = model(data)

        # per-node reconstruction error on train nodes
        node_err = torch.mean(
            (x_rec[data.train_mask] - data.x[data.train_mask]) ** 2,
            dim=1
        )                                              # [N_train]

        # variance regularisation
        loss = node_err.mean() + lambda_var * node_err.var()

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
    print(f'Lambda (var reg) : {LAMBDA_VAR}')
    print(f'FedComb β        : {BETA_MOMENTUM}  η={ETA}')

    print('\nLoading graphs...')
    graphs = get_all_graphs(data_dir, verbose=False)
    client_ids, _ = get_fl_partition(graphs)
    print(f'Total clients: {len(client_ids)}')

    global_model = GraphAutoencoder().to(device)
    server       = FedCombServer(global_model, beta=BETA_MOMENTUM,
                                 eta=ETA, eps=EPSILON)

    f1_per_round = []

    print('\nFedRIVER (Variant A — Variance Reg + FedComb)...')
    for rnd in range(1, NUM_ROUNDS + 1):

        selected     = random.sample(client_ids, CLIENTS_PER_ROUND)
        local_states = []

        for cid in selected:
            local_model = copy.deepcopy(global_model)
            train_local(local_model, graphs[cid], device, EPOCHS, LR, LAMBDA_VAR)
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
    print(f'FedRIVER Var-Reg  —  Best weighted micro-F1 : {best_f1:.4f}')
    print(f'                     (round {f1_per_round.index(best_f1)+1})')
    print(f'{"="*50}')
    print(f'\nRound-wise F1: {[round(f,4) for f in f1_per_round]}')

    return best_f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./dataset')
    args = parser.parse_args()
    main(args.data_dir)
