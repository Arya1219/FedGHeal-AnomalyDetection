# fedavg_ae_dp.py
"""
Federated Autoencoder with Differential Privacy — FedRIVER
-----------------------------------------------------------
FL-AE + FedAvg + Gaussian DP noise injected before aggregation.

DP mechanism (Gaussian mechanism):
  1. Each client clips gradients to L2-norm ≤ CLIP_NORM
  2. Server adds Gaussian noise N(0, (NOISE_MULTIPLIER * CLIP_NORM)²)
     to each averaged parameter before broadcasting

This is the standard DP-FedAvg approach (McMahan et al., 2018).
Expected: DP hurts F1 vs plain FL-AE → motivates non-DP FedRIVER.

Run:
    python fedavg_ae_dp.py --data_dir ./dataset
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
from models import Autoencoder


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

# DP parameters
CLIP_NORM         = 1.0    # L2 gradient clipping threshold
NOISE_MULTIPLIER  = 0.01   # Gaussian noise std = NOISE_MULTIPLIER * CLIP_NORM


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
#  DP-FedAvg helpers
# ─────────────────────────────────────────────

def extreme_error(err: torch.Tensor, p: float = 0.05, alpha: float = 3.0):
    q = torch.quantile(err, 1 - p)
    return torch.where(err > q, err * alpha, err)


def clip_gradients(model, max_norm: float):
    """Clip ALL model parameters' gradients to L2 norm ≤ max_norm."""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def fedavg_with_dp(local_states: list, clip_norm: float, noise_mult: float) -> dict:
    """
    FedAvg aggregation + Gaussian noise injection (DP mechanism).
    Noise std = noise_mult * clip_norm  (per parameter element).
    """
    avg = copy.deepcopy(local_states[0])
    n   = len(local_states)

    for k in avg:
        for i in range(1, n):
            avg[k] = avg[k].float() + local_states[i][k].float()
        avg[k] = avg[k].float() / n

        # Gaussian noise scaled to sensitivity / n
        noise_std = (noise_mult * clip_norm) / n
        noise     = torch.randn_like(avg[k].float()) * noise_std
        avg[k]    = avg[k].float() + noise

    return avg


def train_local_dp(model, data, device, epochs, lr, clip_norm):
    """Train with per-step gradient clipping."""
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.MSELoss()

    data = data.to(device)
    for _ in range(epochs):
        optimizer.zero_grad()
        x     = data.x[data.train_mask]
        x_hat, _ = model(x)
        loss  = loss_fn(x_hat, x)
        loss.backward()
        clip_gradients(model, clip_norm)
        optimizer.step()


def compute_threshold(model, graphs, device, quantile=0.95):
    model.eval()
    all_errors = []
    with torch.no_grad():
        for data in graphs.values():
            data = data.to(device)
            x    = data.x[data.train_mask]
            x_hat, _ = model(x)
            err  = torch.mean((x_hat - x) ** 2, dim=1)
            err  = extreme_error(err, EXTREME_P, EXTREME_ALPHA)
            all_errors.append(err.cpu().numpy())
    return float(np.quantile(np.concatenate(all_errors), quantile))


def evaluate(model, graphs, threshold, device):
    model.eval()
    total_nodes = 0
    weighted_f1 = 0.0
    with torch.no_grad():
        for data in graphs.values():
            data  = data.to(device)
            x_hat, _ = model(data.x)
            err   = torch.mean((x_hat - data.x) ** 2, dim=1)
            err   = extreme_error(err, EXTREME_P, EXTREME_ALPHA)

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
    print(f'Device             : {device}')
    print(f'Rounds             : {NUM_ROUNDS}')
    print(f'Local epochs       : {EPOCHS}')
    print(f'Clients per round  : {CLIENTS_PER_ROUND}')
    print(f'DP clip norm       : {CLIP_NORM}')
    print(f'DP noise multiplier: {NOISE_MULTIPLIER}')
    print(f'DP noise std       : {NOISE_MULTIPLIER * CLIP_NORM:.4f}')

    print('\nLoading graphs...')
    graphs = get_all_graphs(data_dir, verbose=False)
    client_ids, _ = get_fl_partition(graphs)
    print(f'Total clients: {len(client_ids)}')

    global_model = Autoencoder().to(device)
    f1_per_round = []

    print('\nFederated Training (FedAvg AE + DP)...')
    for rnd in range(1, NUM_ROUNDS + 1):

        selected     = random.sample(client_ids, CLIENTS_PER_ROUND)
        local_states = []

        for cid in selected:
            local_model = copy.deepcopy(global_model)
            train_local_dp(local_model, graphs[cid], device, EPOCHS, LR, CLIP_NORM)
            local_states.append(local_model.state_dict())

        agg_state = fedavg_with_dp(local_states, CLIP_NORM, NOISE_MULTIPLIER)
        global_model.load_state_dict(agg_state)

        threshold = compute_threshold(global_model, graphs, device, THRESHOLD_Q)
        f1        = evaluate(global_model, graphs, threshold, device)
        f1_per_round.append(f1)

        print(f'  Round {rnd:2d}/{NUM_ROUNDS}  '
              f'clients={selected}  '
              f'F1={f1:.4f}')

    best_f1 = max(f1_per_round)
    print(f'\n{"="*50}')
    print(f'FedAvg AE + DP  —  Best weighted micro-F1 : {best_f1:.4f}')
    print(f'                   (round {f1_per_round.index(best_f1)+1})')
    print(f'{"="*50}')
    print(f'\nRound-wise F1: {[round(f,4) for f in f1_per_round]}')

    return best_f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./dataset')
    args = parser.parse_args()
    main(args.data_dir)
