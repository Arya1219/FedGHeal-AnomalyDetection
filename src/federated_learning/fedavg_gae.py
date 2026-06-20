# fedavg_gae.py
"""
Federated Graph Autoencoder with FedAvg — FedRIVER
---------------------------------------------------
Federated learning baseline using GAE + FedAvg aggregation.
Uses graph structure (GCN encoder).  This is the direct FL
counterpart of centralized_gae.py and the immediate predecessor
of the FedRIVER variants.

Expected story:
  FL-AE  < Centralized-GAE < FL-GAE < FedRIVER-GAE

Key difference from fedavg_ae.py:
  - Model takes full PyG Data object (uses edge_index)
  - GCN encoder propagates neighbor information
  - Local training must pass the full Data object, not just x

Run:
    python fedavg_gae.py --data_dir ./dataset
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


def fedavg(local_states: list) -> dict:
    avg = copy.deepcopy(local_states[0])
    for k in avg:
        for i in range(1, len(local_states)):
            avg[k] += local_states[i][k]
        avg[k] /= len(local_states)
    return avg


def train_local(model, data, device, epochs, lr):
    """Train a local GAE copy for `epochs` steps on a single client."""
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.MSELoss()

    data = data.to(device)
    for _ in range(epochs):
        optimizer.zero_grad()
        x_rec, _ = model(data)
        # train on train_mask nodes only
        loss = loss_fn(
            x_rec[data.train_mask],
            data.x[data.train_mask]
        )
        loss.backward()
        optimizer.step()


def compute_threshold(model, graphs, device, quantile=0.95):
    """Threshold from TRAINING nodes only — no leakage."""
    model.eval()
    all_errors = []
    with torch.no_grad():
        for data in graphs.values():
            data     = data.to(device)
            x_rec, _ = model(data)
            err = torch.mean((x_rec - data.x) ** 2, dim=1)
            err = extreme_error(err, EXTREME_P, EXTREME_ALPHA)
            train_err = err[data.train_mask]
            all_errors.append(train_err.cpu().numpy())
    return float(np.quantile(np.concatenate(all_errors), quantile))


def evaluate(model, graphs, threshold, device):
    """Node-count-weighted micro-F1 across all machines."""
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

    print('\nLoading graphs...')
    graphs = get_all_graphs(data_dir, verbose=False)
    client_ids, _ = get_fl_partition(graphs)
    print(f'Total clients: {len(client_ids)}')

    global_model = GraphAutoencoder().to(device)

    f1_per_round = []

    print('\nFederated Training (FedAvg GAE)...')
    for rnd in range(1, NUM_ROUNDS + 1):

        selected     = random.sample(client_ids, CLIENTS_PER_ROUND)
        local_states = []

        for cid in selected:
            local_model = copy.deepcopy(global_model)
            train_local(local_model, graphs[cid], device, EPOCHS, LR)
            local_states.append(local_model.state_dict())

        global_model.load_state_dict(fedavg(local_states))

        threshold = compute_threshold(global_model, graphs, device, THRESHOLD_Q)
        f1        = evaluate(global_model, graphs, threshold, device)
        f1_per_round.append(f1)

        print(f'  Round {rnd:2d}/{NUM_ROUNDS}  '
              f'clients={selected}  '
              f'threshold={threshold:.6f}  '
              f'F1={f1:.4f}')

    best_f1 = max(f1_per_round)
    print(f'\n{"="*45}')
    print(f'FedAvg GAE  —  Best weighted micro-F1 : {best_f1:.4f}')
    print(f'              (round {f1_per_round.index(best_f1)+1})')
    print(f'{"="*45}')
    print(f'\nRound-wise F1: {[round(f,4) for f in f1_per_round]}')

    return best_f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./dataset')
    args = parser.parse_args()
    main(args.data_dir)
