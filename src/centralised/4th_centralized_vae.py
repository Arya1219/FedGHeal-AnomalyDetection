# centralized_vae.py
"""
Centralized Variational Autoencoder Baseline — FedRIVER
---------------------------------------------------------
Trains a single VAE on ALL machines simultaneously.
No federation, no graph structure.

For anomaly detection with VAE:
  - At eval time, reparameterization is deterministic (z = mu)
  - Reconstruction error is MSE on the mean path
  - Threshold computed from training node errors only (no leakage)

Run:
    python centralized_vae.py --data_dir ./dataset
"""

import os
import sys
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score

from prep_data_lib import get_all_graphs
from models import VariationalAutoencoder, vae_loss


# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────

SEED          = 42
EPOCHS        = 50
LR            = 0.01
BETA          = 0.5    # KL weight — lower = more focus on reconstruction
EXTREME_P     = 0.05
EXTREME_ALPHA = 3.0
THRESHOLD_Q   = 0.95


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
#  Extreme error amplification
# ─────────────────────────────────────────────

def extreme_error(err: torch.Tensor, p: float = 0.05, alpha: float = 3.0):
    q = torch.quantile(err, 1 - p)
    return torch.where(err > q, err * alpha, err)


# ─────────────────────────────────────────────
#  Train one epoch
# ─────────────────────────────────────────────

def train_one_epoch(model, data, optimizer, device):
    model.train()
    optimizer.zero_grad()

    x = data.x[data.train_mask].to(device)
    x_hat, mu, logvar = model(x)

    loss = vae_loss(x_hat, x, mu, logvar, beta=BETA)
    loss.backward()
    optimizer.step()
    return loss.item()


# ─────────────────────────────────────────────
#  Threshold (train errors only — NO leakage)
# ─────────────────────────────────────────────

def compute_threshold(model, graphs: dict, device, quantile: float = 0.95):
    model.eval()
    all_errors = []

    with torch.no_grad():
        for data in graphs.values():
            data = data.to(device)
            x    = data.x[data.train_mask]

            # deterministic pass: reparameterize returns mu at eval
            x_hat, mu, _ = model(x)

            err = torch.mean((x_hat - x) ** 2, dim=1)
            err = extreme_error(err, p=EXTREME_P, alpha=EXTREME_ALPHA)
            all_errors.append(err.cpu().numpy())

    all_errors = np.concatenate(all_errors)
    return float(np.quantile(all_errors, quantile))


# ─────────────────────────────────────────────
#  Evaluation
# ─────────────────────────────────────────────

def evaluate(model, graphs: dict, threshold: float, device):
    model.eval()
    total_nodes = 0
    weighted_f1 = 0.0

    with torch.no_grad():
        for data in graphs.values():
            data  = data.to(device)
            x_hat, mu, _ = model(data.x)

            err = torch.mean((x_hat - data.x) ** 2, dim=1)
            err = extreme_error(err, p=EXTREME_P, alpha=EXTREME_ALPHA)

            test_mask   = ~data.train_mask
            test_err    = err[test_mask].cpu().numpy()
            test_labels = data.y[test_mask].cpu().numpy()

            y_pred = (test_err    > threshold).astype(int)
            y_true = (test_labels != 0).astype(int)

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
    print(f'Device : {device}')
    print(f'Epochs : {EPOCHS}')
    print(f'Beta   : {BETA}  (KL weight)')

    # ── Load data ──────────────────────────────────────────────────────────
    print('\nLoading graphs...')
    graphs = get_all_graphs(data_dir, verbose=False)
    print(f'Machines loaded: {len(graphs)}')

    # ── Model ──────────────────────────────────────────────────────────────
    model     = VariationalAutoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # ── Training ───────────────────────────────────────────────────────────
    print('\nTraining Centralized VAE...')
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for data in graphs.values():
            total_loss += train_one_epoch(model, data, optimizer, device)

        if epoch % 10 == 0:
            print(f'  Epoch {epoch:3d}/{EPOCHS}  loss={total_loss:.4f}')

    # ── Threshold ──────────────────────────────────────────────────────────
    threshold = compute_threshold(model, graphs, device, quantile=THRESHOLD_Q)
    print(f'\nAnomaly threshold (Q{THRESHOLD_Q*100:.0f} of train errors): {threshold:.6f}')

    # ── Evaluation ─────────────────────────────────────────────────────────
    f1 = evaluate(model, graphs, threshold, device)
    print(f'\n{"="*45}')
    print(f'Centralized VAE  —  Weighted micro-F1 : {f1:.4f}')
    print(f'{"="*45}')

    return f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./dataset')
    args = parser.parse_args()
    main(args.data_dir)
