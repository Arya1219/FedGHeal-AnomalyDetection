# evaluate.py
"""
Clean Unified Evaluation — FedRIVER
-------------------------------------
Load any saved model checkpoint and re-evaluate it cleanly.

Key guarantees:
  - Threshold computed from TRAINING nodes only (zero leakage)
  - Extreme error amplification applied consistently
  - Per-machine breakdown available with --verbose
  - Weighted micro-F1 and per-machine F1 both reported
  - Works for AE, VAE, and GAE checkpoints

Usage:
    # Evaluate a saved GAE checkpoint
    python evaluate.py --model_path ./checkpoints/fedavg_gae.pt \
                       --model_type gae \
                       --data_dir ./dataset

    # With per-machine breakdown
    python evaluate.py --model_path ./checkpoints/fedriver_var.pt \
                       --model_type gae \
                       --data_dir ./dataset \
                       --verbose

Model types:
    ae   → models.Autoencoder
    vae  → models.VariationalAutoencoder
    gae  → models.GraphAutoencoder

Checkpoint format:
    torch.save(model.state_dict(), path)
"""

import argparse
import os

import numpy as np
import torch
from sklearn.metrics import (
    f1_score, precision_score, recall_score, confusion_matrix
)

from prep_data_lib import get_all_graphs
from models import Autoencoder, VariationalAutoencoder, GraphAutoencoder


# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────

EXTREME_P     = 0.05
EXTREME_ALPHA = 3.0
THRESHOLD_Q   = 0.95


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def extreme_error(err: torch.Tensor, p: float = 0.05, alpha: float = 3.0):
    q = torch.quantile(err, 1 - p)
    return torch.where(err > q, err * alpha, err)


def get_reconstruction_error(model, data, model_type: str, device):
    """
    Forward pass → per-node reconstruction error [N] tensor.
    Works for AE, VAE (deterministic eval), GAE.
    """
    model.eval()
    data = data.to(device)

    with torch.no_grad():
        if model_type == 'gae':
            x_rec, _ = model(data)
        elif model_type == 'vae':
            x_hat, mu, _ = model(data.x)   # eval mode → z=mu (deterministic)
            x_rec = x_hat
        else:   # ae
            x_rec, _ = model(data.x)

        err = torch.mean((x_rec - data.x) ** 2, dim=1)   # [N]
    return err


# ─────────────────────────────────────────────
#  Threshold — TRAIN nodes only
# ─────────────────────────────────────────────

def compute_threshold(
    model, graphs: dict, model_type: str, device,
    quantile: float = THRESHOLD_Q,
) -> float:
    """
    Compute anomaly detection threshold from TRAINING nodes only.
    This is the CORRECT approach — test data must not influence threshold.
    """
    all_errors = []
    for data in graphs.values():
        err = get_reconstruction_error(model, data, model_type, device)
        err = extreme_error(err, EXTREME_P, EXTREME_ALPHA)
        all_errors.append(err[data.train_mask].cpu().numpy())

    all_errors = np.concatenate(all_errors)
    threshold  = float(np.quantile(all_errors, quantile))
    return threshold


# ─────────────────────────────────────────────
#  Evaluation — clean, no leakage
# ─────────────────────────────────────────────

def evaluate_all(
    model, graphs: dict, model_type: str, threshold: float, device,
    verbose: bool = False,
) -> dict:
    """
    Evaluate on TEST nodes across all machines.

    Returns
    -------
    dict with keys:
        weighted_f1   : float
        weighted_prec : float
        weighted_rec  : float
        per_machine   : dict { machine_id -> {f1, prec, rec, tp, fp, tn, fn} }
    """
    total_nodes   = 0
    weighted_f1   = 0.0
    weighted_prec = 0.0
    weighted_rec  = 0.0
    per_machine   = {}

    for mid, data in graphs.items():
        err = get_reconstruction_error(model, data, model_type, device)
        err = extreme_error(err, EXTREME_P, EXTREME_ALPHA)

        test_mask   = ~data.train_mask
        test_err    = err[test_mask].cpu().numpy()
        test_labels = data.y[test_mask].cpu().numpy()

        y_pred = (test_err    > threshold).astype(int)
        y_true = (test_labels != 0).astype(int)

        f1   = f1_score(y_true, y_pred, average='micro', zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

        n = data.num_nodes
        weighted_f1   += n * f1
        weighted_prec += n * prec
        weighted_rec  += n * rec
        total_nodes   += n

        per_machine[mid] = {
            'f1': f1, 'prec': prec, 'rec': rec,
            'tp': int(tp), 'fp': int(fp),
            'tn': int(tn), 'fn': int(fn),
            'n_test': int(test_mask.sum()),
            'n_anom': int(y_true.sum()),
        }

    results = {
        'weighted_f1'  : weighted_f1   / total_nodes,
        'weighted_prec': weighted_prec / total_nodes,
        'weighted_rec' : weighted_rec  / total_nodes,
        'per_machine'  : per_machine,
    }

    if verbose:
        print(f'\n{"─"*65}')
        print(f'{"Machine":<12} {"F1":>6} {"Prec":>6} {"Rec":>6} '
              f'{"TP":>6} {"FP":>6} {"TN":>6} {"FN":>6} {"Anom%":>7}')
        print(f'{"─"*65}')
        for mid, m in sorted(per_machine.items()):
            anom_pct = 100 * m['n_anom'] / max(m['n_test'], 1)
            print(f'{mid:<12} {m["f1"]:>6.4f} {m["prec"]:>6.4f} '
                  f'{m["rec"]:>6.4f} {m["tp"]:>6d} {m["fp"]:>6d} '
                  f'{m["tn"]:>6d} {m["fn"]:>6d} {anom_pct:>6.1f}%')
        print(f'{"─"*65}')

    return results


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main(model_path: str, model_type: str, data_dir: str, verbose: bool):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device     : {device}')
    print(f'Model type : {model_type}')
    print(f'Checkpoint : {model_path}')
    print(f'Data dir   : {data_dir}')

    # ── Load model ─────────────────────────────────────────────────────────
    if model_type == 'gae':
        model = GraphAutoencoder()
    elif model_type == 'vae':
        model = VariationalAutoencoder()
    else:
        model = Autoencoder()

    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)
    print(f'Loaded checkpoint — {sum(p.numel() for p in model.parameters()):,} parameters')

    # ── Load data ──────────────────────────────────────────────────────────
    print('\nLoading graphs...')
    graphs = get_all_graphs(data_dir, verbose=False)
    print(f'Machines: {len(graphs)}')

    # ── Threshold (train nodes only — ZERO LEAKAGE) ────────────────────────
    threshold = compute_threshold(model, graphs, model_type, device, THRESHOLD_Q)
    print(f'\nAnomaly threshold (Q{THRESHOLD_Q*100:.0f} train errors): {threshold:.6f}')

    # ── Evaluation ─────────────────────────────────────────────────────────
    results = evaluate_all(model, graphs, model_type, threshold, device,
                           verbose=verbose)

    print(f'\n{"="*45}')
    print(f'Weighted micro-F1   : {results["weighted_f1"]:.4f}')
    print(f'Weighted Precision  : {results["weighted_prec"]:.4f}')
    print(f'Weighted Recall     : {results["weighted_rec"]:.4f}')
    print(f'{"="*45}')

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clean unified model evaluation.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to saved model state_dict (.pt)')
    parser.add_argument('--model_type', type=str, default='gae',
                        choices=['ae', 'vae', 'gae'],
                        help='Model architecture (ae | vae | gae)')
    parser.add_argument('--data_dir',   type=str, default='./dataset',
                        help='Directory containing machine CSVs')
    parser.add_argument('--verbose',    action='store_true',
                        help='Print per-machine breakdown')
    args = parser.parse_args()

    main(args.model_path, args.model_type, args.data_dir, args.verbose)
