# graph_builder.py
"""
Graph construction for FedRIVER — SMD Dataset
----------------------------------------------
Strategy: Adaptive Mutual KNN + RBF edge weights

Each row (timestep) becomes a node.
Node features  : 38 sensor readings (already normalized to [0,1]).
Node labels    : binary anomaly label (0 = normal, 1 = anomaly).
Edges          : mutual k-nearest neighbors in Euclidean feature space.
Edge weights   : RBF kernel  w = exp(-d² / σ²)
                 where σ = median of all kNN distances (adaptive per machine).
Fallback       : isolated nodes (no mutual neighbors) are connected to
                 their single nearest neighbor to guarantee connectivity.

Why this is better than your original k=2 cosine approach:
  - k=7 gives ~4x more edges → richer neighborhood signal
  - Euclidean distance is more appropriate than cosine for
    normalized sensor data where magnitude differences matter
  - RBF weights encode how similar two timesteps are, not just
    whether they are neighbors
  - Mutual KNN keeps the graph symmetric and sparse
  - Adaptive σ (per machine) handles different machines having
    different feature spread

Usage:
    from graph_builder import build_graph
    data = build_graph('path/to/machine-1-1_test_combined.csv', k=7, train_ratio=0.6)
    # data.x          : [N, 38] float tensor
    # data.edge_index : [2, E] long tensor
    # data.edge_attr  : [E]    float tensor (RBF weights)
    # data.y          : [N]    long tensor  (0/1)
    # data.train_mask : [N]    bool tensor
"""

import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.neighbors import BallTree


# ─────────────────────────────────────────────
#  Core graph builder
# ─────────────────────────────────────────────

def build_graph(
    csv_path: str,
    k: int = 7,
    train_ratio: float = 0.6,
    seed: int = 42,
) -> Data:
    """
    Build a PyG Data object from a single machine CSV file.

    Parameters
    ----------
    csv_path    : path to machine CSV  (38 features + 'label' column)
    k           : number of nearest neighbors  (default 7)
    train_ratio : fraction of NORMAL nodes used for training (default 0.6)
    seed        : random seed for reproducible train/test split

    Returns
    -------
    torch_geometric.data.Data
    """
    # ── 1. Load CSV ──────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path, header=0)

    feature_cols = [c for c in df.columns if c != 'label']
    X = df[feature_cols].values.astype(np.float32)   # [N, 38]
    y = df['label'].values.astype(np.float32)         # [N]
    N = len(X)

    # ── 2. KNN via BallTree (memory-efficient, no full distance matrix) ──────
    tree = BallTree(X, metric='euclidean')
    dists, inds = tree.query(X, k=k + 1)   # +1: first result is self

    dists = dists[:, 1:].astype(np.float32)   # [N, k]  exclude self
    inds  = inds[:, 1:]                        # [N, k]

    # ── 3. Adaptive sigma (median of all kNN distances for this machine) ─────
    sigma = float(np.median(dists))
    if sigma < 1e-8:
        sigma = 1.0   # safety fallback

    # ── 4. RBF weights ────────────────────────────────────────────────────────
    rbf = np.exp(-(dists ** 2) / (sigma ** 2))   # [N, k]

    # ── 5. Build mutual KNN edges ─────────────────────────────────────────────
    # neighbor_set[i] = set of k nearest neighbors of node i
    neighbor_set = [set(inds[i].tolist()) for i in range(N)]

    edge_dict = {}   # (i, j) with i < j  →  weight

    for i in range(N):
        for j_pos, j in enumerate(inds[i]):
            j = int(j)
            if i in neighbor_set[j]:          # mutual: j also has i as neighbor
                key = (min(i, j), max(i, j))
                w   = float(rbf[i, j_pos])
                # keep the higher weight if edge already seen from other side
                if key not in edge_dict or w > edge_dict[key]:
                    edge_dict[key] = w

    # ── 6. Fallback: connect isolated nodes to their nearest neighbor ─────────
    has_edge = set()
    for (i, j) in edge_dict:
        has_edge.add(i)
        has_edge.add(j)

    for i in range(N):
        if i not in has_edge:
            j = int(inds[i, 0])
            w = float(rbf[i, 0])
            key = (min(i, j), max(i, j))
            edge_dict[key] = w

    # ── 7. Convert to directed edge_index (both directions) for PyG ──────────
    src_list = [i for (i, j) in edge_dict] + [j for (i, j) in edge_dict]
    dst_list = [j for (i, j) in edge_dict] + [i for (i, j) in edge_dict]
    w_list   = list(edge_dict.values()) + list(edge_dict.values())

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)   # [2, E]
    edge_attr  = torch.tensor(w_list, dtype=torch.float)                 # [E]

    # ── 8. Train mask  ────────────────────────────────────────────────────────
    # Train on a random subset of NORMAL nodes only.
    # Test on ALL remaining nodes (normal + anomaly).
    rng = np.random.default_rng(seed)
    normal_idx = np.where(y == 0)[0]
    n_train    = int(len(normal_idx) * train_ratio)
    train_idx  = rng.choice(normal_idx, size=n_train, replace=False)

    train_mask = torch.zeros(N, dtype=torch.bool)
    train_mask[train_idx] = True

    # ── 9. Pack into PyG Data object ─────────────────────────────────────────
    data = Data(
        x          = torch.tensor(X, dtype=torch.float),
        edge_index = edge_index,
        edge_attr  = edge_attr,
        y          = torch.tensor(y, dtype=torch.long),
        train_mask = train_mask,
    )

    return data


# ─────────────────────────────────────────────
#  Batch builder — builds all 28 machines
# ─────────────────────────────────────────────

def build_all_graphs(
    data_dir: str,
    k: int = 7,
    train_ratio: float = 0.6,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Build graphs for all machines in data_dir.

    Returns
    -------
    dict: { machine_id (str) -> torch_geometric.data.Data }
    Example keys: '1-1', '1-2', ..., '3-11'
    """
    csv_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.csv')
    ])

    graphs = {}

    for fname in csv_files:
        machine_id = (
            fname
            .replace('machine-', '')
            .replace('_test_combined.csv', '')
        )
        csv_path = os.path.join(data_dir, fname)

        if verbose:
            print(f'  Building graph for machine {machine_id} ...', end=' ')

        data = build_graph(csv_path, k=k, train_ratio=train_ratio, seed=seed)

        if verbose:
            n_edges = data.edge_index.shape[1] // 2   # undirected count
            n_anom  = int((data.y != 0).sum())
            print(
                f'nodes={data.num_nodes:6d}  '
                f'edges={n_edges:7d}  '
                f'anomalies={n_anom:5d} '
                f'({100*n_anom/data.num_nodes:.1f}%)'
            )

        graphs[machine_id] = data

    return graphs


# ─────────────────────────────────────────────
#  Quick smoke-test
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else './dataset'

    print(f'\nBuilding graphs from: {DATA_DIR}')
    print(f'k=7, train_ratio=0.6, sigma=adaptive (per machine)\n')

    graphs = build_all_graphs(DATA_DIR, k=7, train_ratio=0.6, verbose=True)

    print(f'\nDone. Total machines: {len(graphs)}')

    # Print one sample
    sample = graphs['1-1']
    print(f'\nSample graph (machine 1-1):')
    print(f'  x.shape       : {sample.x.shape}')
    print(f'  edge_index    : {sample.edge_index.shape}')
    print(f'  edge_attr     : {sample.edge_attr.shape}')
    print(f'  y.shape       : {sample.y.shape}')
    print(f'  train nodes   : {sample.train_mask.sum().item()}')
    print(f'  test  nodes   : {(~sample.train_mask).sum().item()}')
    print(f'  anomaly nodes : {(sample.y != 0).sum().item()}')
    print(f'  edge_attr min : {sample.edge_attr.min():.4f}')
    print(f'  edge_attr max : {sample.edge_attr.max():.4f}')
