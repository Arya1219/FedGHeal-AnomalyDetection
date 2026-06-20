# prep_data_lib.py
"""
Data loading library for FedRIVER — SMD Dataset
------------------------------------------------
Provides two main functions used by ALL other scripts:

    get_all_graphs(data_dir)
        → dict of { machine_id -> PyG Data }
          Builds or loads all 28 machine graphs.

    get_client_ids(data_dir)
        → sorted list of machine_id strings
          e.g. ['1-1', '1-2', ..., '3-11']

All graph construction is delegated to graph_builder.py.
This file handles caching, splitting into FL clients,
and providing a consistent interface to every training script.

Caching
-------
Graphs are expensive to build (BallTree KNN on ~25k nodes).
On first call, built graphs are saved to  data_dir/graph_cache/
as .pt files. Subsequent calls load from cache instantly.
Set  force_rebuild=True  to ignore cache.
"""

import os
import torch
from torch_geometric.data import Data
from graph_builder import build_graph, build_all_graphs


# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────

FEATURE_DIM  = 38
K_NEIGHBORS  = 7
TRAIN_RATIO  = 0.6
RANDOM_SEED  = 42
CACHE_SUBDIR = 'graph_cache'


# ─────────────────────────────────────────────
#  Cache helpers
# ─────────────────────────────────────────────

def _cache_path(data_dir: str, machine_id: str) -> str:
    cache_dir = os.path.join(data_dir, CACHE_SUBDIR)
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f'{machine_id}.pt')


def _save_graph(data: Data, path: str):
    torch.save(data, path)


def _load_graph(path: str) -> Data:
    return torch.load(path, weights_only=False)


# ─────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────

def get_client_ids(data_dir: str) -> list:
    """
    Return sorted list of machine IDs found in data_dir.
    e.g. ['1-1', '1-2', ..., '3-11']
    """
    ids = []
    for fname in os.listdir(data_dir):
        if fname.endswith('_test_combined.csv'):
            mid = fname.replace('machine-', '').replace('_test_combined.csv', '')
            ids.append(mid)
    return sorted(ids)


def get_all_graphs(
    data_dir: str,
    k: int = K_NEIGHBORS,
    train_ratio: float = TRAIN_RATIO,
    seed: int = RANDOM_SEED,
    force_rebuild: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Build (or load from cache) graphs for all machines.

    Parameters
    ----------
    data_dir      : directory containing machine CSVs
    k             : KNN neighbors for graph construction
    train_ratio   : fraction of normal nodes used for training
    seed          : random seed
    force_rebuild : if True, ignore cache and rebuild
    verbose       : print progress

    Returns
    -------
    dict: { machine_id (str) -> torch_geometric.data.Data }
    """
    client_ids = get_client_ids(data_dir)

    if verbose:
        print(f'Found {len(client_ids)} machines in {data_dir}')

    graphs = {}

    for mid in client_ids:
        cache = _cache_path(data_dir, mid)

        if not force_rebuild and os.path.exists(cache):
            if verbose:
                print(f'  [cache] Loading machine {mid}')
            graphs[mid] = _load_graph(cache)
            continue

        csv_path = os.path.join(
            data_dir, f'machine-{mid}_test_combined.csv'
        )

        if verbose:
            print(f'  [build] Building graph for machine {mid} ...', end=' ')

        data = build_graph(
            csv_path,
            k=k,
            train_ratio=train_ratio,
            seed=seed,
        )

        _save_graph(data, cache)

        if verbose:
            n_edges = data.edge_index.shape[1] // 2
            n_anom  = int((data.y != 0).sum())
            print(
                f'nodes={data.num_nodes:6d}  '
                f'edges={n_edges:7d}  '
                f'anomalies={n_anom:5d} ({100*n_anom/data.num_nodes:.1f}%)'
            )

        graphs[mid] = data

    if verbose:
        total_nodes = sum(g.num_nodes for g in graphs.values())
        total_edges = sum(g.edge_index.shape[1] // 2 for g in graphs.values())
        total_anom  = sum(int((g.y != 0).sum()) for g in graphs.values())
        print(f'\nDataset summary:')
        print(f'  Machines : {len(graphs)}')
        print(f'  Nodes    : {total_nodes:,}')
        print(f'  Edges    : {total_edges:,}  (undirected)')
        print(f'  Anomalies: {total_anom:,} ({100*total_anom/total_nodes:.1f}%)')

    return graphs


def get_single_graph(
    data_dir: str,
    machine_id: str,
    k: int = K_NEIGHBORS,
    train_ratio: float = TRAIN_RATIO,
    seed: int = RANDOM_SEED,
    force_rebuild: bool = False,
) -> Data:
    """
    Build (or load) graph for a single machine.

    Parameters
    ----------
    machine_id : e.g. '1-1', '2-3', '3-11'
    """
    cache = _cache_path(data_dir, machine_id)

    if not force_rebuild and os.path.exists(cache):
        return _load_graph(cache)

    csv_path = os.path.join(
        data_dir, f'machine-{machine_id}_test_combined.csv'
    )

    data = build_graph(
        csv_path,
        k=k,
        train_ratio=train_ratio,
        seed=seed,
    )

    _save_graph(data, cache)
    return data


# ─────────────────────────────────────────────
#  FL helper — used by all federated scripts
# ─────────────────────────────────────────────

def get_fl_partition(graphs: dict) -> tuple:
    """
    Returns (client_ids, graphs) as parallel lists,
    consistent ordering guaranteed.

    Usage in FL training:
        client_ids, graph_list = get_fl_partition(graphs)
        selected = random.sample(client_ids, k=7)
        for cid in selected:
            data = graphs[cid]
    """
    client_ids = sorted(graphs.keys())
    return client_ids, graphs


# ─────────────────────────────────────────────
#  Smoke test
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else './dataset'

    print('=' * 60)
    print('prep_data_lib.py — smoke test')
    print('=' * 60)

    # First call: builds and caches
    graphs = get_all_graphs(DATA_DIR, verbose=True)

    print('\nSecond call (should load from cache):')
    graphs2 = get_all_graphs(DATA_DIR, verbose=True)

    # Check single graph access
    g = graphs['1-1']
    print(f'\nMachine 1-1:')
    print(f'  x          : {g.x.shape}')
    print(f'  edge_index : {g.edge_index.shape}')
    print(f'  edge_attr  : {g.edge_attr.shape}')
    print(f'  y          : {g.y.shape}')
    print(f'  train_mask : {g.train_mask.sum().item()} train nodes')
    print(f'  test nodes : {(~g.train_mask).sum().item()}')

    # FL partition
    client_ids, _ = get_fl_partition(graphs)
    print(f'\nFL clients ({len(client_ids)}): {client_ids}')
