# FedGHeal: Graph-Aware MT-FedHealRL

**Federated Graph-Based Anomaly Detection with Differential Privacy for Industrial Time Series**

> Om Jee Pandey, Senior Member, IEEE, and Arya Giri  
> Department of Electronics Engineering, IIT (BHU) Varanasi – 221005, India

---

## Overview

FedGHeal is a unified federated learning framework for privacy-preserving anomaly detection in industrial IoT environments. It addresses three critical shortcomings of standard federated anomaly detection pipelines — simultaneously and within a single cohesive system:

| Challenge | Problem | Our Solution |
|---|---|---|
| **Non-IID Heterogeneity** | Machines have wildly different anomaly rates (0.4% – 15.7%), causing client drift under FedAvg | **FedMAV** — momentum-adaptive variance aggregation |
| **Poisoning Attacks** | Malicious clients can corrupt the global model invisibly in weight space | **Graph-Aware RL Trust Scoring** — detects poisoning via anomaly score distribution shifts |
| **Flat Representations** | Standard autoencoders ignore temporal graph structure, missing context-dependent anomalies | **GAE Local Model** — GCN encoder with adaptive mutual KNN graph construction |

All three contributions operate under formal **(ε, δ)-differential privacy** guarantees via Gaussian noise injection on client updates.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FEDGHEAL SERVER                          │
│                                                                 │
│   ┌──────────────┐    ┌────────────────┐    ┌───────────────┐  │
│   │    FedMAV    │    │  RL Trust Score│    │  DP Gaussian  │  │
│   │  Aggregation │◄───│  (Graph-Aware) │    │  Noise Layer  │  │
│   └──────┬───────┘    └────────────────┘    └───────────────┘  │
│          │ Global θᵗ                                            │
└──────────┼──────────────────────────────────────────────────────┘
           │ Broadcast
    ┌──────┴──────┬──────────────┬─────────────┐
    ▼             ▼              ▼             ▼
┌────────┐  ┌────────┐    ┌────────┐   ┌────────┐
│Client 1│  │Client 2│ …  │Client k│   │  (m=7  │
│ GAE +  │  │ GAE +  │    │ GAE +  │   │  per   │
│MutKNN  │  │MutKNN  │    │MutKNN  │   │ round) │
└────────┘  └────────┘    └────────┘   └────────┘
  Machine 1   Machine 2    Machine 28
```

Each client builds a local graph from its multivariate sensor time series, trains a Graph Autoencoder locally, then transmits privatized weight updates to the server.

---

## Key Contributions

### 1. Adaptive Mutual KNN Graph Construction

Each client constructs a sparse temporal graph from its 38-channel sensor readings:

- **KNN Search** — BallTree index; each node retrieves its *k* = 3 nearest neighbors in Euclidean feature space
- **Adaptive RBF Bandwidth** — σ = median of all pairwise KNN distances within the machine (no manual tuning across heterogeneous machines)
- **RBF Edge Weights** — `w_ij = exp(−‖xᵢ − xⱼ‖² / 2σ²)`
- **Mutual KNN Filter** — edge (i, j) retained only if both `j ∈ Nᵢ` and `i ∈ Nⱼ`, enforcing graph symmetry
- **Isolated Node Fallback** — guarantees full connectivity

Anomalous timesteps form sparse, isolated clusters in feature space. With mutual KNN, they connect primarily to boundary nodes rather than the dense normal cluster, amplifying their GCN reconstruction error.

### 2. Graph Autoencoder (GAE) Local Model

**GCN Encoder:** Two graph convolutional layers + linear projection  
`38 → 26 → 16 → 8` (latent dimension)

**MLP Decoder:** Reconstructs node features without graph structure  
`8 → 16 → 26 → 38`

**Anomaly Scoring:** Per-node MSE reconstruction error with extreme amplification on the top-5% — sharpens the anomaly boundary under severe class imbalance. Per-client threshold τ_k is set at the 0.95 quantile of training node errors.

### 3. FedMAV — Momentum-Adaptive Variance Aggregation

Replaces standard FedAvg with an Adam-style update at the federation level:

```
Δᵗ  = θ̄ᵗ − θᵗ                         (average delta)
mᵗ  = β·mᵗ⁻¹ + (1−β)·Δᵗ               (momentum buffer, β=0.9)
vᵗ  = vᵗ⁻¹ + (mᵗ)²                     (variance accumulator)
θᵗ⁺¹ = θᵗ + η_g · mᵗ / √(vᵗ + ε)     (adaptive update)
```

The momentum buffer smooths noisy round-to-round delta estimates. The variance accumulator dampens overshooting parameters while boosting lagging ones. Under DP noise, FedMAV degrades more gracefully than FedAvg because momentum averaging attenuates round-level noise by (1−β)^(t−t') in subsequent rounds.

### 4. Graph-Aware RL Client Trust Scoring

Extends MT-FedHealRL's RL defense with a graph-aware anomaly consistency signal:

**Anomaly Consistency Score (ACS):**
```
ACS_k = 1 − KS(Eₖ, Eₖᵍ)
```
where KS is the Kolmogorov-Smirnov statistic between the local model's and the global model's anomaly score distributions on client k's graph. Poisoned clients that appear benign in weight space will still exhibit distributional shifts in anomaly scores.

**Joint Trust Score:**
```
Trust_k = γ · WSI_k + (1−γ) · ACS_k     (γ = 0.6)
```

**RL State Representation:**
```
sₖ = [wₖ, Sₖᵈ, Sₖᵐ, ACSₖ, r]
```
Q-learning policy decides `{include, exclude}` for each client each round.

### 5. Differential Privacy

Gaussian noise is added to client updates before transmission:

```
θ̃ᵏₜ = θᵏₜ + N(0, σ²_DP·I)
```

where `σ_DP = C · √(2 ln(1.25/δ)) / ε`, satisfying **(ε, δ)-DP** per the Gaussian mechanism. Gradient clipping norm C = 1.0.

Three privacy regimes evaluated: strong (ε=0.5), moderate (ε=1.0), weak (ε=5.0).

---

## Dataset

**Server Machine Dataset (SMD)** — 28 industrial server machines, 38 sensor channels (CPU, memory, network I/O, disk), mapped directly to 28 federated clients.

| Property | Value |
|---|---|
| FL clients | 28 |
| Sensor channels | 38 |
| Total timesteps (nodes) | 708,420 |
| Min anomaly rate | 0.4% |
| Max anomaly rate | 15.7% |
| Mean anomaly rate | 4.2% |
| Total graph edges (undirected) | 1,535,537 |
| Avg. node degree (k=3) | ≈ 4.2 |
| Train / Test split | 60% / 40% |

---

## Results

Performance is reported as weighted micro-F1 (node-count weighted across all 28 machines).

| Method | Architecture | F1 |
|---|---|---|
| Centralized AE | MLP AE | 0.8959 |
| Centralized VAE | MLP VAE | 0.8833 |
| Centralized GAE | GCN AE | 0.8913 |
| FL-AE (FedAvg) | MLP AE | 0.8861 |
| FL-VAE (FedAvg) | MLP VAE | 0.8846 |
| FL-GAE (FedAvg) | GCN AE | 0.8832 |
| FL-AE + DP | MLP AE | 0.8860 |
| FL-GAE + DP | GCN AE | 0.8845 |
| **FedGHeal (FedMAV)** | **GCN AE** | **TBD** |

Key findings:
- FL-AE (0.8861) closely matches centralized AE (0.8959), confirming that federation incurs only a modest privacy-performance tradeoff.
- Naive graph addition (FL-GAE with high k) can hurt performance via GCN over-smoothing. Mutual KNN at k=3 with per-machine adaptive RBF thresholds recovers the structural advantage.
- FedMAV's convergence advantage over FedAvg grows from round 6 onward as the momentum buffer stabilizes.

---

## Installation

```bash
git clone https://github.com/your-org/FedGHeal.git
cd FedGHeal
pip install -r requirements.txt
```

**Requirements:** Python 3.10, PyTorch 2.0, PyTorch Geometric, NumPy, scikit-learn.

**SMD dataset:** Download from the [OmniAnomaly repository](https://github.com/NetManAIOps/OmniAnomaly) and place under `data/SMD/`.

---

## Project Structure

```
FedGHeal/
├── src/                   # Core framework source
│   ├── graph/             # Adaptive mutual KNN graph construction
│   ├── models/            # GAE (GCN encoder + MLP decoder)
│   ├── federation/        # FedMAV aggregation server
│   ├── defense/           # Graph-aware RL trust scoring
│   └── privacy/           # Differential privacy (Gaussian mechanism)
├── notebooks/             # Experiment notebooks and analysis
├── paper/                 # LaTeX source and figures
├── data/                  # SMD dataset (not tracked)
├── requirements.txt
└── README.md
```

---

## Hyperparameters

| Hyperparameter | Value |
|---|---|
| Local learning rate η | 0.01 |
| Local epochs E | 15 |
| Clients per round m | 7 |
| FL rounds T | 20 |
| KNN neighbors k | 3 |
| Extreme amplification α | 3.0 |
| Extreme fraction p | 0.05 |
| Threshold quantile | 0.95 (per-machine) |
| FedMAV momentum β | 0.9 |
| FedMAV global lr η_g | 0.01 |
| FedMAV ε | 1e-8 |
| DP clipping norm C | 1.0 |
| Random seed | 42 |

---


---

## Future Work

- **Dynamic graph reconstruction** — Rebuild the KNN graph per FL round as the global model improves.
- **Asynchronous federation** — FedMAV with staleness correction for machines with varying compute capacity.
- **Cross-dataset generalization** — Evaluation on SWAT, WADI, and MSL-SMAP benchmarks.
- **Formal convergence analysis** — Theoretical convergence rate bound for FedMAV under non-IID conditions.
- **Hierarchical federation** — Two-tier aggregation aligned with SMD's three machine groups.

---
