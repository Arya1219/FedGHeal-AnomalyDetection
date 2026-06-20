# models.py
"""
Model Architectures for FedRIVER — SMD Dataset
-----------------------------------------------
Contains three model classes:

    1. Autoencoder (AE)
       Plain MLP encoder-decoder. No graph structure.
       Baseline for centralized and federated experiments.

    2. VariationalAutoencoder (VAE)
       MLP-based VAE with reparameterization trick.
       Baseline — tests whether probabilistic latent space helps.

    3. GraphAutoencoder (GAE)
       GCN encoder (2 message-passing layers) + MLP decoder.
       Main model — exploits graph structure between timesteps.

All three follow the same interface convention:
    forward(input) -> (reconstruction, latent)

For VAE forward returns (reconstruction, mu, logvar).
Use vae_loss() for VAE training loss.

Architecture dimensions (matching your original code):
    Input  : 38
    Hidden1: 26
    Hidden2: 16
    Latent :  8
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


# ─────────────────────────────────────────────
#  1. Autoencoder (AE)
# ─────────────────────────────────────────────

class Autoencoder(nn.Module):
    """
    Plain MLP Autoencoder.
    Input: raw node feature tensor  [N, 38]
    Does NOT use graph structure.
    """

    def __init__(
        self,
        input_dim:   int = 38,
        hidden1:     int = 26,
        hidden2:     int = 16,
        latent_dim:  int = 8,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, input_dim),
        )

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : [N, 38] float tensor  (raw features, NOT a PyG Data object)

        Returns
        -------
        x_hat : [N, 38]  reconstruction
        z     : [N,  8]  latent representation
        """
        z     = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


# ─────────────────────────────────────────────
#  2. Variational Autoencoder (VAE)
# ─────────────────────────────────────────────

class VariationalAutoencoder(nn.Module):
    """
    MLP Variational Autoencoder.
    Input: raw node feature tensor  [N, 38]
    Does NOT use graph structure.

    forward() returns (reconstruction, mu, logvar).
    Use vae_loss() below to compute training loss.
    """

    def __init__(
        self,
        input_dim:   int = 38,
        hidden1:     int = 26,
        hidden2:     int = 16,
        latent_dim:  int = 8,
    ):
        super().__init__()

        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1,   hidden2)

        # Latent distribution parameters
        self.fc_mu     = nn.Linear(hidden2, latent_dim)
        self.fc_logvar = nn.Linear(hidden2, latent_dim)

        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden2)
        self.fc4 = nn.Linear(hidden2,    hidden1)
        self.fc5 = nn.Linear(hidden1,    input_dim)

    def encode(self, x: torch.Tensor):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu   # deterministic at eval time

    def decode(self, z: torch.Tensor):
        h = F.relu(self.fc3(z))
        h = F.relu(self.fc4(h))
        return self.fc5(h)

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : [N, 38] float tensor

        Returns
        -------
        x_hat  : [N, 38]  reconstruction
        mu     : [N,  8]  latent mean
        logvar : [N,  8]  latent log-variance
        """
        mu, logvar = self.encode(x)
        z          = self.reparameterize(mu, logvar)
        x_hat      = self.decode(z)
        return x_hat, mu, logvar


def vae_loss(
    x_hat:  torch.Tensor,
    x:      torch.Tensor,
    mu:     torch.Tensor,
    logvar: torch.Tensor,
    beta:   float = 1.0,
) -> torch.Tensor:
    """
    VAE loss = reconstruction MSE + beta * KL divergence.

    beta=1  → standard VAE
    beta<1  → emphasise reconstruction (better for anomaly detection)
    """
    recon = F.mse_loss(x_hat, x, reduction='mean')
    kl    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl


# ─────────────────────────────────────────────
#  3. Graph Autoencoder (GAE)
# ─────────────────────────────────────────────

class GraphAutoencoder(nn.Module):
    """
    Graph Autoencoder with GCN encoder and MLP decoder.
    Input: PyG Data object (uses data.x and data.edge_index).

    Encoder: GCNConv(38->26) -> GCNConv(26->16) -> Linear(16->8)
    Decoder: MLP(8->16->26->38)

    The GCN layers aggregate neighbor information,
    so each node's latent code reflects its local graph context.
    This is the key advantage over plain AE for anomaly detection:
    anomalous nodes will have neighbors that reconstruct differently,
    making their own reconstruction error higher.
    """

    def __init__(
        self,
        in_channels:     int = 38,
        hidden_channels: int = 26,
        out_channels_1:  int = 16,
        latent_dim:      int = 8,
    ):
        super().__init__()

        # GCN encoder
        self.encoder1 = GCNConv(in_channels,     hidden_channels)
        self.encoder2 = GCNConv(hidden_channels, out_channels_1)
        self.encoder3 = nn.Linear(out_channels_1, latent_dim)

        # MLP decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim,     out_channels_1),
            nn.ReLU(),
            nn.Linear(out_channels_1, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, in_channels),
        )

    def forward(self, data):
        """
        Parameters
        ----------
        data : torch_geometric.data.Data
               Must have  data.x  and  data.edge_index

        Returns
        -------
        x_rec  : [N, 38]  reconstruction
        latent : [N,  8]  latent representation
        """
        x, edge_index = data.x, data.edge_index

        x      = F.relu(self.encoder1(x, edge_index))
        x      = F.relu(self.encoder2(x, edge_index))
        latent = self.encoder3(x)

        x_rec  = self.decoder(latent)
        return x_rec, latent


# ─────────────────────────────────────────────
#  Smoke test
# ─────────────────────────────────────────────

if __name__ == '__main__':
    from torch_geometric.data import Data

    print('=' * 55)
    print('models.py — smoke test')
    print('=' * 55)

    N = 100   # dummy nodes

    # ── AE ──────────────────────────────────
    ae  = Autoencoder()
    x   = torch.randn(N, 38)
    out, z = ae(x)
    print(f'\nAutoencoder')
    print(f'  input  : {x.shape}')
    print(f'  output : {out.shape}')
    print(f'  latent : {z.shape}')
    assert out.shape == (N, 38)
    assert z.shape   == (N,  8)
    print('  ✅ OK')

    # ── VAE ─────────────────────────────────
    vae = VariationalAutoencoder()
    out, mu, logvar = vae(x)
    loss = vae_loss(out, x, mu, logvar)
    print(f'\nVariationalAutoencoder')
    print(f'  input  : {x.shape}')
    print(f'  output : {out.shape}')
    print(f'  mu     : {mu.shape}')
    print(f'  logvar : {logvar.shape}')
    print(f'  loss   : {loss.item():.4f}')
    assert out.shape    == (N, 38)
    assert mu.shape     == (N,  8)
    assert logvar.shape == (N,  8)
    print('  ✅ OK')

    # ── GAE ─────────────────────────────────
    gae = GraphAutoencoder()
    # dummy graph: ring topology
    src = list(range(N)) + list(range(1, N)) + [0]
    dst = list(range(1, N)) + [0] + list(range(N))
    edge_index = torch.tensor([src[:N], dst[:N]], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)
    out, latent = gae(data)
    print(f'\nGraphAutoencoder')
    print(f'  input  : {data.x.shape}')
    print(f'  output : {out.shape}')
    print(f'  latent : {latent.shape}')
    assert out.shape    == (N, 38)
    assert latent.shape == (N,  8)
    print('  ✅ OK')

    print('\nAll models passed.')
