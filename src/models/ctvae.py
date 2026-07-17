"""
Conditional Tabular Variational Autoencoder (CTVAE)
-----------------------------------------------------
Architecture:
  - Encoder MLP: (data_dim + cond_dim) -> (mu, log_var) in latent space.
  - Decoder MLP: (latent_dim + cond_dim) -> reconstructed data.
  - Loss = Reconstruction Loss (MSE for continuous, BCE for onehot/label)
           + beta * KL Divergence.
  - Reparameterisation trick for sampling z ~ N(mu, sigma^2).

Production considerations:
  - No BatchNorm in Encoder or Decoder — uses LayerNorm for DP-SGD compatibility.
  - beta parameter (beta-VAE) controls KL weight; set to 1.0 for standard VAE.
  - Conditional vector identical to CTGAN: one-hot across all categorical columns.
  - Training is stable by design (no GAN saddle-point problem).
  - Generates by sampling z ~ N(0, I) + optional conditional vector.

Public API (mirrors TabularCTGAN for interoperability):
    TabularCTVAE.fit(data_tensor, col_meta, ...) -> dict
    TabularCTVAE.sample(n_rows, ...) -> torch.Tensor
    TabularCTVAE.save(path) / TabularCTVAE.load(path)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.ctgan import ColumnMeta, _ConditionalSampler

__all__ = ["TabularCTVAE", "CTVAEEncoder", "CTVAEDecoder"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class CTVAEEncoder(nn.Module):
    """
    MLP Encoder: maps (data, cond) -> (mu, log_var) in latent space.
    Uses LayerNorm throughout for DP-SGD compatibility.
    """

    def __init__(
        self,
        data_dim: int,
        cond_dim: int,
        latent_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()
        input_dim = data_dim + cond_dim

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), nn.LayerNorm(h), nn.ReLU(inplace=True)]
            prev_dim = h

        self.shared = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_log_var = nn.Linear(prev_dim, latent_dim)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(torch.cat([x, cond], dim=1))
        return self.fc_mu(h), self.fc_log_var(h)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class CTVAEDecoder(nn.Module):
    """
    MLP Decoder: maps (z, cond) -> reconstructed data, applying per-column
    activations (tanh for continuous, sigmoid for label, softmax for onehot).
    """

    def __init__(
        self,
        latent_dim: int,
        cond_dim: int,
        col_meta: List[ColumnMeta],
        hidden_dims: Tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()
        self.col_meta = col_meta
        input_dim = latent_dim + cond_dim
        output_dim = sum(c.dim for c in col_meta)

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), nn.LayerNorm(h), nn.ReLU(inplace=True)]
            prev_dim = h

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstructed data.

        Returns **raw logits** for onehot columns so that
        ``F.cross_entropy`` (which internally applies ``log_softmax``)
        receives the correct input during training.  Softmax is applied
        only at sampling time via :meth:`TabularCTVAE.sample`.
        """
        raw = self.net(torch.cat([z, cond], dim=1))

        outputs: List[torch.Tensor] = []
        offset = 0
        for meta in self.col_meta:
            chunk = raw[:, offset: offset + meta.dim]
            if meta.col_type == "continuous":
                outputs.append(torch.tanh(chunk))
            elif meta.col_type == "onehot":
                # Return raw logits — do NOT apply softmax here.
                # F.cross_entropy in _elbo_loss expects unnormalised logits.
                outputs.append(chunk)
            else:  # label
                outputs.append(torch.sigmoid(chunk))
            offset += meta.dim

        return torch.cat(outputs, dim=1)


# ---------------------------------------------------------------------------
# Reparameterisation
# ---------------------------------------------------------------------------

def _reparameterise(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """z = mu + eps * sigma  (eps ~ N(0, I))."""
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std


# ---------------------------------------------------------------------------
# ELBO Loss
# ---------------------------------------------------------------------------

def _elbo_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    col_meta: List[ColumnMeta],
    beta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute ELBO = Reconstruction Loss + beta * KL divergence.

    Reconstruction per column type:
      - continuous: MSE
      - onehot    : CrossEntropy (target is argmax of one-hot chunk)
      - label     : BCE

    Returns:
        total_loss, recon_loss, kl_loss (all scalar tensors).
    """
    recon_loss = torch.tensor(0.0, device=recon.device)
    offset = 0
    for meta in col_meta:
        r_chunk = recon[:, offset: offset + meta.dim]
        t_chunk = target[:, offset: offset + meta.dim]

        if meta.col_type == "continuous":
            recon_loss = recon_loss + F.mse_loss(r_chunk, t_chunk)
        elif meta.col_type == "onehot":
            # Cross-entropy: target is index of the 1 in the one-hot
            target_idx = t_chunk.argmax(dim=1)
            recon_loss = recon_loss + F.cross_entropy(r_chunk, target_idx)
        else:  # label
            recon_loss = recon_loss + F.binary_cross_entropy(r_chunk, t_chunk.clamp(0, 1))

        offset += meta.dim

    # KL: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

    return recon_loss + beta * kl_loss, recon_loss, kl_loss


# ---------------------------------------------------------------------------
# TabularCTVAE — high-level facade
# ---------------------------------------------------------------------------

class TabularCTVAE:
    """
    High-level facade for Conditional Tabular VAE.

    Parameters
    ----------
    col_meta : list of ColumnMeta
    categories : dict {col_name: [cat_val, ...]} for categorical columns.
    latent_dim : int — latent space dimensionality.
    hidden_dims : tuple of int — hidden layer sizes.
    beta : float — weight of KL term in ELBO (1.0 = standard VAE).
    constraint_penalty_weight : float — weight of soft constraint penalty.
    device : str or None — auto-detected if None.
    """

    def __init__(
        self,
        col_meta: List[ColumnMeta],
        categories: Optional[Dict[str, List[str]]] = None,
        latent_dim: int = 128,
        hidden_dims: Tuple[int, ...] = (256, 256),
        beta: float = 1.0,
        constraint_penalty_weight: float = 1.0,
        device: Optional[str] = None,
    ) -> None:
        self.col_meta = col_meta
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.beta = beta
        self.constraint_penalty_weight = constraint_penalty_weight

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        logger.info("TabularCTVAE using device: %s", self.device)

        self._cond_sampler = _ConditionalSampler(categories or {})
        cond_dim = self._cond_sampler.cond_dim
        data_dim = sum(c.dim for c in col_meta)

        self.encoder = CTVAEEncoder(data_dim, cond_dim, latent_dim, hidden_dims).to(self.device)
        self.decoder = CTVAEDecoder(latent_dim, cond_dim, col_meta, hidden_dims).to(self.device)

        self._is_trained = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        data_tensor: torch.Tensor,
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 2e-4,
        weight_decay: float = 1e-6,
        constraints_engine=None,
        col_name_index: Optional[Dict[str, int]] = None,
        dp_trainer=None,
        early_stopping_patience: int = 0,
    ) -> Dict[str, List[float]]:
        """
        Train the CTVAE.

        Args:
            data_tensor          : (N, data_dim) float32 preprocessed tensor.
            epochs               : Training epochs.
            batch_size           : Mini-batch size.
            lr                   : Adam learning rate.
            weight_decay         : L2 regularisation.
            constraints_engine   : ConstraintsEngine for soft loss penalty.
            col_name_index       : {col_name: tensor_col_idx} for penalty mapping.
            dp_trainer           : DPTrainer instance (wraps optimiser for DP-SGD).
            early_stopping_patience: Epochs without improvement before stopping
                                     (0 = disabled; always disabled when dp_trainer set).

        Returns:
            dict with keys 'total_losses', 'recon_losses', 'kl_losses'.
        """
        data_tensor = data_tensor.to(self.device)
        n = data_tensor.size(0)
        steps_per_epoch = max(1, n // batch_size)

        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

        if dp_trainer is not None:
            # Wrap both encoder+decoder as a single module for DP
            combined = nn.ModuleList([self.encoder, self.decoder])
            optimizer = dp_trainer.wrap_optimizer(optimizer, combined)
            # Disable early stopping when DP is active
            early_stopping_patience = 0
            logger.info("DP-SGD active: early stopping disabled for CTVAE.")

        total_losses: List[float] = []
        recon_losses: List[float] = []
        kl_losses: List[float] = []

        best_loss = float("inf")
        patience_counter = 0

        logger.info(
            "CTVAE training started: epochs=%d, batch_size=%d, device=%s",
            epochs, batch_size, self.device,
        )

        for epoch in range(1, epochs + 1):
            perm = torch.randperm(n, device=self.device)
            shuffled = data_tensor[perm]

            ep_total = ep_recon = ep_kl = 0.0

            for step in range(steps_per_epoch):
                start = step * batch_size
                batch = shuffled[start: start + batch_size]
                b = batch.size(0)

                cond = self._cond_sampler.sample(b, self.device)
                mu, log_var = self.encoder(batch, cond)
                z = _reparameterise(mu, log_var)
                recon = self.decoder(z, cond)

                loss, rl, kl = _elbo_loss(recon, batch, mu, log_var, self.col_meta, self.beta)

                # Soft constraint penalty
                # Only continuous columns (dim=1 in col_meta) are passed to the
                # constraint penalty.  Onehot columns (dim > 1) have
                # col_name_index pointing to the *start* of the group; passing
                # only recon[:, idx] would extract only the first element,
                # yielding semantically meaningless penalties.
                if constraints_engine is not None and col_name_index:
                    tensor_dict = {}
                    for meta in self.col_meta:
                        if meta.dim == 1 and meta.name in col_name_index:
                            idx = col_name_index[meta.name]
                            tensor_dict[meta.name] = recon[:, idx]
                    if tensor_dict:
                        penalty = constraints_engine.soft_loss_penalty(tensor_dict)
                        loss = loss + self.constraint_penalty_weight * penalty

                optimizer.zero_grad()
                if dp_trainer is not None:
                    dp_trainer.backward(loss, combined, optimizer)
                else:
                    loss.backward()
                    optimizer.step()

                ep_total += loss.item()
                ep_recon += rl.item()
                ep_kl += kl.item()

            avg_total = ep_total / steps_per_epoch
            avg_recon = ep_recon / steps_per_epoch
            avg_kl = ep_kl / steps_per_epoch
            total_losses.append(avg_total)
            recon_losses.append(avg_recon)
            kl_losses.append(avg_kl)

            if epoch % max(1, epochs // 10) == 0 or epoch == 1:
                logger.info(
                    "Epoch %d/%d — Total=%.4f | Recon=%.4f | KL=%.4f",
                    epoch, epochs, avg_total, avg_recon, avg_kl,
                )

            # Early stopping (disabled under DP)
            if early_stopping_patience > 0:
                if avg_total < best_loss - 1e-6:
                    best_loss = avg_total
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        logger.info(
                            "Early stopping triggered at epoch %d (patience=%d).",
                            epoch, early_stopping_patience,
                        )
                        break

        self._is_trained = True
        logger.info("CTVAE training complete.")
        return {
            "total_losses": total_losses,
            "recon_losses": recon_losses,
            "kl_losses": kl_losses,
        }

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        n_rows: int,
        condition_col: Optional[str] = None,
        condition_val: Optional[str] = None,
        batch_size: int = 2048,
    ) -> torch.Tensor:
        """Generate synthetic rows by sampling z ~ N(0,I) and decoding."""
        if not self._is_trained:
            raise RuntimeError("TabularCTVAE is not trained. Call fit() first.")

        # Only decoder is used during sampling; encoder.eval() is not needed
        # but calling it is harmless. We keep decoder.eval() for correctness.
        self.decoder.eval()
        outputs: List[torch.Tensor] = []

        fixed_cond: Optional[torch.Tensor] = None
        if condition_col is not None and condition_val is not None:
            if condition_col in self._cond_sampler.col_names:
                col_idx = self._cond_sampler.col_names.index(condition_col)
                cats = self._cond_sampler.cat_lists[col_idx]
                if condition_val in cats:
                    cat_idx = cats.index(condition_val)
                    abs_idx = self._cond_sampler._offsets[col_idx] + cat_idx
                    fixed_cond = torch.zeros(1, self._cond_sampler.cond_dim, device=self.device)
                    fixed_cond[0, abs_idx] = 1.0

        with torch.no_grad():
            remaining = n_rows
            while remaining > 0:
                b = min(remaining, batch_size)
                z = torch.randn(b, self.latent_dim, device=self.device)
                cond = (
                    fixed_cond.expand(b, -1)
                    if fixed_cond is not None
                    else self._cond_sampler.zeros(b, self.device)
                )
                raw_recon = self.decoder(z, cond)

                # Post-hoc softmax for onehot columns (decoder returns
                # raw logits during training for correct cross-entropy).
                processed: List[torch.Tensor] = []
                offset = 0
                for meta in self.col_meta:
                    chunk = raw_recon[:, offset: offset + meta.dim]
                    if meta.col_type == "onehot":
                        chunk = F.softmax(chunk, dim=-1)
                    processed.append(chunk)
                    offset += meta.dim

                outputs.append(torch.cat(processed, dim=1).cpu())
                remaining -= b

        self.encoder.train()
        self.decoder.train()
        return torch.cat(outputs, dim=0)[:n_rows]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save(
            {
                "encoder_state": self.encoder.state_dict(),
                "decoder_state": self.decoder.state_dict(),
                "col_meta": self.col_meta,
                "latent_dim": self.latent_dim,
                "hidden_dims": self.hidden_dims,
                "beta": self.beta,
                "constraint_penalty_weight": self.constraint_penalty_weight,
                "categories": {
                    col: cats
                    for col, cats in zip(
                        self._cond_sampler.col_names, self._cond_sampler.cat_lists
                    )
                },
            },
            path,
        )
        logger.info("TabularCTVAE saved to %s", path)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "TabularCTVAE":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        instance = cls(
            col_meta=ckpt["col_meta"],
            categories=ckpt.get("categories", {}),
            latent_dim=ckpt["latent_dim"],
            hidden_dims=ckpt["hidden_dims"],
            beta=ckpt["beta"],
            constraint_penalty_weight=ckpt.get("constraint_penalty_weight", 1.0),
            device=device,
        )
        instance.encoder.load_state_dict(ckpt["encoder_state"])
        instance.decoder.load_state_dict(ckpt["decoder_state"])
        instance._is_trained = True
        logger.info("TabularCTVAE loaded from %s", path)
        return instance
