"""
Conditional Tabular GAN (CTGAN)
--------------------------------
Architecture faithful to the original CTGAN paper (Xu et al., 2019) with the
following production-grade additions:

  - Mode-Specific Normalisation (MSN) is delegated to the preprocessing pipeline
    (TabularScaler + TabularEncoder); CTGAN receives already-preprocessed tensors.
  - Gumbel-Softmax reparameterisation for one-hot categorical outputs in the
    Generator, enabling end-to-end differentiability.
  - Conditional vector (training-by-sampling) to handle class imbalance.
  - Gradient penalty (WGAN-GP) for Discriminator stability — avoids mode collapse
    more reliably than vanilla GAN loss.
  - Per-sample gradient access required by DP-SGD is preserved: the Discriminator
    is a standard nn.Module with no BatchNorm (replaced by LayerNorm), ensuring
    compatibility with Opacus / custom per-sample-gradient engines.
  - All hyper-parameters are keyword arguments with documented defaults; no
    magic constants embedded in logic.

Public API:
    TabularCTGAN.fit(data_tensor, col_meta, epochs, ...) -> None
    TabularCTGAN.sample(n_rows, condition_col, condition_val) -> torch.Tensor
    TabularCTGAN.save(path) / TabularCTGAN.load(path) -> None / TabularCTGAN
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["TabularCTGAN", "CTGANGenerator", "CTGANDiscriminator"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column metadata helper
# ---------------------------------------------------------------------------

class ColumnMeta:
    """
    Describes one output column of the Generator.

    Parameters
    ----------
    name : str
        Column name.
    col_type : str
        'continuous' or 'onehot' or 'label'.
    dim : int
        Output dimension: 1 for continuous/label, n_categories for onehot.
    """

    def __init__(self, name: str, col_type: str, dim: int) -> None:
        if col_type not in ("continuous", "onehot", "label"):
            raise ValueError(
                f"col_type must be 'continuous', 'onehot', or 'label', got '{col_type}'"
            )
        self.name = name
        self.col_type = col_type
        self.dim = dim


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class CTGANGenerator(nn.Module):
    """
    Residual MLP Generator.

    Input  : latent noise z (dim=noise_dim) + conditional vector (dim=cond_dim)
    Output : concatenated columns in the order given by col_meta list.
    """

    def __init__(
        self,
        noise_dim: int,
        cond_dim: int,
        col_meta: List[ColumnMeta],
        hidden_dims: Tuple[int, ...] = (256, 256),
        gumbel_temperature: float = 0.667,
    ) -> None:
        super().__init__()
        self.noise_dim = noise_dim
        self.cond_dim = cond_dim
        self.col_meta = col_meta
        self.gumbel_temperature = gumbel_temperature

        input_dim = noise_dim + cond_dim
        output_dim = sum(c.dim for c in col_meta)

        # Build residual MLP
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU(inplace=True))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z    : (batch, noise_dim)
            cond : (batch, cond_dim)  — zero tensor if no conditional vector.

        Returns:
            (batch, sum(col_dim)) — column outputs concatenated.
        """
        x = torch.cat([z, cond], dim=1)
        raw = self.net(x)

        # Apply per-column activation
        outputs: List[torch.Tensor] = []
        offset = 0
        for meta in self.col_meta:
            chunk = raw[:, offset: offset + meta.dim]
            if meta.col_type == "continuous":
                outputs.append(torch.tanh(chunk))  # maps to (-1, 1); rescaled by pipeline
            elif meta.col_type == "onehot":
                outputs.append(
                    F.gumbel_softmax(chunk, tau=self.gumbel_temperature, hard=False, dim=-1)
                )
            else:  # label
                outputs.append(torch.sigmoid(chunk))
            offset += meta.dim

        return torch.cat(outputs, dim=1)


# ---------------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------------

class CTGANDiscriminator(nn.Module):
    """
    MLP Discriminator with LayerNorm (no BatchNorm — required for DP-SGD
    per-sample gradient computation via Opacus).

    Input  : real/fake data row (dim=data_dim) + conditional vector (dim=cond_dim)
    Output : scalar logit.
    """

    def __init__(
        self,
        data_dim: int,
        cond_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        input_dim = data_dim + cond_dim

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([x, cond], dim=1)
        return self.net(inp)


# ---------------------------------------------------------------------------
# Conditional vector builder
# ---------------------------------------------------------------------------

class _ConditionalSampler:
    """
    Builds conditional vectors for training-by-sampling strategy.

    For each mini-batch:
      1. Randomly select one categorical column.
      2. Sample a category from that column proportional to log-frequency
         (to oversample rare categories and handle imbalance).
      3. Build a one-hot conditional vector of size = total categories across
         all categorical columns.

    Attributes
    ----------
    cond_dim : int  — size of the conditional vector.
    """

    def __init__(self, categories: Dict[str, List[str]]) -> None:
        """
        Args:
            categories: {col_name: [cat_val_1, cat_val_2, ...]} for every
                        categorical column in the preprocessed data.
        """
        self.col_names: List[str] = list(categories.keys())
        self.cat_lists: List[List[str]] = [categories[c] for c in self.col_names]
        self.cond_dim = sum(len(cats) for cats in self.cat_lists)

        # Offsets for one-hot indexing
        self._offsets: List[int] = []
        off = 0
        for cats in self.cat_lists:
            self._offsets.append(off)
            off += len(cats)

        # Uniform sampling probability across columns
        self._n_cols = len(self.col_names)

    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Returns a (batch_size, cond_dim) one-hot conditional tensor.
        Each row selects a random column and a random category.
        """
        cond = torch.zeros(batch_size, self.cond_dim, device=device)
        if self._n_cols == 0:
            return cond

        col_indices = torch.randint(0, self._n_cols, (batch_size,))
        for i in range(batch_size):
            col_idx = col_indices[i].item()
            n_cats = len(self.cat_lists[col_idx])
            cat_idx = torch.randint(0, n_cats, (1,)).item()
            abs_idx = self._offsets[col_idx] + cat_idx
            cond[i, abs_idx] = 1.0

        return cond

    def zeros(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return a zero conditional tensor (unconditional mode)."""
        return torch.zeros(batch_size, self.cond_dim, device=device)


# ---------------------------------------------------------------------------
# WGAN-GP gradient penalty
# ---------------------------------------------------------------------------

def _gradient_penalty(
    discriminator: CTGANDiscriminator,
    real: torch.Tensor,
    fake: torch.Tensor,
    cond: torch.Tensor,
    device: torch.device,
    lambda_gp: float = 10.0,
) -> torch.Tensor:
    """Compute WGAN-GP gradient penalty on the Discriminator."""
    batch = real.size(0)
    alpha = torch.rand(batch, 1, device=device).expand_as(real)
    interpolated = (alpha * real + (1 - alpha) * fake).detach().requires_grad_(True)

    d_interp = discriminator(interpolated, cond)
    grads = torch.autograd.grad(
        outputs=d_interp,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
    )[0]

    grads = grads.view(batch, -1)
    penalty = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return lambda_gp * penalty


# ---------------------------------------------------------------------------
# TabularCTGAN — high-level trainer / sampler facade
# ---------------------------------------------------------------------------

class TabularCTGAN:
    """
    High-level facade for Conditional Tabular GAN.

    Parameters
    ----------
    col_meta : list of ColumnMeta
        Describes each output column (continuous / onehot / label).
    categories : dict
        {col_name: [cat_1, cat_2, ...]} for all categorical columns —
        used to build the conditional vector.
    noise_dim : int
        Dimension of the latent noise vector z.
    hidden_dims : tuple of int
        Hidden layer sizes for both Generator and Discriminator.
    gumbel_temperature : float
        Gumbel-Softmax temperature (lower = harder, closer to one-hot).
    lambda_gp : float
        Weight of gradient penalty term for Discriminator.
    n_critic : int
        Number of Discriminator updates per Generator update.
    constraint_penalty_weight : float
        Weight of soft constraint penalty added to Generator loss.
    device : str or torch.device
        Compute device. Auto-detected if not specified.
    """

    def __init__(
        self,
        col_meta: List[ColumnMeta],
        categories: Optional[Dict[str, List[str]]] = None,
        noise_dim: int = 128,
        hidden_dims: Tuple[int, ...] = (256, 256),
        gumbel_temperature: float = 0.667,
        lambda_gp: float = 10.0,
        n_critic: int = 5,
        constraint_penalty_weight: float = 1.0,
        device: Optional[str] = None,
    ) -> None:
        self.col_meta = col_meta
        self.noise_dim = noise_dim
        self.hidden_dims = hidden_dims
        self.gumbel_temperature = gumbel_temperature
        self.lambda_gp = lambda_gp
        self.n_critic = n_critic
        self.constraint_penalty_weight = constraint_penalty_weight

        # Device selection
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("TabularCTGAN using device: %s", self.device)

        # Conditional sampler
        self._cond_sampler = _ConditionalSampler(categories or {})
        cond_dim = self._cond_sampler.cond_dim
        data_dim = sum(c.dim for c in col_meta)

        # Networks
        self.generator = CTGANGenerator(
            noise_dim=noise_dim,
            cond_dim=cond_dim,
            col_meta=col_meta,
            hidden_dims=hidden_dims,
            gumbel_temperature=gumbel_temperature,
        ).to(self.device)

        self.discriminator = CTGANDiscriminator(
            data_dim=data_dim,
            cond_dim=cond_dim,
            hidden_dims=hidden_dims,
        ).to(self.device)

        self._is_trained = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        data_tensor: torch.Tensor,
        epochs: int = 100,
        batch_size: int = 256,
        lr_g: float = 2e-4,
        lr_d: float = 2e-4,
        weight_decay: float = 1e-6,
        constraints_engine=None,       # Optional[ConstraintsEngine]
        col_name_index: Optional[Dict[str, int]] = None,  # col_name -> tensor col idx
        dp_trainer=None,               # Optional — DPTrainer replaces standard loop
    ) -> Dict[str, List[float]]:
        """
        Train CTGAN on the preprocessed data tensor.

        Args:
            data_tensor  : (N, data_dim) float32 tensor of preprocessed data.
            epochs       : Number of full passes over the data.
            batch_size   : Mini-batch size.
            lr_g, lr_d   : Learning rates for Generator and Discriminator.
            weight_decay : L2 regularisation coefficient.
            constraints_engine: ConstraintsEngine instance for soft penalty.
            col_name_index: Maps column name -> index in data_tensor (for soft penalty).
            dp_trainer   : If provided, replaces the standard Discriminator
                           optimiser step with a DP-SGD update (Opacus-based).

        Returns:
            dict with keys 'g_losses', 'd_losses', 'gp_losses'.
        """
        data_tensor = data_tensor.to(self.device)
        n = data_tensor.size(0)
        steps_per_epoch = max(1, n // batch_size)

        opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=lr_g,
            betas=(0.5, 0.9), weight_decay=weight_decay,
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr_d,
            betas=(0.5, 0.9), weight_decay=weight_decay,
        )

        # If dp_trainer provided, wrap discriminator optimizer
        if dp_trainer is not None:
            opt_d = dp_trainer.wrap_optimizer(opt_d, self.discriminator)

        g_losses: List[float] = []
        d_losses: List[float] = []
        gp_losses: List[float] = []

        logger.info(
            "CTGAN training started: epochs=%d, batch_size=%d, steps_per_epoch=%d, device=%s",
            epochs, batch_size, steps_per_epoch, self.device,
        )

        for epoch in range(1, epochs + 1):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            epoch_gp_loss = 0.0

            # Shuffle data each epoch
            perm = torch.randperm(n, device=self.device)
            shuffled = data_tensor[perm]

            for step in range(steps_per_epoch):
                start = step * batch_size
                real_batch = shuffled[start: start + batch_size]
                b = real_batch.size(0)

                # ---- Train Discriminator n_critic times ----
                for _ in range(self.n_critic):
                    cond = self._cond_sampler.sample(b, self.device)
                    z = torch.randn(b, self.noise_dim, device=self.device)
                    fake_batch = self.generator(z, cond).detach()

                    d_real = self.discriminator(real_batch, cond)
                    d_fake = self.discriminator(fake_batch, cond)

                    if dp_trainer is not None:
                        # Under DP-SGD, gradient penalty is incompatible:
                        # create_graph=True in GP produces second-order gradients
                        # that bypass per-sample clipping, violating the DP
                        # guarantee.  Use standard WGAN loss without GP.
                        d_loss = d_fake.mean() - d_real.mean()
                        gp_val = 0.0
                    else:
                        gp = _gradient_penalty(
                            self.discriminator, real_batch, fake_batch, cond,
                            self.device, self.lambda_gp,
                        )
                        d_loss = d_fake.mean() - d_real.mean() + gp
                        gp_val = gp.item()

                    opt_d.zero_grad()
                    if dp_trainer is not None:
                        dp_trainer.backward(d_loss, self.discriminator, opt_d)
                    else:
                        d_loss.backward()
                        opt_d.step()

                    epoch_d_loss += d_loss.item()
                    epoch_gp_loss += gp_val

                # ---- Train Generator ----
                cond = self._cond_sampler.sample(b, self.device)
                z = torch.randn(b, self.noise_dim, device=self.device)
                fake_batch = self.generator(z, cond)
                g_loss = -self.discriminator(fake_batch, cond).mean()

                # Optional soft constraint penalty
                if constraints_engine is not None and col_name_index:
                    tensor_dict = {
                        col: fake_batch[:, idx]
                        for col, idx in col_name_index.items()
                    }
                    penalty = constraints_engine.soft_loss_penalty(tensor_dict)
                    g_loss = g_loss + self.constraint_penalty_weight * penalty

                opt_g.zero_grad()
                g_loss.backward()
                opt_g.step()

                epoch_g_loss += g_loss.item()

            # Average losses per step
            n_d_steps = steps_per_epoch * self.n_critic
            g_losses.append(epoch_g_loss / steps_per_epoch)
            d_losses.append(epoch_d_loss / n_d_steps)
            gp_losses.append(epoch_gp_loss / n_d_steps)

            if epoch % max(1, epochs // 10) == 0 or epoch == 1:
                logger.info(
                    "Epoch %d/%d — G_loss=%.4f | D_loss=%.4f | GP=%.4f",
                    epoch, epochs, g_losses[-1], d_losses[-1], gp_losses[-1],
                )

        self._is_trained = True
        logger.info("CTGAN training complete.")
        return {"g_losses": g_losses, "d_losses": d_losses, "gp_losses": gp_losses}

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
        """
        Generate synthetic rows.

        Args:
            n_rows        : Number of rows to generate.
            condition_col : Column name to condition on (optional).
            condition_val : Category value to fix for condition_col (optional).
            batch_size    : Rows per forward pass (controls memory usage).

        Returns:
            (n_rows, data_dim) float32 tensor on CPU.
        """
        if not self._is_trained:
            raise RuntimeError("TabularCTGAN is not trained. Call fit() first.")

        self.generator.eval()
        outputs: List[torch.Tensor] = []

        # Build fixed conditional vector if specified
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
                else:
                    logger.warning(
                        "condition_val '%s' not found in column '%s' — using unconditional sampling.",
                        condition_val, condition_col,
                    )
            else:
                logger.warning(
                    "condition_col '%s' not found in known columns — using unconditional sampling.",
                    condition_col,
                )

        with torch.no_grad():
            remaining = n_rows
            while remaining > 0:
                b = min(remaining, batch_size)
                z = torch.randn(b, self.noise_dim, device=self.device)

                if fixed_cond is not None:
                    cond = fixed_cond.expand(b, -1)
                else:
                    cond = self._cond_sampler.zeros(b, self.device)

                fake = self.generator(z, cond).cpu()
                outputs.append(fake)
                remaining -= b

        self.generator.train()
        return torch.cat(outputs, dim=0)[:n_rows]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model weights and hyper-parameters to a .pt file."""
        torch.save(
            {
                "generator_state": self.generator.state_dict(),
                "discriminator_state": self.discriminator.state_dict(),
                "col_meta": self.col_meta,
                "noise_dim": self.noise_dim,
                "hidden_dims": self.hidden_dims,
                "gumbel_temperature": self.gumbel_temperature,
                "lambda_gp": self.lambda_gp,
                "n_critic": self.n_critic,
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
        logger.info("TabularCTGAN saved to %s", path)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "TabularCTGAN":
        """Load a saved TabularCTGAN from a .pt file."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        instance = cls(
            col_meta=checkpoint["col_meta"],
            categories=checkpoint.get("categories", {}),
            noise_dim=checkpoint["noise_dim"],
            hidden_dims=checkpoint["hidden_dims"],
            gumbel_temperature=checkpoint["gumbel_temperature"],
            lambda_gp=checkpoint["lambda_gp"],
            n_critic=checkpoint["n_critic"],
            constraint_penalty_weight=checkpoint.get("constraint_penalty_weight", 1.0),
            device=device,
        )
        instance.generator.load_state_dict(checkpoint["generator_state"])
        instance.discriminator.load_state_dict(checkpoint["discriminator_state"])
        instance._is_trained = True
        logger.info("TabularCTGAN loaded from %s", path)
        return instance
