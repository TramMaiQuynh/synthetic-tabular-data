"""
Tabular Diffusion Model
------------------------
Implements a DDPM-style (Ho et al., 2020) denoising diffusion probabilistic model
adapted for mixed-type tabular data.

Architecture decisions:
  - Denoising network: time-conditioned MLP (sinusoidal time embedding).
  - Forward process: linear beta schedule, T timesteps.
  - Reverse process: DDPM reverse step (predicts noise epsilon at each step).
  - Data dimensionality: same flattened preprocessed tensor as CTGAN/CTVAE.
  - Conditional generation: concatenate conditional vector to network input.
  - No BatchNorm — uses LayerNorm for DP-SGD compatibility.
  - Guided Sampling for constraints: constraint soft-gradient projected onto
    denoising trajectory (classifier-free guidance approximation via penalty).

References:
  Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020.
  Kotelnikov et al., "TabDDPM: Modelling Tabular Data with Diffusion Models", 2022.

Public API:
    TabularDiffusion.fit(data_tensor, ...) -> dict
    TabularDiffusion.sample(n_rows, ...) -> torch.Tensor
    TabularDiffusion.save(path) / TabularDiffusion.load(path)
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.ctgan import ColumnMeta, _ConditionalSampler

__all__ = ["TabularDiffusion"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sinusoidal time embedding
# ---------------------------------------------------------------------------

class _SinusoidalTimeEmbedding(nn.Module):
    """Maps integer timestep t -> (batch, embed_dim) sinusoidal embedding."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (batch,) integer timesteps.
        Returns:
            (batch, embed_dim) float embeddings.
        """
        half = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


# ---------------------------------------------------------------------------
# Denoising MLP (U-Net equivalent for tabular)
# ---------------------------------------------------------------------------

class _DenoisingMLP(nn.Module):
    """
    Time-conditioned MLP denoising network.
    Input: noisy data x_t + time embedding + conditional vector.
    Output: predicted noise epsilon (same dim as data).
    Uses LayerNorm (no BatchNorm) for DP-SGD compatibility.
    """

    def __init__(
        self,
        data_dim: int,
        cond_dim: int,
        hidden_dims: Tuple[int, ...],
        time_embed_dim: int = 128,
    ) -> None:
        super().__init__()
        self.time_embed = _SinusoidalTimeEmbedding(time_embed_dim)
        input_dim = data_dim + time_embed_dim + cond_dim

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), nn.LayerNorm(h), nn.SiLU()]
            prev_dim = h

        layers.append(nn.Linear(prev_dim, data_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        t_emb = self.time_embed(t)
        inp = torch.cat([x_t, t_emb, cond], dim=1)
        return self.net(inp)


# ---------------------------------------------------------------------------
# Beta schedule helpers
# ---------------------------------------------------------------------------

def _linear_beta_schedule(T: int, beta_start: float, beta_end: float) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, T)


def _cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule (Nichol & Dhariwal, 2021) — less aggressive noise early."""
    steps = T + 1
    x = torch.linspace(0, T, steps) / T
    alphas_cumprod = torch.cos((x + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(0.0001, 0.9999)


# ---------------------------------------------------------------------------
# TabularDiffusion — high-level facade
# ---------------------------------------------------------------------------

class TabularDiffusion:
    """
    High-level facade for the Tabular Diffusion Model.

    Parameters
    ----------
    col_meta : list of ColumnMeta
        Describes output columns (used for output shape).
    categories : dict {col_name: [cat_val, ...]}
        For building the conditional vector.
    T : int
        Number of diffusion timesteps.
    beta_schedule : str
        'linear' or 'cosine'.
    beta_start, beta_end : float
        Beta schedule bounds (only for linear schedule).
    hidden_dims : tuple of int
        Hidden layer sizes for the denoising MLP.
    time_embed_dim : int
        Sinusoidal time embedding dimension.
    device : str or None
    """

    def __init__(
        self,
        col_meta: List[ColumnMeta],
        categories: Optional[Dict[str, List[str]]] = None,
        T: int = 1000,
        beta_schedule: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        hidden_dims: Tuple[int, ...] = (512, 512, 256),
        time_embed_dim: int = 128,
        constraint_penalty_weight: float = 1.0,
        device: Optional[str] = None,
    ) -> None:
        self.col_meta = col_meta
        self.T = T
        self.beta_schedule = beta_schedule
        self.hidden_dims = hidden_dims
        self.time_embed_dim = time_embed_dim
        self.constraint_penalty_weight = constraint_penalty_weight
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        logger.info("TabularDiffusion using device: %s", self.device)

        self._cond_sampler = _ConditionalSampler(categories or {})
        cond_dim = self._cond_sampler.cond_dim
        self.data_dim = sum(c.dim for c in col_meta)

        # Build denoising network
        self.denoiser = _DenoisingMLP(
            data_dim=self.data_dim,
            cond_dim=cond_dim,
            hidden_dims=hidden_dims,
            time_embed_dim=time_embed_dim,
        ).to(self.device)

        # Pre-compute diffusion constants and move to device
        self._build_diffusion_constants(T, beta_schedule, beta_start, beta_end)
        self._is_trained = False

    def _build_diffusion_constants(
        self,
        T: int,
        schedule: str,
        beta_start: float,
        beta_end: float,
    ) -> None:
        """Pre-compute and register all diffusion schedule constants."""
        if schedule == "cosine":
            betas = _cosine_beta_schedule(T)
        else:
            betas = _linear_beta_schedule(T, beta_start, beta_end)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register as buffers (serialised with model, moved with .to(device))
        # We store as plain tensors on the device instead of nn.Module buffers
        # (facade class, not nn.Module)
        self._betas = betas.to(self.device)
        self._alphas = alphas.to(self.device)
        self._alphas_cumprod = alphas_cumprod.to(self.device)
        self._alphas_cumprod_prev = alphas_cumprod_prev.to(self.device)
        self._sqrt_alphas_cumprod = alphas_cumprod.sqrt().to(self.device)
        self._sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod).sqrt().to(self.device)
        self._posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ).to(self.device)

    # ------------------------------------------------------------------
    # Forward diffusion (q process)
    # ------------------------------------------------------------------

    def _q_sample(
        self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample x_t given x_0 and timestep t.
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps

        Returns:
            x_t, noise
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_ab = self._sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_one_minus_ab = self._sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        return sqrt_ab * x_0 + sqrt_one_minus_ab * noise, noise

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
        Train the Tabular Diffusion model (simple L2 loss on predicted noise).

        Args:
            data_tensor          : (N, data_dim) float32 preprocessed tensor.
            epochs               : Number of training epochs.
            batch_size           : Mini-batch size.
            lr                   : Adam learning rate.
            weight_decay         : L2 regularisation.
            constraints_engine   : ConstraintsEngine — penalty on denoised output
                                   (Guided Sampling approximation during training).
            col_name_index       : {col_name: tensor_col_idx}.
            dp_trainer           : DPTrainer for DP-SGD (disables early stopping).
            early_stopping_patience: 0 = disabled.

        Returns:
            dict with key 'losses'.
        """
        data_tensor = data_tensor.to(self.device)
        n = data_tensor.size(0)
        steps_per_epoch = max(1, n // batch_size)

        optimizer = torch.optim.Adam(
            self.denoiser.parameters(), lr=lr, weight_decay=weight_decay
        )

        if dp_trainer is not None:
            optimizer = dp_trainer.wrap_optimizer(optimizer, self.denoiser)
            early_stopping_patience = 0
            logger.info("DP-SGD active: early stopping disabled for TabularDiffusion.")

        losses: List[float] = []
        best_loss = float("inf")
        patience_counter = 0

        logger.info(
            "TabularDiffusion training: epochs=%d, batch_size=%d, T=%d, device=%s",
            epochs, batch_size, self.T, self.device,
        )

        for epoch in range(1, epochs + 1):
            perm = torch.randperm(n, device=self.device)
            shuffled = data_tensor[perm]
            ep_loss = 0.0

            for step in range(steps_per_epoch):
                start = step * batch_size
                x_0 = shuffled[start: start + batch_size]
                b = x_0.size(0)

                # Sample random timesteps uniformly
                t = torch.randint(0, self.T, (b,), device=self.device)
                cond = self._cond_sampler.sample(b, self.device)

                # Forward diffusion
                x_t, noise = self._q_sample(x_0, t)

                # Predict noise
                noise_pred = self.denoiser(x_t, t, cond)
                loss = F.mse_loss(noise_pred, noise)

                # Optional guided-sampling constraint penalty
                # IMPORTANT: Only continuous columns (dim=1 in col_meta) are
                # passed to the constraint penalty. Onehot columns (dim > 1)
                # have col_name_index pointing to the *start* of the group;
                # passing only x0_approx[:, idx] would extract only the first
                # element, yielding semantically meaningless penalties.
                if constraints_engine is not None and col_name_index:
                    # Approximate x_0 from current prediction (for penalty signal)
                    sqrt_ab = self._sqrt_alphas_cumprod[t].unsqueeze(-1)
                    sqrt_one_minus_ab = self._sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
                    x0_approx = (x_t - sqrt_one_minus_ab * noise_pred) / sqrt_ab.clamp(min=1e-6)
                    tensor_dict = {}
                    for meta in self.col_meta:
                        if meta.dim == 1 and meta.name in col_name_index:
                            idx = col_name_index[meta.name]
                            tensor_dict[meta.name] = x0_approx[:, idx]
                    if tensor_dict:
                        penalty = constraints_engine.soft_loss_penalty(tensor_dict)
                        loss = loss + self.constraint_penalty_weight * penalty

                optimizer.zero_grad()
                if dp_trainer is not None:
                    dp_trainer.backward(loss, self.denoiser, optimizer)
                else:
                    loss.backward()
                    optimizer.step()

                ep_loss += loss.item()

            avg_loss = ep_loss / steps_per_epoch
            losses.append(avg_loss)

            if epoch % max(1, epochs // 10) == 0 or epoch == 1:
                logger.info("Epoch %d/%d — Loss=%.6f", epoch, epochs, avg_loss)

            if early_stopping_patience > 0:
                if avg_loss < best_loss - 1e-7:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        logger.info(
                            "Early stopping at epoch %d (patience=%d).",
                            epoch, early_stopping_patience,
                        )
                        break

        self._is_trained = True
        logger.info("TabularDiffusion training complete.")
        return {"losses": losses}

    # ------------------------------------------------------------------
    # Reverse process (sampling)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        n_rows: int,
        condition_col: Optional[str] = None,
        condition_val: Optional[str] = None,
        batch_size: int = 512,
    ) -> torch.Tensor:
        """
        Generate synthetic rows via iterative reverse denoising.

        Args:
            n_rows        : Number of rows to generate.
            condition_col : Column name to condition on (optional).
            condition_val : Category value (optional).
            batch_size    : Rows per reverse-diffusion pass.

        Returns:
            (n_rows, data_dim) float32 tensor on CPU.
        """
        if not self._is_trained:
            raise RuntimeError("TabularDiffusion is not trained. Call fit() first.")

        self.denoiser.eval()
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

        remaining = n_rows
        while remaining > 0:
            b = min(remaining, batch_size)
            cond = (
                fixed_cond.expand(b, -1)
                if fixed_cond is not None
                else self._cond_sampler.zeros(b, self.device)
            )

            # Start from pure noise
            x_t = torch.randn(b, self.data_dim, device=self.device)

            # Iterative reverse denoising T -> 0 with clip_denoised to stabilize trajectory
            for t_val in reversed(range(self.T)):
                t = torch.full((b,), t_val, device=self.device, dtype=torch.long)
                eps_pred = self.denoiser(x_t, t, cond)

                beta_t = self._betas[t_val]
                alpha_t = self._alphas[t_val]
                alpha_bar_t = self._alphas_cumprod[t_val]
                alpha_bar_prev = self._alphas_cumprod_prev[t_val]

                # Reconstruct clean x_0 and clamp to training range [0, 1] to avoid out-of-distribution drift
                sqrt_recip_alpha_bar = 1.0 / alpha_bar_t.sqrt()
                sqrt_one_minus_alpha_bar = (1.0 - alpha_bar_t).sqrt()
                x0_approx = sqrt_recip_alpha_bar * x_t - (sqrt_recip_alpha_bar * sqrt_one_minus_alpha_bar) * eps_pred
                x0_approx = x0_approx.clamp(0.0, 1.0)

                # Compute posterior mean using the clamped x0_approx estimate
                coeff1 = (beta_t * alpha_bar_prev.sqrt()) / (1.0 - alpha_bar_t)
                coeff2 = (alpha_t.sqrt() * (1.0 - alpha_bar_prev)) / (1.0 - alpha_bar_t)
                predicted_mean = coeff1 * x0_approx + coeff2 * x_t

                if t_val > 0:
                    posterior_var = self._posterior_variance[t_val]
                    noise = torch.randn_like(x_t)
                    x_t = predicted_mean + posterior_var.sqrt() * noise
                else:
                    x_t = predicted_mean

            # Apply type-aware output activations on the GPU
            activated: List[torch.Tensor] = []
            offset = 0
            for meta in self.col_meta:
                chunk = x_t[:, offset : offset + meta.dim]
                if meta.col_type == "onehot":
                    # Softmax categorical distribution (matches CTGAN/CTVAE output activations)
                    activated.append(torch.softmax(chunk, dim=-1))
                elif meta.col_type == "label":
                    # Bounded label range, clamp strictly to [0, 1] (avoid non-linear compression)
                    activated.append(chunk.clamp(0.0, 1.0))
                else:  # continuous
                    if meta.name.endswith("_is_missing"):
                        # Binarize binary indicator by thresholding directly at 0.5
                        activated.append((chunk > 0.5).float())
                    else:
                        # Continuous feature normalization bounding
                        activated.append(chunk.clamp(0.0, 1.0))
                offset += meta.dim

            x_t_activated = torch.cat(activated, dim=1)
            outputs.append(x_t_activated.cpu())
            remaining -= b

        self.denoiser.train()
        return torch.cat(outputs, dim=0)[:n_rows]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save(
            {
                "denoiser_state": self.denoiser.state_dict(),
                "col_meta": self.col_meta,
                "T": self.T,
                "beta_schedule": self.beta_schedule,
                "beta_start": self.beta_start,
                "beta_end": self.beta_end,
                "constraint_penalty_weight": self.constraint_penalty_weight,
                "hidden_dims": self.hidden_dims,
                "time_embed_dim": self.time_embed_dim,
                "categories": {
                    col: cats
                    for col, cats in zip(
                        self._cond_sampler.col_names, self._cond_sampler.cat_lists
                    )
                },
            },
            path,
        )
        logger.info("TabularDiffusion saved to %s", path)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "TabularDiffusion":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        instance = cls(
            col_meta=ckpt["col_meta"],
            categories=ckpt.get("categories", {}),
            T=ckpt["T"],
            beta_schedule=ckpt["beta_schedule"],
            beta_start=ckpt.get("beta_start", 1e-4),
            beta_end=ckpt.get("beta_end", 0.02),
            hidden_dims=ckpt["hidden_dims"],
            time_embed_dim=ckpt["time_embed_dim"],
            constraint_penalty_weight=ckpt.get("constraint_penalty_weight", 1.0),
            device=device,
        )
        instance.denoiser.load_state_dict(ckpt["denoiser_state"])
        instance._is_trained = True
        logger.info("TabularDiffusion loaded from %s", path)
        return instance
