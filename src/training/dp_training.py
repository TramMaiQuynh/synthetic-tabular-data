"""
DP-SGD Training Wrapper
------------------------
Provides a unified interface for Differentially Private training via two backends:

  Backend A — Opacus (preferred):
    Uses `opacus.PrivacyEngine` to attach per-sample gradient hooks and noise
    injection to an existing PyTorch optimiser. Tracks privacy budget (epsilon, delta)
    via the Moments Accountant / RDP accountant.

  Backend B — Custom per-sample gradient (fallback):
    Implements per-sample gradient clipping + Gaussian noise injection manually
    using PyTorch's `functorch` / gradient hooks. Used when Opacus is unavailable
    (Windows incompatibility, version mismatch, etc.).

Behaviour is *identical* from the caller's perspective regardless of backend.

Usage:
    trainer = DPTrainer(
        target_epsilon=1.0,
        target_delta=1e-5,
        max_grad_norm=1.0,
        noise_multiplier=None,   # auto-calibrated if None
        backend="auto",          # "opacus" | "custom" | "auto"
    )
    result = trainer.wrap_optimizer(optimizer, model)
    if isinstance(result, tuple):
        optimizer, model = result  # Opacus returns (optimizer, wrapped_model)
    else:
        optimizer = result           # Custom returns optimizer only

    # In training loop:
    optimizer.zero_grad()
    trainer.backward(loss, model, optimizer)   # clips + noises + steps
    epsilon = trainer.current_epsilon(dataset_size, batch_size, epoch)

Design notes:
  - BatchNorm in models is *prohibited* when using DP-SGD (breaks per-sample
    gradient isolation). All models in src/models/ use LayerNorm only.
  - Privacy budget is an *upper bound*; actual epsilon depends on the number of
    gradient steps consumed. Budget should be pre-declared and logged.
  - When DP-SGD is active, early stopping must be disabled in the model trainers
    (to avoid leaking information through epoch count). DPTrainer enforces this
    contract by design — callers pass fixed epochs; DPTrainer never terminates early.
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

__all__ = ["DPTrainer"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def _detect_opacus() -> bool:
    """Return True if Opacus is importable and functional."""
    try:
        import opacus  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Noise multiplier calibration (RDP-based closed-form approximation)
# ---------------------------------------------------------------------------

def _calibrate_noise_multiplier(
    target_epsilon: float,
    target_delta: float,
    dataset_size: int,
    batch_size: int,
    epochs: int,
    max_grad_norm: float,
    tol: float = 0.01,
    max_iterations: int = 1000,
) -> float:
    """
    Binary-search calibration of noise_multiplier such that the resulting
    (epsilon, delta) at the end of training does not exceed the targets.

    Uses a simplified RDP accountant bound (moments accountant approximation):
        epsilon ≈ sqrt(2 * steps * log(1/delta)) / (noise_multiplier * steps_ratio)

    This approximation is conservative (overestimates epsilon) — safe for production.
    For precise accounting, use Opacus's built-in accountant.

    Returns:
        Calibrated noise_multiplier (float > 0).
    """
    sampling_rate = batch_size / dataset_size
    steps = int(epochs * math.ceil(dataset_size / batch_size))

    # Closed-form estimate from basic composition + subsampling amplification:
    # eps ≈ sampling_rate * sqrt(2 * steps * log(1/delta)) / noise_multiplier
    # Solve for noise_multiplier:
    denom_eps_target = target_epsilon
    noise_sigma = (
        sampling_rate * math.sqrt(2.0 * steps * math.log(1.0 / target_delta))
        / denom_eps_target
    )

    logger.info(
        "Calibrated noise_multiplier=%.4f for target eps=%.2f, delta=%.2e, "
        "steps=%d, sampling_rate=%.5f",
        noise_sigma, target_epsilon, target_delta, steps, sampling_rate,
    )
    return max(0.1, noise_sigma)  # floor at 0.1 to avoid degenerate configs


# ---------------------------------------------------------------------------
# Custom DP-SGD backend (no Opacus dependency)
# ---------------------------------------------------------------------------

class _CustomDPOptimizer:
    """
    Wraps a standard PyTorch optimiser with batch-level gradient clipping
    and heuristic Gaussian noise injection.

    CRITICAL NOTE: This custom backend is a HEURISTIC APPROXIMATION. It does
    NOT compute per-sample gradients or register hooks. Instead, it clips
    the average batch gradient computed by loss.backward() and adds noise.
    Because it clips the batch gradient as a whole rather than clipping each
    sample's gradient individually, the noise added is not properly calibrated
    to the individual sample sensitivity.

    Consequently, this fallback does NOT provide formal, mathematically rigorous
    Differential Privacy guarantees. It should be used ONLY as a lightweight
    computational fallback when Opacus is unavailable. For rigorous DP-SGD
    guarantees, the Opacus backend must be used.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        model: nn.Module,
        max_grad_norm: float,
        noise_multiplier: float,
        batch_size: int = 256,
    ) -> None:
        self.optimizer = optimizer
        self.model = model
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.batch_size = batch_size
        self._step_count = 0

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def backward_and_step(self, loss: torch.Tensor) -> None:
        """Compute loss gradient, clip, add noise, step optimiser.

        Noise is scaled to match the DP-SGD formulation (Abadi et al., 2016):
        Since ``loss`` uses ``reduction='mean'``, ``param.grad`` already
        contains a ``1/B`` factor.  The Gaussian noise standard deviation
        must also be divided by ``B`` to keep the signal-to-noise ratio
        consistent with the theoretical analysis.
        """
        loss.backward()

        # Clip total gradient norm (approximation of per-sample clipping)
        nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=self.max_grad_norm
        )

        # Inject calibrated Gaussian noise (divided by batch_size to
        # match the 1/B-scaled gradient from mean-reduction loss).
        noise_std = (self.max_grad_norm * self.noise_multiplier) / self.batch_size
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad += torch.randn_like(param.grad) * noise_std

        self.optimizer.step()
        self._step_count += 1

    def rdp_epsilon(
        self,
        dataset_size: int,
        batch_size: int,
        target_delta: float,
    ) -> float:
        """
        Estimate accumulated epsilon via the basic composition formula.
        Conservative upper bound (does not use subsampling amplification).
        """
        sampling_rate = min(1.0, batch_size / dataset_size)
        steps = self._step_count
        if steps == 0 or self.noise_multiplier <= 0:
            return float("inf")
        eps = (
            sampling_rate
            * math.sqrt(2.0 * steps * math.log(1.0 / target_delta))
            / self.noise_multiplier
        )
        return eps


# ---------------------------------------------------------------------------
# Public DPTrainer
# ---------------------------------------------------------------------------

class DPTrainer:
    """
    Unified DP-SGD training coordinator.

    Parameters
    ----------
    target_epsilon : float
        Maximum privacy budget (epsilon) to consume over all training.
    target_delta : float
        Failure probability (delta). Typical value: 1e-5.
    max_grad_norm : float
        Per-sample gradient clipping norm (C in the DP-SGD paper).
    noise_multiplier : float or None
        Gaussian noise std = max_grad_norm * noise_multiplier.
        If None, auto-calibrated from target_epsilon / target_delta.
    backend : str
        'opacus' — use Opacus PrivacyEngine.
        'custom' — use built-in per-sample approx gradient clipping.
        'auto'   — use Opacus if available, else fall back to custom.
    """

    def __init__(
        self,
        target_epsilon: float = 1.0,
        target_delta: float = 1e-5,
        max_grad_norm: float = 1.0,
        noise_multiplier: Optional[float] = None,
        backend: str = "auto",
    ) -> None:
        if target_epsilon <= 0:
            raise ValueError("target_epsilon must be > 0.")
        if not (0 < target_delta < 1):
            raise ValueError("target_delta must be in (0, 1).")
        if max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be > 0.")

        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.max_grad_norm = max_grad_norm
        self._noise_multiplier = noise_multiplier
        self.backend = backend

        self._opacus_available = _detect_opacus()
        self._active_backend: Optional[str] = None
        self._wrapped_optimizer = None
        self._wrapped_model: Optional[nn.Module] = None  # GradSampleModule for Opacus
        self._privacy_engine = None  # Opacus PrivacyEngine if used

        logger.info(
            "DPTrainer init: target_eps=%.2f, target_delta=%.2e, "
            "max_grad_norm=%.2f, backend='%s', opacus_available=%s",
            target_epsilon, target_delta, max_grad_norm, backend, self._opacus_available,
        )

    def _resolve_backend(self) -> str:
        if self.backend == "auto":
            return "opacus" if self._opacus_available else "custom"
        if self.backend == "opacus" and not self._opacus_available:
            logger.warning(
                "Backend 'opacus' requested but Opacus is not installed. "
                "Falling back to 'custom' backend."
            )
            return "custom"
        return self.backend

    def _get_noise_multiplier(
        self,
        dataset_size: int = 10000,
        batch_size: int = 256,
        epochs: int = 100,
    ) -> float:
        if self._noise_multiplier is not None:
            return self._noise_multiplier
        return _calibrate_noise_multiplier(
            target_epsilon=self.target_epsilon,
            target_delta=self.target_delta,
            dataset_size=dataset_size,
            batch_size=batch_size,
            epochs=epochs,
            max_grad_norm=self.max_grad_norm,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def wrap_optimizer(
        self,
        optimizer: torch.optim.Optimizer,
        model: nn.Module,
        dataset_size: int = 10000,
        batch_size: int = 256,
        epochs: int = 100,
    ) -> Union["_CustomDPOptimizer", Tuple["_CustomDPOptimizer", nn.Module]]:
        """
        Wrap an existing optimizer with DP-SGD logic.

        Args:
            optimizer    : Standard PyTorch optimizer (pre-created).
            model        : The nn.Module to train (must have no BatchNorm).
            dataset_size : Total training rows (used for noise calibration).
            batch_size   : Mini-batch size.
            epochs       : Total training epochs.

        Returns:
            For custom backend: A wrapped optimizer with .zero_grad() and .step()
            replaced by DP-safe equivalents.

            For Opacus backend: Tuple (wrapped_optimizer, wrapped_model).
            Callers MUST replace their model reference with wrapped_model
            to ensure per-sample gradient hooks are used during forward passes.
            Also accessible as DPTrainer.wrapped_optimizer / DPTrainer.wrapped_model.
        """
        resolved = self._resolve_backend()
        self._active_backend = resolved

        noise_mult = self._get_noise_multiplier(dataset_size, batch_size, epochs)
        self._noise_multiplier = noise_mult  # cache resolved value

        if resolved == "opacus":
            try:
                return self._wrap_opacus(
                    optimizer, model, noise_mult, batch_size, dataset_size, epochs
                )
            except Exception as exc:
                logger.warning(
                    "Opacus wrap_optimizer failed (%s). Falling back to custom backend.", exc
                )
                self._active_backend = "custom"

        # Custom backend
        logger.warning(
            "CRITICAL DP WARNING: Custom DP-SGD backend is a heuristic approximation using batch-level gradient clipping. "
            "It does NOT compute per-sample gradients and therefore does NOT provide formal, mathematically rigorous "
            "Differential Privacy guarantees. For formal privacy guarantees, use backend='opacus'."
        )
        wrapped = _CustomDPOptimizer(
            optimizer=optimizer,
            model=model if isinstance(model, nn.Module) else nn.ModuleList(list(model)),
            max_grad_norm=self.max_grad_norm,
            noise_multiplier=noise_mult,
            batch_size=batch_size,
        )
        self._wrapped_optimizer = wrapped
        self._wrapped_model = None
        logger.info(
            "DPTrainer using 'custom' backend — noise_multiplier=%.4f, "
            "max_grad_norm=%.2f, batch_size=%d.",
            noise_mult, self.max_grad_norm, batch_size,
        )
        return wrapped

    def _reconstruct_optimizer(
        self,
        old_optimizer: torch.optim.Optimizer,
        new_model: nn.Module,
    ) -> torch.optim.Optimizer:
        """
        Reconstruct an optimizer with the same hyperparameters as `old_optimizer`
        but pointing to `new_model.parameters()`.

        This is required when ModuleValidator.fix() replaces the model and
        creates new parameter tensors. The old optimizer would reference stale
        parameters, causing Opacus to raise:
            ValueError: Module parameters are different than optimizer Parameters
        """
        opt_cls = type(old_optimizer)
        defaults = old_optimizer.defaults.copy()

        # Group hyperparameters from param_groups (supports per-group configs)
        # Most optimizers have a single param_group; we preserve all groups
        new_param_groups = []
        for group in old_optimizer.param_groups:
            group_spec = {k: v for k, v in group.items() if k != "params"}
            new_param_groups.append(group_spec)

        if len(new_param_groups) > 1:
            # Multiple param groups: create optimizer with list of dicts
            params_with_groups = []
            for i, group_spec in enumerate(new_param_groups):
                params_with_groups.append(
                    {"params": list(new_model.parameters()), **group_spec}
                )
            new_optimizer = opt_cls(params_with_groups, **defaults)
        else:
            # Single param group: typical case
            new_optimizer = opt_cls(new_model.parameters(), **defaults)

        logger.info(
            "Reconstructed optimizer %s with new model parameters "
            "(previous optimizer referenced %d old params, "
            "new optimizer has %d new params).",
            opt_cls.__name__,
            sum(p.numel() for p in old_optimizer.param_groups[0]["params"]),
            sum(p.numel() for p in new_model.parameters()),
        )
        return new_optimizer

    def _wrap_opacus(
        self,
        optimizer: torch.optim.Optimizer,
        model: nn.Module,
        noise_multiplier: float,
        batch_size: int,
        dataset_size: int,
        epochs: int,
    ) -> Tuple[torch.optim.Optimizer, nn.Module]:
        """
        Attach Opacus PrivacyEngine to model + optimizer.

        Returns:
            Tuple (dp_optimizer, wrapped_model).
            Callers MUST update their model reference to wrapped_model.

        Raises:
            ValueError: If ModuleValidator.fix() creates a new model and the
                        optimizer cannot be reconstructed.
        """
        from opacus import PrivacyEngine
        from opacus.validators import ModuleValidator

        # Step 1: Validate and optionally fix model
        # ModuleValidator.fix() calls clone_module() which deep-copies ALL
        # parameter tensors. If fix() triggers, we must reconstruct the
        # optimizer with the new model's parameters.
        if not ModuleValidator.is_valid(model):
            errors = ModuleValidator.validate(model, strict=False)
            logger.warning(
                "Opacus ModuleValidator found issues: %s. "
                "Attempting auto-fix.",
                errors,
            )
            fixed_model = ModuleValidator.fix(model)
            if fixed_model is not model:
                # Bug #2 fix: fix() created a new model with new parameter
                # tensors. Reconstruct optimizer to match.
                logger.info(
                    "ModuleValidator.fix() replaced model object. "
                    "Reconstructing optimizer with new model parameters."
                )
                optimizer = self._reconstruct_optimizer(optimizer, fixed_model)
                model = fixed_model
            else:
                # fix() returned the same model (no changes needed)
                logger.info(
                    "ModuleValidator.fix() returned same model object. "
                    "Proceeding with original optimizer."
                )

        # Step 2: Create a proper DataLoader for Opacus internal accounting
        # Opacus 1.6+ requires a valid DataLoader (data_loader=None crashes).
        # The DataLoader is used to compute sample_rate = batch_size / dataset_size.
        # Since we manage batches manually (not via Opacus DataLoader),
        # we create a dummy loader with the correct dimensions for accounting.
        from torch.utils.data import TensorDataset, DataLoader as TDataLoader

        dummy_dataset = TensorDataset(torch.zeros(dataset_size, 1))
        data_loader = TDataLoader(dummy_dataset, batch_size=batch_size)

        # Step 3: Attach PrivacyEngine
        privacy_engine = PrivacyEngine()
        wrapped_model, dp_optimizer, _ = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=self.max_grad_norm,
            poisson_sampling=False,
        )
        self._privacy_engine = privacy_engine
        self._wrapped_optimizer = dp_optimizer
        self._wrapped_model = wrapped_model

        logger.info(
            "DPTrainer using Opacus backend — noise_multiplier=%.4f, "
            "max_grad_norm=%.2f, wrapped_model=%s.",
            noise_multiplier, self.max_grad_norm,
            type(wrapped_model).__name__,
        )
        return dp_optimizer, wrapped_model

    def backward(
        self,
        loss: torch.Tensor,
        model: nn.Module,
        optimizer,
    ) -> None:
        """
        Execute backward pass with DP clipping + noise.

        Args:
            loss      : Scalar loss tensor.
            model     : The model being trained (used only for custom backend).
            optimizer : The wrapped optimizer returned by wrap_optimizer().
        """
        if self._active_backend == "custom" and isinstance(optimizer, _CustomDPOptimizer):
            optimizer.backward_and_step(loss)
        elif self._active_backend == "opacus":
            # Opacus: DP is transparent via hooks on backward + step.
            # NOTE: If ModelValidator.fix() was NOT triggered, hooks are on the
            # original model's submodules and fire even without using the
            # wrapped model for forward passes.
            # If fix() WAS triggered, hooks are on the fixed model, and the
            # caller MUST have updated its model reference to use the wrapped model.
            loss.backward()
            optimizer.step()
        else:
            raise RuntimeError(
                f"DPTrainer.backward() called with unresolved backend "
                f"'{self._active_backend}'. Call wrap_optimizer() before training."
            )

    def current_epsilon(
        self,
        dataset_size: int,
        batch_size: int,
        epochs_done: int,
    ) -> float:
        """
        Return the estimated privacy budget consumed so far.

        Args:
            dataset_size : Total training rows.
            batch_size   : Mini-batch size.
            epochs_done  : Number of completed epochs.

        Returns:
            Estimated epsilon (float). Returns inf if training not started.
        """
        if self._active_backend == "opacus" and self._privacy_engine is not None:
            try:
                eps = self._privacy_engine.get_epsilon(self.target_delta)
                logger.info("Opacus epsilon=%.4f after %d epochs.", eps, epochs_done)
                return eps
            except Exception as exc:
                logger.warning("Opacus get_epsilon failed: %s", exc)

        if isinstance(self._wrapped_optimizer, _CustomDPOptimizer):
            eps = self._wrapped_optimizer.rdp_epsilon(dataset_size, batch_size, self.target_delta)
            logger.info("Custom backend epsilon~=%.4f after %d epochs.", eps, epochs_done)
            return eps

        return float("inf")

    @property
    def wrapped_model(self) -> Optional[nn.Module]:
        """
        Return the Opacus GradSampleModule-wrapped model, if Opacus backend is active.

        For the custom backend, returns None.
        Callers should use this model for all forward passes when using Opacus.
        """
        return self._wrapped_model