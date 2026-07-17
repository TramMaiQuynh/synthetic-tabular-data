"""
Hyperparameter Optimisation (HPO)
----------------------------------
Uses Optuna with Bayesian Optimisation (TPE sampler) to search for the
optimal training configuration for any supported generative model type.

Objective function:
    Weighted combination of statistical fidelity (Wasserstein distance,
    marginal distributions) and constraint satisfaction rate on a held-out
    validation subset. Lower is better.

Privacy-aware HPO:
    When DP is enabled, each HPO trial consumes a share of the overall
    privacy budget. A pre-allocated HPO budget fraction (default 20%) is
    deducted from the total epsilon before handing the remaining budget
    to the final training run. The noise_multiplier is recalibrated for
    each trial to stay within the per-trial budget.

Output:
    Best hyper-parameters are logged and optionally written to
    configs/<dataset_name>/model_config.yaml for reproducibility.

Usage:
    hpo = HPORunner(
        model_type="ctgan",
        dataset_name="telco_customer_churn",
        artifacts_root="artifacts",
        n_trials=30,
    )
    best_params = hpo.run(preprocessed_df, continuous_cols, categorical_cols)
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["HPORunner"]


# ---------------------------------------------------------------------------
# Fidelity metric (lightweight proxy — no Module 3 dependency)
# ---------------------------------------------------------------------------

def _wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    """Compute 1D Wasserstein-1 distance between two empirical distributions."""
    a_sorted = np.sort(a[np.isfinite(a)])
    b_sorted = np.sort(b[np.isfinite(b)])

    if len(a_sorted) == 0 or len(b_sorted) == 0:
        return 0.0

    # Interpolate to common grid via quantile matching
    n = max(len(a_sorted), len(b_sorted))
    q = np.linspace(0, 1, n)
    a_q = np.quantile(a_sorted, q)
    b_q = np.quantile(b_sorted, q)
    return float(np.mean(np.abs(a_q - b_q)))


def _compute_fidelity_score(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    continuous_cols: List[str],
) -> float:
    """
    Compute mean Wasserstein-1 distance across all continuous columns.
    Lower = better fidelity.
    """
    scores = []
    for col in continuous_cols:
        if col in real_df.columns and col in synth_df.columns:
            real_vals = pd.to_numeric(real_df[col], errors="coerce").dropna().values
            synth_vals = pd.to_numeric(synth_df[col], errors="coerce").dropna().values
            scores.append(_wasserstein_1d(real_vals, synth_vals))
    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# HPORunner
# ---------------------------------------------------------------------------

class HPORunner:
    """
    Optuna-based hyperparameter optimiser for generative tabular models.

    Parameters
    ----------
    model_type : str
        'ctgan', 'ctvae', or 'diffusion'.
    dataset_name : str
        Dataset identifier (for checkpoint / config paths).
    artifacts_root : str
        Root path for saving checkpoints.
    n_trials : int
        Number of Optuna trials.
    timeout : float or None
        Optional wall-clock timeout for the study in seconds.
    val_fraction : float
        Fraction of data held out as validation for fidelity scoring.
    fidelity_weight : float
        Weight of fidelity score in the objective (rest goes to constraint violation).
    enable_dp : bool
        Whether the final training run will use DP-SGD.
    total_epsilon : float
        Total privacy budget. HPO will consume hpo_budget_fraction of this.
    target_delta : float
        DP failure probability.
    hpo_budget_fraction : float
        Fraction of total_epsilon allocated to HPO trials.
    n_jobs : int
        Optuna parallel jobs (1 = sequential; -1 = auto).
    """

    def __init__(
        self,
        model_type: str,
        dataset_name: str,
        artifacts_root: str,
        n_trials: int = 20,
        timeout: Optional[float] = None,
        val_fraction: float = 0.15,
        fidelity_weight: float = 0.7,
        enable_dp: bool = False,
        total_epsilon: float = 1.0,
        target_delta: float = 1e-5,
        hpo_budget_fraction: float = 0.2,
        n_jobs: int = 1,
    ) -> None:
        from src.training.trainer import ModelTrainer  # deferred to avoid circular import

        self.model_type = model_type
        self.dataset_name = dataset_name
        self.artifacts_root = artifacts_root
        self.n_trials = n_trials
        self.timeout = timeout
        self.val_fraction = val_fraction
        self.fidelity_weight = fidelity_weight
        self.enable_dp = enable_dp
        self.total_epsilon = total_epsilon
        self.target_delta = target_delta
        self.hpo_budget_fraction = hpo_budget_fraction
        self.n_jobs = n_jobs

        # Remaining epsilon after HPO allocation
        self._epsilon_for_hpo = total_epsilon * hpo_budget_fraction
        self._epsilon_for_training = total_epsilon * (1.0 - hpo_budget_fraction)

        logger.info(
            "HPORunner: model='%s', n_trials=%d, enable_dp=%s, "
            "epsilon_for_hpo=%.3f, epsilon_for_training=%.3f.",
            model_type, n_trials, enable_dp,
            self._epsilon_for_hpo, self._epsilon_for_training,
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        preprocessed_df: pd.DataFrame,
        continuous_cols: List[str],
        categorical_cols: List[str],
        constraint_expressions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run the HPO study and return the best hyperparameter configuration.

        Args:
            preprocessed_df     : Full preprocessed DataFrame.
            continuous_cols     : Original continuous column names.
            categorical_cols    : Original categorical column names.
            constraint_expressions: Constraint expressions for satisfaction scoring.

        Returns:
            dict of best hyperparameters ready to pass to ModelTrainer.train().
        """
        try:
            import optuna
        except ImportError as exc:
            raise ImportError(
                "Optuna is required for HPO. Install it with: pip install optuna"
            ) from exc

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Train/val split (stratified by row index — no label leakage)
        # Use a fixed seed for reproducibility of HPO results
        rng = np.random.RandomState(42)
        n = len(preprocessed_df)
        val_n = max(1, int(n * self.val_fraction))
        val_idx = rng.choice(n, size=val_n, replace=False)
        train_mask = np.ones(n, dtype=bool)
        train_mask[val_idx] = False

        train_df = preprocessed_df.iloc[train_mask].reset_index(drop=True)
        val_df = preprocessed_df.iloc[val_idx].reset_index(drop=True)

        logger.info(
            "HPO train/val split: %d train rows, %d val rows.",
            len(train_df), len(val_df),
        )

        def objective(trial: "optuna.Trial") -> float:
            params = self._suggest_params(trial, len(train_df))

            # Per-trial DP epsilon budget
            per_trial_eps = (
                self._epsilon_for_hpo / self.n_trials if self.enable_dp else float("inf")
            )

            try:
                from src.training.trainer import ModelTrainer

                trainer = ModelTrainer(
                    model_type=self.model_type,
                    dataset_name=self.dataset_name,
                    artifacts_root=self.artifacts_root,
                )
                result = trainer.train(
                    preprocessed_df=train_df,
                    continuous_cols=continuous_cols,
                    categorical_cols=categorical_cols,
                    epochs=params["epochs"],
                    batch_size=params["batch_size"],
                    lr=params["lr"],
                    weight_decay=params["weight_decay"],
                    enable_dp=self.enable_dp,
                    target_epsilon=per_trial_eps,
                    target_delta=self.target_delta,
                    max_grad_norm=params.get("max_grad_norm", 1.0),
                    noise_multiplier=params.get("noise_multiplier"),
                    constraint_expressions=constraint_expressions,
                    model_kwargs=params.get("model_kwargs", {}),
                )

                # Generate a small synthetic sample for fidelity scoring
                model = trainer.model
                col_meta = trainer.col_meta
                n_synth = min(len(val_df), 500)
                synth_tensor = model.sample(n_synth)
                synth_df = self._tensor_to_df(synth_tensor, col_meta, val_df.columns.tolist())

                # Fidelity score (lower = better)
                fidelity = _compute_fidelity_score(val_df, synth_df, continuous_cols)

                # Constraint violation rate (lower = better)
                constraint_score = 0.0
                if constraint_expressions:
                    from src.models.constraints import ConstraintsEngine
                    engine = ConstraintsEngine(constraint_expressions)
                    constraint_score = engine.violation_rate(synth_df)

                objective_value = (
                    self.fidelity_weight * fidelity
                    + (1.0 - self.fidelity_weight) * constraint_score
                )
                logger.info(
                    "Trial %d: fidelity=%.4f, constraint_violation=%.4f, objective=%.4f | %s",
                    trial.number, fidelity, constraint_score, objective_value, params,
                )
                return objective_value

            except Exception as exc:
                logger.warning("Trial %d failed: %s", trial.number, exc)
                return float("inf")

        # Run study
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs,
            show_progress_bar=False,
        )

        best_trial = study.best_trial
        best_params = best_trial.params
        logger.info(
            "HPO complete. Best trial #%d — objective=%.4f | params=%s",
            best_trial.number, best_trial.value, best_params,
        )

        # Annotate with residual epsilon for the final training run
        best_params["_epsilon_for_training"] = self._epsilon_for_training
        return best_params

    # ------------------------------------------------------------------
    # Parameter suggestion space
    # ------------------------------------------------------------------

    def _suggest_params(self, trial: Any, dataset_size: int) -> Dict[str, Any]:
        """Define the Optuna search space per model type."""
        # Common hyper-parameters
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-4, log=True)

        # DP-aware batch size: larger batch reduces sensitivity noise
        # Heuristic: batch_size >= sqrt(N) when DP is active
        if self.enable_dp:
            min_batch = max(64, int(math.sqrt(dataset_size)))
            batch_size = trial.suggest_categorical(
                "batch_size",
                [b for b in [128, 256, 512, 1024] if b >= min_batch][:4] or [min_batch],
            )
        else:
            batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])

        epochs = trial.suggest_int("epochs", 20, 200, step=20)

        model_kwargs: Dict[str, Any] = {}

        if self.model_type in ("ctgan", "ctvae"):
            h_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])
            n_layers = trial.suggest_int("n_layers", 1, 3)
            hidden_dims = tuple([h_dim] * n_layers)
            model_kwargs["hidden_dims"] = hidden_dims

            if self.model_type == "ctgan":
                model_kwargs["noise_dim"] = trial.suggest_categorical("noise_dim", [64, 128, 256])
                model_kwargs["n_critic"] = trial.suggest_int("n_critic", 3, 10)
                model_kwargs["lambda_gp"] = trial.suggest_float("lambda_gp", 5.0, 20.0)
            else:  # ctvae
                model_kwargs["latent_dim"] = trial.suggest_categorical("latent_dim", [64, 128, 256])
                model_kwargs["beta"] = trial.suggest_float("beta", 0.5, 4.0)

        elif self.model_type == "diffusion":
            h_dim = trial.suggest_categorical("hidden_dim", [256, 512])
            n_layers = trial.suggest_int("n_layers", 2, 4)
            model_kwargs["hidden_dims"] = tuple([h_dim] * n_layers)
            model_kwargs["T"] = trial.suggest_categorical("T", [500, 1000])
            model_kwargs["beta_schedule"] = trial.suggest_categorical(
                "beta_schedule", ["cosine", "linear"]
            )

        params: Dict[str, Any] = {
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "epochs": epochs,
            "model_kwargs": model_kwargs,
        }

        if self.enable_dp:
            params["max_grad_norm"] = trial.suggest_float("max_grad_norm", 0.5, 5.0)

        return params

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _tensor_to_df(
        tensor: "torch.Tensor",
        col_meta: List[Any],
        reference_columns: List[str],
    ) -> pd.DataFrame:
        """Convert a generated tensor to a DataFrame with reference column names.

        Correctly reconstructs onehot column groups by matching prefix-based
        naming conventions while avoiding prefix collisions from columns whose
        names are strict supersets of a onehot group prefix (e.g. continuous
        column ``education_num`` could incorrectly match the ``education_``
        onehot prefix).

        The algorithm is: a reference column ``c`` belongs to the onehot group
        of ``meta.name`` iff:

        1. ``c`` starts with ``meta.name + '_'``.
        2. ``c`` does NOT end with ``_is_missing`` (reserved for imputation).
        3. There is no *other* column in ``col_meta`` whose name is a strict
           prefix of ``c`` and longer than ``meta.name`` (this prevents
           ``education_num`` from being assigned to the ``education`` group
           when ``education_num`` is actually a continuous column, BUT also
           when ``education_num`` is a separate column with no onehot group).
        4. ``c`` is not a continuous / label column name that happens to start
           with the onehot prefix (e.g. when the dataset genuinely contains a
           column named ``education_num`` next to ``education_HS-grad``).
        """
        arr = tensor.detach().cpu().numpy()
        cols: List[str] = []

        # Build a set of all non-onehot column names in col_meta so we can
        # reject reference columns that are actually continuous/label columns
        # whose names overlap with a onehot prefix.
        other_names: Dict[str, str] = {}
        for m in col_meta:
            if m.col_type != "onehot":
                other_names[m.name] = m.col_type

        for meta in col_meta:
            if meta.col_type == "onehot":
                onehot_prefix = meta.name + "_"
                matching = []
                for c in reference_columns:
                    if not c.startswith(onehot_prefix):
                        continue
                    if c.endswith("_is_missing"):
                        continue
                    # Reject if c is actually a continuous or label column
                    # (occurs when a column name like "education_num" starts
                    # with "education_" but is its own independent column).
                    if c in other_names:
                        continue
                    # Check for longer-match collision with another col_meta
                    # whose name is a strict prefix of c and longer than ours.
                    has_longer_match = False
                    for other in col_meta:
                        if other.name != meta.name and c.startswith(other.name + "_") and len(other.name) > len(meta.name):
                            has_longer_match = True
                            break
                    if not has_longer_match:
                        matching.append(c)
                cols.extend(sorted(matching)[:meta.dim])
            else:
                cols.append(meta.name)

        # Pad or truncate if needed
        n_cols = arr.shape[1]
        if len(cols) < n_cols:
            cols += [f"__col_{i}" for i in range(len(cols), n_cols)]
        elif len(cols) > n_cols:
            cols = cols[:n_cols]

        return pd.DataFrame(arr, columns=cols)
