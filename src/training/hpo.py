"""
Hyperparameter Optimisation (HPO)
----------------------------------
Uses Optuna with Bayesian Optimisation (TPE sampler) to search for the
optimal training configuration for any supported generative model type.

Objective function:
    Column-weighted combination of:
      - Wasserstein-1 distance on continuous columns (MinMax-scaled using
        the real data's range for scale invariance across features).
      - Jensen-Shannon Divergence (squared JS distance, base=2) on
        categorical columns (category union alignment handles missing
        categories between real and synthetic distributions).
      - Constraint violation rate on inverse-transformed synthetic data
        (ensures constraint expressions reference original column names
        and original-domain values for correct semantic evaluation).
    Lower is better.

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
        seed=42,
    )
    best_params = hpo.run(
        preprocessed_df,
        continuous_cols,
        categorical_cols,
        pipeline=pipeline,  # fitted PreprocessingPipeline for inverse_transform
    )
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon

# Defensive import: PEP 563 (from __future__ import annotations) makes string
# annotations lazy, so "torch.Tensor" in _tensor_to_df's type hint does not
# crash at definition time. However, runtime reflection APIs such as
# typing.get_type_hints() will fail with NameError if torch is not imported.
# This import also serves as documentation that this module depends on torch.
import torch  # noqa: F401  # used by _tensor_to_df type hint

if TYPE_CHECKING:
    from src.preprocessing.pipeline import PreprocessingPipeline

logger = logging.getLogger(__name__)

__all__ = ["HPORunner"]


# ---------------------------------------------------------------------------
# Fidelity metric (lightweight proxy — no Module 3 dependency)
# ---------------------------------------------------------------------------

# NOTE: The Wasserstein-1 distance here is computed via scipy.stats.wasserstein_distance
# which is the EXACT closed-form 1D optimal transport distance (equation 7 in
# Kantorovich's formulation, 1942; solved trivially in 1D as W1 = ∫|F⁻¹(t) - G⁻¹(t)|dt).
# This matches src/evaluation/fidelity.py's implementation.
#
# The Jensen-Shannon Divergence is computed with base=2 (scipy default is base=e,
# which would give max ≈ 0.832 instead of 1.0). This matches src/evaluation/fidelity.py
# line 86: jensenshannon(..., base=2.0).


def _wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute exact 1D Wasserstein-1 distance between two empirical distributions.

    Optimal Transport Theory (Kantorovich, 1942; Monge, 1781):
        W_1(P, Q) = inf_{γ ∈ Π(P,Q)} ∫ |x - y| dγ(x,y)

    In 1D, this simplifies to (Vallender, 1974):
        W_1(P, Q) = ∫_{-∞}^{∞} |F(x) - G(x)| dx  =  ∫₀¹ |F⁻¹(t) - G⁻¹(t)| dt

    Normalisation: both real (a) and synthetic (b) values are normalised to [0, 1]
    using the real data's observed range (min/max). This ensures:
      - Scale invariance across columns with different units (e.g., dollars vs months).
      - Commensurability with JS divergence values (also in [0, 1]).
      - Out-of-range synthetic values are correctly penalised (b_normed may exceed 1.0
        if b contains values beyond the real data's range, which is a memorisation
        or drift issue).

    Args:
        a: 1-D numpy array of real (validation) values.
        b: 1-D numpy array of synthetic values.

    Returns:
        Wasserstein-1 distance in normalised space (float >= 0).
    """
    a_clean = a[np.isfinite(a)]
    b_clean = b[np.isfinite(b)]

    if len(a_clean) == 0 or len(b_clean) == 0:
        # Penalty: returning 0.0 would reward a completely collapsed model
        # (all-NaN synthetic output) as "perfect" under a minimize objective.
        # Return maximum penalty (1.0) since both Wasserstein and JS are
        # normalised to [0, 1] — this ensures failed trials are penalised,
        # not rewarded.
        return 1.0

    a_min, a_max = float(a_clean.min()), float(a_clean.max())
    denom = a_max - a_min
    if denom == 0:
        # Constant column: distance is the absolute difference between the
        # constant values (normalised to [0,1] by treating the constant as
        # the only value). If both are the same constant, distance is 0.
        b_const = b_clean[0] if len(b_clean) > 0 else 0.0
        a_const = a_clean[0] if len(a_clean) > 0 else 0.0
        return float(abs(b_const - a_const))

    a_normed = (a_clean - a_min) / denom
    b_normed = (b_clean - a_min) / denom
    return float(wasserstein_distance(a_normed, b_normed))


def _js_divergence(
    real_cat: pd.Series, synth_cat: pd.Series, base: float = 2.0
) -> float:
    """
    Compute Jensen-Shannon Divergence (squared JS distance) for a categorical column.

    Information Theoretic Definition (Lin, 1991):
        D_JS(P ‖ Q) = ½ D_KL(P ‖ M) + ½ D_KL(Q ‖ M)
    where M = ½(P + Q) is the midpoint distribution.

    The result is the squared JS distance (base=2), yielding values in [0, 1].

    CRITICAL: scipy.spatial.distance.jensenshannon defaults to base=e (natural
    logarithm), which gives max ≈ 0.832 instead of 1.0. The base=2 parameter is
    explicitly set here to ensure consistency with:
      - The mathematical definition in the literature.
      - src/evaluation/fidelity.py (line 86: base=2.0).
      - The [0, 1] bound assumed by all downstream thresholds.

    Args:
        real_cat: Real categorical series.
        synth_cat: Synthetic categorical series.
        base: Logarithm base (MUST be 2.0 to match fidelity.py).

    Returns:
        JS divergence in [0, 1].
    """
    r_counts = real_cat.dropna().astype(str).value_counts()
    s_counts = synth_cat.dropna().astype(str).value_counts()

    # Union of all categories across both real and synthetic distributions.
    # This handles the case where synthetic data either:
    #   (a) omits a rare category that exists in the real data, or
    #   (b) hallucinates a category that does not exist in the real data.
    union_cats = list(set(r_counts.index) | set(s_counts.index))
    if not union_cats:
        return 0.0

    try:
        union_cats.sort()
    except TypeError:
        union_cats.sort(key=str)

    p = np.array([r_counts.get(cat, 0) for cat in union_cats], dtype=np.float64)
    q = np.array([s_counts.get(cat, 0) for cat in union_cats], dtype=np.float64)

    p_sum = p.sum()
    q_sum = q.sum()
    if p_sum == 0 or q_sum == 0:
        # If one distribution has zero mass, they are maximally divergent.
        return 1.0 if (p_sum > 0 or q_sum > 0) else 0.0

    p = p / p_sum
    q = q / q_sum

    # jensenshannon returns JS distance = sqrt(JSD), so we square it.
    js_dist = jensenshannon(p, q, base=base)
    return float(js_dist**2)


def _compute_fidelity_score(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    continuous_cols: List[str],
    categorical_cols: List[str],
) -> float:
    """
    Compute column-weighted fidelity score across all features.

    Score formulation:
        score = mean(W_1(col_1), W_1(col_2), ..., JS(col_k), ...)

    This is a COLUMN-WEIGHTED mean (not type-weighted). Each feature contributes
    equally regardless of its type (continuous vs categorical). This design ensures:
      - Datasets with many categorical columns (e.g., Telco: 16 cat, 3 cont) are
        not dominated by either type.
      - The score is interpretable as "average per-feature distributional distance".

    Args:
        real_df: Real DataFrame in ORIGINAL domain (after inverse_transform).
        synth_df: Synthetic DataFrame in the same domain as real_df.
        continuous_cols: List of continuous column names.
        categorical_cols: List of categorical column names.

    Returns:
        Mean per-feature distributional distance (float >= 0, lower = better).
    """
    scores: List[float] = []

    # Continuous columns: exact Wasserstein-1 distance (optimal transport).
    # Each column is normalised to [0, 1] using the real column's range,
    # ensuring scale invariance across columns with different units.
    for col in continuous_cols:
        if col in real_df.columns and col in synth_df.columns:
            real_vals = pd.to_numeric(real_df[col], errors="coerce").dropna().values
            synth_vals = pd.to_numeric(synth_df[col], errors="coerce").dropna().values
            if len(real_vals) > 0 and len(synth_vals) > 0:
                scores.append(_wasserstein_1d(real_vals, synth_vals))

    # Categorical columns: JS divergence (information theory).
    for col in categorical_cols:
        if col in real_df.columns and col in synth_df.columns:
            scores.append(
                _js_divergence(real_df[col], synth_df[col], base=2.0)
            )

    if not scores:
        # Penalty: returning 0.0 would reward a completely collapsed model
        # (all-NaN synthetic output across all columns) as "perfect" under a
        # minimize objective. Return maximum penalty (1.0) since both
        # Wasserstein and JS are normalised to [0, 1].
        return 1.0

    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# HPORunner
# ---------------------------------------------------------------------------


class HPORunner:
    """
    Optuna-based hyperparameter optimiser for generative tabular models.

    Uses the Tree-structured Parzen Estimator (TPE) sampler (Bergstra et al., 2011),
    which models P(score | hyperparams) and P(hyperparams) to guide the search
    toward promising regions of parameter space. TPE is preferred over random
    search or grid search because generative model training is expensive and
    the parameter space is high-dimensional.

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
    seed : int
        Random seed for reproducibility of the train/val split and the TPE
        sampler. In a K-Fold CV setting, each fold should pass a different
        seed (e.g. base_seed + fold) to ensure independent HPO runs.
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
        seed: int = 42,
    ) -> None:
        from src.training.trainer import ModelTrainer  # noqa: F401, deferred to avoid circular import

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
        self.seed = seed

        # Remaining epsilon after HPO allocation
        self._epsilon_for_hpo = total_epsilon * hpo_budget_fraction
        self._epsilon_for_training = total_epsilon * (1.0 - hpo_budget_fraction)

        logger.info(
            "HPORunner: model='%s', n_trials=%d, enable_dp=%s, "
            "epsilon_for_hpo=%.3f, epsilon_for_training=%.3f, seed=%d.",
            model_type, n_trials, enable_dp,
            self._epsilon_for_hpo, self._epsilon_for_training, seed,
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
        pipeline: Optional["PreprocessingPipeline"] = None,
    ) -> Dict[str, Any]:
        """
        Run the HPO study and return the best hyperparameter configuration.

        Args:
            preprocessed_df     : Full preprocessed DataFrame (output of
                                  PreprocessingPipeline.fit_transform()).
            continuous_cols     : Original continuous column names.
            categorical_cols    : Original categorical column names.
            constraint_expressions: Constraint expressions for satisfaction scoring.
            pipeline            : Fitted PreprocessingPipeline. REQUIRED when
                                  constraint_expressions are provided (constraint
                                  expressions reference original column names, which
                                  only exist after inverse_transform). If pipeline is
                                  NOT provided but constraints ARE provided, the
                                  constraint check will silently produce wrong results
                                  (0.0 violation rate) because column names in the
                                  encoded space do not match constraint expression
                                  column names.

        Returns:
            dict of best hyperparameters ready to pass to ModelTrainer.train().
        """
        # ------------------------------------------------------------------
        # Guard: constraints REQUIRE inverse_transform
        # ------------------------------------------------------------------
        # Constraint expressions reference ORIGINAL column names (e.g.,
        # "TotalCharges", "tenure", "MonthlyCharges"). In the encoded feature
        # space, categorical columns are one-hot encoded (e.g.,
        # "education_10th", "education_11th"), and continuous columns are
        # MinMax-scaled to [0, 1] or [-1, 1]. Evaluating constraints on
        # encoded data produces SILENTLY WRONG results: either column names
        # are not found (categorical) or values are on the wrong scale
        # (continuous). We therefore RAISE an error if constraints are
        # provided without a pipeline.
        if constraint_expressions and pipeline is None:
            raise ValueError(
                "Constraint expressions were provided but no preprocessing pipeline "
                "was given. Constraint expressions reference original column names "
                "(e.g., 'TotalCharges = tenure * MonthlyCharges'), which only exist "
                "after inverse_transform. Pass a fitted PreprocessingPipeline via "
                "the pipeline= parameter."
            )

        try:
            import optuna
        except ImportError as exc:
            raise ImportError(
                "Optuna is required for HPO. Install it with: pip install optuna"
            ) from exc

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # ---------------------------------------------------------------
        # Train/val split
        # ---------------------------------------------------------------
        # NOTE: This split is on the PREPROCESSED (encoded/scaled) data.
        #
        # If pipeline is provided, val_df is inverse-transformed once before
        # the trial loop (avoids redundant inverse_transform calls across
        # trials). If pipeline is NOT provided:
        #   - Continuous columns: Wasserstein on MinMax-scaled values. This
        #     is acceptable because both real and synthetic use the same
        #     scaling parameters (fit only on real data).
        #   - Categorical columns: one-hot/label-encoded. JS divergence
        #     CANNOT be computed because original category labels are lost.
        #     Categorical columns are skipped in this case.
        rng = np.random.RandomState(self.seed)
        n = len(preprocessed_df)
        val_n = max(1, int(n * self.val_fraction))
        val_idx = rng.choice(n, size=val_n, replace=False)
        train_mask = np.ones(n, dtype=bool)
        train_mask[val_idx] = False

        train_df = preprocessed_df.iloc[train_mask].reset_index(drop=True)
        val_df_encoded = preprocessed_df.iloc[val_idx].reset_index(drop=True)

        logger.info(
            "HPO train/val split: %d train rows, %d val rows (seed=%d).",
            len(train_df), len(val_df_encoded), self.seed,
        )

        # Pre-compute inverse-transformed validation data if pipeline available.
        val_df_decoded: Optional[pd.DataFrame] = None
        if pipeline is not None:
            val_df_decoded = pipeline.inverse_transform(val_df_encoded)
            logger.info(
                "Validation data inverse-transformed: shape=%s.",
                val_df_decoded.shape,
            )

        # ---------------------------------------------------------------
        # Objective function
        # ---------------------------------------------------------------

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
                _ = trainer.train(
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

                # -------------------------------------------------------
                # Generate synthetic samples for fidelity scoring
                # -------------------------------------------------------
                model = trainer.model
                col_meta = trainer.col_meta
                n_synth = min(len(val_df_encoded), 500)
                # Defensive: wrap sample() in torch.no_grad() to prevent
                # computation graph accumulation across HPO trials, which
                # would lead to GPU OOM. Even though individual model.sample()
                # implementations already use no_grad internally, this provides
                # a defence-in-depth guarantee at the HPO layer.
                with torch.no_grad():
                    synth_tensor = model.sample(n_synth)
                synth_df_encoded = self._tensor_to_df(
                    synth_tensor, col_meta, val_df_encoded.columns.tolist()
                )

                # -------------------------------------------------------
                # Fidelity score
                # -------------------------------------------------------
                if val_df_decoded is not None and pipeline is not None:
                    # Evaluate fidelity on INVERSE-TRANSFORMED data (original domain).
                    # Categorical columns have original string labels → JS works.
                    # Continuous columns are in original units → Wasserstein computed
                    # with within-column normalisation using real column range.
                    synth_df_decoded = pipeline.inverse_transform(synth_df_encoded)
                    fidelity_score = _compute_fidelity_score(
                        val_df_decoded,
                        synth_df_decoded,
                        continuous_cols,
                        categorical_cols,
                    )
                else:
                    # Fallback: evaluate on ENCODED data (pipeline not provided).
                    # Categorical columns skipped because one-hot encoding destroys
                    # original category labels.
                    if categorical_cols:
                        logger.debug(
                            "HPO trial %d: categorical fidelity skipped because "
                            "no pipeline was provided for inverse_transform.",
                            trial.number,
                        )
                    fidelity_score = _compute_fidelity_score(
                        val_df_encoded,
                        synth_df_encoded,
                        continuous_cols,
                        [],  # no categorical columns in encoded space
                    )

                # -------------------------------------------------------
                # Constraint violation rate
                # -------------------------------------------------------
                constraint_score = 0.0
                if constraint_expressions:
                    # Guard ensures pipeline is not None here (see ValueError above).
                    from src.models.constraints import ConstraintsEngine

                    synth_df_for_constraints = pipeline.inverse_transform(
                        synth_df_encoded
                    )
                    engine = ConstraintsEngine(constraint_expressions)
                    constraint_score = engine.violation_rate(synth_df_for_constraints)

                # -------------------------------------------------------
                # Combined objective (weighted sum)
                # -------------------------------------------------------
                objective_value = (
                    self.fidelity_weight * fidelity_score
                    + (1.0 - self.fidelity_weight) * constraint_score
                )
                logger.info(
                    "Trial %d: fidelity=%.4f, constraint_violation=%.4f, "
                    "objective=%.4f | %s",
                    trial.number,
                    fidelity_score,
                    constraint_score,
                    objective_value,
                    params,
                )
                return objective_value

            except Exception as exc:
                logger.warning("Trial %d failed: %s", trial.number, exc)
                return float("inf")

        # ---------------------------------------------------------------
        # Run Optuna study
        # ---------------------------------------------------------------
        sampler = optuna.samplers.TPESampler(seed=self.seed)
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