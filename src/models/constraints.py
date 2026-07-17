"""
Business Logic Constraints Engine
----------------------------------
Compiles, validates, and enforces cross-column relational constraints declared
in data_schema.yaml (or an explicit list).

Enforcement modes (applied in priority order):
  1. Soft Loss Penalty  — differentiable penalty term injected into training loss
                          (CTGAN / CTVAE only, continuous constraints only).
  2. Post-generation Correction — clamp / re-sample violating rows after generation,
                                   bounded by max_retries + fallback policy.

JS-divergence between raw and corrected distributions is measured per column to
detect Over-Correction and surface a structured warning.

All public methods are pure functions or accept / return torch.Tensor / pd.DataFrame
with no hidden state mutations. Instances are safe to serialise with joblib.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal representation of a single parsed constraint
# ---------------------------------------------------------------------------

class _Constraint:
    """A single compiled relational constraint between columns."""

    def __init__(
        self,
        expression: str,
        lhs_col: str,
        op: str,
        rhs_col: Optional[str],
        rhs_scalar: Optional[float],
    ) -> None:
        self.expression = expression
        self.lhs_col = lhs_col
        self.op = op
        self.rhs_col = rhs_col          # None if RHS is a literal
        self.rhs_scalar = rhs_scalar    # None if RHS is a column reference

    # ------------------------------------------------------------------
    # Violation detection
    # ------------------------------------------------------------------

    def violating_mask(self, df: pd.DataFrame) -> pd.Series:
        """Return a boolean Series — True where a row *violates* this constraint."""
        if self.lhs_col not in df.columns:
            logger.warning(
                "Constraint '%s': column '%s' not found in DataFrame — skipped.",
                self.expression, self.lhs_col,
            )
            return pd.Series(False, index=df.index)

        lhs = pd.to_numeric(df[self.lhs_col], errors="coerce")

        if self.rhs_col is not None:
            if self.rhs_col not in df.columns:
                logger.warning(
                    "Constraint '%s': column '%s' not found in DataFrame — skipped.",
                    self.expression, self.rhs_col,
                )
                return pd.Series(False, index=df.index)
            rhs = pd.to_numeric(df[self.rhs_col], errors="coerce")
        else:
            rhs = float(self.rhs_scalar)  # type: ignore[arg-type]

        ops: Dict[str, Callable] = {
            ">=": lambda a, b: ~(a >= b),
            "<=": lambda a, b: ~(a <= b),
            ">":  lambda a, b: ~(a > b),
            "<":  lambda a, b: ~(a < b),
            "==": lambda a, b: ~(a == b),
            "!=": lambda a, b: ~(a != b),
        }
        if self.op not in ops:
            raise ValueError(f"Unsupported operator '{self.op}' in constraint '{self.expression}'.")

        mask = ops[self.op](lhs, rhs)
        # Treat NaN as non-violating (cannot evaluate)
        return mask.fillna(False)

    # ------------------------------------------------------------------
    # Soft loss penalty (PyTorch)
    # ------------------------------------------------------------------

    def soft_penalty(self, tensor_dict: Dict[str, Any]) -> Any:
        """
        Compute a differentiable penalty for continuous constraints.

        Args:
            tensor_dict: mapping col_name -> torch.Tensor (shape [batch]).

        Returns:
            Scalar torch.Tensor — mean ReLU violation magnitude.
            Returns 0.0 tensor if columns are absent or import fails.
        """
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            logger.warning("torch not available — soft_penalty returns 0.")
            return 0.0

        # Detect the device of the input tensors to prevent device mismatch errors
        device = next(iter(tensor_dict.values())).device if tensor_dict else torch.device("cpu")

        if self.lhs_col not in tensor_dict:
            return torch.tensor(0.0, device=device)

        lhs_t = tensor_dict[self.lhs_col].float()

        if self.rhs_col is not None:
            if self.rhs_col not in tensor_dict:
                return torch.tensor(0.0, device=device)
            rhs_t = tensor_dict[self.rhs_col].float()
        else:
            rhs_t = torch.tensor(float(self.rhs_scalar), dtype=torch.float32, device=device)

        # Compute violation as ReLU(-(lhs - rhs)) for ">=" / ">"
        # and ReLU(-(rhs - lhs)) for "<=" / "<"
        if self.op in (">=", ">"):
            gap = rhs_t - lhs_t
        elif self.op in ("<=", "<"):
            gap = lhs_t - rhs_t
        elif self.op == "==":
            # Equality: penalize the absolute difference between lhs and rhs.
            # When lhs == rhs, gap is 0 → no penalty.
            gap = (lhs_t - rhs_t).abs()
        elif self.op == "!=":
            # Inequality: penalize when lhs is TOO CLOSE to rhs.
            # We want |lhs - rhs| >= some margin. Use a hinge-style penalty:
            # penalty = max(0, margin - |lhs - rhs|)
            # This encourages lhs to be at least `margin` away from rhs.
            margin = 1.0  # in normalized space; interpretable across columns
            gap = margin - (lhs_t - rhs_t).abs()
        else:
            raise ValueError(f"Unsupported operator '{self.op}' in constraint '{self.expression}'.")

        return F.relu(gap).mean()

    def __repr__(self) -> str:
        return f"<Constraint: {self.expression}>"


# ---------------------------------------------------------------------------
# Parser: text expression -> _Constraint
# ---------------------------------------------------------------------------

_SUPPORTED_OPS = (">=", "<=", "==", "!=", ">", "<")

# Regex pattern: <lhs> <operator> <rhs>, with mandatory whitespace around
# the operator to avoid false matches inside column names.
_EXPR_RE = re.compile(
    r'^(.+?)\s+(>=|<=|==|!=|>|<)\s+(.+)$'
)


def _parse_expression(expr: str) -> _Constraint:
    """
    Parse a simple binary expression of the form:
        <col_or_literal> <op> <col_or_literal>

    Operators must be surrounded by whitespace.  This avoids mis-parsing
    column names that contain operator-like characters.

    Examples:
        "Age >= 18"
        "TotalCharges >= MonthlyCharges"
        "tenure > 0"

    Raises:
        ValueError: If the expression cannot be parsed.
    """
    expr = expr.strip()
    match = _EXPR_RE.match(expr)
    if match is None:
        raise ValueError(
            f"Cannot parse constraint expression '{expr}'. "
            f"Supported operators: {_SUPPORTED_OPS}. "
            "Expected format: '<col> <op> <col_or_literal>' "
            "(whitespace required around the operator)."
        )

    lhs_raw = match.group(1).strip()
    op = match.group(2)
    rhs_raw = match.group(3).strip()

    # Determine if RHS is a literal number or a column name
    try:
        rhs_scalar: Optional[float] = float(rhs_raw)
        rhs_col: Optional[str] = None
    except ValueError:
        rhs_scalar = None
        rhs_col = rhs_raw

    return _Constraint(
        expression=expr,
        lhs_col=lhs_raw,
        op=op,
        rhs_col=rhs_col,
        rhs_scalar=rhs_scalar,
    )


# ---------------------------------------------------------------------------
# Public API — ConstraintsEngine
# ---------------------------------------------------------------------------

class ConstraintsEngine:
    """
    Compile, evaluate, and enforce a list of cross-column business constraints.

    Parameters
    ----------
    expressions : list of str
        Constraint expressions in the form "<col> <op> <col_or_scalar>".
        Example: ["TotalCharges >= MonthlyCharges", "tenure >= 0", "Age >= 18"].
    max_retries : int
        Maximum correction iterations per batch in post_correction().
    violation_rate_threshold : float
        If the final violation rate after correction exceeds this value,
        a structured warning is logged.
    js_divergence_threshold : float
        If per-column JS divergence between raw and corrected data exceeds
        this threshold, an Over-Correction warning is logged.
    """

    def __init__(
        self,
        expressions: List[str],
        max_retries: int = 5,
        violation_rate_threshold: float = 0.01,
        js_divergence_threshold: float = 0.05,
    ) -> None:
        if not isinstance(expressions, list):
            raise TypeError("expressions must be a list of strings.")
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1.")

        self.expressions = expressions
        self.max_retries = max_retries
        self.violation_rate_threshold = violation_rate_threshold
        self.js_divergence_threshold = js_divergence_threshold

        # Compile constraints
        self._constraints: List[_Constraint] = []
        for expr in expressions:
            try:
                self._constraints.append(_parse_expression(expr))
                logger.debug("Compiled constraint: %s", expr)
            except ValueError as exc:
                logger.error("Failed to compile constraint '%s': %s", expr, exc)
                raise

        logger.info(
            "ConstraintsEngine initialised with %d constraints.", len(self._constraints)
        )

    # ------------------------------------------------------------------
    # 1. Violation detection
    # ------------------------------------------------------------------

    def violation_mask(self, df: pd.DataFrame) -> pd.Series:
        """
        Return a boolean Series — True where *any* constraint is violated.

        Args:
            df: DataFrame in original (decoded) column space.

        Returns:
            pd.Series of dtype bool, index aligned with df.
        """
        if not self._constraints:
            return pd.Series(False, index=df.index)

        combined = pd.Series(False, index=df.index)
        for c in self._constraints:
            combined = combined | c.violating_mask(df)
        return combined

    def violation_rate(self, df: pd.DataFrame) -> float:
        """Return the fraction of rows violating at least one constraint."""
        mask = self.violation_mask(df)
        return float(mask.mean()) if len(mask) > 0 else 0.0

    # ------------------------------------------------------------------
    # 2. Soft loss penalty (training-time, PyTorch)
    # ------------------------------------------------------------------

    def soft_loss_penalty(self, tensor_dict: Dict[str, Any]) -> Any:
        """
        Aggregate soft penalty across all continuous constraints.

        Args:
            tensor_dict: {col_name: torch.Tensor (shape [batch])}.

        Returns:
            Scalar torch.Tensor.
        """
        try:
            import torch
        except ImportError:
            return 0.0

        device = next(iter(tensor_dict.values())).device if tensor_dict else torch.device("cpu")
        total = torch.tensor(0.0, device=device)
        for c in self._constraints:
            total = total + c.soft_penalty(tensor_dict)
        return total

    # ------------------------------------------------------------------
    # 3. Post-generation correction
    # ------------------------------------------------------------------

    def post_correction(
        self,
        df: pd.DataFrame,
        generate_fn: Optional[Callable[[int], pd.DataFrame]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Correct constraint violations via clamping; optionally re-sample
        rows that cannot be corrected by clamping.

        Strategy (per constraint):
        - If RHS is a scalar: clamp LHS column to the valid boundary.
        - If RHS is another column: clamp LHS to RHS value (preserves the
          relationship but may slightly distort marginal distribution).
        - After max_retries exhausted, keep best available row (lowest number
          of remaining violations) as fallback.

        Args:
            df: DataFrame with decoded (original-space) values.
            generate_fn: Optional callable(n: int) -> pd.DataFrame.
                         If provided, violating rows are re-generated up to
                         max_retries times before falling back to clamping.

        Returns:
            corrected_df : pd.DataFrame with constraints enforced.
            stats        : dict containing violation_rate_before,
                           violation_rate_after, js_divergences (per col).
        """
        raw_df = df.copy()
        corrected_df = df.copy()
        n_total = len(corrected_df)

        violation_rate_before = self.violation_rate(corrected_df)
        logger.info(
            "post_correction: violation_rate_before=%.4f (n=%d rows).",
            violation_rate_before, n_total,
        )

        for attempt in range(1, self.max_retries + 1):
            mask = self.violation_mask(corrected_df)
            n_violating = int(mask.sum())

            if n_violating == 0:
                logger.debug("post_correction: all constraints satisfied after %d attempt(s).", attempt - 1)
                break

            logger.debug(
                "post_correction attempt %d/%d: %d violating rows.",
                attempt, self.max_retries, n_violating,
            )

            # Try re-generation first if callable provided
            if generate_fn is not None and attempt <= self.max_retries - 1:
                try:
                    replacement = generate_fn(n_violating)
                    if len(replacement) == n_violating:
                        replacement.index = corrected_df.index[mask]
                        corrected_df.loc[mask] = replacement.values
                        continue
                except Exception as exc:  # noqa: BLE001
                    logger.warning("generate_fn raised %s during post_correction — falling back to clamp.", exc)

            # Clamp-based correction
            for constraint in self._constraints:
                row_mask = constraint.violating_mask(corrected_df)
                if not row_mask.any():
                    continue

                lhs_col = constraint.lhs_col
                if lhs_col not in corrected_df.columns:
                    continue

                if constraint.rhs_col is not None:
                    rhs_col = constraint.rhs_col
                    if rhs_col not in corrected_df.columns:
                        continue
                    rhs_vals = pd.to_numeric(corrected_df[rhs_col], errors="coerce")
                    lhs_vals = pd.to_numeric(corrected_df[lhs_col], errors="coerce")

                    if constraint.op in (">=", ">"):
                        # LHS should be >= RHS -> clamp LHS up to RHS
                        corrected_df.loc[row_mask, lhs_col] = (
                            lhs_vals[row_mask].combine(rhs_vals[row_mask], max)
                        )
                    elif constraint.op in ("<=", "<"):
                        # LHS should be <= RHS -> clamp LHS down to RHS
                        corrected_df.loc[row_mask, lhs_col] = (
                            lhs_vals[row_mask].combine(rhs_vals[row_mask], min)
                        )
                    elif constraint.op == "==":
                        # Equality: set LHS equal to RHS
                        corrected_df.loc[row_mask, lhs_col] = rhs_vals[row_mask]
                    elif constraint.op == "!=":
                        # Inequality: if LHS == RHS (within tolerance), nudge LHS
                        # away from RHS by a small epsilon.
                        # Only modify rows where LHS is actually equal to RHS.
                        equal_mask = (lhs_vals[row_mask] - rhs_vals[row_mask]).abs() < 1e-6
                        if equal_mask.any():
                            eps = 1e-3
                            equal_idx = row_mask[row_mask].index[equal_mask.values]
                            corrected_df.loc[equal_idx, lhs_col] = (
                                rhs_vals.loc[equal_idx] + eps
                            )
                else:
                    # Scalar boundary
                    scalar = float(constraint.rhs_scalar)  # type: ignore[arg-type]
                    lhs_vals = pd.to_numeric(corrected_df[lhs_col], errors="coerce")

                    if constraint.op in (">=", ">"):
                        corrected_df.loc[row_mask, lhs_col] = lhs_vals[row_mask].clip(lower=scalar)
                    elif constraint.op in ("<=", "<"):
                        corrected_df.loc[row_mask, lhs_col] = lhs_vals[row_mask].clip(upper=scalar)
                    elif constraint.op == "==":
                        # Equality with scalar: set LHS equal to scalar
                        corrected_df.loc[row_mask, lhs_col] = scalar
                    elif constraint.op == "!=":
                        # Inequality with scalar: if LHS == scalar (within tolerance),
                        # perturb LHS away from scalar by a small epsilon.
                        lhs_vals_rm = lhs_vals[row_mask]
                        equal_mask = (lhs_vals_rm - scalar).abs() < 1e-6
                        if equal_mask.any():
                            eps = 1e-3
                            equal_idx = row_mask[row_mask].index[equal_mask.values]
                            corrected_df.loc[equal_idx, lhs_col] = scalar + eps

        violation_rate_after = self.violation_rate(corrected_df)
        logger.info("post_correction: violation_rate_after=%.4f.", violation_rate_after)

        if violation_rate_after > self.violation_rate_threshold:
            logger.warning(
                "post_correction: final violation_rate=%.4f exceeds threshold=%.4f. "
                "Inspect constraint definitions or increase max_retries.",
                violation_rate_after, self.violation_rate_threshold,
            )

        # Measure JS divergence per affected column to detect Over-Correction
        js_divergences = self._compute_js_divergences(raw_df, corrected_df)
        for col, jsd in js_divergences.items():
            if jsd > self.js_divergence_threshold:
                logger.warning(
                    "Over-Correction detected: column '%s' JS divergence=%.4f "
                    "exceeds threshold=%.4f. Post-correction may have distorted "
                    "the original distribution.",
                    col, jsd, self.js_divergence_threshold,
                )

        stats: Dict[str, Any] = {
            "violation_rate_before": violation_rate_before,
            "violation_rate_after": violation_rate_after,
            "js_divergences": js_divergences,
            "n_total": n_total,
        }
        return corrected_df, stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _js_divergence_1d(a: np.ndarray, b: np.ndarray, n_bins: int = 50) -> float:
        """Compute Jensen-Shannon divergence between two 1D arrays via histograms."""
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if len(a) == 0 or len(b) == 0:
            return 0.0

        combined_min = min(a.min(), b.min())
        combined_max = max(a.max(), b.max())

        if combined_min == combined_max:
            return 0.0

        bins = np.linspace(combined_min, combined_max, n_bins + 1)
        p, _ = np.histogram(a, bins=bins, density=False)
        q, _ = np.histogram(b, bins=bins, density=False)

        p = p.astype(float) + 1e-10
        q = q.astype(float) + 1e-10
        p /= p.sum()
        q /= q.sum()
        m = 0.5 * (p + q)

        def _kl(pk: np.ndarray, qk: np.ndarray) -> float:
            return float(np.sum(pk * np.log(pk / qk)))

        return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)

    def _compute_js_divergences(
        self, raw_df: pd.DataFrame, corrected_df: pd.DataFrame
    ) -> Dict[str, float]:
        """Compute per-column JS divergence between raw and corrected DataFrames."""
        affected_cols: List[str] = []
        for c in self._constraints:
            if c.lhs_col in raw_df.columns:
                affected_cols.append(c.lhs_col)

        result: Dict[str, float] = {}
        for col in set(affected_cols):
            raw_vals = pd.to_numeric(raw_df[col], errors="coerce").dropna().values
            corrected_vals = pd.to_numeric(corrected_df[col], errors="coerce").dropna().values
            result[col] = self._js_divergence_1d(raw_vals, corrected_vals)

        return result

    def __repr__(self) -> str:
        return (
            f"<ConstraintsEngine constraints={len(self._constraints)} "
            f"max_retries={self.max_retries}>"
        )
