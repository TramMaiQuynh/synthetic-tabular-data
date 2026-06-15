"""
Privacy Auditor
---------------
Audits privacy leakage risks of synthetic datasets:
1. DCR (Distance to Closest Record): Minimum L2 distance to real records.
2. NNDR (Nearest Neighbor Distance Ratio): Ratio of distance to closest vs second closest.
3. Attacker Simulation:
   - Membership Inference Attack (MIA) via distance-based method (Stadler et al., 2022).
   - Attribute Inference Attack (AIA) accuracy/F1 score for a sensitive column.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, mean_squared_error, r2_score
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)


def compute_dcr_nndr(
    real_numeric: np.ndarray,
    synth_numeric: np.ndarray,
    max_samples: int = 5000,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute DCR (Distance to Closest Record) and NNDR (Nearest Neighbor Distance Ratio).
    
    To maintain scalability on CPU, we can downsample the inputs to max_samples.
    Uses a fixed random seed for reproducibility in audit settings.
    """
    # Use a deterministic RNG for reproducible downsampling (H-4 fix)
    rng = np.random.RandomState(random_state)
    
    # Downsample if necessary to prevent long CPU calculations
    if len(real_numeric) > max_samples:
        idx = rng.choice(len(real_numeric), max_samples, replace=False)
        real_numeric = real_numeric[idx]
    if len(synth_numeric) > max_samples:
        idx = rng.choice(len(synth_numeric), max_samples, replace=False)
        synth_numeric = synth_numeric[idx]
        
    # Fit Nearest Neighbors on real data
    # We query 2 nearest neighbors (for DCR and NNDR calculations)
    nn = NearestNeighbors(n_neighbors=2, algorithm="auto", metric="minkowski", p=2)
    nn.fit(real_numeric)
    
    distances, _ = nn.kneighbors(synth_numeric)
    
    d1 = distances[:, 0]
    d2 = distances[:, 1]
    
    # NNDR = d1 / d2. Handle d2 == 0 by setting NNDR to 1.0 (no leakage indicator)
    nndr = np.where(d2 > 0.0, d1 / d2, 1.0)
    
    return d1, nndr


class PrivacyAuditor:
    """Audits privacy leakage metrics of synthetic tabular data."""
    
    def __init__(
        self,
        continuous_cols: List[str],
        categorical_cols: List[str],
        sensitive_col: Optional[str] = None,
        max_samples: int = 5000,
    ) -> None:
        self.continuous_cols = continuous_cols
        self.categorical_cols = categorical_cols
        self.sensitive_col = sensitive_col
        self.max_samples = max_samples
        
    def evaluate(
        self,
        real_train_df: pd.DataFrame,
        real_test_df: pd.DataFrame,
        synth_df: pd.DataFrame,
        pipeline_loader_fn, # function that takes a df and returns its numeric scaled/encoded numpy array
    ) -> Dict[str, Any]:
        """
        Evaluate DCR, NNDR, MIA, and AIA.
        
        Args:
            real_train_df: Real data used in training the generator.
            real_test_df: Holdout real data not used in training.
            synth_df: Generated synthetic data.
            pipeline_loader_fn: Function to convert raw df to processed numeric numpy array.
                The function MUST return MinMax-normalized data in [0, 1] for
                DCR thresholds and MIA distance metrics to be meaningful.
            
        Returns:
            Dict containing privacy metrics.
        """
        results: Dict[str, Any] = {}
        
        # 1. Convert dataframes to normalized numeric representation
        real_train_num = pipeline_loader_fn(real_train_df)
        real_test_num = pipeline_loader_fn(real_test_df)
        synth_num = pipeline_loader_fn(synth_df)
        
        # 2. Geometric Metrics (DCR and NNDR) using real_train as reference
        dcr, nndr = compute_dcr_nndr(real_train_num, synth_num, self.max_samples)
        
        results["dcr_vals"] = dcr
        results["nndr_vals"] = nndr
        results["dcr_mean"] = float(np.mean(dcr))
        results["dcr_min"] = float(np.min(dcr))
        results["nndr_mean"] = float(np.mean(nndr))
        results["nndr_min"] = float(np.min(nndr))
        
        # H-3 fix: Use 5th-percentile based leakage detection instead of hard-coded 0.01.
        # Records whose DCR falls below the 5th percentile of the holdout-to-train
        # DCR distribution are considered suspiciously close (potential memorization).
        holdout_dcr, _ = compute_dcr_nndr(real_train_num, real_test_num, self.max_samples)
        leakage_threshold = float(np.percentile(holdout_dcr, 5)) if len(holdout_dcr) > 0 else 0.01
        # Ensure threshold is at least a small epsilon to avoid division issues
        leakage_threshold = max(leakage_threshold, 1e-8)
        
        results["dcr_leakage_threshold"] = leakage_threshold
        results["dcr_leakage_pct"] = float(np.mean(dcr < leakage_threshold) * 100)
        
        # 3. Membership Inference Attack (MIA) Simulation — distance-based (C-2 fix)
        results["mia_auc"] = self._simulate_mia(real_train_num, real_test_num, synth_num)
        
        # 4. Attribute Inference Attack (AIA) Simulation
        if self.sensitive_col and self.sensitive_col in real_train_df.columns:
            results["aia"] = self._simulate_aia(real_train_df, real_test_df, synth_df)
            
        return results
        
    def _simulate_mia(
        self,
        real_train_num: np.ndarray,
        real_test_num: np.ndarray,
        synth_num: np.ndarray,
    ) -> float:
        """
        Simulate a Membership Inference Attack using the distance-based method.
        
        This follows the approach of Stadler et al. (2022): for each real record,
        compute the distance to its nearest synthetic record. Records that were
        used to train the generator ("members") tend to be closer to synthetic
        data than records that were not ("non-members"), because the generator
        memorizes patterns from its training set.
        
        The attacker's hypothesis:
            member records have smaller distance to synthetic data.
        
        We use negative distance as a "membership score" and compute ROC-AUC
        to quantify the attacker's distinguishing power.
        
        Returns:
            ROC-AUC score: ~0.5 = random guess (strong privacy),
                           >0.6 = moderate leakage, >0.8 = severe leakage.
        """
        # Cap samples for computational efficiency
        max_eval = min(2500, self.max_samples)
        
        n_train = len(real_train_num)
        n_test = len(real_test_num)
        
        if n_train < 10 or n_test < 10:
            # Not enough data for meaningful evaluation
            return 0.5
        
        # Subsample deterministically
        rng = np.random.RandomState(42)
        
        train_idx = rng.choice(n_train, min(n_train, max_eval), replace=False)
        test_idx = rng.choice(n_test, min(n_test, max_eval), replace=False)
        
        eval_train = real_train_num[train_idx]
        eval_test = real_test_num[test_idx]
        
        # Subsample synthetic data for the NN index
        n_synth = len(synth_num)
        synth_sample = synth_num
        if n_synth > self.max_samples:
            synth_idx = rng.choice(n_synth, self.max_samples, replace=False)
            synth_sample = synth_num[synth_idx]
        
        # Fit nearest neighbor on synthetic data
        nn_synth = NearestNeighbors(n_neighbors=1, algorithm="auto", metric="euclidean")
        nn_synth.fit(synth_sample)
        
        # Compute distance to nearest synthetic record for each real record
        dist_train, _ = nn_synth.kneighbors(eval_train)
        dist_test, _ = nn_synth.kneighbors(eval_test)
        
        # Membership labels: 1 = member (train), 0 = non-member (test)
        y_true = np.array([1] * len(dist_train) + [0] * len(dist_test))
        
        # Membership score: negative distance (closer = higher score = more likely member)
        scores = np.concatenate([-dist_train.ravel(), -dist_test.ravel()])
        
        try:
            auc = roc_auc_score(y_true, scores)
            return float(auc)
        except Exception:
            return 0.5
            
    def _simulate_aia(
        self,
        real_train_df: pd.DataFrame,
        real_test_df: pd.DataFrame,
        synth_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Simulate an Attribute Inference Attack.
        
        Train a predictor on synthetic data to guess the sensitive column,
        and evaluate its performance on the real holdout test data.
        """
        col = self.sensitive_col
        if col is None:
            return {}
            
        # Split features and target
        def get_xy(df):
            X = df.drop(columns=[col])
            # One-hot encode categorical features for the sklearn model
            # Do NOT use drop_first=True to avoid semantic drift when
            # real and synthetic have different category sets (M-2 aligned fix)
            X = pd.get_dummies(X, drop_first=False)
            y = df[col]
            return X, y
            
        try:
            # We align the columns of real and synthetic features
            X_s, y_s = get_xy(synth_df)
            X_test, y_test = get_xy(real_test_df)
            
            # M-1 fix: Use join="outer" to preserve all columns from both sides
            # fill_value=0 ensures missing categories get zero-filled instead of dropped
            X_s, X_test = X_s.align(X_test, join="outer", axis=1, fill_value=0)
            
            if X_s.empty or len(y_s) == 0 or len(y_test) == 0:
                return {}
                
            # Determine task type
            # H-2 fix: Replace deprecated is_categorical_dtype with isinstance check
            is_classification = (
                col in self.categorical_cols or
                isinstance(real_train_df[col].dtype, pd.CategoricalDtype) or
                pd.api.types.is_object_dtype(real_train_df[col]) or
                len(real_train_df[col].unique()) <= 10
            )
            
            if is_classification:
                # Classify
                clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
                clf.fit(X_s, y_s.astype(str))
                preds = clf.predict(X_test)
                
                acc = accuracy_score(y_test.astype(str), preds)
                try:
                    f1 = f1_score(y_test.astype(str), preds, average="macro")
                except Exception:
                    f1 = acc
                    
                return {
                    "task": "classification",
                    "accuracy": float(acc),
                    "f1_score": float(f1),
                }
            else:
                # Regress
                reg = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
                reg.fit(X_s, y_s)
                preds = reg.predict(X_test)
                
                mse = mean_squared_error(y_test, preds)
                r2 = r2_score(y_test, preds)
                
                return {
                    "task": "regression",
                    "mse": float(mse),
                    "r2_score": float(r2),
                }
        except Exception as exc:
            logger.warning("Attribute Inference Attack simulation failed: %s", exc)
            return {"error": str(exc)}
