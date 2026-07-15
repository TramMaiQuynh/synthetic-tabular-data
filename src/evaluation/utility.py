"""
ML Utility Evaluator (TSTR Framework)
--------------------------------------
Validates synthetic dataset usefulness for machine learning tasks:
1. Performs Train-on-Synthetic, Test-on-Real (TSTR).
2. Performs Train-on-Real, Test-on-Real (TRTR) as a baseline.
3. Automatically detects Classification vs Regression tasks.
4. Uses RandomForest, GradientBoosting, and Linear/Logistic Regression.
"""

import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)


class UtilityEvaluator:
    """Evaluates ML utility of synthetic data using the TSTR vs TRTR framework."""
    
    def __init__(
        self,
        target_col: str,
        continuous_cols: List[str],
        categorical_cols: List[str],
    ) -> None:
        self.target_col = target_col
        self.continuous_cols = continuous_cols
        self.categorical_cols = categorical_cols
        
    def evaluate(
        self,
        real_train_df: pd.DataFrame,
        real_test_df: pd.DataFrame,
        synth_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Run TSTR and TRTR and return performance comparison metrics.
        
        Args:
            real_train_df: Real training data.
            real_test_df: Real holdout test data.
            synth_df: Synthetic training data.
            
        Returns:
            Dict containing TSTR vs TRTR comparison results.
        """
        results: Dict[str, Any] = {
            "task": "classification",
            "target_column": self.target_col,
            "metrics": {},
        }
        
        if self.target_col not in real_train_df.columns or self.target_col not in real_test_df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in training/testing datasets.")
            
        # 1. Determine task type
        # H-2 fix: Replace deprecated is_categorical_dtype with isinstance check
        is_classification = (
            self.target_col in self.categorical_cols or
            isinstance(real_train_df[self.target_col].dtype, pd.CategoricalDtype) or
            pd.api.types.is_object_dtype(real_train_df[self.target_col]) or
            len(real_train_df[self.target_col].unique()) <= 10
        )
        
        results["task"] = "classification" if is_classification else "regression"
        
        # 2. Separate features and target, and align one-hot column representations
        X_real_train, y_real_train = self._prepare_xy(real_train_df, is_classification)
        X_real_test, y_real_test = self._prepare_xy(real_test_df, is_classification)
        X_synth, y_synth = self._prepare_xy(synth_df, is_classification)
        
        # Align features (some categories may not exist in one set, creating different get_dummies columns)
        X_real_train, X_real_test = X_real_train.align(X_real_test, join="outer", axis=1, fill_value=0)
        X_synth, _ = X_synth.align(X_real_train, join="right", axis=1, fill_value=0) # Match synthetic features exactly to train
        
        # Double check alignment with test set
        X_synth, X_real_test = X_synth.align(X_real_test, join="inner", axis=1)
        X_real_train, X_real_test = X_real_train.align(X_real_test, join="inner", axis=1)

        # Scale features using StandardScaler to ensure optimizer convergence (especially Logistic Regression)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_real_train = pd.DataFrame(
            scaler.fit_transform(X_real_train), columns=X_real_train.columns, index=X_real_train.index
        )
        X_real_test = pd.DataFrame(
            scaler.transform(X_real_test), columns=X_real_test.columns, index=X_real_test.index
        )
        X_synth = pd.DataFrame(
            scaler.transform(X_synth), columns=X_synth.columns, index=X_synth.index
        )
        
        # 3. Define models based on task
        if is_classification:
            models = {
                "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
                "GradientBoosting": GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
                "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
            }
        else:
            models = {
                "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
                "GradientBoosting": GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
                "LinearRegression": LinearRegression(),
            }
            
        # 4. Evaluate each model
        for name, model in models.items():
            logger.info("Evaluating ML Utility model: %s (%s)", name, results["task"])
            
            # TRTR (Train-on-Real, Test-on-Real)
            try:
                model.fit(X_real_train, y_real_train)
                preds_trtr = model.predict(X_real_test)
                probs_trtr = model.predict_proba(X_real_test) if is_classification and hasattr(model, "predict_proba") else None
                trtr_scores = self._compute_scores(y_real_test, preds_trtr, probs_trtr, is_classification)
            except Exception as exc:
                logger.error("TRTR fit failed for %s: %s", name, exc)
                trtr_scores = {}
                
            # TSTR (Train-on-Synthetic, Test-on-Real)
            try:
                # Re-instantiate the model to clear state
                model_cls = model.__class__
                model_params = model.get_params()
                model_tstr = model_cls(**model_params)
                
                model_tstr.fit(X_synth, y_synth)
                preds_tstr = model_tstr.predict(X_real_test)
                probs_tstr = model_tstr.predict_proba(X_real_test) if is_classification and hasattr(model_tstr, "predict_proba") else None
                tstr_scores = self._compute_scores(y_real_test, preds_tstr, probs_tstr, is_classification)
            except Exception as exc:
                logger.error("TSTR fit failed for %s: %s", name, exc)
                tstr_scores = {}
                
            # Store in results
            results["metrics"][name] = {
                "TRTR": trtr_scores,
                "TSTR": tstr_scores,
            }
            
        return results

    def _prepare_xy(self, df: pd.DataFrame, is_classification: bool) -> Tuple[pd.DataFrame, pd.Series]:
        """Separate features and target, and get_dummies for categorical variables."""
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        
        # L-2 fix: Fill missing values BEFORE astype(str) to avoid converting
        # NaN to the literal string "nan" which makes fillna a no-op.
        if y.isnull().any():
            if is_classification:
                mode_val = y.mode().iloc[0] if not y.mode().empty else "missing"
                y = y.fillna(mode_val).astype(str)
            else:
                y = y.fillna(y.median())
        elif is_classification:
            y = y.astype(str)
                
        # M-2 fix: Do NOT use drop_first=True. When real and synthetic have
        # different category sets, drop_first drops different categories on
        # each side, causing semantic drift after column alignment.
        X = pd.get_dummies(X, drop_first=False)
        
        # Fill remaining continuous NaNs with 0.0 for safety
        X = X.fillna(0.0)
        
        return X, y

    def _compute_scores(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray],
        is_classification: bool,
    ) -> Dict[str, float]:
        """Compute performance scores for predictions."""
        scores = {}
        if is_classification:
            # Classification Metrics
            y_true_str = y_true.astype(str)
            y_pred_str = y_pred.astype(str)
            
            scores["accuracy"] = float(accuracy_score(y_true_str, y_pred_str))
            try:
                scores["f1_macro"] = float(f1_score(y_true_str, y_pred_str, average="macro"))
            except Exception:
                scores["f1_macro"] = 0.0
                
            # If binary classification and probabilities are provided, compute ROC-AUC
            unique_classes = np.unique(y_true_str)
            if len(unique_classes) == 2 and y_prob is not None:
                try:
                    # Find probability column for class '1' or the second class
                    # Align y_true binary labels to match classes of the model
                    probs = y_prob[:, 1]
                    scores["auc_roc"] = float(roc_auc_score(y_true_str, probs))
                except Exception:
                    scores["auc_roc"] = 0.5
            else:
                scores["auc_roc"] = 0.5
        else:
            # Regression Metrics
            scores["mse"] = float(mean_squared_error(y_true, y_pred))
            scores["mae"] = float(mean_absolute_error(y_true, y_pred))
            scores["r2"] = float(r2_score(y_true, y_pred))
            
        return scores
