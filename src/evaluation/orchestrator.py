"""
Evaluation Suite Orchestrator
----------------------------
Coordinates the entire evaluation process.
Exposes a clean API to load data, run assessments, generate visual plots,
and compile HTML/Markdown compliance reports.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Callable

from src.evaluation.fidelity import FidelityAssessor
from src.evaluation.privacy import PrivacyAuditor
from src.evaluation.utility import UtilityEvaluator
from src.evaluation.visual import VisualOverlayGenerator
from src.evaluation.report import ComplianceReporter
from src.config.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class EvaluationSuite:
    """Orchestrates Module 3 evaluation: fidelity, privacy, utility, visual, and reporting."""
    
    def __init__(
        self,
        dataset_name: str,
        artifacts_root: Optional[str] = None,
        eval_dir: Optional[str] = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.config = ConfigLoader.load_config(dataset_name)
        self.schema = ConfigLoader.load_schema(dataset_name)
        
        # Read column lists from schema
        self.continuous_cols = list(self.schema.get("continuous_features", {}).keys())
        self.categorical_cols = list(self.schema.get("categorical_features", {}).keys())
        self.target_col = self.schema.get("target_column", "")
        self.pii_columns = self.schema.get("PII_columns_to_drop", [])
        
        # Resolve output directories
        if eval_dir is not None:
            self.eval_dir = os.path.abspath(eval_dir)
        else:
            if artifacts_root is None:
                # Default root relative to workspace
                artifacts_root = os.path.abspath(os.path.join(
                    os.path.dirname(__file__), "..", "..", "artifacts"
                ))
            self.eval_dir = os.path.join(artifacts_root, dataset_name, "evaluation")
            
        self.plots_dir = os.path.join(self.eval_dir, "plots")
        
        os.makedirs(self.plots_dir, exist_ok=True)
        
    def run_evaluation(
        self,
        real_df: pd.DataFrame,
        synth_df: pd.DataFrame,
        real_train_df: Optional[pd.DataFrame] = None,
        real_test_df: Optional[pd.DataFrame] = None,
        target_col: Optional[str] = None,
        sensitive_col: Optional[str] = None,
        pipeline_loader_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    ) -> Dict[str, Any]:
        """
        Execute full evaluation workflow.
        
        Args:
            real_df: Complete real dataframe.
            synth_df: Generated synthetic dataframe.
            real_train_df: Real train dataframe (optional, auto-split if not provided).
            real_test_df: Real test dataframe (optional, auto-split if not provided).
            target_col: ML utility target column (optional, read from schema if not provided).
            sensitive_col: AIA sensitive column (optional, defaults to target_col).
            pipeline_loader_fn: Function mapping df -> numeric values (optional).
            
        Returns:
            Dict containing evaluation results.
        """
        logger.info("Starting EvaluationSuite run for dataset '%s'...", self.dataset_name)
        
        # 1. Align column schema
        # Filter out dropped PII columns
        active_continuous = [c for c in self.continuous_cols if c not in self.pii_columns]
        active_categorical = [c for c in self.categorical_cols if c not in self.pii_columns]
        
        # Coerce continuous columns to numeric to prevent conversion errors during metrics calculation
        real_df = real_df.copy()
        synth_df = synth_df.copy()
        for col in active_continuous:
            if col in real_df.columns:
                real_df[col] = pd.to_numeric(real_df[col], errors='coerce')
            if col in synth_df.columns:
                synth_df[col] = pd.to_numeric(synth_df[col], errors='coerce')
        
        # Resolve target and sensitive columns
        t_col = target_col or self.target_col
        if not t_col:
            # Fallback if schema doesn't specify a target: use the last column in df
            t_col = real_df.columns[-1]
            logger.info("No target column specified or in schema; falling back to last column: '%s'", t_col)
            
        s_col = sensitive_col or t_col
        
        # 2. Split real data if not provided.
        # -----------------------------------------------------------------
        # IMPORTANT: This fallback split is only valid when EvaluationSuite
        # is used STANDALONE (i.e. the generative model was trained on the
        # same 80 % partition that this split produces, by coincidence).
        #
        # When called from run_pipeline.py the caller ALWAYS supplies
        # real_train_df and real_test_df (split before generator training),
        # so this block should never execute in a correct full-pipeline run.
        #
        # If this warning appears in the logs, it means either:
        #   a) EvaluationSuite is being used directly without a pre-split, OR
        #   b) run_pipeline.py forgot to pass real_train_df / real_test_df.
        #
        # In case (b), MIA results are INVALID: the generator has seen the
        # entire real_df, so there are no true non-members to compare against.
        # -----------------------------------------------------------------
        if real_train_df is None or real_test_df is None:
            logger.warning(
                "FALLBACK SPLIT TRIGGERED inside EvaluationSuite. "
                "real_train_df / real_test_df were not supplied by the caller. "
                "If the generative model was trained on all of real_df, "
                "MIA and DCR-Leakage results will be UNRELIABLE because the "
                "test split contains no true non-members. "
                "Pass real_train_df and real_test_df from run_pipeline.py to suppress this."
            )
            n = len(real_df)
            train_size = int(n * 0.8)
            real_shuffled = real_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
            real_train_df = real_shuffled.iloc[:train_size].reset_index(drop=True)
            real_test_df = real_shuffled.iloc[train_size:].reset_index(drop=True)
        else:
            real_train_df = real_train_df.copy()
            real_test_df = real_test_df.copy()
            for col in active_continuous:
                if col in real_train_df.columns:
                    real_train_df[col] = pd.to_numeric(real_train_df[col], errors='coerce')
                if col in real_test_df.columns:
                    real_test_df[col] = pd.to_numeric(real_test_df[col], errors='coerce')
            
        # 3. Setup default pipeline loader function if not provided
        if pipeline_loader_fn is None:
            logger.info("No pipeline_loader_fn provided. Instantiating a preprocessing pipeline helper...")
            from src.preprocessing.pipeline import PreprocessingPipeline
            
            # Setup a helper pipeline
            helper_pipeline = PreprocessingPipeline(self.dataset_name)
            # Drop PII
            clean_real = real_df.drop(columns=[c for c in self.pii_columns if c in real_df.columns], errors="ignore")
            # Fit and transform
            helper_pipeline.fit_transform(clean_real)
            
            def default_loader(df: pd.DataFrame):
                # Apply preprocessing pipeline and return values as float32 array
                processed = helper_pipeline.transform(df)
                return processed.values.astype("float32")
                
            pipeline_loader_fn = default_loader
            
        # 4. Assess Statistical Fidelity
        logger.info("[1/5] Evaluating Statistical Fidelity...")
        fidelity_assessor = FidelityAssessor(active_continuous, active_categorical)
        fidelity_results = fidelity_assessor.evaluate(real_df, synth_df)
        
        # 5. Audit Privacy Leakage
        logger.info("[2/5] Evaluating Privacy Leakage (DCR, NNDR, MIA, AIA)...")
        privacy_auditor = PrivacyAuditor(
            active_continuous, active_categorical, sensitive_col=s_col
        )
        privacy_results = privacy_auditor.evaluate(
            real_train_df, real_test_df, synth_df, pipeline_loader_fn
        )
        
        # 6. Evaluate Machine Learning Utility
        # Read utility_exclude_features from schema (e.g. duration for Bank Marketing)
        # These features are dropped before TSTR/TRTR to avoid leakage/ceiling effect
        utility_exclude = self.schema.get("utility_exclude_features", [])
        if utility_exclude:
            logger.info("Utility evaluation will exclude leakage features: %s", utility_exclude)
        logger.info("[3/5] Evaluating Machine Learning Utility (TSTR vs TRTR)...")
        utility_evaluator = UtilityEvaluator(
            t_col, active_continuous, active_categorical,
            exclude_features=utility_exclude,
        )
        utility_results = utility_evaluator.evaluate(real_train_df, real_test_df, synth_df)
        
        # 7. Generate Visual Overlays
        logger.info("[4/5] Generating Visual Plots...")
        visual_generator = VisualOverlayGenerator(
            active_continuous, active_categorical, self.plots_dir
        )
        
        dist_grid_path = visual_generator.plot_distributions(real_df, synth_df)
        corr_comp_path = visual_generator.plot_correlation_difference(
            fidelity_results["real_corr"], fidelity_results["synth_corr"]
        )
        dcr_leakage_threshold = privacy_results.get("dcr_leakage_threshold", 0.01)
        dcr_dist_path = visual_generator.plot_dcr_distribution(
            privacy_results["dcr_vals"],
            leakage_threshold=dcr_leakage_threshold
        )
        
        # L-4 fix: Safe relpath helper that handles cross-drive paths on Windows
        # (os.path.relpath raises ValueError when paths are on different drives)
        def _safe_relpath(path: str, base: str) -> str:
            try:
                return os.path.relpath(path, base)
            except ValueError:
                return path  # Fallback to absolute path
        
        # Create relative paths from eval_dir to display correctly in reports
        rel_plots = {
            "distributions": _safe_relpath(dist_grid_path, self.eval_dir),
            "correlation": _safe_relpath(corr_comp_path, self.eval_dir),
            "dcr": _safe_relpath(dcr_dist_path, self.eval_dir),
        }
        
        # 8. Generate Compliance Reports
        logger.info("[5/5] Generating Compliance Reports...")
        reporter = ComplianceReporter(self.dataset_name, self.eval_dir)
        md_path, html_path = reporter.generate_report(
            fidelity_results,
            privacy_results,
            utility_results,
            rel_plots,
            target_col=t_col,
            sensitive_col=s_col,
        )
        
        logger.info("EvaluationSuite run complete. Saved reports to %s", self.eval_dir)
        
        # Compile summary
        # Use NaN instead of 0.0 when there are no columns of a given type,
        # so that downstream consumers (reports, experiment runner) can
        # distinguish "no data" from "perfect score".
        return {
            "fidelity": {
                "avg_js": float(np.mean(list(fidelity_results["js_divergence"].values()))) if fidelity_results["js_divergence"] else float("nan"),
                "avg_wasserstein": float(np.mean(list(fidelity_results["wasserstein"].values()))) if fidelity_results["wasserstein"] else float("nan"),
                "correlation_difference": fidelity_results["correlation_difference"],
            },
            "privacy": {
                "dcr_mean": privacy_results["dcr_mean"],
                "dcr_leakage_pct": privacy_results["dcr_leakage_pct"],
                "mia_auc": privacy_results["mia_auc"],
            },
            "utility": {
                "task": utility_results["task"],
                "metrics": {
                    model: {
                        "TRTR": metrics["TRTR"],
                        "TSTR": metrics["TSTR"],
                    }
                    for model, metrics in utility_results["metrics"].items()
                }
            },
            "report_paths": {
                "markdown": md_path,
                "html": html_path,
            }
        }
