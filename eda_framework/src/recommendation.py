"""
EDA Framework - Stage 6 & 7: Recommendations and Blueprints
-----------------------------------------------------------
Aggregates all profiling analytics and privacy audits to construct a dynamic preprocessing
blueprint, writing schema and pipeline configurations in YAML.
"""

import os
import logging
import pandas as pd
from typing import Dict, Any, Optional
from eda_framework.utils.helpers import load_yaml, save_yaml
from eda_framework.src.validation import RawDataValidator
from eda_framework.src.privacy_audit import PrivacyAuditor
from eda_framework.src.missing_analysis import MissingValueAnalyzer
from eda_framework.src.profiling import FeatureProfiler
from eda_framework.src.association import RelationshipAnalyzer

logger = logging.getLogger(__name__)

class PreprocessingRecommender:
    def __init__(self, eda_config_path: str):
        self.eda_config = load_yaml(eda_config_path)
        self.validator = None
        self.auditor = PrivacyAuditor(self.eda_config)
        self.missing_analyzer = MissingValueAnalyzer(self.eda_config)
        self.profiler = FeatureProfiler(self.eda_config)
        self.relationship_analyzer = RelationshipAnalyzer()

    def run_all(self, file_path: str, dataset_name: str, target_col: Optional[str] = None) -> Dict[str, Any]:
        """Run entire EDA audit and construct data schema and preprocessing blueprint."""
        # 1. Validation & Loading
        # Ensure project root is in sys.path for reliable imports regardless of CWD
        import sys
        _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        if _project_root not in sys.path:
            sys.path.insert(0, _project_root)
        from src.config.config_loader import ConfigLoader
        config = ConfigLoader.load_config(dataset_name)

        self.validator = RawDataValidator(file_path)
        raw_struct = self.validator.inspect_raw_structure()
        
        read_kwargs = {}
        sep = raw_struct["delimiter"]
        if hasattr(config, "ingestion"):
            sep = getattr(config.ingestion, "separator", ",")
            if hasattr(config.ingestion, "has_header") and not config.ingestion.has_header:
                read_kwargs["header"] = None
                if hasattr(config.ingestion, "columns") and config.ingestion.columns:
                    read_kwargs["names"] = config.ingestion.columns
            if hasattr(config.ingestion, "na_values") and config.ingestion.na_values:
                read_kwargs["na_values"] = config.ingestion.na_values

        df, quality_report = self.validator.load_and_profile(
            sep=sep, 
            encoding=raw_struct["encoding"],
            **read_kwargs
        )

        # Load existing schema if it exists to preserve feature types
        project_config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "config"))
        existing_schema_path = os.path.join(project_config_dir, dataset_name, "data_schema.yaml")
        existing_schema = load_yaml(existing_schema_path) or {}

        continuous_cols = []
        categorical_cols = []
        
        existing_cont_feats = existing_schema.get("continuous_features", {})
        existing_cat_feats = existing_schema.get("categorical_features", {})
        
        if existing_cont_feats or existing_cat_feats:
            for col in df.columns:
                if col in existing_cont_feats:
                    continuous_cols.append(col)
                elif col in existing_cat_feats:
                    categorical_cols.append(col)
                else:
                    if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 20:
                        continuous_cols.append(col)
                    else:
                        categorical_cols.append(col)
        else:
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 20:
                    continuous_cols.append(col)
                else:
                    categorical_cols.append(col)

        # Coerce continuous columns to numeric to avoid type errors in analysis (e.g. skewness with string dtype)
        for col in continuous_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 2. Privacy Audit
        privacy_report = self.auditor.audit(df)

        # 3. Missing Value Analysis
        missing_report = self.missing_analyzer.analyze(df)

        # Filter out dropped columns before profiling
        recomm_drops = set(privacy_report["recommended_drops"] + missing_report["columns_to_drop_high_missing"])
        active_continuous = [c for c in continuous_cols if c not in recomm_drops]
        active_categorical = [c for c in categorical_cols if c not in recomm_drops]

        # 4. Feature Profiling
        profiling_report = self.profiler.profile(df, active_continuous, active_categorical)

        # 5. Relationship Analysis
        relationship_report = self.relationship_analyzer.analyze(df, active_continuous, active_categorical)

        # 6. Build the Data Schema blueprint matching existing project format
        resolved_target = target_col or existing_schema.get("target_column", "")
        
        categorical_schema = {}
        for col in active_categorical:
            # Sort unique values for consistency
            cats = sorted([str(val) for val in df[col].dropna().unique()])
            categorical_schema[col] = cats

        continuous_schema = {}
        for col in active_continuous:
            series = df[col].dropna()
            continuous_schema[col] = {
                "min": float(series.min()) if not series.empty else 0.0,
                "max": float(series.max()) if not series.empty else 1.0
            }

        new_data_schema = {
            "PII_columns_to_drop": list(recomm_drops),
            "categorical_features": categorical_schema,
            "continuous_features": continuous_schema,
            "target_column": resolved_target
        }

        # 7. Build the Preprocessing Blueprint
        impute_strategy = {}
        # Fill with missing analyzer recommendations
        for col, strategy in missing_report["imputation_recommendations"].items():
            impute_strategy[col] = strategy
        # Fallback defaults for columns with missing potential
        for col in active_continuous:
            if col not in impute_strategy:
                impute_strategy[col] = "median"
        for col in active_categorical:
            if col not in impute_strategy:
                impute_strategy[col] = "mode"

        encoding_strategy = {}
        for col in active_categorical:
            encoding_strategy[col] = profiling_report["categorical_features"][col]["encoding_recommendation"]

        scaling_strategy = {}
        for col in active_continuous:
            scaling_strategy[col] = profiling_report["continuous_features"][col]["scaling_recommendation"]

        pipeline_config = {
            "dataset_name": dataset_name,
            "imputation_strategy": impute_strategy,
            "encoding_strategy": encoding_strategy,
            "scaling_strategy": scaling_strategy
        }

        # Save configs to appropriate locations
        self._deploy_configs(dataset_name, new_data_schema, pipeline_config)

        return {
            "quality_report": quality_report,
            "privacy_report": privacy_report,
            "missing_report": missing_report,
            "profiling_report": profiling_report,
            "relationship_report": relationship_report,
            "generated_data_schema": new_data_schema,
            "generated_pipeline_config": pipeline_config
        }

    def _deploy_configs(self, dataset_name: str, data_schema: Dict[str, Any], pipeline_config: Dict[str, Any]) -> None:
        """Deploys configurations to eda_framework outputs dir only.
        
        EDA framework is a RECOMMENDATION ENGINE. It writes draft configurations
        to eda_framework/outputs/<dataset>/ for user review. The user then reviews
        and manually copies approved configs to config/<dataset>/ (the single 
        source of truth). NEVER overwrite config/<dataset>/ files directly, as 
        they may contain user edits that would be lost.
        """
        eda_output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", dataset_name))
        os.makedirs(eda_output_dir, exist_ok=True)

        # Save Schema (PII columns are ONLY in data_schema.yaml — single source of truth)
        save_yaml(data_schema, os.path.join(eda_output_dir, "data_schema.yaml"))

        # Save Pipeline Config (no PII columns — they belong in data_schema.yaml only)
        save_yaml(pipeline_config, os.path.join(eda_output_dir, "pipeline_config.yaml"))

        logger.info(
            "EDA recommendations written to %s. "
            "Review and manually copy to config/%s/ to apply.",
            eda_output_dir, dataset_name,
        )
