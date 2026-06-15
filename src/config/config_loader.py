"""
Configuration Loader
--------------------
Enterprise-grade config loader using Pydantic for strict schema validation.
Merges default_config.yaml with dataset-specific configurations.
"""

import os
import logging
import yaml
from pydantic import BaseModel, Field, conint, confloat, ValidationError
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class IngestionConfig(BaseModel):
    model_config = {"extra": "forbid"}
    columns: List[str]
    pii_columns_to_drop: List[str] = Field(default_factory=list)
    quasi_identifiers: List[str] = Field(default_factory=list)
    target_column: str
    separator: str = ","
    has_header: bool = True
    na_values: Optional[List[str]] = None
    read_excel: bool = False
    skiprows: Optional[int] = None
    max_onehot_cardinality: int = 10

class PrivacyConfig(BaseModel):
    model_config = {"extra": "forbid"}
    enable_differential_privacy: bool = True
    target_epsilon: confloat(gt=0) = 1.0
    target_delta: confloat(ge=0, lt=1) = 1e-5
    max_grad_norm: float = 1.0

class ModelConfig(BaseModel):
    model_config = {"extra": "forbid"}
    model_type: str = "diffusion"  # ctgan, ctvae, diffusion
    epochs: conint(gt=0) = 100
    batch_size: conint(gt=0) = 256
    learning_rate: float = 2e-4
    max_ram_gb: float = 8.0

class AppConfig(BaseModel):
    model_config = {"extra": "forbid"}
    ingestion: IngestionConfig
    privacy: PrivacyConfig
    model: ModelConfig

def deep_merge(dict1: dict, dict2: dict) -> dict:
    """Recursively merges dict2 into dict1 and returns the merged dictionary."""
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result

class ConfigLoader:
    @staticmethod
    def load_schema(dataset_name: str) -> dict:
        """
        Loads data_schema.yaml for the given dataset.
        
        Args:
            dataset_name: Name of the dataset folder under configs/
            
        Returns:
            dict: The loaded data schema dictionary.
        """
        configs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "configs"))
        schema_path = os.path.join(configs_dir, dataset_name, "data_schema.yaml")
        if not os.path.exists(schema_path):
            raise FileNotFoundError(f"data_schema.yaml not found at {schema_path}")
            
        with open(schema_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def load_config(dataset_name: str) -> AppConfig:
        """
        Loads the global default configuration, merges it with the dataset-specific
        configurations, validates the merged result against the AppConfig Pydantic model,
        and returns it.
        """
        configs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "configs"))
        default_path = os.path.join(configs_dir, "default_config.yaml")
        
        # 1. Load global default config
        if not os.path.exists(default_path):
            raise FileNotFoundError(f"Global default configuration file not found at {default_path}")
            
        with open(default_path, "r", encoding="utf-8") as f:
            default_dict = yaml.safe_load(f) or {}
            
        # 2. Find dataset subfolder
        dataset_dir = os.path.join(configs_dir, dataset_name)
        if not os.path.exists(dataset_dir):
            raise ValueError(f"Dataset configuration folder not found at {dataset_dir}")
            
        # 3. Load dataset-specific configs if they exist
        dataset_dict = {}
        for config_name in ["ingestion", "privacy", "model"]:
            config_path = os.path.join(dataset_dir, f"{config_name}_config.yaml")
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    content = yaml.safe_load(f) or {}
                    dataset_dict[config_name] = content
                    
        # 4. Deep merge
        merged_dict = deep_merge(default_dict, dataset_dict)
        
        # 5. Pydantic validation (Fail-Fast)
        try:
            config = AppConfig.model_validate(merged_dict)
            return config
        except ValidationError as e:
            logger.error("Config validation failed for dataset '%s':", dataset_name)
            logger.error(e)
            raise e
