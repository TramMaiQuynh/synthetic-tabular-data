"""
Training Package
-----------------
Exports the model training orchestrator, DP-SGD wrapper, and HPO runner.
"""

from src.training.trainer import ModelTrainer, build_col_meta
from src.training.dp_training import DPTrainer
from src.training.hpo import HPORunner

__all__ = [
    "ModelTrainer",
    "build_col_meta",
    "DPTrainer",
    "HPORunner",
]
