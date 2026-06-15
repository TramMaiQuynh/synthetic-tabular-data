"""
Evaluation, Validation & Audit Package
---------------------------------------
Exports modules for analyzing statistical fidelity, empirical privacy,
machine learning utility, visual overlay generation, and compliance reporting.
"""

from src.evaluation.fidelity import FidelityAssessor
from src.evaluation.privacy import PrivacyAuditor
from src.evaluation.utility import UtilityEvaluator
from src.evaluation.visual import VisualOverlayGenerator
from src.evaluation.report import ComplianceReporter
from src.evaluation.orchestrator import EvaluationSuite

__all__ = [
    "FidelityAssessor",
    "PrivacyAuditor",
    "UtilityEvaluator",
    "VisualOverlayGenerator",
    "ComplianceReporter",
    "EvaluationSuite",
]
