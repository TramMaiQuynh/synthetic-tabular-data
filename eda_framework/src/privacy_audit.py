"""
EDA Framework - Stage 2: Privacy Audit
-------------------------------------
Scans for Personally Identifiable Information (PII) keywords in column names and detects potential unique identifier columns.
"""

import logging
import re
import pandas as pd
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


def _pre_split_compounds(col: str) -> str:
    """Pre-split common closed compound words in column names to avoid suffix false positives."""
    col_lower = col.lower()
    replacements = {
        "firstname": "first_name",
        "lastname": "last_name",
        "username": "user_name",
        "surname": "sur_name",
        "phonenumber": "phone_number",
        "creditcard": "credit_card",
        "emailaddress": "email_address",
    }
    for target, repl in replacements.items():
        if target in col_lower:
            col = re.sub(re.escape(target), repl, col, flags=re.IGNORECASE)
    return col


def _kw_matches(kw: str, tokens: set) -> bool:
    """Return True if kw is an exact token."""
    return kw.lower() in tokens


def _col_tokens(col: str) -> set:
    """Tokenise a column name handling camelCase, snake_case, UPPER_CASE, and closed compounds."""
    # Pre-split closed compound words
    col_split = _pre_split_compounds(col)
    # Insert _ between lowercase/digit and uppercase (camelCase split)
    words = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', col_split)
    # Split on any non-alphanumeric boundary; drop empty strings
    return set(t for t in re.split(r'[^a-zA-Z0-9]', words.lower()) if t)


class PrivacyAuditor:
    def __init__(self, config: Dict[str, Any]):
        self.cardinality_threshold = config.get("pii", {}).get("cardinality_threshold", 0.95)
        self.keywords = config.get("pii", {}).get("name_keywords", [])

    def audit(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform PII and unique identifier audit on the loaded DataFrame."""
        n_rows = len(df)
        if n_rows == 0:
            return {"PII_columns": [], "Quasi_identifiers": [], "comments": "Empty DataFrame"}

        pii_columns = []
        quasi_identifiers = []
        recomm_drop = []

        for col in df.columns:
            # Tokenise column name (camelCase + snake_case aware)
            tokens = _col_tokens(col)
            keyword_match = any(_kw_matches(kw, tokens) for kw in self.keywords)
            
            # Uniqueness check
            unique_count = df[col].nunique(dropna=True)
            uniqueness_ratio = unique_count / n_rows

            is_high_card = uniqueness_ratio >= self.cardinality_threshold

            # Simple logic: If it matches name keywords and has high cardinality, or is string and unique, it's PII/ID
            if keyword_match:
                if is_high_card:
                    pii_columns.append(col)
                    recomm_drop.append(col)
                else:
                    # Low cardinality matching name: e.g. age/gender are not PII but might be Quasi-identifiers
                    # Let's see: target or sensitive columns.
                    quasi_identifiers.append(col)
            elif is_high_card and not pd.api.types.is_numeric_dtype(df[col]):
                # String ID column like UUID
                pii_columns.append(col)
                recomm_drop.append(col)

        # Quasi-identifier heuristics: apply the same token-based matching
        # to prevent false positives like 'state' in 'estimated', 'sex' in 'proxy_sex_ratio'
        common_qis = {'age', 'gender', 'sex', 'race', 'education', 'marital', 'zip', 'postcode', 'country', 'state'}
        for col in df.columns:
            if col not in pii_columns and col not in quasi_identifiers:
                col_tokens = _col_tokens(col)
                if any(_kw_matches(qi, col_tokens) for qi in common_qis):
                    quasi_identifiers.append(col)

        return {
            "PII_columns_detected": pii_columns,
            "Quasi_identifiers_detected": quasi_identifiers,
            "recommended_drops": recomm_drop
        }
