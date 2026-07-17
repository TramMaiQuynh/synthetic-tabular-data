"""
Helper Utilities for EDA Framework
----------------------------------
Provides file loading, separator detection, encoding checks, and configuration persistence.
"""

import os
import yaml
import json
import logging
import pandas as pd
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML file safely."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.error("Failed to load YAML from %s: %s", path, e)
        return {}

def save_yaml(data: Dict[str, Any], path: str) -> None:
    """Save dictionary to a YAML file with nice formatting."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
        logger.info("Saved config to %s", path)
    except Exception as e:
        logger.error("Failed to save YAML to %s: %s", path, e)

def detect_file_properties(file_path: str) -> Tuple[str, str]:
    """
    Detect separator and encoding of a raw text/CSV file.
    Delimiters checked: ',', ';', '\t', '|'
    Encodings checked: 'utf-8', 'utf-8-sig', 'latin-1'
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Detect encoding first
    possible_encodings = ['utf-8', 'utf-8-sig', 'latin-1']
    detected_encoding = 'utf-8'
    
    for encoding in possible_encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                f.read(4096)  # Try reading a chunk
            detected_encoding = encoding
            break
        except (UnicodeDecodeError, LookupError):
            continue

    # Detect delimiter
    delimiters = [',', ';', '\t', '|', '  ']
    detected_sep = ','
    max_cols = 0
    
    try:
        with open(file_path, 'r', encoding=detected_encoding) as f:
            lines = [f.readline() for _ in range(5)]
            lines = [line for line in lines if line.strip()]
            
        for sep in delimiters:
            if len(lines) > 0:
                cols = len(lines[0].split(sep))
                # Check consistency across lines
                consistent = True
                for line in lines[1:]:
                    if len(line.split(sep)) != cols:
                        consistent = False
                        break
                if consistent and cols > max_cols:
                    max_cols = cols
                    detected_sep = sep
    except Exception as e:
        logger.warning("Error detecting delimiter, defaulting to ',': %s", e)

    return detected_sep, detected_encoding
