"""Re-export uplift feature selection from utils (shared with feature_select CLI)."""
from __future__ import annotations

from utils.uplift_feature_selection import (
    merge_csv_and_protected_features,
    resolve_protected_features,
    run_two_stage_filter,
)

__all__ = [
    "merge_csv_and_protected_features",
    "resolve_protected_features",
    "run_two_stage_filter",
]
