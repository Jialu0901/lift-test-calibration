"""
Two-stage CausalML FilterSelect for uplift (Val coarse, Train fine).
Forced-retention lists align with MODELING_DATA_SPEC.md §4.1.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from utils.feature_select import run_filter_select

logger = logging.getLogger(__name__)


def resolve_protected_features(
    candidate_features: list[str],
    explicit: list[str] | None,
    prefixes: list[str] | None,
) -> list[str]:
    """
    Subset of candidate_features forced through coarse/fine top-k (MODELING_DATA_SPEC §4.1).
    Preserves candidate_features order. A column matches if it is in explicit or starts with
    any prefix string.
    """
    ex = set(explicit or ())
    pfx = tuple(prefixes or ())
    return [c for c in candidate_features if c in ex or any(c.startswith(p) for p in pfx)]


def run_two_stage_filter(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    candidate_features: list[str],
    t_col: str,
    y_col: str,
    *,
    method: str = "F",
    order: int = 1,
    null_impute: str = "mean",
    coarse_top_n: int,
    fine_top_k: int,
    out_selected_path: Path,
    protected_features: list[str] | None = None,
) -> tuple[list[str], dict]:
    """
    1) Val: FilterSelect on all candidates, keep top coarse_top_n by rank.
    2) Train: FilterSelect on coarse set, keep top fine_top_k.
    Columns in protected_features (subset of cand, typically from SPEC §4.1) are always kept;
    remaining slots are filled by importance order.
    Writes selected feature names to out_selected_path; full importance CSVs and summary JSON
    alongside it in the same directory.
    """
    cand = [c for c in candidate_features if c in train_df.columns and c in val_df.columns]
    if not cand:
        raise ValueError("No overlapping candidate features between train/val")

    protected_ordered = [c for c in (protected_features or []) if c in cand]
    protected_set = set(protected_ordered)
    n_prot = len(protected_ordered)

    _, imp_v_full, method_val = run_filter_select(
        val_df,
        cand,
        t_col,
        y_col,
        method=method,
        order=order,
        null_impute=null_impute,
        top_k=None,
    )
    if len(imp_v_full) > 0:
        ranked_v = [str(x) for x in imp_v_full["feature"].tolist()]
        others_v = [c for c in ranked_v if c not in protected_set]
        if n_prot > coarse_top_n:
            logger.warning(
                "Protected feature count (%d) exceeds coarse_top_n (%d); keeping all protected",
                n_prot,
                coarse_top_n,
            )
        n_other_slots = max(0, coarse_top_n - n_prot)
        sel_v = protected_ordered + others_v[:n_other_slots]
    else:
        logger.warning("Val FilterSelect returned empty importance; falling back to cand order")
        rest = [c for c in cand if c not in protected_set]
        cap = max(0, min(coarse_top_n, len(cand)) - n_prot)
        sel_v = protected_ordered + rest[:cap]

    logger.info(
        "Feature coarse (Val): %d candidates -> %d features (top_n=%d, protected=%d)",
        len(cand),
        len(sel_v),
        coarse_top_n,
        n_prot,
    )
    if not sel_v:
        sel_v = cand[: min(coarse_top_n, len(cand))]

    _, imp_t_full, method_train = run_filter_select(
        train_df,
        sel_v,
        t_col,
        y_col,
        method=method,
        order=order,
        null_impute=null_impute,
        top_k=None,
    )
    protected_in_coarse = [c for c in protected_ordered if c in sel_v]
    n_prot_f = len(protected_in_coarse)

    if len(imp_t_full) > 0:
        ranked_t = [str(x) for x in imp_t_full["feature"].tolist()]
        others_t = [c for c in ranked_t if c not in protected_set]
        if n_prot_f > fine_top_k:
            logger.warning(
                "Protected feature count in coarse set (%d) exceeds fine_top_k (%d); keeping all protected",
                n_prot_f,
                fine_top_k,
            )
        n_other_f = max(0, fine_top_k - n_prot_f)
        sel_t = protected_in_coarse + others_t[:n_other_f]
    else:
        logger.warning("Train FilterSelect returned empty importance; falling back to sel_v order")
        rest_t = [c for c in sel_v if c not in protected_set]
        cap_t = max(0, min(fine_top_k, len(sel_v)) - n_prot_f)
        sel_t = protected_in_coarse + rest_t[:cap_t]

    logger.info(
        "Feature fine (Train): %d -> %d features (top_k=%d, protected=%d)",
        len(sel_v),
        len(sel_t),
        fine_top_k,
        n_prot_f,
    )
    if not sel_t:
        sel_t = sel_v[: min(fine_top_k, len(sel_v))]

    out_dir = out_selected_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    imp_v_full.to_csv(out_dir / "feature_importance_val.csv", index=False)
    imp_t_full.to_csv(out_dir / "feature_importance_train_on_coarse.csv", index=False)

    summary = {
        "n_candidates": len(cand),
        "coarse_top_n": coarse_top_n,
        "fine_top_k": fine_top_k,
        "n_after_coarse": len(sel_v),
        "n_selected": len(sel_t),
        "n_protected": n_prot,
        "protected_resolved": protected_ordered,
        "effective_method_val": method_val,
        "effective_method_train": method_train,
    }
    with open(out_dir / "feature_selection_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    out_selected_path.write_text("\n".join(sel_t) + "\n", encoding="utf-8")
    logger.info("Wrote %s (%d features)", out_selected_path, len(sel_t))

    meta = {
        "coarse_n": len(sel_v),
        "fine_n": len(sel_t),
        "n_protected": n_prot,
        "protected_resolved": protected_ordered,
        "importance_val_head": imp_v_full.head(20).to_dict(orient="records") if len(imp_v_full) else [],
        "importance_train_head": imp_t_full.head(20).to_dict(orient="records") if len(imp_t_full) else [],
        "effective_method_val": method_val,
        "effective_method_train": method_train,
        "feature_importance_val_path": str(out_dir / "feature_importance_val.csv"),
        "feature_importance_train_path": str(out_dir / "feature_importance_train_on_coarse.csv"),
        "feature_selection_summary_path": str(out_dir / "feature_selection_summary.json"),
    }
    return sel_t, meta


def merge_csv_and_protected_features(
    csv_features: list[str],
    protected_ordered: list[str],
) -> list[str]:
    """Order: protected first (dedupe), then CSV names not already included (preserve CSV order)."""
    seen: set[str] = set()
    out: list[str] = []
    for c in protected_ordered:
        if c not in seen:
            seen.add(c)
            out.append(c)
    for c in csv_features:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out
