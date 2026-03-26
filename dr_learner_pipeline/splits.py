"""Train / Val / Test split by outcome_date lists (JSON file or in-memory dict)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _outcome_date_as_str_series(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce")
    return dt.dt.strftime("%Y-%m-%d")


def _norm_date_list(xs: list) -> list[str]:
    out: list[str] = []
    for x in xs:
        if x is None:
            continue
        s = str(x).strip()
        if not s:
            continue
        dtp = pd.to_datetime(s, errors="coerce")
        if pd.isna(dtp):
            raise ValueError(f"Invalid date string in split file: {x!r}")
        out.append(dtp.strftime("%Y-%m-%d"))
    return out


def _coerce_split_dates_dict(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"split_dates must be a JSON object, got {type(raw).__name__}")
    for k in ("train", "val", "test"):
        if k not in raw:
            raise ValueError(f"split_dates must contain '{k}' key")
    return raw


def date_range_for_split_dict(raw: dict[str, Any]) -> tuple[str, str]:
    """Min/max date across train+val+test lists for SQL date_range."""
    raw = _coerce_split_dates_dict(raw)
    all_dates: list[str] = []
    for k in ("train", "val", "test"):
        all_dates.extend(_norm_date_list(raw[k]))
    if not all_dates:
        raise ValueError("No dates in split spec")
    sorted_d = sorted(all_dates)
    return sorted_d[0], sorted_d[-1]


def split_train_val_test_by_dates_from_dict(
    df: pd.DataFrame,
    raw: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Same semantics as :func:`split_train_val_test_by_dates` but ``raw`` is already a dict
    (e.g. from notebook ``SPLIT_DATES``).
    """
    raw = _coerce_split_dates_dict(raw)
    train_dates = set(_norm_date_list(raw["train"]))
    val_dates = set(_norm_date_list(raw["val"]))
    test_dates = set(_norm_date_list(raw["test"]))

    for a, b, name_a, name_b in [
        (train_dates, val_dates, "train", "val"),
        (train_dates, test_dates, "train", "test"),
        (val_dates, test_dates, "val", "test"),
    ]:
        inter = a & b
        if inter:
            raise ValueError(
                f"split_dates: '{name_a}' and '{name_b}' overlap: "
                + ", ".join(sorted(inter)[:10])
                + (" ..." if len(inter) > 10 else "")
            )

    if not train_dates or not val_dates or not test_dates:
        raise ValueError("train, val, test must each contain at least one valid date")

    if "outcome_date" not in df.columns:
        raise ValueError("DataFrame must contain outcome_date")

    date_str = _outcome_date_as_str_series(df["outcome_date"])
    if date_str.isna().any():
        raise ValueError(f"{int(date_str.isna().sum())} row(s) have invalid outcome_date")

    m_tr = date_str.isin(train_dates)
    m_va = date_str.isin(val_dates)
    m_te = date_str.isin(test_dates)

    train_df = df.loc[m_tr].sort_values("outcome_date").reset_index(drop=True)
    val_df = df.loc[m_va].sort_values("outcome_date").reset_index(drop=True)
    test_df = df.loc[m_te].sort_values("outcome_date").reset_index(drop=True)

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            "After date filtering, one of train/val/test is empty — check split JSON vs data"
        )

    return train_df, val_df, test_df


def split_train_val_test_by_dates(
    df: pd.DataFrame,
    path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    JSON keys: train, val, test — each a list of date strings.
    Lists must be pairwise disjoint.
    """
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    return split_train_val_test_by_dates_from_dict(df, raw)


def date_range_for_load(split_path: str | Path) -> tuple[str, str]:
    """Min/max date across train+val+test lists for SQL date_range."""
    path = Path(split_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    return date_range_for_split_dict(raw)
