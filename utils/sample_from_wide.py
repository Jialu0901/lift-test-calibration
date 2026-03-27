"""
从宽表 DataFrame 构建建模样本（id + X + T_* + Y）。
供 notebook / train_dr 共用，避免 notebook 必须依赖 train_dr 内联定义。
"""
from __future__ import annotations

import pandas as pd

from utils.config import CHANNELS, TARGET, build_treatment, get_feature_columns_from_df


def build_sample_from_wide_df(
    df: pd.DataFrame,
    *,
    sample_weight_column: str | None = None,
) -> pd.DataFrame:
    """
    从宽表 DataFrame（parquet 或 DB）构建建模样本：id + X + T_* + Y。
    与 MODELING_DATA_SPEC 一致。

    If ``sample_weight_column`` is set, that column must exist on ``df``; values are copied as float
    into a unified ``sample_weight`` column (sanitized during training).
    """
    if df.empty:
        raise ValueError("Empty wide_table")
    if TARGET not in df.columns:
        raise ValueError(f"Target column {TARGET} not found")

    df = build_treatment(df)
    feature_cols = get_feature_columns_from_df(df)
    if not feature_cols:
        raise ValueError("No feature columns found")

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    X = df[feature_cols].fillna(0).astype(float)
    Y = df[TARGET].fillna(0).astype(int).rename("Y")
    t_cols = [f"T_{c}" for c in CHANNELS]
    id_cols = ["outcome_date", "tenant_id", "user_id"]
    id_avail = [c for c in id_cols if c in df.columns]
    sample = df[id_avail].copy() if id_avail else pd.DataFrame(index=df.index)
    sample = pd.concat([sample, X, df[t_cols], Y], axis=1)
    if sample_weight_column:
        if sample_weight_column not in df.columns:
            raise ValueError(
                f"sample_weight_column={sample_weight_column!r} not found in wide table columns"
            )
        sample["sample_weight"] = pd.to_numeric(df[sample_weight_column], errors="coerce").astype(
            float
        )
    return sample
