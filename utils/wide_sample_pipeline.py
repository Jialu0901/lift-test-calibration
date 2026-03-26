"""Load wide table (DB or parquet), filter by date range, build modeling sample DataFrame."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.config import CHANNELS, is_treatment_column
from utils.sample_from_wide import build_sample_from_wide_df
from utils.wide_table_db import load_wide_table_from_db


def candidate_features_from_df(df: pd.DataFrame) -> list[str]:
    exclude = {
        "batch_id",
        "outcome_date",
        "tenant_id",
        "user_id",
        "Y",
        "is_converted",
        "sample_weight",
        "sampling_rate",
        "campaign_cnt",
    } | {f"T_{c}" for c in CHANNELS}
    return [
        c
        for c in df.columns
        if c not in exclude
        and not is_treatment_column(c)
        and pd.api.types.is_numeric_dtype(df[c])
        and df[c].dtype != bool
    ]


def load_and_build_sample(
    date_start: str,
    date_end: str,
    *,
    db_table: str | None = None,
    wide_parquet: str | Path | None = None,
    sample_limit: int | None = None,
    chunk_days: int | None = 1,
) -> pd.DataFrame:
    """
    Load raw wide rows with outcome_date in [date_start, date_end], then build_sample_from_wide_df.
    Exactly one of db_table or wide_parquet must be set.
    """
    if bool(db_table) == bool(wide_parquet):
        raise ValueError("Set exactly one of db_table or wide_parquet")

    d0, d1 = date_start, date_end

    if wide_parquet is not None:
        wp = Path(wide_parquet)
        if wp.is_dir():
            merged = wp / "wide_table.parquet"
            if not merged.exists():
                raise FileNotFoundError(f"No wide_table.parquet in {wp}")
            raw = pd.read_parquet(merged)
        else:
            raw = pd.read_parquet(wp)
        if raw.empty:
            raise ValueError("Empty wide table")
        raw = raw[
            (pd.to_datetime(raw["outcome_date"], errors="coerce") >= pd.Timestamp(d0))
            & (pd.to_datetime(raw["outcome_date"], errors="coerce") <= pd.Timestamp(d1))
        ]
        return build_sample_from_wide_df(raw)

    raw = load_wide_table_from_db(
        table_name=db_table,
        limit=sample_limit,
        date_range=(d0, d1),
        chunk_days=chunk_days if chunk_days else None,
    )
    return build_sample_from_wide_df(raw)
