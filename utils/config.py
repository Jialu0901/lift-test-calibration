"""
Benchmark Model（Channel-level ITE）配置。
与 data_merge 的 PLATFORMS 一致。
"""
from __future__ import annotations

# Channels（与 data_merge PLATFORMS + other 一致）
CHANNELS = [
    "google", "meta", "tiktok", "pinterest", "microsoft",
    "snapchat", "applovin", "other",
]

# Sample ID 列
ID_COLS = ["batch_id", "outcome_date", "tenant_id", "user_id"]

# Target
TARGET = "is_converted"

# 明确排除的列（不作为 X）
EXCLUDE_FROM_X = {
    "batch_id", "outcome_date", "tenant_id", "user_id",
    "is_converted", "sample_weight",
    "campaign_cnt",
}


def get_treatment_cols_for_channel(c: str) -> tuple[str, str, str]:
    """返回 (is_ads_col, click_col, impr_col)。"""
    return (
        f"is_{c}_ads",
        f"{c}_14d_click_cnt",
        f"{c}_14d_impr_cnt",
    )


def is_treatment_column(col: str) -> bool:
    """是否为 treatment 相关列，应从 X 中排除。"""
    if col.startswith("T_") and col[2:] in CHANNELS:
        return True
    if col == "campaign_cnt":
        return True
    for prefix in ("is_google_ads", "is_meta_ads", "is_tiktok_ads", "is_pinterest_ads",
                   "is_microsoft_ads", "is_snapchat_ads", "is_applovin_ads", "is_other_ads"):
        if col == prefix:
            return True
    for p in ("google_14d_", "meta_14d_", "tiktok_14d_", "pinterest_14d_",
              "microsoft_14d_", "snapchat_14d_", "applovin_14d_", "other_14d_"):
        if col.startswith(p):
            return True
    return False


def get_feature_columns_from_df(df):
    """
    从 wide_table DataFrame 推导可用的 feature 列。
    排除 ID、target、treatment 相关列，保留 numeric 列。
    """
    import pandas as pd
    excluded = set(ID_COLS) | {TARGET} | EXCLUDE_FROM_X
    feature_cols = []
    for col in df.columns:
        if col in excluded or is_treatment_column(col):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if df[col].dtype == "bool":
            continue
        feature_cols.append(col)
    return feature_cols


def build_treatment(df):
    """为每个 channel 派生二值 T_c：T=1 当 is_ads==1 或 click_cnt>0 或 impr_cnt>0。"""
    import pandas as pd
    out = df.copy()
    for c in CHANNELS:
        is_col, click_col, impr_col = get_treatment_cols_for_channel(c)
        t = pd.Series(0, index=df.index)
        if is_col in df.columns:
            t = t | (df[is_col].fillna(0) >= 1)
        if click_col in df.columns:
            t = t | (df[click_col].fillna(0) > 0)
        if impr_col in df.columns:
            t = t | (df[impr_col].fillna(0) > 0)
        out[f"T_{c}"] = t.astype(int)
    return out
