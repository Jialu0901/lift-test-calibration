"""
CausalML FilterSelect 封装，供 train_dr 等入口做 uplift 相关特征筛选。
参见: https://causalml.readthedocs.io/en/latest/causalml.html#module-causalml.feature_selection
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from causalml.feature_selection import FilterSelect

logger = logging.getLogger(__name__)

FS_METHODS = frozenset({"F", "LR", "KL", "ED", "Chi"})
NULL_IMPUTE_CHOICES = frozenset({"mean", "median", "most_frequent"})


def run_filter_select(
    df: pd.DataFrame,
    feature_names: list[str],
    treatment_col: str,
    y_name: str,
    *,
    method: str = "F",
    n_bins: int = 5,
    top_k: int | None = None,
    score_threshold: float | None = None,
    order: int = 1,
    null_impute: str = "mean",
    disp: bool = False,
) -> tuple[list[str], pd.DataFrame, str]:
    """
    在单二元处理列上运行 FilterSelect.get_importance，返回筛选后的特征名、importance 表、实际使用的方法名。

    Args:
        df: 含特征列、treatment_col、y_name 的 DataFrame（通常为 train 子集）
        feature_names: 参与筛选的特征列名（须为 df 的列）
        treatment_col: 0/1 处理列名；内部映射为 control/treatment 字符串
        y_name: 结果列名（须为 df 的列）
        method: F | LR | KL | ED | Chi
        n_bins: KL / ED / Chi 分箱数
        top_k: 按 rank 取前 k 个；None 表示不截断
        score_threshold: 仅对 F/LR 有效，保留 score >= 阈值的特征
        order: F/LR 下 treatment×特征交互阶数（1=线性，2=含二次，3=至三次）
        null_impute: mean | median | most_frequent（数据中不能有 NaN 且为 None 时会报错）
        disp: LR 方法是否打印收敛信息
    """
    if method not in FS_METHODS:
        raise ValueError(f"method must be one of {sorted(FS_METHODS)}, got {method}")
    if null_impute not in NULL_IMPUTE_CHOICES:
        raise ValueError(f"null_impute must be one of {sorted(NULL_IMPUTE_CHOICES)}, got {null_impute}")

    names = [c for c in feature_names if c in df.columns]
    if not names:
        return [], pd.DataFrame(), method

    work = df[names + [treatment_col, y_name]].copy()
    t = work[treatment_col].values
    work["__treatment_group_key__"] = np.where(np.asarray(t).ravel() == 1, "treatment", "control")
    work["__outcome__"] = work[y_name].values

    fs = FilterSelect()

    def _call_get_importance(mth: str, kw_extra: dict[str, Any]) -> pd.DataFrame:
        return fs.get_importance(
            data=work,
            features=names,
            y_name="__outcome__",
            method=mth,
            experiment_group_column="__treatment_group_key__",
            control_group="control",
            treatment_group="treatment",
            n_bins=n_bins,
            null_impute=null_impute,
            **kw_extra,
        )

    kw: dict[str, Any] = {}
    if method in ("F", "LR"):
        kw["order"] = order
        if method == "LR":
            kw["disp"] = disp

    imp: pd.DataFrame
    effective_method = method
    try:
        imp = _call_get_importance(method, kw)
    except TypeError as e:
        # 兼容旧版 causalml（无 order/disp 等参数）
        logger.warning("FilterSelect.get_importance fallback without optional kwargs: %s", e)
        imp = fs.get_importance(
            data=work,
            features=names,
            y_name="__outcome__",
            method=method,
            experiment_group_column="__treatment_group_key__",
            control_group="control",
            treatment_group="treatment",
            n_bins=n_bins,
            null_impute=null_impute,
        )
    except ValueError as e:
        # 新版 statsmodels/patsy 下 CausalML filter_F 的 f_test 约束维数可能不匹配（wrong shape for coefs）
        err_msg = str(e).lower()
        if "wrong shape for coefs" in err_msg:
            logger.warning(
                "FilterSelect method=%s failed (%s); retrying with LR (then KL if needed)",
                method,
                e,
            )
            try:
                kw_lr: dict[str, Any] = {"order": order, "disp": disp}
                imp = _call_get_importance("LR", kw_lr)
                effective_method = "LR"
            except Exception as e2:
                logger.warning("FilterSelect LR fallback failed (%s); retrying with KL", e2)
                imp = _call_get_importance("KL", {})
                effective_method = "KL"
        else:
            raise
    except Exception as e:
        # 其他实现差异：同样尝试 LR → KL
        err_msg = str(e).lower()
        if method == "F" and ("f_test" in err_msg or "wald" in err_msg or "coefs" in err_msg):
            logger.warning(
                "FilterSelect method=F failed (%s); retrying with LR then KL",
                e,
            )
            try:
                imp = _call_get_importance("LR", {"order": order, "disp": disp})
                effective_method = "LR"
            except Exception:
                imp = _call_get_importance("KL", {})
                effective_method = "KL"
        else:
            raise

    method = effective_method
    imp = imp.sort_values("rank", ascending=True)
    if score_threshold is not None and effective_method in ("F", "LR"):
        imp = imp[imp["score"] >= score_threshold]
    if top_k is not None and top_k > 0:
        imp = imp.head(top_k)
    selected = [str(x) for x in imp["feature"].tolist()]
    return selected, imp, effective_method


def union_features_per_channel(
    train_df: pd.DataFrame,
    channels: list[str],
    candidate_features: list[str],
    y_col: str,
    *,
    method: str = "F",
    n_bins: int = 5,
    top_k: int | None = None,
    score_threshold: float | None = None,
    order: int = 1,
    null_impute: str = "mean",
    disp: bool = False,
) -> tuple[list[str], dict[str, Any]]:
    """
    对每个 channel 在 train 上用 T_channel 做 FilterSelect，特征集合为 candidate_features 与 train 列的交集。
    返回：按 candidate_features 原顺序过滤后的并集、以及每 channel 的 importance 与诊断信息。
    """
    candidates = [c for c in candidate_features if c in train_df.columns]
    if not candidates:
        return [], {"by_channel": {}, "channels_skipped": []}

    union: set[str] = set()
    by_channel: dict[str, dict[str, Any]] = {}
    skipped: list[str] = []

    for ch in channels:
        t_col = f"T_{ch}"
        if t_col not in train_df.columns:
            skipped.append(ch)
            continue
        T = train_df[t_col].values
        if T.sum() == 0 or T.sum() == len(T):
            skipped.append(ch)
            continue
        try:
            sel, imp, _eff = run_filter_select(
                train_df,
                candidates,
                t_col,
                y_col,
                method=method,
                n_bins=n_bins,
                top_k=top_k,
                score_threshold=score_threshold,
                order=order,
                null_impute=null_impute,
                disp=disp,
            )
        except Exception as e:
            logger.warning("FilterSelect failed for channel %s: %s", ch, e)
            skipped.append(ch)
            continue
        union.update(sel)
        by_channel[ch] = {
            "n_selected": len(sel),
            "importance": imp.to_dict(orient="records"),
        }

    if not union:
        logger.warning("FilterSelect produced empty union; keeping all candidate features")
        return candidates, {"by_channel": by_channel, "channels_skipped": skipped, "fallback": True}

    ordered = [c for c in candidate_features if c in union]
    meta: dict[str, Any] = {
        "by_channel": by_channel,
        "channels_skipped": skipped,
        "n_union": len(ordered),
        "fallback": False,
    }
    return ordered, meta
