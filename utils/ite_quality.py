"""
ITE 预测质量评估：宏观 (ATE 校准)、中观 (AUUC/Qini)、微观 (Decile 一致性)。
"""
from __future__ import annotations

from typing import Any

import numpy as np


def compute_ate_calibration(
    pred_ite: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
) -> dict[str, float]:
    """
    宏观 ATE 校准：pred_mean 与 obs_ate 的差异。
    obs_ate = E[Y|T=1] - E[Y|T=0]（简单差分，需 T 有 variation）。
    """
    T = np.asarray(T).ravel()
    Y = np.asarray(Y).ravel()
    pred_ite = np.asarray(pred_ite).ravel()
    pred_mean = float(np.mean(pred_ite))

    n1 = (T == 1).sum()
    n0 = (T == 0).sum()
    if n1 == 0 or n0 == 0:
        return {"pred_mean": pred_mean, "obs_ate": float("nan"), "ate_diff": float("nan")}

    y1_mean = float(np.mean(Y[T == 1]))
    y0_mean = float(np.mean(Y[T == 0]))
    obs_ate = y1_mean - y0_mean
    ate_diff = pred_mean - obs_ate
    return {"pred_mean": pred_mean, "obs_ate": obs_ate, "ate_diff": ate_diff}


def compute_qini_auc(
    uplift: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
) -> float:
    """
    中观 Ranking：Qini 系数（归一化 AUUC）。
    按 uplift 降序，计算累积 gain 曲线下面积，再除以最优曲线面积归一化。
    Qini(φ) = n_{t,y=1}(φ)/N_t - n_{c,y=1}(φ)/N_c
    """
    uplift = np.asarray(uplift).ravel()
    T = np.asarray(T).ravel().astype(int)
    Y = np.asarray(Y).ravel().astype(int)

    order = np.argsort(-uplift)  # 降序
    T_s = T[order]
    Y_s = Y[order]

    N_t = max(1, T.sum())
    N_c = max(1, (T == 0).sum())
    n = len(T)

    # 累积 n_{t,y=1}(k), n_{c,y=1}(k) 到前 k 个
    ct = np.cumsum((T_s == 1) & (Y_s == 1))
    cc = np.cumsum((T_s == 0) & (Y_s == 1))
    # Qini at step k = ct[k]/N_t - cc[k]/N_c
    qini_curve = ct.astype(float) / N_t - cc.astype(float) / N_c
    # 随机曲线：线性 (0,0) -> (n, baseline)
    baseline = np.mean(Y[T == 1]) - np.mean(Y[T == 0]) if N_t and N_c else 0
    random_curve = np.linspace(0, baseline, n + 1)[1:]
    # 面积：梯形积分
    area_model = float(np.trapz(qini_curve, dx=1.0 / n))
    area_random = float(np.trapz(random_curve, dx=1.0 / n))
    # 最优曲线：把所有 T=1且Y=1 放前面，T=0且Y=1 放后面
    opt_order = np.argsort(-((T == 1) & (Y == 1)).astype(float) + ((T == 0) & (Y == 1)).astype(float) * 0.5)
    T_opt = T[opt_order]
    Y_opt = Y[opt_order]
    ct_opt = np.cumsum((T_opt == 1) & (Y_opt == 1))
    cc_opt = np.cumsum((T_opt == 0) & (Y_opt == 1))
    qini_opt = ct_opt.astype(float) / N_t - cc_opt.astype(float) / N_c
    area_opt = float(np.trapz(qini_opt, dx=1.0 / n))
    if abs(area_opt - area_random) < 1e-12:
        return 0.0
    qini_coef = (area_model - area_random) / (area_opt - area_random)
    return float(np.clip(qini_coef, -1, 1))


def compute_decile_consistency(
    pred_ite: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    """
    微观 Decile 一致性：按 predicted ITE 分十分位，每档内观测效应与预测的相关系数 R。
    """
    pred_ite = np.asarray(pred_ite).ravel()
    T = np.asarray(T).ravel()
    Y = np.asarray(Y).ravel()
    n = len(pred_ite)

    if n < n_bins * 2:
        return {"decile_r": float("nan"), "decile_pred": [], "decile_obs": []}

    qs = np.linspace(0, 100, n_bins + 1)[1:-1]
    edges = np.percentile(pred_ite, qs)
    edges = np.concatenate([[-np.inf], edges, [np.inf]])
    pred_bins = []
    obs_bins = []
    for i in range(len(edges) - 1):
        mask = (pred_ite >= edges[i]) & (pred_ite < edges[i + 1])
        if mask.sum() == 0:
            pred_bins.append(0.0)
            obs_bins.append(0.0)
            continue
        pred_bins.append(float(np.mean(pred_ite[mask])))
        t_sub = T[mask]
        y_sub = Y[mask]
        n1 = (t_sub == 1).sum()
        n0 = (t_sub == 0).sum()
        if n1 and n0:
            obs_bins.append(float(np.mean(y_sub[t_sub == 1]) - np.mean(y_sub[t_sub == 0])))
        else:
            obs_bins.append(0.0)
    pred_bins = np.array(pred_bins)
    obs_bins = np.array(obs_bins)
    if pred_bins.std() < 1e-12 or obs_bins.std() < 1e-12:
        r = 0.0
    else:
        r = float(np.corrcoef(pred_bins, obs_bins)[0, 1])
    return {
        "decile_r": r,
        "decile_pred": [float(x) for x in pred_bins],
        "decile_obs": [float(x) for x in obs_bins],
    }


def compute_ite_quality_metrics(
    pred_ite: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
) -> dict[str, Any]:
    """汇总三个维度的 ITE 质量指标。"""
    ate = compute_ate_calibration(pred_ite, T, Y)
    qini = compute_qini_auc(pred_ite, T, Y)
    dec = compute_decile_consistency(pred_ite, T, Y)
    return {
        "macro_ate": ate,
        "auuc_qini": qini,
        "micro_decile": {"decile_r": dec["decile_r"]},
    }
