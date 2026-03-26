"""Total ITE ranking metrics: conversion by rank, conversion-Qini vs random, decile R²."""
from __future__ import annotations

import numpy as np
import pandas as pd


def conversion_by_bins(score: np.ndarray, y: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Deciles by score descending: bin index, n, mean(score), mean(y)."""
    score = np.asarray(score, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = len(score)
    order = np.argsort(-score)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(n)
    bin_id = np.minimum((ranks * n_bins) // n, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = bin_id == b
        if not m.any():
            rows.append({"bin": b, "n": 0, "mean_score": np.nan, "conversion_rate": np.nan})
        else:
            rows.append(
                {
                    "bin": b,
                    "n": int(m.sum()),
                    "mean_score": float(np.mean(score[m])),
                    "conversion_rate": float(np.mean(y[m])),
                }
            )
    return pd.DataFrame(rows)


def decile_r2_score(score: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """R² of OLS mean(Y|bin) ~ mean(score|bin) across bins (unweighted)."""
    df = conversion_by_bins(score, y, n_bins=n_bins)
    df = df[df["n"] > 0]
    if len(df) < 3:
        return float("nan")
    x = df["mean_score"].values
    yy = df["conversion_rate"].values
    x_mean = x.mean()
    y_mean = yy.mean()
    beta = np.sum((x - x_mean) * (yy - y_mean)) / (np.sum((x - x_mean) ** 2) + 1e-12)
    alpha = y_mean - beta * x_mean
    pred = alpha + beta * x
    ss_res = np.sum((yy - pred) ** 2)
    ss_tot = np.sum((yy - y_mean) ** 2)
    if ss_tot < 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def conversion_qini_vs_random(
    score: np.ndarray,
    y: np.ndarray,
    n_random: int = 20,
    seed: int = 42,
) -> tuple[float, dict]:
    """
    Area under cumulative conversion-rate curve (model order) vs mean random order,
    normalized by optimal order. Returns (qini_coef, detail).
    """
    y = np.asarray(y, dtype=float).ravel()
    score = np.asarray(score, dtype=float).ravel()
    n = len(y)
    if n < 10:
        return float("nan"), {"reason": "n too small"}

    order = np.argsort(-score)
    y_s = y[order]
    k = np.arange(1, n + 1, dtype=float)
    cum_y = np.cumsum(y_s)
    curve_m = cum_y / k
    dx = 1.0 / n
    area_model = float(np.trapz(curve_m, dx=dx))

    rng = np.random.default_rng(seed)
    random_areas = []
    for _ in range(n_random):
        perm = rng.permutation(n)
        y_r = y[perm]
        cum_r = np.cumsum(y_r)
        curve_r = cum_r / k
        random_areas.append(float(np.trapz(curve_r, dx=dx)))
    area_random = float(np.mean(random_areas))

    y_opt = np.sort(y)[::-1]
    cum_opt = np.cumsum(y_opt)
    curve_opt = cum_opt / k
    area_opt = float(np.trapz(curve_opt, dx=dx))

    if abs(area_opt - area_random) < 1e-12:
        qini = 0.0
    else:
        qini = (area_model - area_random) / (area_opt - area_random)

    return float(np.clip(qini, -1.0, 1.0)), {
        "area_model": area_model,
        "area_random_mean": area_random,
        "area_opt": area_opt,
        "n_random": n_random,
    }


def cumulative_conversion_curve(score: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fractions 1..n/n vs cumulative conversion rate in top-k by score."""
    y = np.asarray(y, dtype=float).ravel()
    score = np.asarray(score, dtype=float).ravel()
    n = len(y)
    order = np.argsort(-score)
    y_s = y[order]
    cum_y = np.cumsum(y_s)
    k = np.arange(1, n + 1, dtype=float)
    frac = k / n
    conv_rate = cum_y / k
    return frac, conv_rate
