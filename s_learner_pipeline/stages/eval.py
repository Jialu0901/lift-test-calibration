"""Per-channel test metrics, plots, grouped ATE, placebo; helpers for plotting."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.ite_quality import compute_ite_quality_metrics, compute_qini_auc

logger = logging.getLogger(__name__)


def qini_curve_points(
    uplift: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (fraction, qini_curve) for plotting."""
    uplift = np.asarray(uplift).ravel()
    T = np.asarray(T).ravel().astype(int)
    Y = np.asarray(Y).ravel().astype(int)
    order = np.argsort(-uplift)
    T_s = T[order]
    Y_s = Y[order]
    N_t = max(1, T.sum())
    N_c = max(1, (T == 0).sum())
    n = len(T)
    ct = np.cumsum((T_s == 1) & (Y_s == 1))
    cc = np.cumsum((T_s == 0) & (Y_s == 1))
    qini_curve = ct.astype(float) / N_t - cc.astype(float) / N_c
    frac = np.arange(1, n + 1, dtype=float) / n
    return frac, qini_curve


def save_qini_plot(
    uplift: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    out_path: Path,
    title: str = "Qini curve",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frac, curve = qini_curve_points(uplift, T, Y)
    plt.figure(figsize=(6, 4))
    plt.plot(frac, curve, label="model")
    plt.xlabel("Fraction of population")
    plt.ylabel("Qini")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def grouped_ate_table(
    pred_ite: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    pred_ite = np.asarray(pred_ite).ravel()
    T = np.asarray(T).ravel()
    Y = np.asarray(Y, dtype=float).ravel()
    n = len(pred_ite)
    order = np.argsort(-pred_ite)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(n)
    bin_id = np.minimum((ranks * n_bins) // n, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = bin_id == b
        if not m.any():
            rows.append(
                {
                    "decile": b,
                    "n": 0,
                    "mean_pred_ite": np.nan,
                    "obs_ate": np.nan,
                }
            )
            continue
        t_sub = T[m]
        y_sub = Y[m]
        n1 = (t_sub == 1).sum()
        n0 = (t_sub == 0).sum()
        if n1 and n0:
            ate = float(np.mean(y_sub[t_sub == 1]) - np.mean(y_sub[t_sub == 0]))
        else:
            ate = float("nan")
        rows.append(
            {
                "decile": b,
                "n": int(m.sum()),
                "mean_pred_ite": float(np.mean(pred_ite[m])),
                "obs_ate": ate,
            }
        )
    return pd.DataFrame(rows)


def placebo_shuffle_t(
    uplift: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    n_shuffles: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    uplift = np.asarray(uplift).ravel()
    T = np.asarray(T).ravel()
    Y = np.asarray(Y).ravel()
    base = compute_qini_auc(uplift, T, Y)
    shuffled = []
    for i in range(n_shuffles):
        Tp = T.copy()
        rng.shuffle(Tp)
        shuffled.append(compute_qini_auc(uplift, Tp, Y))
    return {
        "qini_observed": float(base),
        "qini_shuffled_mean": float(np.mean(shuffled)) if shuffled else float("nan"),
        "qini_shuffled_std": float(np.std(shuffled)) if shuffled else float("nan"),
        "n_shuffles": n_shuffles,
    }


def evaluate_channel_test(
    pred: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    ch_dir: Path,
    channel: str,
) -> dict:
    ch_dir.mkdir(parents=True, exist_ok=True)
    plots = ch_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    metrics = {}
    try:
        metrics = compute_ite_quality_metrics(pred, T, Y)
    except Exception as e:
        logger.warning("ite_quality failed for %s: %s", channel, e)
        metrics = {}

    qini_val = float("nan")
    try:
        qini_val = float(compute_qini_auc(pred, T, Y))
    except Exception:
        pass

    save_qini_plot(pred, T, Y, plots / "auuc_qini.png", title=f"Qini {channel}")

    gdf = grouped_ate_table(pred, T, Y)
    gdf.to_csv(ch_dir / "grouped_ate.csv", index=False)

    out = {
        "channel": channel,
        "ite_quality": metrics,
        "qini": qini_val,
    }
    with open(ch_dir / "ite_quality_test.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    return out


def save_conversion_rank_plot(
    frac: np.ndarray,
    conv_rate: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(frac, conv_rate, label="cumulative conv. rate (top fraction)")
    plt.xlabel("Fraction of population (ranked by ITE_total)")
    plt.ylabel("Cumulative conversion rate in top-k")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
