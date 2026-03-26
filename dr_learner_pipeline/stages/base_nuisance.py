"""Train candidate propensity and outcome models; pick best on validation."""
from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score
from sklearn.ensemble import RandomForestRegressor

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

logger = logging.getLogger(__name__)

EPS = 1e-10


def _ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error (simple binning)."""
    probs = np.clip(np.asarray(probs, dtype=float).ravel(), EPS, 1 - EPS)
    y = np.asarray(y).astype(int).ravel()
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y)
    for i in range(n_bins):
        m = (probs >= bins[i]) & (probs < bins[i + 1])
        if i == n_bins - 1:
            m = (probs >= bins[i]) & (probs <= bins[i + 1])
        cnt = int(m.sum())
        if cnt == 0:
            continue
        conf = float(np.mean(probs[m]))
        acc = float(np.mean(y[m]))
        ece += (cnt / n) * abs(conf - acc)
    return float(ece)


def make_propensity(kind: str, random_state: int) -> Any:
    k = kind.lower()
    if k == "lr":
        return LogisticRegression(max_iter=1000, random_state=random_state)
    if k == "xgb":
        if not HAS_XGB:
            raise ImportError("xgboost required for propensity xgb")
        kw: dict[str, Any] = dict(
            n_estimators=120,
            max_depth=5,
            learning_rate=0.1,
            random_state=random_state,
            verbosity=0,
        )
        try:
            return xgb.XGBClassifier(**kw, eval_metric="logloss")
        except TypeError:
            return xgb.XGBClassifier(**kw)
    raise ValueError(f"Unknown propensity kind: {kind}")


def make_outcome(kind: str, random_state: int) -> Any:
    k = kind.lower()
    if k == "lgbm":
        if not HAS_LGB:
            raise ImportError("lightgbm required for outcome lgbm")
        return lgb.LGBMRegressor(
            n_estimators=120,
            max_depth=5,
            learning_rate=0.1,
            random_state=random_state,
            verbose=-1,
        )
    if k == "rf":
        return RandomForestRegressor(
            n_estimators=120,
            max_depth=8,
            random_state=random_state,
        )
    raise ValueError(f"Unknown outcome kind: {kind}")


def fit_propensity(model: Any, X: pd.DataFrame, T: np.ndarray) -> Any:
    T = np.asarray(T).ravel()
    if len(np.unique(T)) < 2:
        return None
    model.fit(X, T)
    return model


def predict_propensity_proba(model: Any | None, X: pd.DataFrame, n: int) -> np.ndarray:
    if model is None:
        return np.full(n, 0.5)
    p = model.predict_proba(X)
    if p.shape[1] == 2:
        return p[:, 1].astype(float)
    return p.ravel().astype(float)


def fit_outcome_arm(model: Any, X: pd.DataFrame, Y: np.ndarray, mask: np.ndarray) -> Any:
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() > 0:
        model.fit(X.iloc[np.flatnonzero(mask)], Y[mask])
    else:
        model.fit(X, np.zeros(len(Y)))
    return model


def score_propensity_on_val(
    model: Any | None,
    X_val: pd.DataFrame,
    T_val: np.ndarray,
) -> dict[str, float]:
    T_val = np.asarray(T_val).ravel()
    n = len(T_val)
    if model is None or len(np.unique(T_val)) < 2:
        return {"log_loss": float("nan"), "ece": float("nan")}
    p = predict_propensity_proba(model, X_val, n)
    p = np.clip(p, EPS, 1 - EPS)
    ll = float(log_loss(T_val, p))
    ece = _ece(p, T_val)
    return {"log_loss": ll, "ece": ece}


def score_outcome_pair_on_val(
    m1: Any,
    m0: Any,
    X_val: pd.DataFrame,
    T_val: np.ndarray,
    Y_val: np.ndarray,
) -> dict[str, float]:
    T_val = np.asarray(T_val).ravel()
    Y_val = np.asarray(Y_val, dtype=float).ravel()
    m1p = m1.predict(X_val)
    m0p = m0.predict(X_val)
    mT = np.where(T_val == 1, m1p, m0p)
    mse = float(mean_squared_error(Y_val, mT))
    yhat = np.clip(mT, 0.0, 1.0)
    auc = 0.5
    if len(np.unique(Y_val)) >= 2:
        try:
            auc = float(roc_auc_score(Y_val, yhat))
        except Exception:
            auc = 0.5
    return {"mse": mse, "auc": auc}


def select_base_models(
    X_tr: pd.DataFrame,
    T_tr: np.ndarray,
    Y_tr: np.ndarray,
    X_va: pd.DataFrame,
    T_va: np.ndarray,
    Y_va: np.ndarray,
    propensity_kinds: list[str],
    outcome_kinds: list[str],
    random_state: int,
) -> dict[str, Any]:
    """
    Returns dict with best_propensity_kind, best_outcome_kind, fitted models on train,
    and scores table.
    """
    T_tr = np.asarray(T_tr).ravel()
    Y_tr = np.asarray(Y_tr, dtype=float).ravel()

    prop_scores: list[dict] = []
    best_prop = "none"
    best_prop_model = None
    best_ll = float("inf")

    for pk in propensity_kinds:
        try:
            m = make_propensity(pk, random_state)
        except Exception as e:
            logger.warning("Skip propensity %s: %s", pk, e)
            continue
        fitted = fit_propensity(m, X_tr, T_tr)
        sc = score_propensity_on_val(fitted, X_va, T_va)
        prop_scores.append({"kind": pk, **sc})
        if fitted is not None and sc["log_loss"] == sc["log_loss"] and sc["log_loss"] < best_ll:
            best_ll = sc["log_loss"]
            best_prop = pk
            best_prop_model = fitted

    if best_prop_model is None:
        best_prop = "none"
        logger.warning("No valid propensity model; using e=0.5")

    out_scores: list[dict] = []
    best_ok = None
    best_mse = float("inf")

    for ok in outcome_kinds:
        try:
            m1 = make_outcome(ok, random_state)
            m0 = make_outcome(ok, random_state)
        except Exception as e:
            logger.warning("Skip outcome %s: %s", ok, e)
            continue
        mask1 = T_tr == 1
        mask0 = T_tr == 0
        fit_outcome_arm(m1, X_tr, Y_tr, mask1)
        fit_outcome_arm(m0, X_tr, Y_tr, mask0)
        sc = score_outcome_pair_on_val(m1, m0, X_va, T_va, Y_va)
        out_scores.append({"kind": ok, **sc})
        if sc["mse"] < best_mse:
            best_mse = sc["mse"]
            best_ok = (ok, m1, m0)

    if best_ok is None:
        raise RuntimeError("No valid outcome model could be trained")

    ok, m1_f, m0_f = best_ok
    logger.info(
        "Base models selected: propensity=%s (val log_loss=%s), outcome=%s (val mse=%.6f)",
        best_prop,
        f"{best_ll:.4f}" if best_ll == best_ll else "nan",
        ok,
        best_mse,
    )

    return {
        "best_propensity_kind": best_prop,
        "best_outcome_kind": ok,
        "propensity_model": best_prop_model,
        "outcome_m1": m1_f,
        "outcome_m0": m0_f,
        "propensity_scores": prop_scores,
        "outcome_scores": out_scores,
    }
