"""Lead estimator: XGB / LGBM / RF with optional Optuna on validation pseudo-target."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

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

try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    optuna = None  # type: ignore
    HAS_OPTUNA = False

logger = logging.getLogger(__name__)


def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_squared_error(y_true, y_pred))


def _val_mse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: np.ndarray | None,
) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if sample_weight is None:
        return _mse(y_true, y_pred)
    w = np.asarray(sample_weight, dtype=float).ravel()
    den = float(np.sum(w))
    if den <= 0:
        return float("inf")
    return float(np.sum(w * (y_true - y_pred) ** 2) / den)


def _fit_sample_weight_kw(sample_weight_tr: np.ndarray | None) -> dict[str, np.ndarray]:
    if sample_weight_tr is None:
        return {}
    return {"sample_weight": np.asarray(sample_weight_tr, dtype=float).ravel()}


def tune_and_fit_lgbm(
    X_tr: pd.DataFrame,
    pseudo_tr: np.ndarray,
    X_va: pd.DataFrame,
    pseudo_va: np.ndarray,
    n_trials: int,
    random_state: int,
    timeout: float | None,
    sample_weight_tr: np.ndarray | None = None,
    sample_weight_va: np.ndarray | None = None,
) -> tuple[Any, dict]:
    if not HAS_LGB:
        raise ImportError("lightgbm required")
    best_params: dict[str, Any] = {}
    best_mse = float("inf")
    best_model = None

    def default_model(p: dict) -> Any:
        return lgb.LGBMRegressor(
            random_state=random_state,
            verbose=-1,
            **p,
        )

    fkw = _fit_sample_weight_kw(sample_weight_tr)
    if not HAS_OPTUNA or n_trials <= 0:
        p = dict(n_estimators=150, max_depth=6, learning_rate=0.08, num_leaves=31)
        m = default_model(p)
        m.fit(X_tr, pseudo_tr, **fkw)
        pred = m.predict(X_va)
        return m, {**p, "val_mse": _val_mse(pseudo_va, pred, sample_weight_va)}

    def objective(trial: "optuna.Trial") -> float:
        p = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        }
        m = default_model(p)
        m.fit(X_tr, pseudo_tr, **fkw)
        pred = m.predict(X_va)
        return _val_mse(pseudo_va, pred, sample_weight_va)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    best_params = study.best_params.copy()
    best_mse = float(study.best_value)
    best_model = default_model(best_params)
    best_model.fit(X_tr, pseudo_tr, **fkw)
    best_params["val_mse"] = best_mse
    return best_model, best_params


def tune_and_fit_xgb(
    X_tr: pd.DataFrame,
    pseudo_tr: np.ndarray,
    X_va: pd.DataFrame,
    pseudo_va: np.ndarray,
    n_trials: int,
    random_state: int,
    timeout: float | None,
    sample_weight_tr: np.ndarray | None = None,
    sample_weight_va: np.ndarray | None = None,
) -> tuple[Any, dict]:
    if not HAS_XGB:
        raise ImportError("xgboost required")

    def default_model(p: dict) -> Any:
        return xgb.XGBRegressor(
            random_state=random_state,
            verbosity=0,
            **p,
        )

    fkw = _fit_sample_weight_kw(sample_weight_tr)
    if not HAS_OPTUNA or n_trials <= 0:
        p = dict(n_estimators=150, max_depth=5, learning_rate=0.08, subsample=0.9)
        m = default_model(p)
        m.fit(X_tr, pseudo_tr, **fkw)
        pred = m.predict(X_va)
        return m, {**p, "val_mse": _val_mse(pseudo_va, pred, sample_weight_va)}

    def objective(trial: "optuna.Trial") -> float:
        p = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        m = default_model(p)
        m.fit(X_tr, pseudo_tr, **fkw)
        pred = m.predict(X_va)
        return _val_mse(pseudo_va, pred, sample_weight_va)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    best_params = study.best_params.copy()
    best_mse = float(study.best_value)
    best_model = default_model(best_params)
    best_model.fit(X_tr, pseudo_tr, **fkw)
    best_params["val_mse"] = best_mse
    return best_model, best_params


def tune_and_fit_rf(
    X_tr: pd.DataFrame,
    pseudo_tr: np.ndarray,
    X_va: pd.DataFrame,
    pseudo_va: np.ndarray,
    n_trials: int,
    random_state: int,
    timeout: float | None,
    sample_weight_tr: np.ndarray | None = None,
    sample_weight_va: np.ndarray | None = None,
) -> tuple[Any, dict]:
    def default_model(p: dict) -> Any:
        return RandomForestRegressor(**p)

    fkw = _fit_sample_weight_kw(sample_weight_tr)
    if not HAS_OPTUNA or n_trials <= 0:
        p = dict(n_estimators=150, max_depth=10, min_samples_leaf=5, random_state=random_state)
        m = default_model(p)
        m.fit(X_tr, pseudo_tr, **fkw)
        pred = m.predict(X_va)
        out = {k: v for k, v in p.items() if k != "random_state"}
        out["val_mse"] = _val_mse(pseudo_va, pred, sample_weight_va)
        return m, out

    def objective(trial: "optuna.Trial") -> float:
        p = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 300),
            "max_depth": trial.suggest_int("max_depth", 4, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "random_state": random_state,
        }
        m = default_model(p)
        m.fit(X_tr, pseudo_tr, **fkw)
        pred = m.predict(X_va)
        return _val_mse(pseudo_va, pred, sample_weight_va)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    best_params = study.best_params.copy()
    best_mse = float(study.best_value)
    best_model = RandomForestRegressor(**best_params, random_state=random_state)
    best_model.fit(X_tr, pseudo_tr, **fkw)
    best_params["val_mse"] = best_mse
    return best_model, best_params


def select_best_lead(
    X_tr: pd.DataFrame,
    pseudo_tr: np.ndarray,
    X_va: pd.DataFrame,
    pseudo_va: np.ndarray,
    families: list[str],
    n_trials: int,
    random_state: int,
    timeout: float | None,
    sample_weight_tr: np.ndarray | None = None,
    sample_weight_va: np.ndarray | None = None,
) -> tuple[Any, str, dict, list[dict]]:
    """
    Returns (model, winning_family, best_params_dict, leaderboard).
    """
    leaderboard: list[dict] = []
    best_mse = float("inf")
    best_model = None
    best_family = ""
    best_params: dict = {}

    runners = {
        "lgbm": tune_and_fit_lgbm,
        "xgb": tune_and_fit_xgb,
        "rf": tune_and_fit_rf,
    }

    for fam in families:
        fam = fam.lower()
        if fam not in runners:
            logger.warning("Unknown lead family %s, skip", fam)
            continue
        try:
            m, params = runners[fam](
                X_tr,
                pseudo_tr,
                X_va,
                pseudo_va,
                n_trials,
                random_state,
                timeout,
                sample_weight_tr,
                sample_weight_va,
            )
        except Exception as e:
            logger.warning("Lead family %s failed: %s", fam, e)
            continue
        vmse = params.get("val_mse", float("inf"))
        leaderboard.append({"family": fam, "val_mse": vmse, "params": params})
        if vmse < best_mse:
            best_mse = vmse
            best_model = m
            best_family = fam
            best_params = {"family": fam, **params}

    if best_model is None:
        raise RuntimeError("No lead estimator succeeded")
    logger.info("Best lead estimator: %s val_mse=%.6f", best_family, best_mse)
    return best_model, best_family, best_params, leaderboard
