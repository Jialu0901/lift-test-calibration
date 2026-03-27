"""Optuna tuning for S-learner base regressors; val objective = (weighted) MSE on Y with X_aug = [X, T]."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error

def _try_import_lgb() -> tuple[Any, bool]:
    try:
        import lightgbm as lgb

        return lgb, True
    except Exception:
        return None, False


def _try_import_xgb() -> tuple[Any, bool]:
    try:
        import xgboost as xgb

        return xgb, True
    except Exception:
        return None, False

try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    optuna = None  # type: ignore
    HAS_OPTUNA = False

logger = logging.getLogger(__name__)


def hstack_X_T(X: pd.DataFrame | np.ndarray, T: np.ndarray) -> np.ndarray:
    Xv = np.asarray(X, dtype=np.float64)
    Tcol = np.asarray(T, dtype=np.float64).reshape(-1, 1)
    return np.hstack([Xv, Tcol])


def val_mse(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray | None) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if w is None:
        return float(mean_squared_error(y_true, y_pred))
    w = np.asarray(w, dtype=np.float64).ravel()
    sw = float(np.sum(w))
    if sw <= 0:
        return float(mean_squared_error(y_true, y_pred))
    return float(np.sum(w * (y_true - y_pred) ** 2) / sw)


def fit_aug_learner(
    model: Any,
    X_aug_tr: np.ndarray,
    y_tr: np.ndarray,
    sample_weight: np.ndarray | None,
) -> None:
    if sample_weight is not None:
        model.fit(X_aug_tr, y_tr, sample_weight=sample_weight)
    else:
        model.fit(X_aug_tr, y_tr)


def fit_base_sregressor(
    learner: Any,
    X: np.ndarray,
    treatment: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None,
) -> Any:
    from causalml.inference.meta import BaseSRegressor

    s = BaseSRegressor(learner=learner)
    if sample_weight is not None:
        try:
            s.fit(X, treatment=treatment, y=y, sample_weight=sample_weight)
            return s
        except TypeError:
            logger.warning(
                "BaseSRegressor.fit does not accept sample_weight in this causalml version; "
                "fitting without sample_weight (Optuna still used weighted X_aug path)."
            )
    s.fit(X, treatment=treatment, y=y)
    return s


def tune_lgbm(
    X_tr: pd.DataFrame,
    T_tr: np.ndarray,
    Y_tr: np.ndarray,
    w_tr: np.ndarray | None,
    X_va: pd.DataFrame,
    T_va: np.ndarray,
    Y_va: np.ndarray,
    w_va: np.ndarray | None,
    n_trials: int,
    random_state: int,
    timeout: float | None,
) -> tuple[Any, dict[str, Any]]:
    lgb, ok = _try_import_lgb()
    if not ok or lgb is None:
        raise ImportError("lightgbm required for lgbm family")

    Xa_tr = hstack_X_T(X_tr, T_tr)
    Xa_va = hstack_X_T(X_va, T_va)
    y_tr = np.asarray(Y_tr, dtype=np.float64).ravel()
    y_va = np.asarray(Y_va, dtype=np.float64).ravel()

    def make_model(p: dict) -> Any:
        return lgb.LGBMRegressor(random_state=random_state, verbose=-1, **p)

    if not HAS_OPTUNA or n_trials <= 0:
        p = dict(n_estimators=150, max_depth=6, learning_rate=0.08, num_leaves=31)
        m = make_model(p)
        fit_aug_learner(m, Xa_tr, y_tr, w_tr)
        pred = m.predict(Xa_va)
        vm = val_mse(y_va, pred, w_va)
        learner = make_model(p)
        s = fit_base_sregressor(learner, np.asarray(X_tr, dtype=np.float64), T_tr, Y_tr, w_tr)
        return s, {**p, "val_mse": vm}

    def objective(trial: "optuna.Trial") -> float:
        p = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        }
        m = make_model(p)
        fit_aug_learner(m, Xa_tr, y_tr, w_tr)
        pred = m.predict(Xa_va)
        return val_mse(y_va, pred, w_va)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    best_params = study.best_params.copy()
    vm = float(study.best_value)
    learner = make_model(best_params)
    s = fit_base_sregressor(learner, np.asarray(X_tr, dtype=np.float64), T_tr, Y_tr, w_tr)
    best_params["val_mse"] = vm
    return s, best_params


def tune_xgb(
    X_tr: pd.DataFrame,
    T_tr: np.ndarray,
    Y_tr: np.ndarray,
    w_tr: np.ndarray | None,
    X_va: pd.DataFrame,
    T_va: np.ndarray,
    Y_va: np.ndarray,
    w_va: np.ndarray | None,
    n_trials: int,
    random_state: int,
    timeout: float | None,
) -> tuple[Any, dict[str, Any]]:
    xgb, ok = _try_import_xgb()
    if not ok or xgb is None:
        raise ImportError("xgboost required for xgb family")

    Xa_tr = hstack_X_T(X_tr, T_tr)
    Xa_va = hstack_X_T(X_va, T_va)
    y_tr = np.asarray(Y_tr, dtype=np.float64).ravel()
    y_va = np.asarray(Y_va, dtype=np.float64).ravel()

    def make_model(p: dict) -> Any:
        return xgb.XGBRegressor(random_state=random_state, verbosity=0, **p)

    if not HAS_OPTUNA or n_trials <= 0:
        p = dict(n_estimators=150, max_depth=5, learning_rate=0.08, subsample=0.9)
        m = make_model(p)
        fit_aug_learner(m, Xa_tr, y_tr, w_tr)
        pred = m.predict(Xa_va)
        vm = val_mse(y_va, pred, w_va)
        learner = make_model(p)
        s = fit_base_sregressor(learner, np.asarray(X_tr, dtype=np.float64), T_tr, Y_tr, w_tr)
        return s, {**p, "val_mse": vm}

    def objective(trial: "optuna.Trial") -> float:
        p = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        m = make_model(p)
        fit_aug_learner(m, Xa_tr, y_tr, w_tr)
        pred = m.predict(Xa_va)
        return val_mse(y_va, pred, w_va)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    best_params = study.best_params.copy()
    vm = float(study.best_value)
    learner = make_model(best_params)
    s = fit_base_sregressor(learner, np.asarray(X_tr, dtype=np.float64), T_tr, Y_tr, w_tr)
    best_params["val_mse"] = vm
    return s, best_params


def tune_rf(
    X_tr: pd.DataFrame,
    T_tr: np.ndarray,
    Y_tr: np.ndarray,
    w_tr: np.ndarray | None,
    X_va: pd.DataFrame,
    T_va: np.ndarray,
    Y_va: np.ndarray,
    w_va: np.ndarray | None,
    n_trials: int,
    random_state: int,
    timeout: float | None,
) -> tuple[Any, dict[str, Any]]:
    Xa_tr = hstack_X_T(X_tr, T_tr)
    Xa_va = hstack_X_T(X_va, T_va)
    y_tr = np.asarray(Y_tr, dtype=np.float64).ravel()
    y_va = np.asarray(Y_va, dtype=np.float64).ravel()

    def make_model(p: dict) -> Any:
        return RandomForestRegressor(**p)

    if not HAS_OPTUNA or n_trials <= 0:
        p = dict(n_estimators=150, max_depth=10, min_samples_leaf=5, random_state=random_state)
        m = make_model(p)
        fit_aug_learner(m, Xa_tr, y_tr, w_tr)
        pred = m.predict(Xa_va)
        vm = val_mse(y_va, pred, w_va)
        out = {k: v for k, v in p.items() if k != "random_state"}
        learner = RandomForestRegressor(**p)
        s = fit_base_sregressor(learner, np.asarray(X_tr, dtype=np.float64), T_tr, Y_tr, w_tr)
        return s, {**out, "val_mse": vm}

    def objective(trial: "optuna.Trial") -> float:
        p = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 300),
            "max_depth": trial.suggest_int("max_depth", 4, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "random_state": random_state,
        }
        m = make_model(p)
        fit_aug_learner(m, Xa_tr, y_tr, w_tr)
        pred = m.predict(Xa_va)
        return val_mse(y_va, pred, w_va)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    best_params = study.best_params.copy()
    vm = float(study.best_value)
    learner = RandomForestRegressor(**best_params, random_state=random_state)
    s = fit_base_sregressor(learner, np.asarray(X_tr, dtype=np.float64), T_tr, Y_tr, w_tr)
    best_params["val_mse"] = vm
    return s, best_params


def tune_elasticnet(
    X_tr: pd.DataFrame,
    T_tr: np.ndarray,
    Y_tr: np.ndarray,
    w_tr: np.ndarray | None,
    X_va: pd.DataFrame,
    T_va: np.ndarray,
    Y_va: np.ndarray,
    w_va: np.ndarray | None,
    n_trials: int,
    random_state: int,
    timeout: float | None,
) -> tuple[Any, dict[str, Any]]:
    Xa_tr = hstack_X_T(X_tr, T_tr)
    Xa_va = hstack_X_T(X_va, T_va)
    y_tr = np.asarray(Y_tr, dtype=np.float64).ravel()
    y_va = np.asarray(Y_va, dtype=np.float64).ravel()

    def make_model(p: dict) -> Any:
        return ElasticNet(random_state=random_state, max_iter=5000, **p)

    if not HAS_OPTUNA or n_trials <= 0:
        p = dict(alpha=0.01, l1_ratio=0.5)
        m = make_model(p)
        fit_aug_learner(m, Xa_tr, y_tr, w_tr)
        pred = m.predict(Xa_va)
        vm = val_mse(y_va, pred, w_va)
        learner = make_model(p)
        s = fit_base_sregressor(learner, np.asarray(X_tr, dtype=np.float64), T_tr, Y_tr, w_tr)
        return s, {**p, "val_mse": vm}

    def objective(trial: "optuna.Trial") -> float:
        p = {
            "alpha": trial.suggest_float("alpha", 1e-4, 1.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.05, 0.95),
        }
        m = make_model(p)
        fit_aug_learner(m, Xa_tr, y_tr, w_tr)
        pred = m.predict(Xa_va)
        return val_mse(y_va, pred, w_va)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    best_params = study.best_params.copy()
    vm = float(study.best_value)
    learner = make_model(best_params)
    s = fit_base_sregressor(learner, np.asarray(X_tr, dtype=np.float64), T_tr, Y_tr, w_tr)
    best_params["val_mse"] = vm
    return s, best_params


def select_best_slearner(
    X_tr: pd.DataFrame,
    T_tr: np.ndarray,
    Y_tr: np.ndarray,
    w_tr: np.ndarray | None,
    X_va: pd.DataFrame,
    T_va: np.ndarray,
    Y_va: np.ndarray,
    w_va: np.ndarray | None,
    families: list[str],
    n_trials: int,
    random_state: int,
    timeout: float | None,
) -> tuple[Any, str, dict[str, Any], list[dict[str, Any]]]:
    """
    Returns (BaseSRegressor fitted on train, winning_family, best_params_dict, leaderboard).
    """
    leaderboard: list[dict[str, Any]] = []
    best_mse = float("inf")
    best_model: Any = None
    best_family = ""
    best_params: dict[str, Any] = {}

    runners: dict[str, Any] = {
        "lgbm": tune_lgbm,
        "xgb": tune_xgb,
        "rf": tune_rf,
        "elasticnet": tune_elasticnet,
    }

    for fam in families:
        fam = fam.lower()
        if fam not in runners:
            logger.warning("Unknown learner family %s, skip", fam)
            continue
        try:
            m, params = runners[fam](
                X_tr,
                T_tr,
                Y_tr,
                w_tr,
                X_va,
                T_va,
                Y_va,
                w_va,
                n_trials,
                random_state,
                timeout,
            )
        except Exception as e:
            logger.warning("S-learner family %s failed: %s", fam, e)
            continue
        vmse = float(params.get("val_mse", float("inf")))
        leaderboard.append({"family": fam, "val_mse": vmse, "params": params})
        if vmse < best_mse:
            best_mse = vmse
            best_model = m
            best_family = fam
            best_params = {"family": fam, **params}

    if best_model is None:
        raise RuntimeError("No S-learner estimator succeeded")
    logger.info("Best S-learner: %s val_mse=%.6f", best_family, best_mse)
    return best_model, best_family, best_params, leaderboard


def make_learner_for_family(family: str, params: dict[str, Any], random_state: int) -> Any:
    """Build an unfitted sklearn-compatible learner from tuning ``params`` (may include ``family`` / ``val_mse``)."""
    p = {k: v for k, v in params.items() if k not in ("family", "val_mse")}
    fam = family.lower()
    if fam == "lgbm":
        lgb, ok = _try_import_lgb()
        if not ok or lgb is None:
            raise ImportError("lightgbm required")
        return lgb.LGBMRegressor(random_state=random_state, verbose=-1, **p)
    if fam == "xgb":
        xgb, ok = _try_import_xgb()
        if not ok or xgb is None:
            raise ImportError("xgboost required")
        return xgb.XGBRegressor(random_state=random_state, verbosity=0, **p)
    if fam == "rf":
        if "random_state" not in p:
            p = {**p, "random_state": random_state}
        return RandomForestRegressor(**p)
    if fam == "elasticnet":
        return ElasticNet(random_state=random_state, max_iter=5000, **p)
    raise ValueError(f"Unknown family: {family}")


def refit_slearner_train_val_from_params(
    best_family: str,
    best_params: dict[str, Any],
    X_tr: pd.DataFrame,
    X_va: pd.DataFrame,
    T_tr: np.ndarray,
    T_va: np.ndarray,
    Y_tr: np.ndarray,
    Y_va: np.ndarray,
    w_tr: np.ndarray | None,
    w_va: np.ndarray | None,
    random_state: int,
) -> Any:
    X_tv = np.vstack([np.asarray(X_tr, dtype=np.float64), np.asarray(X_va, dtype=np.float64)])
    T_tv = np.concatenate([np.asarray(T_tr).ravel(), np.asarray(T_va).ravel()])
    Y_tv = np.concatenate([np.asarray(Y_tr, dtype=np.float64).ravel(), np.asarray(Y_va, dtype=np.float64).ravel()])
    if w_tr is not None and w_va is not None:
        w_tv = np.concatenate([np.asarray(w_tr).ravel(), np.asarray(w_va).ravel()])
    else:
        w_tv = None
    learner = make_learner_for_family(best_family, best_params, random_state)
    return fit_base_sregressor(learner, X_tv, T_tv, Y_tv, w_tv)
