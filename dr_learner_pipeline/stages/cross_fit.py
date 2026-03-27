"""Cross-fitted DR pseudo-outcomes using selected nuisance model types."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from utils.dr_model import EPS, _dr_pseudo_outcome

from .base_nuisance import (
    fit_outcome_arm,
    fit_propensity,
    make_outcome,
    make_propensity,
    predict_propensity_proba,
)

logger = logging.getLogger(__name__)


def _effective_k(n: int, n_cf_folds: int, T: np.ndarray) -> int:
    T = np.asarray(T).ravel()
    n0 = int((T == 0).sum())
    n1 = int((T == 1).sum())
    if n_cf_folds <= 1 or min(n0, n1) < 2:
        return 1
    max_splits = min(n_cf_folds, n0, n1)
    if max_splits < 2:
        return 1
    if n < max_splits * 2:
        max_splits = max(2, min(max_splits, n // 2))
    return max_splits


def _fit_nuisance_bundle(
    X_df: pd.DataFrame,
    T: np.ndarray,
    Y: np.ndarray,
    prop_kind: str,
    out_kind: str,
    random_state: int,
    sample_weight: np.ndarray | None = None,
):
    T = np.asarray(T).ravel()
    Y = np.asarray(Y, dtype=float).ravel()
    pk = (prop_kind or "").lower()
    prop = None
    prop_fitted = None
    if pk and pk != "none":
        prop = make_propensity(prop_kind, random_state)
        prop_fitted = fit_propensity(prop, X_df, T, sample_weight=sample_weight)
    m1 = make_outcome(out_kind, random_state)
    m0 = make_outcome(out_kind, random_state)
    mask1 = T == 1
    mask0 = T == 0
    fit_outcome_arm(m1, X_df, Y, mask1, sample_weight=sample_weight)
    fit_outcome_arm(m0, X_df, Y, mask0, sample_weight=sample_weight)
    ok = prop_fitted is not None
    return prop_fitted, m1, m0, ok


def cross_fit_pseudo_tau(
    X_df: pd.DataFrame,
    T: np.ndarray,
    Y: np.ndarray,
    *,
    prop_kind: str,
    out_kind: str,
    n_cf_folds: int,
    random_state: int,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    n = len(X_df)
    T = np.asarray(T).ravel()
    Y = np.asarray(Y, dtype=float).ravel()
    pseudo = np.zeros(n, dtype=float)
    k = _effective_k(n, n_cf_folds, T)

    if k <= 1:
        prop, m1, m0, ok = _fit_nuisance_bundle(
            X_df, T, Y, prop_kind, out_kind, random_state, sample_weight=sample_weight
        )
        e = predict_propensity_proba(prop, X_df, n) if ok else np.full(n, 0.5)
        e = np.clip(e, EPS, 1 - EPS)
        m1p = m1.predict(X_df)
        m0p = m0.predict(X_df)
        return _dr_pseudo_outcome(e, m1p, m0p, T, Y)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    for fold_id, (train_idx, val_idx) in enumerate(skf.split(np.zeros(n), T)):
        X_tr = X_df.iloc[train_idx]
        T_tr = T[train_idx]
        Y_tr = Y[train_idx]
        X_val = X_df.iloc[val_idx]
        T_val = T[val_idx]
        Y_val = Y[val_idx]
        sw_tr = None
        if sample_weight is not None:
            sw_tr = np.asarray(sample_weight, dtype=float).ravel()[train_idx]
        prop, m1, m0, ok = _fit_nuisance_bundle(
            X_tr, T_tr, Y_tr, prop_kind, out_kind, random_state + fold_id, sample_weight=sw_tr
        )
        e_val = predict_propensity_proba(prop, X_val, len(val_idx))
        if not ok:
            logger.warning("cross_fit fold %d: propensity degenerate, e=0.5", fold_id)
            e_val = np.full(len(val_idx), 0.5)
        m1v = m1.predict(X_val)
        m0v = m0.predict(X_val)
        pseudo[val_idx] = _dr_pseudo_outcome(e_val, m1v, m0v, T_val, Y_val)

    return pseudo


def refit_nuisance_full_train(
    X_df: pd.DataFrame,
    T: np.ndarray,
    Y: np.ndarray,
    prop_kind: str,
    out_kind: str,
    random_state: int,
    sample_weight: np.ndarray | None = None,
):
    """Refit nuisances on all training rows (for Val pseudo targets)."""
    return _fit_nuisance_bundle(
        X_df, T, Y, prop_kind, out_kind, random_state, sample_weight=sample_weight
    )


def dr_pseudo_on_split(
    prop_model,
    m1,
    m0,
    prop_ok: bool,
    X_df: pd.DataFrame,
    T: np.ndarray,
    Y: np.ndarray,
) -> np.ndarray:
    n = len(X_df)
    T = np.asarray(T).ravel()
    Y = np.asarray(Y, dtype=float).ravel()
    e = predict_propensity_proba(prop_model, X_df, n) if prop_ok else np.full(n, 0.5)
    m1p = m1.predict(X_df)
    m0p = m0.predict(X_df)
    return _dr_pseudo_outcome(e, m1p, m0p, T, Y)
