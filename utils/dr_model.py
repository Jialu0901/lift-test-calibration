"""
单 channel 的 DR-Learner：倾向 + 结果模型 (m_1, m_0) + 交叉拟合 DR 伪结果 + X→ITE 第二阶段。
推理时 ITE 仅依赖 X（部署路径）；可选 debug 方法用全样本 nuisance 算单位级 DR 公式。
"""
from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

logger = logging.getLogger(__name__)

EPS = 0.01  # propensity clip

# LightGBM 不支持 feature name 中的 JSON 特殊字符；仅允许 [a-zA-Z0-9_]
_JSON_UNSAFE_PATTERN = re.compile(r"[^a-zA-Z0-9_]")


def _sanitize_feature_names(names: list[str]) -> list[str]:
    """将 feature 名中的特殊字符替换为下划线，确保 LightGBM 可接受。"""
    seen: dict[str, int] = {}
    out = []
    for n in names:
        safe = _JSON_UNSAFE_PATTERN.sub("_", str(n)) or "f"
        safe = re.sub(r"_+", "_", safe).strip("_") or "f"  # 合并连续 _，去首尾
        if not safe[0].isalnum() and safe[0] != "_":
            safe = "f" + safe
        cnt = seen.get(safe, 0)
        seen[safe] = cnt + 1
        out.append(f"{safe}_{cnt}" if cnt else safe)
    return out


def _dr_pseudo_outcome(
    e: np.ndarray,
    m1: np.ndarray,
    m0: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
) -> np.ndarray:
    """DR 伪结果: m1-m0 + (T-e)/(e(1-e)) * (Y - m_T)。"""
    e = np.clip(np.asarray(e, dtype=float).ravel(), EPS, 1 - EPS)
    T = np.asarray(T).ravel()
    Y = np.asarray(Y, dtype=float).ravel()
    m1 = np.asarray(m1, dtype=float).ravel()
    m0 = np.asarray(m0, dtype=float).ravel()
    mT = np.where(T == 1, m1, m0)
    return (m1 - m0) + (T - e) / (e * (1 - e)) * (Y - mT)


class DRChannelModel:
    """
    单 channel 的 DR-Learner：全 train 上拟合 nuisance（诊断用）；
    ITE 预测为第二阶段 regressor，仅在 fit 时消费 Y/T。
    """

    def __init__(self, channel: str, use_lightgbm: bool = True) -> None:
        self.channel = channel
        self.use_lightgbm = use_lightgbm and HAS_LGB
        self.propensity_model: Any = None
        self.outcome_m1: Any = None
        self.outcome_m0: Any = None
        self.ite_model: Any = None
        self.scaler: StandardScaler | None = None
        self._fitted = False
        self._n_cf_folds_used: int = 1

    def _get_propensity_model(self):
        if self.use_lightgbm:
            return lgb.LGBMClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                random_state=42, verbose=-1,
            )
        return LogisticRegression(max_iter=1000, random_state=42)

    def _get_outcome_model(self):
        if self.use_lightgbm:
            return lgb.LGBMRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                random_state=42, verbose=-1,
            )
        from sklearn.linear_model import Ridge
        return Ridge(alpha=1.0, random_state=42)

    def _get_ite_model(self):
        if self.use_lightgbm:
            return lgb.LGBMRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                random_state=42, verbose=-1,
            )
        from sklearn.linear_model import Ridge
        return Ridge(alpha=1.0, random_state=42)

    def _fit_nuisance_on_subset(
        self,
        X_df: pd.DataFrame,
        T: np.ndarray,
        Y: np.ndarray,
    ) -> tuple[Any, Any, Any, bool]:
        """
        在子集上拟合倾向 + m1/m0。若 T 无变异则 propensity 未训练，返回 propensity_fitted=False。
        """
        T = np.asarray(T).ravel()
        Y = np.asarray(Y).ravel()
        prop = self._get_propensity_model()
        m1_m = self._get_outcome_model()
        m0_m = self._get_outcome_model()
        mask1 = T == 1
        mask0 = T == 0
        propensity_fitted = len(np.unique(T)) >= 2
        if propensity_fitted:
            prop.fit(X_df, T)
        if mask1.sum() > 0:
            m1_m.fit(X_df.iloc[np.flatnonzero(mask1)], Y[mask1])
        else:
            m1_m.fit(X_df, np.zeros(len(Y)))
        if mask0.sum() > 0:
            m0_m.fit(X_df.iloc[np.flatnonzero(mask0)], Y[mask0])
        else:
            m0_m.fit(X_df, np.zeros(len(Y)))
        return prop, m1_m, m0_m, propensity_fitted

    def _predict_propensity_or_constant(
        self,
        prop: Any,
        X_df: pd.DataFrame,
        propensity_fitted: bool,
    ) -> np.ndarray:
        n = len(X_df)
        if not propensity_fitted:
            return np.full(n, 0.5, dtype=float)
        proba = prop.predict_proba(X_df)
        if proba.shape[1] == 2:
            return proba[:, 1].astype(float)
        return proba.ravel().astype(float)

    def _effective_cf_folds(self, n: int, n_cf_folds: int, T: np.ndarray) -> int:
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

    def _cross_fit_pseudo_tau(
        self,
        X_scaled_df: pd.DataFrame,
        T: np.ndarray,
        Y: np.ndarray,
        n_cf_folds: int,
    ) -> np.ndarray:
        n = len(X_scaled_df)
        T = np.asarray(T).ravel()
        Y = np.asarray(Y).ravel()
        pseudo = np.zeros(n, dtype=float)
        k = self._effective_cf_folds(n, n_cf_folds, T)
        self._n_cf_folds_used = k

        if k <= 1:
            prop, m1_m, m0_m, ok = self._fit_nuisance_on_subset(X_scaled_df, T, Y)
            e = self._predict_propensity_or_constant(prop, X_scaled_df, ok)
            m1 = m1_m.predict(X_scaled_df)
            m0 = m0_m.predict(X_scaled_df)
            pseudo = _dr_pseudo_outcome(e, m1, m0, T, Y)
            if n_cf_folds > 1:
                logger.warning(
                    "Channel %s: cross-fitting disabled (folds=%s, n0=%s, n1=%s); "
                    "pseudo-tau uses in-sample nuisance (biased).",
                    self.channel, n_cf_folds, int((T == 0).sum()), int((T == 1).sum()),
                )
            return pseudo

        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        for fold_id, (train_idx, val_idx) in enumerate(skf.split(np.zeros(n), T)):
            X_tr = X_scaled_df.iloc[train_idx]
            T_tr = T[train_idx]
            Y_tr = Y[train_idx]
            X_val = X_scaled_df.iloc[val_idx]
            T_val = T[val_idx]
            Y_val = Y[val_idx]
            prop, m1_m, m0_m, ok = self._fit_nuisance_on_subset(X_tr, T_tr, Y_tr)
            if not ok:
                logger.warning(
                    "Channel %s: fold %d complement has single T level; using e=0.5 for holdout pseudo.",
                    self.channel, fold_id,
                )
            e_val = self._predict_propensity_or_constant(prop, X_val, ok)
            m1_val = m1_m.predict(X_val)
            m0_val = m0_m.predict(X_val)
            pseudo[val_idx] = _dr_pseudo_outcome(e_val, m1_val, m0_val, T_val, Y_val)

        return pseudo

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        n_cf_folds: int = 5,
    ) -> None:
        """
        全 train 上：交叉拟合 DR 伪结果 → 拟合 ite_model(X)；再在全数据上拟合 nuisance（诊断/metrics）。
        """
        if hasattr(X, "columns"):
            raw_names = list(X.columns)
            X = np.asarray(X)
        else:
            raw_names = [f"f{i}" for i in range(X.shape[1])]
            X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        T = np.asarray(T).ravel()
        Y = np.asarray(Y).ravel()

        self._feature_names = _sanitize_feature_names(raw_names)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=self._feature_names)

        pseudo_tau = self._cross_fit_pseudo_tau(X_scaled_df, T, Y, n_cf_folds)
        pseudo_tau = np.nan_to_num(pseudo_tau, nan=0.0, posinf=0.0, neginf=0.0)

        self.ite_model = self._get_ite_model()
        self.ite_model.fit(X_scaled_df, pseudo_tau)

        # 全 train nuisance（与 compute_metrics / diagnostics 一致）
        self.propensity_model = self._get_propensity_model()
        self.propensity_model.fit(X_scaled_df, T)

        mask1 = T == 1
        mask0 = T == 0
        self.outcome_m1 = self._get_outcome_model()
        self.outcome_m0 = self._get_outcome_model()
        if mask1.sum() > 0:
            self.outcome_m1.fit(X_scaled_df.iloc[np.flatnonzero(mask1)], Y[mask1])
        else:
            self.outcome_m1.fit(X_scaled_df, np.zeros_like(Y))
        if mask0.sum() > 0:
            self.outcome_m0.fit(X_scaled_df.iloc[np.flatnonzero(mask0)], Y[mask0])
        else:
            self.outcome_m0.fit(X_scaled_df, np.zeros_like(Y))

        self._fitted = True

    def _transform_X(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        X = np.asarray(X) if not isinstance(X, np.ndarray) else X
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        if getattr(self, "_feature_names", None) is not None:
            return pd.DataFrame(X, columns=self._feature_names)
        return pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])

    def predict_propensity(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """P(T=1|X)。"""
        if not self._fitted:
            raise RuntimeError("Model not fitted")
        Xs = self._transform_X(X)
        proba = self.propensity_model.predict_proba(Xs)
        if proba.shape[1] == 2:
            return proba[:, 1]
        return proba.ravel()

    def predict_outcome(self, X: pd.DataFrame | np.ndarray, T: np.ndarray) -> np.ndarray:
        """E[Y|T,X]，根据 T 选 m_1 或 m_0。"""
        if not self._fitted:
            raise RuntimeError("Model not fitted")
        Xs = self._transform_X(X)
        T = np.asarray(T).ravel()
        pred = np.zeros(len(T), dtype=float)
        m1 = self.outcome_m1.predict(Xs)
        m0 = self.outcome_m0.predict(Xs)
        pred[T == 1] = m1[T == 1]
        pred[T == 0] = m0[T == 0]
        return pred

    def predict_ite(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """DR-Learner 第二阶段：仅由 X 预测 ITE（部署路径，不依赖 Y/T）。"""
        if not self._fitted or self.ite_model is None:
            raise RuntimeError("Model not fitted")
        Xs = self._transform_X(X)
        return np.asarray(self.ite_model.predict(Xs), dtype=float).ravel()

    def dr_pseudo_outcome_for_debug(
        self,
        X: pd.DataFrame | np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
    ) -> np.ndarray:
        """
        用已拟合的全样本 nuisance 计算单位级 DR 伪结果（有偏、仅供对照；非默认 ITE）。
        ITE = m_1 - m_0 + (T - e) / (e(1-e)) * (Y - m_T)
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted")
        Xs = self._transform_X(X)
        T = np.asarray(T).ravel()
        Y = np.asarray(Y).ravel()
        e = np.clip(self.predict_propensity(X), EPS, 1 - EPS)
        m1 = self.outcome_m1.predict(Xs)
        m0 = self.outcome_m0.predict(Xs)
        return _dr_pseudo_outcome(e, m1, m0, T, Y)

    def compute_metrics(
        self,
        X: pd.DataFrame | np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
    ) -> dict[str, Any]:
        """在给定数据上计算 propensity、outcome、Y 预测的评估指标。"""
        X = np.asarray(X) if not isinstance(X, np.ndarray) else X
        T = np.asarray(T).ravel()
        Y = np.asarray(Y).ravel()

        e = self.predict_propensity(X)
        e_clip = np.clip(e, 1e-10, 1 - 1e-10)
        T_pred = (e > 0.5).astype(int)

        metrics: dict[str, Any] = {}

        acc = (T_pred == T).mean()
        metrics["propensity"] = {"accuracy": float(acc), "log_loss": float(log_loss(T, e_clip))}
        try:
            if len(np.unique(T)) >= 2:
                metrics["propensity"]["auc"] = float(roc_auc_score(T, e))
            else:
                metrics["propensity"]["auc"] = 0.5
        except Exception:
            metrics["propensity"]["auc"] = 0.5

        mask1 = T == 1
        mask0 = T == 0
        Xs = self._transform_X(X)
        m1 = self.outcome_m1.predict(Xs)
        m0 = self.outcome_m0.predict(Xs)

        for name, mask, pred in [("outcome_m1", mask1, m1), ("outcome_m0", mask0, m0)]:
            if mask.sum() > 0:
                y_sub = Y[mask]
                p_sub = pred[mask]
                metrics[name] = {
                    "mse": float(mean_squared_error(y_sub, p_sub)),
                    "mae": float(mean_absolute_error(y_sub, p_sub)),
                    "r2": float(1 - mean_squared_error(y_sub, p_sub) / (y_sub.var() + 1e-10)),
                }
            else:
                metrics[name] = {"mse": 0.0, "mae": 0.0, "r2": 0.0}

        mT = np.where(T == 1, m1, m0)
        if np.unique(Y).size <= 2:
            y_pred = (mT > 0.5).astype(int)
            metrics["y_pred"] = {"accuracy": float((y_pred == Y).mean())}
            try:
                if len(np.unique(Y)) >= 2:
                    metrics["y_pred"]["auc"] = float(roc_auc_score(Y, np.clip(mT, 0, 1)))
                else:
                    metrics["y_pred"]["auc"] = 0.5
            except Exception:
                metrics["y_pred"]["auc"] = 0.5
        else:
            metrics["y_pred"] = {
                "mse": float(mean_squared_error(Y, mT)),
                "mae": float(mean_absolute_error(Y, mT)),
            }

        return metrics
