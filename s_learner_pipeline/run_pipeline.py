"""
S-Learner experiment pipeline (CausalML BaseSRegressor + Optuna).

Run from model_build directory:
  python -m s_learner_pipeline.run_pipeline \\
    --config s_learner_pipeline/config/pipeline_s_learner.yaml \\
    --split_dates_path path/to/split_dates.json \\
    --db-config-json path/to/db_config.json \\
    --db-table your_schema.your_wide_table

Notebook: ``DB_CONFIG``, ``split_dates`` dict, ``PIPELINE_CONFIG`` / overrides →
``load_wide_and_split`` + ``run_pipeline_training_from_splits`` or ``run_pipeline_notebook``.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TypeAlias

import joblib
import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from utils.config import CHANNELS
from utils.dr_model import _sanitize_feature_names
from utils.uplift_feature_selection import merge_csv_and_protected_features, resolve_protected_features
from utils.wide_sample_pipeline import candidate_features_from_df, load_and_build_sample

from s_learner_pipeline.logging_utils import setup_run_logging
from s_learner_pipeline.metrics_total import (
    conversion_by_bins,
    conversion_qini_vs_random,
    cumulative_conversion_curve,
    decile_r2_score,
)
from s_learner_pipeline.repro import copy_artifacts, write_config_backup
from s_learner_pipeline.splits import (
    date_range_for_load,
    date_range_for_split_dict,
    split_train_val_test_by_dates,
    split_train_val_test_by_dates_from_dict,
)
from s_learner_pipeline.stages.eval import (
    evaluate_channel_test,
    placebo_shuffle_t,
    save_conversion_rank_plot,
)
from s_learner_pipeline.stages.slearner_tune import (
    refit_slearner_train_val_from_params,
    select_best_slearner,
)

logger = logging.getLogger(__name__)

SplitSpec: TypeAlias = Path | dict[str, Any]


def _deep_merge_dict(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def merge_pipeline_config(
    base_cfg: dict[str, Any],
    overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    if not overrides:
        return dict(base_cfg)
    return _deep_merge_dict(dict(base_cfg), overrides)


def _load_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except ImportError as e:
        raise RuntimeError("Install PyYAML: pip install pyyaml") from e
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_under_model_build(p: str | Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return (_root / path).resolve()


def _load_selected_features_csv(path: Path) -> list[str]:
    tab = pd.read_csv(path)
    if "selected_feature" not in tab.columns:
        raise SystemExit(f"CSV must contain column 'selected_feature': {path}")
    out: list[str] = []
    for x in tab["selected_feature"].dropna():
        s = str(x).strip()
        if s:
            out.append(s)
    return out


def _merged_intersect_usable(merged_names: list[str], usable: set[str]) -> list[str]:
    out: list[str] = []
    for c in merged_names:
        if c not in usable:
            logger.warning("Skipping selected feature not in train/val/test columns: %s", c)
            continue
        out.append(c)
    return out


def _feat_list_all_candidates(
    cand: list[str],
    protected_resolved: list[str],
    usable: set[str],
) -> list[str]:
    prot_set = set(protected_resolved)
    rest = [c for c in cand if c not in prot_set]
    merged = merge_csv_and_protected_features(rest, protected_resolved)
    return _merged_intersect_usable(merged, usable)


def _prepare_X(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feat_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, None, list[str]]:
    X_tr = train_df[feat_cols].fillna(0).astype(float)
    X_va = val_df[feat_cols].fillna(0).astype(float)
    X_te = test_df[feat_cols].fillna(0).astype(float)
    raw_names = list(X_tr.columns)
    safe = _sanitize_feature_names(raw_names)
    X_tr_df = X_tr.copy()
    X_va_df = X_va.copy()
    X_te_df = X_te.copy()
    X_tr_df.columns = safe
    X_va_df.columns = safe
    X_te_df.columns = safe
    return X_tr_df, X_va_df, X_te_df, None, safe


def _parse_row_weights(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, str | None]:
    """
    One set of row weights for the whole run (all channels). Returns (w_tr, w_va, w_te, active, column_name).
    """
    use_sw = bool(cfg.get("use_sample_weight", True))
    col = str(cfg.get("sample_weight_column") or "sample_weight")
    n1, n2, n3 = len(train_df), len(val_df), len(test_df)
    ones_tr = np.ones(n1, dtype=np.float64)
    ones_va = np.ones(n2, dtype=np.float64)
    ones_te = np.ones(n3, dtype=np.float64)
    if not use_sw:
        return ones_tr, ones_va, ones_te, False, None
    if col not in train_df.columns or col not in val_df.columns or col not in test_df.columns:
        logger.info(
            "use_sample_weight true but column %r missing on a split; using unit weights",
            col,
        )
        return ones_tr, ones_va, ones_te, False, col

    def one(d: pd.DataFrame) -> np.ndarray:
        w = pd.to_numeric(d[col], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64)
        w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
        return w

    w_tr, w_va, w_te = one(train_df), one(val_df), one(test_df)
    return w_tr, w_va, w_te, True, col


def _predict_slearner(m: Any, X_df: pd.DataFrame) -> np.ndarray:
    X = np.asarray(X_df, dtype=np.float64)
    try:
        out = m.predict(X)
    except TypeError:
        out = m.predict(X=X)
    arr = np.asarray(out, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[1] >= 1:
        return arr[:, 0].ravel()
    return arr.ravel()


def _inner_fitted_estimator(slearner: Any) -> Any:
    for attr in ("model", "_model", "estimator"):
        o = getattr(slearner, attr, None)
        if o is not None:
            return o
    return slearner


@dataclass(frozen=True)
class PipelineDataBundle:
    df: pd.DataFrame
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    date_load_start: str
    date_load_end: str


def _progress(msg: str, *, verbose: bool) -> None:
    if verbose:
        print(f"[S pipeline] {msg}", flush=True)


def _estimator_params_dict(model: Any) -> dict[str, Any] | None:
    if model is None:
        return None
    getp = getattr(model, "get_params", None)
    if getp is None:
        return {"_type": type(model).__name__, "_note": "no get_params"}
    try:
        return dict(getp(deep=True))
    except Exception as e:
        return {"_type": type(model).__name__, "_get_params_error": str(e)}


def load_wide_and_split(cfg: dict, split_spec: SplitSpec) -> PipelineDataBundle:
    db_table = cfg.get("db_table")
    wide_path = cfg.get("wide_table_path")
    wide_ok = bool(wide_path and str(wide_path).strip())
    db_ok = bool(db_table and str(db_table).strip())
    if not wide_ok and not db_ok:
        raise SystemExit(
            "Data source not configured: set wide_table_path in YAML for parquet, or set db_table in YAML "
            "or pass --db-table for MySQL."
        )

    if isinstance(split_spec, dict):
        d0, d1 = date_range_for_split_dict(split_spec)
    else:
        d0, d1 = date_range_for_load(split_spec)
    limit = cfg.get("sample_limit")
    chunk_days = cfg.get("chunk_days", 1)

    _obsolete_fs = [
        k
        for k in ("filter_method", "filter_order", "filter_null_impute", "coarse_top_n", "fine_top_k")
        if k in cfg and cfg.get(k) is not None
    ]
    if _obsolete_fs:
        logger.warning(
            "Ignoring keys (use `python -m feature_select` instead): %s",
            ", ".join(_obsolete_fs),
        )

    if wide_ok:
        df = load_and_build_sample(
            d0,
            d1,
            db_table=None,
            wide_parquet=Path(str(wide_path).strip()),
            sample_limit=limit,
            chunk_days=chunk_days if chunk_days else None,
        )
    else:
        df = load_and_build_sample(
            d0,
            d1,
            db_table=db_table,
            wide_parquet=None,
            sample_limit=limit,
            chunk_days=chunk_days if chunk_days else None,
        )

    if isinstance(split_spec, dict):
        train_df, val_df, test_df = split_train_val_test_by_dates_from_dict(df, split_spec)
    else:
        train_df, val_df, test_df = split_train_val_test_by_dates(df, split_spec)
    logger.info("Rows train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))
    return PipelineDataBundle(
        df=df,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        date_load_start=d0,
        date_load_end=d1,
    )


def run_pipeline_training_from_splits(
    cfg: dict,
    split_spec: SplitSpec,
    config_yaml_path: Path,
    cli_overrides: dict[str, Any],
    bundle: PipelineDataBundle,
    *,
    verbose: bool = False,
) -> Path:
    rs = int(cfg.get("random_seed", 42))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(cfg.get("output_root", "output"))
    exp_dir = _root / out_root / ts
    exp_dir.mkdir(parents=True, exist_ok=True)

    co = dict(cli_overrides)
    if isinstance(split_spec, dict):
        split_art_path = exp_dir / "notebook_split_dates.json"
        split_art_path.write_text(
            json.dumps(split_spec, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        co["split_dates_source"] = "inline_dict"
        co["notebook_split_dates_path"] = str(split_art_path.resolve())
    else:
        split_art_path = Path(split_spec).resolve()
        co.setdefault("split_dates_path", str(split_art_path))

    setup_run_logging(exp_dir / "process.log")
    logger.info("Experiment directory %s", exp_dir)
    _progress(f"experiment directory: {exp_dir}", verbose=verbose)

    write_config_backup(exp_dir, cfg, co)
    extra_art = [config_yaml_path, split_art_path]
    csv_raw = cfg.get("selected_features_csv")
    if csv_raw and str(csv_raw).strip():
        cr = _resolve_under_model_build(csv_raw)
        if cr.is_file() or cr.is_dir():
            extra_art.append(cr)
    copy_artifacts(
        exp_dir,
        Path(__file__).resolve(),
        extra_art,
    )

    df = bundle.df
    train_df = bundle.train_df
    val_df = bundle.val_df
    test_df = bundle.test_df
    logger.info(
        "Training phase: rows train=%d val=%d test=%d (load range %s .. %s)",
        len(train_df),
        len(val_df),
        len(test_df),
        bundle.date_load_start,
        bundle.date_load_end,
    )
    _progress(
        f"training phase: train={len(train_df)} val={len(val_df)} test={len(test_df)} "
        f"(loaded {bundle.date_load_start} .. {bundle.date_load_end})",
        verbose=verbose,
    )

    w_tr, w_va, w_te, weights_active, sw_col = _parse_row_weights(train_df, val_df, test_df, cfg)
    (exp_dir / "sample_weight_run_meta.json").write_text(
        json.dumps(
            {
                "use_sample_weight_config": bool(cfg.get("use_sample_weight", True)),
                "sample_weight_column": sw_col,
                "weights_active": weights_active,
                "sum_w_train": float(np.sum(w_tr)),
                "sum_w_val": float(np.sum(w_va)),
                "sum_w_test": float(np.sum(w_te)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    cand = candidate_features_from_df(df)
    prot_explicit = list(cfg.get("feature_selection_protected_features") or [])
    prot_prefixes = list(cfg.get("feature_selection_protected_prefixes") or [])
    protected_resolved = resolve_protected_features(cand, prot_explicit, prot_prefixes)

    usable = set(train_df.columns) & set(val_df.columns) & set(test_df.columns)
    csv_key = cfg.get("selected_features_csv")
    csv_key_str = str(csv_key).strip() if csv_key is not None else ""

    ch_list = cfg.get("channels")
    if ch_list is None:
        ch_list = list(CHANNELS)

    feat_sel_global: list[str]
    feat_sel_per_channel: dict[str, list[str]] | None = None
    per_channel_csv_detail: dict[str, dict[str, object]] | None = None
    feature_mode: str
    resolved_feature_path: Path | None = None
    csv_path_single: Path | None = None
    csv_feats_single: list[str] = []
    n_csv_in_model_global = 0
    n_prot_in_model_global = 0

    if not csv_key_str:
        feature_mode = "all_candidates"
        feat_sel_global = _feat_list_all_candidates(cand, protected_resolved, usable)
        if not feat_sel_global:
            raise SystemExit("No candidate features left after protected merge and column intersection")
        n_prot_in_model_global = len([c for c in protected_resolved if c in feat_sel_global])
        logger.info(
            "Feature list: n=%d protected_in_model=%d source=all_candidates",
            len(feat_sel_global),
            n_prot_in_model_global,
        )
    else:
        resolved = _resolve_under_model_build(csv_key)
        if resolved.is_file():
            feature_mode = "csv_file"
            csv_path_single = resolved
            resolved_feature_path = resolved
            csv_feats_single = _load_selected_features_csv(resolved)
            merged_names = merge_csv_and_protected_features(csv_feats_single, protected_resolved)
            feat_sel_global = _merged_intersect_usable(merged_names, usable)
            if not feat_sel_global:
                raise SystemExit("No features left after CSV + protected merge and column intersection")
            n_csv_in_model_global = len([c for c in csv_feats_single if c in feat_sel_global])
            n_prot_in_model_global = len([c for c in protected_resolved if c in feat_sel_global])
            logger.info(
                "Feature list: n=%d (from CSV rows in model=%d, protected in model=%d) source=csv_file path=%s",
                len(feat_sel_global),
                n_csv_in_model_global,
                n_prot_in_model_global,
                resolved,
            )
        elif resolved.is_dir():
            feature_mode = "csv_dir_per_channel"
            resolved_feature_path = resolved
            fallback_list = _feat_list_all_candidates(cand, protected_resolved, usable)
            if not fallback_list:
                raise SystemExit("No candidate features for directory-mode fallback")
            feat_sel_per_channel = {}
            per_channel_csv_detail = {}
            for ch in ch_list:
                sub = resolved / ch / "selected_features.csv"
                if sub.is_file():
                    csv_feats_ch = _load_selected_features_csv(sub)
                    merged_ch = merge_csv_and_protected_features(csv_feats_ch, protected_resolved)
                    fl = _merged_intersect_usable(merged_ch, usable)
                    if not fl:
                        logger.warning(
                            "No usable features for channel %s from %s; using all candidates",
                            ch,
                            sub,
                        )
                        feat_sel_per_channel[ch] = list(fallback_list)
                        per_channel_csv_detail[ch] = {
                            "path": str(sub),
                            "fallback_all": True,
                        }
                    else:
                        feat_sel_per_channel[ch] = fl
                        per_channel_csv_detail[ch] = {
                            "path": str(sub),
                            "fallback_all": False,
                        }
                else:
                    logger.warning(
                        "Missing %s; using all candidate features for channel %s",
                        sub,
                        ch,
                    )
                    feat_sel_per_channel[ch] = list(fallback_list)
                    per_channel_csv_detail[ch] = {"path": None, "fallback_all": True}
            feat_sel_global = fallback_list
            logger.info(
                "Feature list: per-channel CSVs under %s (fallback = all candidates when missing)",
                resolved,
            )
        else:
            raise SystemExit(
                f"selected_features_csv must be a file, an existing directory, or empty/null: {resolved}"
            )

    w_tr_arg = w_tr if weights_active else None
    w_va_arg = w_va if weights_active else None

    test_preds_by_ch: dict[str, np.ndarray] = {}
    id_cols = [c for c in ["outcome_date", "tenant_id", "user_id"] if c in test_df.columns]
    _progress(
        f"feature lists resolved (mode={feature_mode!r}); weights_active={weights_active}; "
        f"iterating {len(ch_list)} channel(s)",
        verbose=verbose,
    )

    for channel in ch_list:
        t_col = f"T_{channel}"
        if t_col not in train_df.columns:
            logger.warning("Skip %s: no %s", channel, t_col)
            _progress(f"skip channel {channel}: missing {t_col}", verbose=verbose)
            continue
        T_tr = train_df[t_col].values
        if T_tr.sum() == 0 or T_tr.sum() == len(T_tr):
            logger.warning("Skip %s: T degenerate on train", channel)
            _progress(f"skip channel {channel}: treatment degenerate on train", verbose=verbose)
            continue

        ch_dir = exp_dir / channel
        ch_dir.mkdir(parents=True, exist_ok=True)
        _progress(f"channel {channel}: start", verbose=verbose)

        if feature_mode == "csv_dir_per_channel" and feat_sel_per_channel is not None:
            feat_sel = feat_sel_per_channel[channel]
            det = (per_channel_csv_detail or {}).get(channel, {})
            fs_meta = {
                "source": "csv_dir_per_channel",
                "run_root": str(resolved_feature_path) if resolved_feature_path else None,
                "channel_csv": det.get("path"),
                "fallback_all": bool(det.get("fallback_all")),
                "n_protected_in_model": len([c for c in protected_resolved if c in feat_sel]),
                "n_selected": len(feat_sel),
            }
        elif feature_mode == "csv_file":
            feat_sel = feat_sel_global
            fs_meta = {
                "source": "csv_file",
                "csv_path": str(csv_path_single) if csv_path_single else None,
                "n_from_csv_rows_in_model": n_csv_in_model_global,
                "n_protected_in_model": n_prot_in_model_global,
                "n_selected": len(feat_sel),
            }
        else:
            feat_sel = feat_sel_global
            fs_meta = {
                "source": "all_candidates",
                "n_protected_in_model": n_prot_in_model_global,
                "n_selected": len(feat_sel),
            }
        (ch_dir / "feature_selection_meta.json").write_text(
            json.dumps(fs_meta, indent=2), encoding="utf-8"
        )
        (ch_dir / "selected_features.txt").write_text("\n".join(feat_sel) + "\n", encoding="utf-8")
        logger.info("Channel %s feature list: %s", channel, fs_meta)

        X_tr_df, X_va_df, X_te_df, scaler, _safe = _prepare_X(train_df, val_df, test_df, feat_sel)
        Y_tr = train_df["Y"].values.astype(float)
        Y_va = val_df["Y"].values.astype(float)
        Y_te = test_df["Y"].values.astype(float)
        T_va = val_df[t_col].values
        T_te = test_df[t_col].values

        _progress(
            f"channel {channel}: Optuna S-learner (trials={int(cfg.get('optuna_n_trials', 15))})...",
            verbose=verbose,
        )
        try:
            lead_model, fam, best_params, leaderboard = select_best_slearner(
                X_tr_df,
                T_tr,
                Y_tr,
                w_tr_arg,
                X_va_df,
                T_va,
                Y_va,
                w_va_arg,
                list(cfg.get("learner_families", ["lgbm", "xgb", "rf"])),
                int(cfg.get("optuna_n_trials", 15)),
                rs,
                cfg.get("optuna_timeout"),
            )
            with open(ch_dir / "lead_leaderboard.json", "w", encoding="utf-8") as f:
                json.dump(leaderboard, f, indent=2, default=str)
            with open(ch_dir / "best_model_params.json", "w", encoding="utf-8") as f:
                json.dump(best_params, f, indent=2, default=str)
            logger.info("Channel %s S-learner: %s params=%s", channel, fam, best_params)
            _progress(f"channel {channel}: S-learner OK (family={fam})", verbose=verbose)
        except Exception as e:
            logger.exception("S-learner training failed %s: %s", channel, e)
            _progress(f"channel {channel}: FAILED S-learner ({e!r})", verbose=verbose)
            continue

        if cfg.get("refit_on_train_val"):
            _progress(f"channel {channel}: refit S-learner on train+val", verbose=verbose)
            lead_model = refit_slearner_train_val_from_params(
                fam,
                best_params,
                X_tr_df,
                X_va_df,
                T_tr,
                T_va,
                Y_tr,
                Y_va,
                w_tr_arg,
                w_va_arg,
                rs,
            )

        joblib.dump(
            {
                "model": lead_model,
                "scaler": scaler,
                "feature_names": feat_sel,
                "sanitized_names": list(X_tr_df.columns),
                "family": fam,
                "use_sample_weight": weights_active,
                "sample_weight_column": sw_col if weights_active else None,
            },
            ch_dir / "lead_model.pkl",
        )
        with open(ch_dir / "lead_model_params.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "family": fam,
                    "tuning_summary": best_params,
                    "use_sample_weight": weights_active,
                    "sample_weight_column": sw_col if weights_active else None,
                    "fitted_estimator_get_params": _estimator_params_dict(
                        _inner_fitted_estimator(lead_model)
                    ),
                    "refit_on_train_val": bool(cfg.get("refit_on_train_val")),
                },
                f,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

        _progress(f"channel {channel}: test predictions + metrics + placebo...", verbose=verbose)
        pred_te = _predict_slearner(lead_model, X_te_df)
        test_preds_by_ch[channel] = pred_te

        id_df = test_df[id_cols].copy() if id_cols else pd.DataFrame(index=test_df.index)
        out_te = id_df.copy()
        out_te["Y"] = Y_te
        out_te[t_col] = T_te
        out_te["pred_ite"] = pred_te
        if weights_active and sw_col:
            out_te["sample_weight"] = w_te
        out_te.to_csv(ch_dir / "test_predictions.csv", index=False)

        evaluate_channel_test(pred_te, T_te, Y_te, ch_dir, channel)

        eval_on = str(cfg.get("eval_placebo_on", "val"))
        if eval_on == "val":
            pred_va = _predict_slearner(lead_model, X_va_df)
            pb = placebo_shuffle_t(
                pred_va,
                T_va,
                Y_va,
                int(cfg.get("placebo_shuffle_repeats", 20)),
                rs,
            )
            with open(ch_dir / "placebo_report.json", "w", encoding="utf-8") as f:
                json.dump(pb, f, indent=2)

        _progress(f"channel {channel}: done", verbose=verbose)

    _progress("channel loop finished; aggregating total ITE (if any predictions)...", verbose=verbose)
    if test_preds_by_ch:
        n_te = len(test_df)
        ite_total = np.zeros(n_te, dtype=float)
        for c, arr in test_preds_by_ch.items():
            if len(arr) != n_te:
                logger.warning("Length mismatch test preds %s", c)
                continue
            ite_total += arr
        total_df = test_df[id_cols].copy() if id_cols else pd.DataFrame(index=test_df.index)
        total_df["Y"] = test_df["Y"].values
        for c, arr in test_preds_by_ch.items():
            total_df[f"pred_ite_{c}"] = arr
        total_df["ITE_total"] = ite_total
        if weights_active and sw_col:
            total_df["sample_weight"] = w_te
        total_df.to_csv(exp_dir / "test_predictions_total.csv", index=False)

        y_te = test_df["Y"].values.astype(float)
        n_bins = int(cfg.get("n_rank_bins", 10))
        cdf = conversion_by_bins(ite_total, y_te, n_bins=n_bins)
        cdf.to_csv(exp_dir / "conversion_by_rank_total.csv", index=False)

        qini_t, qdet = conversion_qini_vs_random(
            ite_total,
            y_te,
            n_random=int(cfg.get("placebo_shuffle_repeats", 20)),
            seed=rs,
        )
        r2_d = decile_r2_score(ite_total, y_te, n_bins=n_bins)
        try:
            c = np.corrcoef(ite_total, y_te)[0, 1]
            corr_sq = float(c**2) if c == c else float("nan")
        except Exception:
            corr_sq = float("nan")

        frac, conv = cumulative_conversion_curve(ite_total, y_te)
        save_conversion_rank_plot(
            frac,
            conv,
            exp_dir / "plots" / "conversion_rank_curve_total.png",
            "Total ITE ranking vs cumulative conversion rate",
        )

        placebo_total = {"note": "shuffle ITE_total scores"}
        rng = np.random.default_rng(rs + 999)
        shuff_qinis = []
        for _ in range(int(cfg.get("placebo_shuffle_repeats", 20))):
            s = ite_total.copy()
            rng.shuffle(s)
            q, _ = conversion_qini_vs_random(s, y_te, n_random=5, seed=rs)
            shuff_qinis.append(q)
        placebo_total["conversion_qini_shuffled_mean"] = float(np.nanmean(shuff_qinis))
        placebo_total["conversion_qini_shuffled_std"] = float(np.nanstd(shuff_qinis))
        placebo_total["conversion_qini_observed"] = qini_t

        with open(exp_dir / "placebo_report_total.json", "w", encoding="utf-8") as f:
            json.dump(placebo_total, f, indent=2, default=str)

        with open(exp_dir / "ite_quality_total.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "conversion_qini": qini_t,
                    "conversion_qini_detail": qdet,
                    "decile_r2_mean_ite_vs_mean_y": r2_d,
                    "sample_corr_sq": corr_sq,
                    "n_bins": n_bins,
                },
                f,
                indent=2,
                default=str,
            )
        logger.info(
            "Total metrics: conversion_qini=%.4f decile_r2=%.4f",
            qini_t if qini_t == qini_t else float("nan"),
            r2_d if r2_d == r2_d else float("nan"),
        )

    logger.info("Done. Artifacts under %s", exp_dir)
    _progress(f"all done -> {exp_dir}", verbose=verbose)
    return exp_dir


def run_pipeline_core(
    cfg: dict,
    split_dates_path: Path,
    config_yaml_path: Path,
    cli_overrides: dict[str, Any],
) -> Path:
    bundle = load_wide_and_split(cfg, split_dates_path)
    return run_pipeline_training_from_splits(
        cfg, split_dates_path, config_yaml_path, cli_overrides, bundle, verbose=False
    )


def run_pipeline_notebook(
    *,
    config_yaml: str | Path,
    split_dates_path: str | Path | None = None,
    split_dates: dict[str, Any] | None = None,
    db_config: dict[str, Any] | None = None,
    db_config_json_path: str | Path | None = None,
    output_root: str | None = None,
    selected_features_csv: str | Path | None = None,
    db_table: str | None = None,
    pipeline_overrides: dict[str, Any] | None = None,
    progress_prints: bool = False,
) -> Path:
    from utils.db_config import read_db_config_json, set_db_config_inline

    if (split_dates_path is None) == (split_dates is None):
        raise ValueError("Pass exactly one of split_dates_path or split_dates")
    if db_config_json_path is not None and db_config is not None:
        raise ValueError("Pass at most one of db_config and db_config_json_path")
    if db_config_json_path is not None:
        db_config = read_db_config_json(db_config_json_path)
    if db_config is not None:
        set_db_config_inline(db_config)
    config_yaml_path = Path(config_yaml).resolve()
    split_spec: SplitSpec = split_dates if split_dates is not None else Path(split_dates_path).resolve()

    cfg = _load_yaml(config_yaml_path)
    if pipeline_overrides is not None:
        cfg = merge_pipeline_config(cfg, pipeline_overrides)
    if output_root is not None:
        cfg["output_root"] = output_root
    if selected_features_csv is not None:
        cfg["selected_features_csv"] = str(selected_features_csv)
    if db_table is not None:
        cfg["db_table"] = db_table

    cli_overrides: dict[str, Any] = {}
    if split_dates_path is not None:
        cli_overrides["split_dates_path"] = str(Path(split_dates_path).resolve())
    if split_dates is not None:
        cli_overrides["split_dates_source"] = "inline_dict"
    if output_root is not None:
        cli_overrides["output_root"] = output_root
    if selected_features_csv is not None:
        cli_overrides["selected_features_csv"] = str(selected_features_csv)
    if db_table is not None:
        cli_overrides["db_table"] = db_table
    if db_config_json_path is not None:
        cli_overrides["db_config_json_path"] = str(Path(db_config_json_path).resolve())

    use_inline_split = split_dates is not None
    if progress_prints or use_inline_split:
        bundle = load_wide_and_split(cfg, split_spec)
        return run_pipeline_training_from_splits(
            cfg,
            split_spec,
            config_yaml_path,
            cli_overrides,
            bundle,
            verbose=progress_prints,
        )
    return run_pipeline_core(cfg, Path(split_dates_path).resolve(), config_yaml_path, cli_overrides)


def main() -> None:
    p = argparse.ArgumentParser(description="S-Learner pipeline (CausalML + Optuna)")
    p.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config" / "pipeline_s_learner.yaml",
        help="YAML config path",
    )
    p.add_argument("--split_dates_path", type=Path, required=True, help="JSON with train/val/test date lists")
    p.add_argument(
        "--db-table",
        dest="db_table",
        type=str,
        default=None,
        help="Override YAML db_table when loading from MySQL (ignored when wide_table_path is set)",
    )
    p.add_argument("--output_root", type=str, default=None, help="Override output root (default from config)")
    p.add_argument(
        "--selected_features_csv",
        type=Path,
        default=None,
        help="Override YAML selected_features_csv",
    )
    p.add_argument(
        "--db-config-json",
        dest="db_config_json",
        type=Path,
        default=None,
        help="MySQL connection JSON path",
    )
    args = p.parse_args()

    if args.db_config_json is not None:
        from utils.db_config import read_db_config_json, set_db_config_inline

        set_db_config_inline(read_db_config_json(args.db_config_json))

    cfg = _load_yaml(args.config)
    if args.output_root:
        cfg["output_root"] = args.output_root
    if args.selected_features_csv is not None:
        cfg["selected_features_csv"] = str(args.selected_features_csv)
    if args.db_table is not None:
        cfg["db_table"] = args.db_table

    cli_overrides = {
        k: getattr(args, k)
        for k in ["split_dates_path", "output_root", "selected_features_csv", "db_table"]
        if getattr(args, k, None) is not None
    }
    if args.db_config_json is not None:
        cli_overrides["db_config_json"] = str(Path(args.db_config_json).resolve())
    run_pipeline_core(cfg, args.split_dates_path, args.config, cli_overrides)


if __name__ == "__main__":
    main()
