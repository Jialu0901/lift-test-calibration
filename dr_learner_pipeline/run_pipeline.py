"""
Six-stage DR-Learner experiment pipeline entrypoint.

Run from model_build directory:
  python -m dr_learner_pipeline.run_pipeline \\
    --config dr_learner_pipeline/config/pipeline_grid.yaml \\
    --split_dates_path path/to/split_dates.json \\
    --db-config-json path/to/db_config.json \\
    --db-table your_schema.your_wide_table

Notebook (same repo on sys.path): global ``DB_CONFIG`` dict → ``run_pipeline_notebook(..., db_config=DB_CONFIG)``,
or ``db_config_json_path=`` (same effect as CLI ``--db-config-json``). See ``run_pipeline_notebook`` docstring.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

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

from dr_learner_pipeline.logging_utils import setup_run_logging
from dr_learner_pipeline.metrics_total import (
    conversion_by_bins,
    conversion_qini_vs_random,
    cumulative_conversion_curve,
    decile_r2_score,
)
from dr_learner_pipeline.repro import copy_artifacts, write_config_backup
from dr_learner_pipeline.splits import date_range_for_load, split_train_val_test_by_dates
from dr_learner_pipeline.stages.base_nuisance import select_base_models
from dr_learner_pipeline.stages.cross_fit import (
    cross_fit_pseudo_tau,
    dr_pseudo_on_split,
    refit_nuisance_full_train,
)
from dr_learner_pipeline.stages.eval import (
    placebo_shuffle_t,
    save_conversion_rank_plot,
    evaluate_channel_test,
)
from dr_learner_pipeline.stages.lead_optuna import select_best_lead

logger = logging.getLogger(__name__)


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
    """Build X matrices: fillna(0), float, sanitize names. No StandardScaler (wide table is pre-scaled upstream)."""
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


def run_pipeline_core(
    cfg: dict,
    split_dates_path: Path,
    config_yaml_path: Path,
    cli_overrides: dict[str, Any],
) -> Path:
    db_table = cfg.get("db_table")
    wide_path = cfg.get("wide_table_path")
    wide_ok = bool(wide_path and str(wide_path).strip())
    db_ok = bool(db_table and str(db_table).strip())
    if not wide_ok and not db_ok:
        raise SystemExit(
            "Data source not configured: set wide_table_path in YAML for parquet, or set db_table in YAML "
            "or pass --db-table for MySQL."
        )

    rs = int(cfg.get("random_seed", 42))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(cfg.get("output_root", "output"))
    exp_dir = _root / out_root / ts
    exp_dir.mkdir(parents=True, exist_ok=True)

    setup_run_logging(exp_dir / "process.log")
    logger.info("Experiment directory %s", exp_dir)

    write_config_backup(exp_dir, cfg, cli_overrides)
    extra_art = [config_yaml_path, split_dates_path]
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

    d0, d1 = date_range_for_load(split_dates_path)
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

    train_df, val_df, test_df = split_train_val_test_by_dates(df, split_dates_path)
    logger.info("Rows train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))

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

    test_preds_by_ch: dict[str, np.ndarray] = {}
    id_cols = [c for c in ["outcome_date", "tenant_id", "user_id"] if c in test_df.columns]

    for channel in ch_list:
        t_col = f"T_{channel}"
        if t_col not in train_df.columns:
            logger.warning("Skip %s: no %s", channel, t_col)
            continue
        T_tr = train_df[t_col].values
        if T_tr.sum() == 0 or T_tr.sum() == len(T_tr):
            logger.warning("Skip %s: T degenerate on train", channel)
            continue

        ch_dir = exp_dir / channel
        ch_dir.mkdir(parents=True, exist_ok=True)

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

        try:
            base = select_base_models(
                X_tr_df,
                T_tr,
                Y_tr,
                X_va_df,
                T_va,
                Y_va,
                list(cfg.get("propensity_candidates", ["lr", "xgb"])),
                list(cfg.get("outcome_candidates", ["lgbm", "rf"])),
                rs,
            )
            with open(ch_dir / "base_model_scores.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "propensity_scores": base["propensity_scores"],
                        "outcome_scores": base["outcome_scores"],
                        "best_propensity_kind": base["best_propensity_kind"],
                        "best_outcome_kind": base["best_outcome_kind"],
                    },
                    f,
                    indent=2,
                    default=str,
                )
        except Exception as e:
            logger.exception("Base model selection failed %s: %s", channel, e)
            continue

        pk = base["best_propensity_kind"]
        okind = base["best_outcome_kind"]
        pseudo_tr = cross_fit_pseudo_tau(
            X_tr_df,
            T_tr,
            Y_tr,
            prop_kind=pk,
            out_kind=okind,
            n_cf_folds=int(cfg.get("n_cf_folds", 5)),
            random_state=rs,
        )
        pseudo_tr = np.nan_to_num(pseudo_tr, nan=0.0, posinf=0.0, neginf=0.0)

        prop_m, m1, m0, p_ok = refit_nuisance_full_train(
            X_tr_df, T_tr, Y_tr, pk, okind, rs
        )
        pseudo_va = dr_pseudo_on_split(prop_m, m1, m0, p_ok, X_va_df, T_va, Y_va)
        pseudo_va = np.nan_to_num(pseudo_va, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            lead_model, fam, best_params, leaderboard = select_best_lead(
                X_tr_df,
                pseudo_tr,
                X_va_df,
                pseudo_va,
                list(cfg.get("lead_families", ["lgbm", "xgb", "rf"])),
                int(cfg.get("optuna_n_trials", 15)),
                rs,
                cfg.get("optuna_timeout"),
            )
            with open(ch_dir / "lead_leaderboard.json", "w", encoding="utf-8") as f:
                json.dump(leaderboard, f, indent=2, default=str)
            with open(ch_dir / "best_model_params.json", "w", encoding="utf-8") as f:
                json.dump(best_params, f, indent=2, default=str)
            joblib.dump(
                {"model": lead_model, "scaler": scaler, "feature_names": feat_sel, "sanitized_names": list(X_tr_df.columns)},
                ch_dir / "lead_model.pkl",
            )
            logger.info("Channel %s lead: %s params=%s", channel, fam, best_params)
        except Exception as e:
            logger.exception("Lead training failed %s: %s", channel, e)
            continue

        if cfg.get("refit_lead_on_train_val"):
            X_tv = pd.concat([X_tr_df, X_va_df], axis=0)
            y_tv = np.concatenate([pseudo_tr, pseudo_va])
            lead_model.fit(X_tv, y_tv)

        pred_te = lead_model.predict(X_te_df)
        test_preds_by_ch[channel] = pred_te

        id_df = test_df[id_cols].copy() if id_cols else pd.DataFrame(index=test_df.index)
        out_te = id_df.copy()
        out_te["Y"] = Y_te
        out_te[t_col] = T_te
        out_te["pred_ite"] = pred_te
        out_te.to_csv(ch_dir / "test_predictions.csv", index=False)

        evaluate_channel_test(pred_te, T_te, Y_te, ch_dir, channel)

        eval_on = str(cfg.get("eval_placebo_on", "val"))
        if eval_on == "val":
            pred_va = lead_model.predict(X_va_df)
            pb = placebo_shuffle_t(
                pred_va,
                T_va,
                Y_va,
                int(cfg.get("placebo_shuffle_repeats", 20)),
                rs,
            )
            with open(ch_dir / "placebo_report.json", "w", encoding="utf-8") as f:
                json.dump(pb, f, indent=2)

    # ----- Total ITE on test -----
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
    return exp_dir


def run_pipeline_notebook(
    *,
    config_yaml: str | Path,
    split_dates_path: str | Path,
    db_config: dict[str, Any] | None = None,
    db_config_json_path: str | Path | None = None,
    output_root: str | None = None,
    selected_features_csv: str | Path | None = None,
    db_table: str | None = None,
) -> Path:
    """
    Run the full pipeline in-process (e.g. Jupyter). Logs go to console and output dir process.log.

    Database config (at most one):

    - ``db_config_json_path``: load via ``read_db_config_json(path)`` (same as CLI ``--db-config-json``).
    - ``db_config``: dict from ``read_db_config_json`` or a literal (e.g. notebook global ``DB_CONFIG``).
    - Both ``None``: use ``get_db_config()`` (default JSON / ``LIFT_DB_CONFIG_JSON``).
    """
    from utils.db_config import read_db_config_json, set_db_config_inline

    if db_config_json_path is not None and db_config is not None:
        raise ValueError("Pass at most one of db_config and db_config_json_path")
    if db_config_json_path is not None:
        db_config = read_db_config_json(db_config_json_path)
    if db_config is not None:
        set_db_config_inline(db_config)
    config_yaml_path = Path(config_yaml).resolve()
    split_p = Path(split_dates_path).resolve()
    cfg = _load_yaml(config_yaml_path)
    if output_root is not None:
        cfg["output_root"] = output_root
    if selected_features_csv is not None:
        cfg["selected_features_csv"] = str(selected_features_csv)
    if db_table is not None:
        cfg["db_table"] = db_table

    cli_overrides: dict[str, Any] = {"split_dates_path": split_p}
    if output_root is not None:
        cli_overrides["output_root"] = output_root
    if selected_features_csv is not None:
        cli_overrides["selected_features_csv"] = str(selected_features_csv)
    if db_table is not None:
        cli_overrides["db_table"] = db_table
    if db_config_json_path is not None:
        cli_overrides["db_config_json_path"] = str(Path(db_config_json_path).resolve())

    return run_pipeline_core(cfg, split_p, config_yaml_path, cli_overrides)


def main() -> None:
    p = argparse.ArgumentParser(description="Six-stage DR-Learner pipeline")
    p.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config" / "pipeline_grid.yaml",
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
        help="Override YAML selected_features_csv: omit/null = all numeric candidates; file = union CSV; dir = <dir>/<channel>/selected_features.csv",
    )
    p.add_argument(
        "--db-config-json",
        dest="db_config_json",
        type=Path,
        default=None,
        help="MySQL connection JSON path (host, user, password, database, port). Applied before pipeline run.",
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
