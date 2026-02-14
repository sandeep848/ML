#!/usr/bin/env python3
"""
src/train_models.py

Train and evaluate tabular models for land-cover composition prediction:
- Baseline: Ridge regression (interpretable)
- High-performance: LightGBM regressor + Optuna tuning
- Evaluation:
    1) Spatial holdout via GroupKFold(block_id)
    2) Temporal holdout via year swap (train 2020 test 2021, and vice versa)
- Uncertainty (optional but recommended):
    - Ensemble of LightGBM models via different random seeds -> prediction mean/std

Inputs
------
- Training table parquet produced by src/build_training_table.py
  Must include:
    cell_id, block_id, year,
    features: B2_med, B3_med, B4_med, B8_med, ndvi_med, ndvi_std, ndwi_med, ndwi_std, valid_frac
    targets : built_prop, veg_prop, water_prop, other_prop

Outputs
-------
- models/baseline/ridge_{target}.pkl
- models/lightgbm/lgbm_{target}.txt
- models/lightgbm/lgbm_{target}_features.json
- models/optuna/study_{target}.db   (sqlite)
- models/metrics_spatial_cv.csv
- models/metrics_temporal_swap.csv
- (optional) models/lightgbm/ensemble_{target}/... (ensemble members)

Usage (Windows one-liner)
-------------------------
python src/train_models.py --train data/processed/tables/train_table.parquet --outdir models --spatial-folds 5 --optuna-trials 50 --ensemble-size 7
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from optuna.samplers import TPESampler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = [
    "B2_med",
    "B3_med",
    "B4_med",
    "B8_med",
    "ndvi_med",
    "ndvi_std",
    "ndwi_med",
    "ndwi_std",
    "valid_frac",
]
TARGET_COLS = ["built_prop", "veg_prop", "water_prop", "other_prop"]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": _rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _change_metrics(y_true_a: np.ndarray, y_true_b: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> Dict[str, float]:
    """
    Change metric: Δ = B - A
    """
    d_true = y_true_b - y_true_a
    d_pred = y_pred_b - y_pred_a
    out = {
        "delta_mae": float(mean_absolute_error(d_true, d_pred)),
        "delta_rmse": _rmse(d_true, d_pred),
    }
    return out


def _load_train(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Training table not found: {path}")
    df = pd.read_parquet(path)
    needed = ["cell_id", "block_id", "year"] + FEATURE_COLS + TARGET_COLS
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Training table missing columns: {missing}")
    df = df.copy()
    df["cell_id"] = df["cell_id"].astype(np.int64)
    df["block_id"] = df["block_id"].astype(np.int64)
    df["year"] = df["year"].astype(np.int32)
    return df


def _ridge_pipeline(alpha: float) -> Pipeline:
    """
    Interpretable baseline with standardization.
    """
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ]
    )
    pre = ColumnTransformer([("num", numeric, FEATURE_COLS)], remainder="drop")
    model = Ridge(alpha=float(alpha), random_state=42)
    pipe = Pipeline([("pre", pre), ("model", model)])
    return pipe


def _lgbm_model(params: Dict) -> LGBMRegressor:
    return LGBMRegressor(**params)


def _default_lgbm_params(seed: int = 42) -> Dict:
    # Strong defaults; Optuna will tune key ones
    return {
        "n_estimators": 5000,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 40,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "random_state": seed,
        "n_jobs": -1,
        "objective": "regression",
        "boosting_type": "gbdt",
        "verbosity": -1,
    }


def _make_optuna_study(db_path: str, study_name: str) -> optuna.Study:
    storage = f"sqlite:///{db_path}"
    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
    )
    return study


@dataclass
class Split:
    train_idx: np.ndarray
    valid_idx: np.ndarray


def _spatial_splits(df: pd.DataFrame, n_splits: int) -> List[Split]:
    gkf = GroupKFold(n_splits=n_splits)
    X = df[FEATURE_COLS].to_numpy()
    groups = df["block_id"].to_numpy()
    splits: List[Split] = []
    for tr, va in gkf.split(X, groups=groups):
        splits.append(Split(train_idx=tr, valid_idx=va))
    return splits


def _fit_lgbm_with_early_stop(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    params: Dict,
) -> LGBMRegressor:
    model = _lgbm_model(params)
    # LightGBM scikit wrapper supports early stopping via callbacks
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="l2",
        callbacks=[],
    )
    return model


def _optuna_objective_factory(df: pd.DataFrame, target: str, n_splits: int) -> callable:
    splits = _spatial_splits(df, n_splits=n_splits)

    X_all = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_all = df[target].to_numpy(dtype=np.float32)

    def objective(trial: optuna.Trial) -> float:
        params = _default_lgbm_params(seed=42)

        # Tune key parameters
        params.update(
            {
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 16, 256),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
                "n_estimators": 5000,
            }
        )

        fold_rmses = []
        for sp in splits:
            X_tr, y_tr = X_all[sp.train_idx], y_all[sp.train_idx]
            X_va, y_va = X_all[sp.valid_idx], y_all[sp.valid_idx]

            # Simple median impute for LGBM
            med = np.nanmedian(X_tr, axis=0)
            X_tr_i = np.where(np.isnan(X_tr), med, X_tr)
            X_va_i = np.where(np.isnan(X_va), med, X_va)

            model = LGBMRegressor(**params)
            model.fit(
                X_tr_i,
                y_tr,
                eval_set=[(X_va_i, y_va)],
                eval_metric="l2",
                callbacks=[],
            )
            pred = model.predict(X_va_i)
            fold_rmses.append(_rmse(y_va, pred))

        return float(np.mean(fold_rmses))

    return objective


def _evaluate_spatial_cv_ridge(df: pd.DataFrame, target: str, n_splits: int, alpha: float) -> pd.DataFrame:
    splits = _spatial_splits(df, n_splits=n_splits)
    X = df[FEATURE_COLS]
    y = df[target].to_numpy()

    rows = []
    for k, sp in enumerate(splits):
        pipe = _ridge_pipeline(alpha=alpha)
        pipe.fit(X.iloc[sp.train_idx], y[sp.train_idx])
        pred = pipe.predict(X.iloc[sp.valid_idx])
        m = _metrics(y[sp.valid_idx], pred)
        rows.append({"model": "ridge", "target": target, "fold": k, **m})
    return pd.DataFrame(rows)


def _evaluate_spatial_cv_lgbm(df: pd.DataFrame, target: str, n_splits: int, best_params: Dict) -> pd.DataFrame:
    splits = _spatial_splits(df, n_splits=n_splits)
    X_all = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_all = df[target].to_numpy(dtype=np.float32)

    rows = []
    for k, sp in enumerate(splits):
        X_tr, y_tr = X_all[sp.train_idx], y_all[sp.train_idx]
        X_va, y_va = X_all[sp.valid_idx], y_all[sp.valid_idx]

        med = np.nanmedian(X_tr, axis=0)
        X_tr_i = np.where(np.isnan(X_tr), med, X_tr)
        X_va_i = np.where(np.isnan(X_va), med, X_va)

        model = LGBMRegressor(**best_params)
        model.fit(
            X_tr_i,
            y_tr,
            eval_set=[(X_va_i, y_va)],
            eval_metric="l2",
            callbacks=[],
        )
        pred = model.predict(X_va_i)
        m = _metrics(y_va, pred)
        rows.append({"model": "lgbm", "target": target, "fold": k, **m})
    return pd.DataFrame(rows)


def _fit_final_lgbm(df: pd.DataFrame, target: str, best_params: Dict) -> Tuple[LGBMRegressor, np.ndarray]:
    """
    Fit on all rows. Returns model + imputation median vector.
    """
    X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y = df[target].to_numpy(dtype=np.float32)
    med = np.nanmedian(X, axis=0)
    X_i = np.where(np.isnan(X), med, X)

    model = LGBMRegressor(**best_params)
    model.fit(X_i, y)
    return model, med


def _save_lgbm(model: LGBMRegressor, path_txt: str) -> None:
    _ensure_dir(os.path.dirname(path_txt))
    model.booster_.save_model(path_txt)


def _temporal_swap_eval(
    df: pd.DataFrame,
    target: str,
    ridge_alpha: float,
    lgbm_params: Dict,
) -> Dict[str, float]:
    years = sorted(df["year"].unique().tolist())
    if len(years) != 2:
        raise ValueError(f"Temporal swap evaluation requires exactly 2 label years, got {years}")

    y0, y1 = years[0], years[1]
    df0 = df[df["year"] == y0].copy()
    df1 = df[df["year"] == y1].copy()

    # Ridge
    ridge = _ridge_pipeline(alpha=ridge_alpha)
    ridge.fit(df0[FEATURE_COLS], df0[target].to_numpy())
    pred1_ridge = ridge.predict(df1[FEATURE_COLS])
    m_ridge_0to1 = _metrics(df1[target].to_numpy(), pred1_ridge)

    ridge2 = _ridge_pipeline(alpha=ridge_alpha)
    ridge2.fit(df1[FEATURE_COLS], df1[target].to_numpy())
    pred0_ridge = ridge2.predict(df0[FEATURE_COLS])
    m_ridge_1to0 = _metrics(df0[target].to_numpy(), pred0_ridge)

    # LGBM with simple impute
    X0 = df0[FEATURE_COLS].to_numpy(dtype=np.float32)
    X1 = df1[FEATURE_COLS].to_numpy(dtype=np.float32)
    y0t = df0[target].to_numpy(dtype=np.float32)
    y1t = df1[target].to_numpy(dtype=np.float32)

    med0 = np.nanmedian(X0, axis=0)
    X0i = np.where(np.isnan(X0), med0, X0)
    X1i = np.where(np.isnan(X1), med0, X1)

    lgbm0 = LGBMRegressor(**lgbm_params)
    lgbm0.fit(X0i, y0t)
    pred1_lgbm = lgbm0.predict(X1i)
    m_lgbm_0to1 = _metrics(y1t, pred1_lgbm)

    med1 = np.nanmedian(X1, axis=0)
    X1i2 = np.where(np.isnan(X1), med1, X1)
    X0i2 = np.where(np.isnan(X0), med1, X0)

    lgbm1 = LGBMRegressor(**lgbm_params)
    lgbm1.fit(X1i2, y1t)
    pred0_lgbm = lgbm1.predict(X0i2)
    m_lgbm_1to0 = _metrics(y0t, pred0_lgbm)

    # Change metrics (Δ = year1 - year0) for each model
    delta_ridge = _change_metrics(
        y_true_a=df0[target].to_numpy(),
        y_true_b=df1[target].to_numpy(),
        y_pred_a=pred0_ridge,   # ridge trained on df1 predicts df0
        y_pred_b=pred1_ridge,   # ridge trained on df0 predicts df1
    )
    delta_lgbm = _change_metrics(
        y_true_a=y0t,
        y_true_b=y1t,
        y_pred_a=pred0_lgbm,
        y_pred_b=pred1_lgbm,
    )

    # Flatten into one dict
    out = {
        "target": target,
        "year0": y0,
        "year1": y1,
        "ridge_mae_0to1": m_ridge_0to1["mae"],
        "ridge_rmse_0to1": m_ridge_0to1["rmse"],
        "ridge_r2_0to1": m_ridge_0to1["r2"],
        "ridge_mae_1to0": m_ridge_1to0["mae"],
        "ridge_rmse_1to0": m_ridge_1to0["rmse"],
        "ridge_r2_1to0": m_ridge_1to0["r2"],
        "lgbm_mae_0to1": m_lgbm_0to1["mae"],
        "lgbm_rmse_0to1": m_lgbm_0to1["rmse"],
        "lgbm_r2_0to1": m_lgbm_0to1["r2"],
        "lgbm_mae_1to0": m_lgbm_1to0["mae"],
        "lgbm_rmse_1to0": m_lgbm_1to0["rmse"],
        "lgbm_r2_1to0": m_lgbm_1to0["r2"],
        "ridge_delta_mae": delta_ridge["delta_mae"],
        "ridge_delta_rmse": delta_ridge["delta_rmse"],
        "lgbm_delta_mae": delta_lgbm["delta_mae"],
        "lgbm_delta_rmse": delta_lgbm["delta_rmse"],
    }
    return out


def _train_ensemble(
    df: pd.DataFrame,
    target: str,
    best_params: Dict,
    ensemble_size: int,
    outdir: str,
) -> None:
    """
    Train multiple LGBM models with different seeds and save them.
    Also saves the imputation median vector used (from full data) once.
    """
    if ensemble_size <= 1:
        return

    ens_dir = os.path.join(outdir, "lightgbm", f"ensemble_{target}")
    _ensure_dir(ens_dir)

    X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y = df[target].to_numpy(dtype=np.float32)
    med = np.nanmedian(X, axis=0)
    X_i = np.where(np.isnan(X), med, X)

    # Save median vector once
    with open(os.path.join(ens_dir, "imputer_median.json"), "w", encoding="utf-8") as f:
        json.dump({"feature_cols": FEATURE_COLS, "median": med.tolist()}, f, indent=2)

    for i in range(ensemble_size):
        seed = 1000 + i
        params = dict(best_params)
        params["random_state"] = seed

        model = LGBMRegressor(**params)
        model.fit(X_i, y)

        model_path = os.path.join(ens_dir, f"member_{i:02d}.txt")
        model.booster_.save_model(model_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Ridge + LightGBM models with spatial/temporal evaluation.")
    p.add_argument("--train", type=str, required=True, help="Path to train_table.parquet")
    p.add_argument("--outdir", type=str, default="models", help="Output models directory (default: models)")
    p.add_argument("--spatial-folds", type=int, default=5, help="GroupKFold splits (default: 5)")
    p.add_argument("--optuna-trials", type=int, default=50, help="Optuna trials per target (default: 50)")
    p.add_argument("--ridge-alpha", type=float, default=1.0, help="Ridge alpha (default: 1.0)")
    p.add_argument("--ensemble-size", type=int, default=1, help="LGBM ensemble size for uncertainty (default: 1 = off)")
    p.add_argument(
        "--min-valid-frac",
        type=float,
        default=0.0,
        help="Optional filter: keep only rows with valid_frac >= threshold (default: 0.0)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_dir(args.outdir)
    _ensure_dir(os.path.join(args.outdir, "baseline"))
    _ensure_dir(os.path.join(args.outdir, "lightgbm"))
    _ensure_dir(os.path.join(args.outdir, "optuna"))

    df = _load_train(args.train)

    # Optional valid_frac filter
    if float(args.min_valid_frac) > 0:
        before = len(df)
        df = df[df["valid_frac"] >= float(args.min_valid_frac)].copy()
        after = len(df)
        print(f"Filtered by valid_frac >= {args.min_valid_frac}: {before:,} -> {after:,}")

    # Ensure exactly 2 label years for temporal swap evaluation (WorldCover 2020 & 2021)
    years = sorted(df["year"].unique().tolist())
    if len(years) != 2:
        raise ValueError(
            f"This training script expects exactly 2 label years for temporal swap eval (e.g., 2020 and 2021). "
            f"Found years={years}. If you have more label years, we can generalize temporal evaluation."
        )

    spatial_rows = []
    temporal_rows = []

    for target in TARGET_COLS:
        print(f"\n==============================")
        print(f"Target: {target}")
        print(f"==============================")

        # --- Ridge baseline: spatial CV evaluation ---
        ridge_cv = _evaluate_spatial_cv_ridge(df, target=target, n_splits=int(args.spatial_folds), alpha=float(args.ridge_alpha))
        spatial_rows.append(ridge_cv)

        # Fit final ridge on all data + save
        ridge_pipe = _ridge_pipeline(alpha=float(args.ridge_alpha))
        ridge_pipe.fit(df[FEATURE_COLS], df[target].to_numpy())
        ridge_path = os.path.join(args.outdir, "baseline", f"ridge_{target}.pkl")
        joblib.dump(ridge_pipe, ridge_path)
        print(f"Saved ridge -> {ridge_path}")

        # --- LightGBM: Optuna tuning ---
        study_db = os.path.join(args.outdir, "optuna", f"study_{target}.db")
        study = _make_optuna_study(study_db, study_name=f"lgbm_{target}")
        objective = _optuna_objective_factory(df, target=target, n_splits=int(args.spatial_folds))

        # Run remaining trials if study already has some
        n_existing = len(study.trials)
        n_total = int(args.optuna_trials)
        n_to_run = max(0, n_total - n_existing)
        if n_to_run > 0:
            print(f"Optuna: running {n_to_run} new trials (existing={n_existing}, target_total={n_total})...")
            study.optimize(objective, n_trials=n_to_run, show_progress_bar=True)
        else:
            print(f"Optuna: study already has {n_existing} trials (>= {n_total}). Reusing best params.")

        best = dict(study.best_params)
        best_params = _default_lgbm_params(seed=42)
        best_params.update(best)
        # keep strong n_estimators; no early stop callback here, but we use many trees + low LR
        best_params["n_estimators"] = 5000
        best_params["verbosity"] = -1

        # --- LightGBM: spatial CV eval with best params ---
        lgbm_cv = _evaluate_spatial_cv_lgbm(df, target=target, n_splits=int(args.spatial_folds), best_params=best_params)
        spatial_rows.append(lgbm_cv)

        # Fit final LGBM on all data + save model + imputer median
        lgbm_model, med = _fit_final_lgbm(df, target=target, best_params=best_params)
        lgbm_path = os.path.join(args.outdir, "lightgbm", f"lgbm_{target}.txt")
        _save_lgbm(lgbm_model, lgbm_path)
        print(f"Saved lgbm -> {lgbm_path}")

        # Save metadata (feature order + median imputer + params)
        meta = {
            "target": target,
            "feature_cols": FEATURE_COLS,
            "imputer_median": med.tolist(),
            "params": best_params,
        }
        meta_path = os.path.join(args.outdir, "lightgbm", f"lgbm_{target}_features.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved lgbm metadata -> {meta_path}")

        # Optional ensemble for uncertainty
        if int(args.ensemble_size) > 1:
            print(f"Training ensemble ({args.ensemble_size} members) for uncertainty...")
            _train_ensemble(df, target=target, best_params=best_params, ensemble_size=int(args.ensemble_size), outdir=args.outdir)
            print(f"Saved ensemble -> {os.path.join(args.outdir, 'lightgbm', f'ensemble_{target}') }")

        # Temporal swap evaluation (ridge + lgbm + delta metrics)
        temp = _temporal_swap_eval(df, target=target, ridge_alpha=float(args.ridge_alpha), lgbm_params=best_params)
        temporal_rows.append(pd.DataFrame([temp]))

    # Save spatial CV results
    spatial_df = pd.concat(spatial_rows, ignore_index=True)
    spatial_out = os.path.join(args.outdir, "metrics_spatial_cv.csv")
    spatial_df.to_csv(spatial_out, index=False)

    # Aggregate spatial CV summary
    spatial_summary = (
        spatial_df.groupby(["model", "target"])
        .agg(mae_mean=("mae", "mean"), mae_std=("mae", "std"), rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"), r2_mean=("r2", "mean"))
        .reset_index()
        .sort_values(["model", "target"])
    )
    spatial_summary_out = os.path.join(args.outdir, "metrics_spatial_cv_summary.csv")
    spatial_summary.to_csv(spatial_summary_out, index=False)

    # Save temporal swap results
    temporal_df = pd.concat(temporal_rows, ignore_index=True)
    temporal_out = os.path.join(args.outdir, "metrics_temporal_swap.csv")
    temporal_df.to_csv(temporal_out, index=False)

    print("\n✅ Training complete")
    print(f"Spatial CV metrics -> {spatial_out}")
    print(f"Spatial CV summary -> {spatial_summary_out}")
    print(f"Temporal swap metrics -> {temporal_out}")
    print(f"Models saved under -> {args.outdir}")


if __name__ == "__main__":
    main()
