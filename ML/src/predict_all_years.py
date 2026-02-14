#!/usr/bin/env python3
"""
src/predict_all_years.py

Use trained models to generate per-grid-cell land-cover composition predictions
for multiple Sentinel years, and save to parquet for Streamlit.

Supports:
- LightGBM single model per target (saved as .txt + metadata json with median imputer)
- Optional ensemble uncertainty (if you trained ensemble_{target}/member_*.txt)

Inputs
------
- Base grid (for block_id optional join, but predictions key is cell_id)
- Features parquet per year: data/processed/features/features_YYYY.parquet
- Trained model files under models/ (from src/train_models.py)

Outputs
-------
- Parquet per year: data/processed/predictions/pred_YYYY.parquet with columns:
    cell_id, year,
    built_pred, veg_pred, water_pred, other_pred,
    (optional uncertainty if ensemble exists)
    built_std, veg_std, water_std, other_std

Notes
-----
- Predictions are clipped to [0,1] and renormalized to sum=1 per row.
- If ensemble exists, std is computed across ensemble member predictions.

Usage (Windows one-liner)
-------------------------
python src/predict_all_years.py --features-dir data/processed/features --years 2019 2020 2021 2022 2023 --model-dir models --output-dir data/processed/predictions
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from lightgbm import Booster


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
TARGETS = ["built_prop", "veg_prop", "water_prop", "other_prop"]
PRED_COLS = ["built_pred", "veg_pred", "water_pred", "other_pred"]
STD_COLS = ["built_std", "veg_std", "water_std", "other_std"]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _ensure_dir_for_file(filepath: str) -> None:
    d = os.path.dirname(os.path.abspath(filepath))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _load_features(features_path: str, year: int) -> pd.DataFrame:
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file not found: {features_path}")
    df = pd.read_parquet(features_path)
    if "cell_id" not in df.columns:
        raise ValueError(f"Features missing 'cell_id': {features_path}")
    for c in FEATURE_COLS:
        if c not in df.columns:
            raise ValueError(f"Features missing '{c}' in {features_path}")
    df = df.copy()
    df["cell_id"] = df["cell_id"].astype(np.int64)
    df["year"] = int(year)
    return df[["cell_id", "year"] + FEATURE_COLS].copy()


@dataclass
class ModelBundle:
    target: str
    booster: Booster
    feature_cols: List[str]
    imputer_median: np.ndarray


def _load_single_lgbm(model_dir: str, target: str) -> ModelBundle:
    """
    Load:
      models/lightgbm/lgbm_{target}.txt
      models/lightgbm/lgbm_{target}_features.json
    """
    txt_path = os.path.join(model_dir, "lightgbm", f"lgbm_{target}.txt")
    meta_path = os.path.join(model_dir, "lightgbm", f"lgbm_{target}_features.json")

    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Missing model file: {txt_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing model metadata: {meta_path}")

    booster = Booster(model_file=txt_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    feature_cols = meta.get("feature_cols")
    median = meta.get("imputer_median")

    if not isinstance(feature_cols, list) or len(feature_cols) == 0:
        raise ValueError(f"Invalid feature_cols in {meta_path}")
    if not isinstance(median, list) or len(median) != len(feature_cols):
        raise ValueError(f"Invalid imputer_median in {meta_path}")

    return ModelBundle(
        target=target,
        booster=booster,
        feature_cols=[str(x) for x in feature_cols],
        imputer_median=np.array(median, dtype=np.float32),
    )


def _load_ensemble_members(model_dir: str, target: str) -> Tuple[List[Booster], Optional[np.ndarray], Optional[List[str]]]:
    """
    Load ensemble members if present:
      models/lightgbm/ensemble_{target}/member_*.txt
      models/lightgbm/ensemble_{target}/imputer_median.json

    Returns (members, median_vector, feature_cols). If no members, returns ([], None, None).
    """
    ens_dir = os.path.join(model_dir, "lightgbm", f"ensemble_{target}")
    if not os.path.isdir(ens_dir):
        return [], None, None

    member_paths = sorted(glob.glob(os.path.join(ens_dir, "member_*.txt")))
    if not member_paths:
        return [], None, None

    median_path = os.path.join(ens_dir, "imputer_median.json")
    if not os.path.exists(median_path):
        raise FileNotFoundError(f"Ensemble found but missing imputer_median.json: {median_path}")

    with open(median_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_cols = meta.get("feature_cols")
    median = meta.get("median")

    if not isinstance(feature_cols, list) or len(feature_cols) == 0:
        raise ValueError(f"Invalid feature_cols in {median_path}")
    if not isinstance(median, list) or len(median) != len(feature_cols):
        raise ValueError(f"Invalid median vector in {median_path}")

    members = [Booster(model_file=p) for p in member_paths]
    return members, np.array(median, dtype=np.float32), [str(x) for x in feature_cols]


def _impute(X: np.ndarray, median: np.ndarray) -> np.ndarray:
    # median shape: (n_features,)
    return np.where(np.isnan(X), median[None, :], X).astype(np.float32, copy=False)


def _clip_and_renorm(preds: np.ndarray) -> np.ndarray:
    """
    preds shape: (n,4) for built,veg,water,other
    Clip to [0,1], then renormalize rows to sum=1 where sum>0.
    """
    p = np.clip(preds, 0.0, 1.0)
    s = p.sum(axis=1, keepdims=True)
    mask = s.squeeze() > 0
    p[mask] = p[mask] / s[mask]
    return p


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict composition for multiple years using trained models.")
    p.add_argument("--features-dir", type=str, required=True, help="Directory containing features_YYYY.parquet files.")
    p.add_argument("--years", type=int, nargs="+", required=True, help="Years to predict (e.g., 2019 2020 2021 2022 2023).")
    p.add_argument("--model-dir", type=str, default="models", help="Model directory from training (default: models).")
    p.add_argument("--output-dir", type=str, default="data/processed/predictions", help="Output directory for pred_YYYY.parquet")
    p.add_argument("--features-pattern", type=str, default="features_{year}.parquet", help="Pattern for feature files.")
    p.add_argument("--include-uncertainty", action="store_true", help="If set, compute std using ensemble models if available.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_dir(args.output_dir)

    # Load single models (always required)
    bundles = {
        "built": _load_single_lgbm(args.model_dir, "built_prop"),
        "veg": _load_single_lgbm(args.model_dir, "veg_prop"),
        "water": _load_single_lgbm(args.model_dir, "water_prop"),
        "other": _load_single_lgbm(args.model_dir, "other_prop"),
    }

    # Optionally load ensembles
    ensembles = {}
    if bool(args.include_uncertainty):
        for key, tgt in [("built", "built_prop"), ("veg", "veg_prop"), ("water", "water_prop"), ("other", "other_prop")]:
            members, med, cols = _load_ensemble_members(args.model_dir, tgt)
            ensembles[key] = (members, med, cols)

    for year in args.years:
        feat_path = os.path.join(args.features_dir, args.features_pattern.format(year=year))
        feats = _load_features(feat_path, year=year)

        # Use the feature order from metadata (should match FEATURE_COLS)
        # We'll align columns explicitly, then impute using each target's median vector
        X_base = feats[FEATURE_COLS].to_numpy(dtype=np.float32)

        preds_4 = []
        stds_4 = []

        for key, bundle in [("built", bundles["built"]), ("veg", bundles["veg"]), ("water", bundles["water"]), ("other", bundles["other"])]:
            # Align columns for safety (metadata may specify same list)
            # Here we assume FEATURE_COLS already correct; if not, we reorder from feats.
            cols = bundle.feature_cols
            X = feats[cols].to_numpy(dtype=np.float32)
            X_i = _impute(X, bundle.imputer_median)

            y_pred = bundle.booster.predict(X_i)
            preds_4.append(y_pred.astype(np.float32))

            if bool(args.include_uncertainty):
                members, ens_med, ens_cols = ensembles.get(key, ([], None, None))
                if members and ens_med is not None and ens_cols is not None:
                    X_e = feats[ens_cols].to_numpy(dtype=np.float32)
                    X_ei = _impute(X_e, ens_med)
                    member_preds = np.stack([m.predict(X_ei).astype(np.float32) for m in members], axis=0)
                    std = member_preds.std(axis=0).astype(np.float32)
                else:
                    std = np.full(shape=(len(feats),), fill_value=np.nan, dtype=np.float32)
                stds_4.append(std)

        preds = np.stack(preds_4, axis=1)  # (n,4)
        preds = _clip_and_renorm(preds)

        out = pd.DataFrame(
            {
                "cell_id": feats["cell_id"].to_numpy(dtype=np.int64),
                "year": feats["year"].to_numpy(dtype=np.int32),
                "built_pred": preds[:, 0],
                "veg_pred": preds[:, 1],
                "water_pred": preds[:, 2],
                "other_pred": preds[:, 3],
            }
        )

        if bool(args.include_uncertainty):
            stds = np.stack(stds_4, axis=1)  # (n,4)
            # Optional: clip negative/NaN not needed; std is nonnegative
            out["built_std"] = stds[:, 0]
            out["veg_std"] = stds[:, 1]
            out["water_std"] = stds[:, 2]
            out["other_std"] = stds[:, 3]

        out = out.sort_values("cell_id").reset_index(drop=True)

        out_path = os.path.join(args.output_dir, f"pred_{year}.parquet")
        _ensure_dir_for_file(out_path)
        out.to_parquet(out_path, index=False)

        print(f"✅ Saved predictions for {year} -> {out_path} (rows={len(out):,})")

    print("✅ All predictions complete")


if __name__ == "__main__":
    main()
