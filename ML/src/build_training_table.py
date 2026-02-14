#!/usr/bin/env python3
"""
src/build_training_table.py

Merge per-year Sentinel features with per-year WorldCover labels to build a single
training table for tabular ML.

Inputs
------
- Base grid GeoPackage (provides cell_id + block_id for spatial CV)
- Features parquet files (e.g., features_2020.parquet, features_2021.parquet)
- Label parquet files (e.g., labels_2020.parquet, labels_2021.parquet)

Outputs
-------
- A single Parquet table:
    data/processed/tables/train_table.parquet

Columns include (at minimum)
---------------------------
- cell_id
- block_id
- year
- Features: B2_med, B3_med, B4_med, B8_med, ndvi_med, ndvi_std, ndwi_med, ndwi_std, valid_frac
- Targets: built_prop, veg_prop, water_prop, other_prop

Also writes a small CSV summary to help you debug class balance / missingness.

Usage (Windows one-liner)
-------------------------
python src/build_training_table.py --grid data/processed/grid/grid_100m.gpkg --features-dir data/processed/features --labels-dir data/processed/labels --years 2020 2021 --output data/processed/tables/train_table.parquet
"""

from __future__ import annotations

import argparse
import os
from typing import List

import geopandas as gpd
import numpy as np
import pandas as pd


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


def _ensure_dir_for_file(filepath: str) -> None:
    d = os.path.dirname(os.path.abspath(filepath))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _load_grid_block_ids(grid_path: str, grid_layer: str | None) -> pd.DataFrame:
    gdf = gpd.read_file(grid_path, layer=grid_layer) if grid_layer else gpd.read_file(grid_path)
    if gdf.empty:
        raise ValueError("Grid file is empty.")
    for col in ["cell_id", "block_id"]:
        if col not in gdf.columns:
            raise ValueError(f"Grid must include '{col}'. Re-run make_grid.py to ensure block_id exists.")
    df = gdf[["cell_id", "block_id"]].copy()
    df["cell_id"] = df["cell_id"].astype(np.int64)
    df["block_id"] = df["block_id"].astype(np.int64)
    return df


def _load_features(features_path: str, year: int) -> pd.DataFrame:
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file not found for year={year}: {features_path}")
    df = pd.read_parquet(features_path)
    if "cell_id" not in df.columns:
        raise ValueError(f"Features missing 'cell_id': {features_path}")
    if "year" not in df.columns:
        df["year"] = year
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Features file missing columns {missing}: {features_path}")
    df["cell_id"] = df["cell_id"].astype(np.int64)
    df["year"] = df["year"].astype(np.int32)
    return df[["cell_id", "year"] + FEATURE_COLS].copy()


def _load_labels(labels_path: str, year: int) -> pd.DataFrame:
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found for year={year}: {labels_path}")
    df = pd.read_parquet(labels_path)
    if "cell_id" not in df.columns:
        raise ValueError(f"Labels missing 'cell_id': {labels_path}")
    if "year" not in df.columns:
        df["year"] = year
    missing = [c for c in TARGET_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Labels file missing columns {missing}: {labels_path}")
    df["cell_id"] = df["cell_id"].astype(np.int64)
    df["year"] = df["year"].astype(np.int32)
    return df[["cell_id", "year"] + TARGET_COLS].copy()


def _clip_props(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clip proportion targets to [0,1] for safety, and renormalize to sum=1 where possible.
    """
    out = df.copy()
    for c in TARGET_COLS:
        out[c] = out[c].clip(lower=0.0, upper=1.0)

    s = out[TARGET_COLS].sum(axis=1)
    mask = s > 0
    out.loc[mask, TARGET_COLS] = out.loc[mask, TARGET_COLS].div(s[mask], axis=0)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build training table by merging features and labels.")
    p.add_argument("--grid", type=str, required=True, help="Path to base grid gpkg (must include block_id).")
    p.add_argument("--grid-layer", type=str, default=None, help="Optional grid layer name for gpkg.")
    p.add_argument("--features-dir", type=str, required=True, help="Directory containing features_YYYY.parquet.")
    p.add_argument("--labels-dir", type=str, required=True, help="Directory containing labels_YYYY.parquet.")
    p.add_argument(
        "--years",
        type=int,
        nargs="+",
        required=True,
        help="Label years to include (e.g., 2020 2021).",
    )
    p.add_argument(
        "--features-pattern",
        type=str,
        default="features_{year}.parquet",
        help="Filename pattern in features-dir (default: features_{year}.parquet).",
    )
    p.add_argument(
        "--labels-pattern",
        type=str,
        default="labels_{year}.parquet",
        help="Filename pattern in labels-dir (default: labels_{year}.parquet).",
    )
    p.add_argument(
        "--output",
        type=str,
        default="data/processed/tables/train_table.parquet",
        help="Output parquet path.",
    )
    p.add_argument(
        "--min-valid-frac",
        type=float,
        default=0.0,
        help="Optional filter: drop rows with valid_frac < threshold (default: 0 = keep all).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    years: List[int] = [int(y) for y in args.years]
    years = sorted(years)

    grid_df = _load_grid_block_ids(args.grid, args.grid_layer)

    tables = []
    for y in years:
        feat_path = os.path.join(args.features_dir, args.features_pattern.format(year=y))
        lab_path = os.path.join(args.labels_dir, args.labels_pattern.format(year=y))

        feats = _load_features(feat_path, y)
        labs = _load_labels(lab_path, y)

        # Merge on cell_id + year
        df = feats.merge(labs, on=["cell_id", "year"], how="inner")
        # Add block_id for spatial CV
        df = df.merge(grid_df, on="cell_id", how="left")

        if df["block_id"].isna().any():
            missing = int(df["block_id"].isna().sum())
            raise ValueError(
                f"{missing} rows missing block_id after merge. "
                "This usually means your grid does not contain those cell_ids."
            )

        df["block_id"] = df["block_id"].astype(np.int64)

        # Optional filter by valid_frac
        if float(args.min_valid_frac) > 0:
            before = len(df)
            df = df.loc[df["valid_frac"] >= float(args.min_valid_frac)].copy()
            after = len(df)
            print(f"Year {y}: filtered by valid_frac >= {args.min_valid_frac} ({before:,} -> {after:,})")

        # Clean/clip targets
        df = _clip_props(df)

        # Drop rows with NaNs in key columns
        needed = ["cell_id", "year", "block_id"] + FEATURE_COLS + TARGET_COLS
        before = len(df)
        df = df.dropna(subset=needed).copy()
        after = len(df)
        if after < before:
            print(f"Year {y}: dropped NaN rows ({before:,} -> {after:,})")

        tables.append(df)

    if not tables:
        raise ValueError("No data loaded. Check input years and file paths.")

    train = pd.concat(tables, ignore_index=True)

    # Final sanity: ensure targets sum ~ 1
    s = train[TARGET_COLS].sum(axis=1)
    mean_sum = float(s.mean())
    min_sum = float(s.min())
    max_sum = float(s.max())

    # Save
    _ensure_dir_for_file(args.output)
    train.to_parquet(args.output, index=False)

    # Write a summary CSV for quick inspection
    summary_path = os.path.splitext(args.output)[0] + "_summary.csv"
    summary = (
        train.groupby("year")[TARGET_COLS + ["valid_frac"]]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    summary.to_csv(summary_path, index=False)

    print("✅ Training table built successfully")
    print(f"Years: {years}")
    print(f"Rows:  {len(train):,}")
    print(f"Output: {args.output}")
    print(f"Summary: {summary_path}")
    print(f"Target sum stats: mean={mean_sum:.6f}, min={min_sum:.6f}, max={max_sum:.6f}")


if __name__ == "__main__":
    main()
