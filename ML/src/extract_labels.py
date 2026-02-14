#!/usr/bin/env python3
"""
src/extract_labels.py

Extract land-cover composition labels (proportions) per base-grid cell from ESA WorldCover
(categorical raster) for a given year.

Inputs
------
- WorldCover GeoTIFF (categorical codes)
- Base grid GeoPackage produced by src/make_grid.py (cell_id, geometry)

Outputs
-------
- Parquet table with one row per grid cell:
    cell_id,
    built_prop, veg_prop, water_prop, other_prop,
    built_count, veg_count, water_count, other_count,
    total_count,
    year

Important
---------
- This script is "assignment compliant" for tabular ML:
  labels are proportions per cell (targets for regression models).

WorldCover code mapping
-----------------------
ESA WorldCover (v100/v200) uses these common classes (codes):
10 Tree cover
20 Shrubland
30 Grassland
40 Cropland
50 Built-up
60 Bare / sparse vegetation
70 Snow and ice
80 Permanent water bodies
90 Herbaceous wetland
95 Mangroves
100 Moss and lichen

We group them into 4 targets:
- built_prop  : code 50
- water_prop  : code 80
- veg_prop    : 10,20,30,40,90,95,100  (vegetation + wetlands/mangroves)
- other_prop  : everything else (60,70, ...)

If you want a different grouping, edit the sets below (BUILT_CODES, WATER_CODES, VEG_CODES).

Example
-------
python src/extract_labels.py \
  --raster data/raw/worldcover/wc_2020.tif \
  --grid data/processed/grid/grid_100m.gpkg \
  --output data/processed/labels/labels_2020.parquet \
  --year 2020
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Dict, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
from rasterio.features import geometry_mask
from shapely.geometry import mapping, box
from tqdm import tqdm


# === WorldCover code groupings (edit if needed) ===
BUILT_CODES = {50}
WATER_CODES = {80}
VEG_CODES = {10, 20, 30, 40, 90, 95, 100}
# everything else -> OTHER


def _ensure_dir_for_file(filepath: str) -> None:
    d = os.path.dirname(os.path.abspath(filepath))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _infer_year_from_filename(path: str) -> Optional[int]:
    name = os.path.basename(path)
    m = re.search(r"(20[1-3][0-9])", name)
    if m:
        y = int(m.group(1))
        if 2015 <= y <= 2035:
            return y
    return None


def _read_masked_labels(
    ds: rasterio.io.DatasetReader,
    geom_geojson: dict,
    all_touched: bool,
    pad: int = 0,
) -> np.ndarray:
    """
    Read the label raster within geometry window and return a flattened array of valid pixels
    inside the polygon.
    """
    window = rasterio.features.geometry_window(ds, [geom_geojson], pad_x=pad, pad_y=pad, north_up=True)
    if window.width <= 0 or window.height <= 0:
        return np.array([], dtype=np.int32)

    transform = ds.window_transform(window)

    mask_outside = geometry_mask(
        geometries=[geom_geojson],
        out_shape=(int(window.height), int(window.width)),
        transform=transform,
        invert=False,  # True outside polygon
        all_touched=all_touched,
    )

    arr = ds.read(1, window=window)
    nodata = ds.nodata
    if nodata is not None:
        arr = np.where(arr == nodata, 0, arr)

    # keep only pixels inside polygon and nonzero
    inside = ~mask_outside
    vals = arr[inside]
    vals = vals[vals != 0]
    return vals.astype(np.int32, copy=False)


def _counts_and_props(vals: np.ndarray) -> Dict[str, float]:
    """
    Convert a 1D array of class codes into grouped counts + proportions.
    """
    total = int(vals.size)
    if total == 0:
        return {
            "built_count": 0,
            "veg_count": 0,
            "water_count": 0,
            "other_count": 0,
            "total_count": 0,
            "built_prop": float("nan"),
            "veg_prop": float("nan"),
            "water_prop": float("nan"),
            "other_prop": float("nan"),
        }

    built = int(np.isin(vals, list(BUILT_CODES)).sum())
    water = int(np.isin(vals, list(WATER_CODES)).sum())
    veg = int(np.isin(vals, list(VEG_CODES)).sum())

    # everything else
    other = total - (built + water + veg)
    if other < 0:
        # Defensive guard (shouldn't happen)
        other = 0

    built_prop = built / total
    veg_prop = veg / total
    water_prop = water / total
    other_prop = other / total

    return {
        "built_count": built,
        "veg_count": veg,
        "water_count": water,
        "other_count": other,
        "total_count": total,
        "built_prop": float(built_prop),
        "veg_prop": float(veg_prop),
        "water_prop": float(water_prop),
        "other_prop": float(other_prop),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract WorldCover proportion labels per grid cell.")
    p.add_argument("--raster", type=str, required=True, help="Path to WorldCover GeoTIFF (categorical).")
    p.add_argument("--grid", type=str, required=True, help="Path to base grid GeoPackage/GeoJSON (must include cell_id).")
    p.add_argument(
        "--grid-layer",
        type=str,
        default=None,
        help="Optional layer name for GeoPackage grids. If omitted, geopandas reads the default layer.",
    )
    p.add_argument("--output", type=str, required=True, help="Output parquet path.")
    p.add_argument("--year", type=int, default=None, help="Year for this raster (if omitted, inferred from filename).")
    p.add_argument("--all-touched", action="store_true", help="Count pixels touched by polygon edges.")
    p.add_argument("--chunk-size", type=int, default=8000, help="Chunk size for processing (default: 8000).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.raster):
        raise FileNotFoundError(f"Raster not found: {args.raster}")
    if not os.path.exists(args.grid):
        raise FileNotFoundError(f"Grid not found: {args.grid}")

    year = args.year if args.year is not None else _infer_year_from_filename(args.raster)
    if year is None:
        raise ValueError("Could not infer --year from filename. Please pass --year YYYY explicitly.")

    grid = gpd.read_file(args.grid, layer=args.grid_layer) if args.grid_layer else gpd.read_file(args.grid)
    if grid.empty:
        raise ValueError("Grid file is empty.")
    if "cell_id" not in grid.columns:
        raise ValueError("Grid must contain a 'cell_id' column.")
    if grid.crs is None:
        raise ValueError("Grid has no CRS.")
    grid = grid[["cell_id", "geometry"]].copy()
    grid["cell_id"] = grid["cell_id"].astype(np.int64)

    with rasterio.open(args.raster) as ds:
        if ds.crs is None:
            raise ValueError("WorldCover raster has no CRS.")

        # Reproject grid to raster CRS
        grid_r = grid.to_crs(ds.crs) if str(grid.crs) != str(ds.crs) else grid

        # Filter by raster extent
        rb = ds.bounds
        raster_bbox = box(rb.left, rb.bottom, rb.right, rb.top)
        grid_r = grid_r.loc[grid_r.geometry.intersects(raster_bbox)].copy()
        if grid_r.empty:
            raise ValueError("No grid cells intersect WorldCover raster extent. Check CRS and inputs.")

        n = len(grid_r)
        chunk_size = max(1, int(args.chunk_size))

        records = []
        pbar = tqdm(total=n, desc=f"Extracting labels {year}", unit="cell")

        for start in range(0, n, chunk_size):
            chunk = grid_r.iloc[start : start + chunk_size]
            for cid, geom in zip(chunk["cell_id"].to_numpy(), chunk.geometry.to_list()):
                geom_json = mapping(geom)
                try:
                    vals = _read_masked_labels(ds, geom_json, all_touched=bool(args.all_touched), pad=0)
                    stats = _counts_and_props(vals)
                    rec = {"cell_id": int(cid), **stats, "year": int(year)}
                except Exception:
                    rec = {
                        "cell_id": int(cid),
                        "built_count": 0,
                        "veg_count": 0,
                        "water_count": 0,
                        "other_count": 0,
                        "total_count": 0,
                        "built_prop": float("nan"),
                        "veg_prop": float("nan"),
                        "water_prop": float("nan"),
                        "other_prop": float("nan"),
                        "year": int(year),
                    }
                records.append(rec)
                pbar.update(1)

        pbar.close()

    df = pd.DataFrame.from_records(records)

    # Deduplicate safely if needed
    if df["cell_id"].duplicated().any():
        # For proportions, average; for counts, sum; for year, take first
        df = (
            df.groupby("cell_id", as_index=False)
            .agg(
                {
                    "built_count": "sum",
                    "veg_count": "sum",
                    "water_count": "sum",
                    "other_count": "sum",
                    "total_count": "sum",
                    "built_prop": "mean",
                    "veg_prop": "mean",
                    "water_prop": "mean",
                    "other_prop": "mean",
                    "year": "first",
                }
            )
        )

    # Recompute proportions from summed counts if desired (more consistent)
    mask_total = df["total_count"] > 0
    df.loc[mask_total, "built_prop"] = df.loc[mask_total, "built_count"] / df.loc[mask_total, "total_count"]
    df.loc[mask_total, "veg_prop"] = df.loc[mask_total, "veg_count"] / df.loc[mask_total, "total_count"]
    df.loc[mask_total, "water_prop"] = df.loc[mask_total, "water_count"] / df.loc[mask_total, "total_count"]
    df.loc[mask_total, "other_prop"] = df.loc[mask_total, "other_count"] / df.loc[mask_total, "total_count"]

    df = df.sort_values("cell_id").reset_index(drop=True)

    _ensure_dir_for_file(args.output)
    df.to_parquet(args.output, index=False)

    # Quick sanity: show mean sum of props (should be ~1 for non-empty)
    prop_sum = (df["built_prop"] + df["veg_prop"] + df["water_prop"] + df["other_prop"])
    mean_prop_sum = float(np.nanmean(prop_sum.to_numpy())) if len(df) else float("nan")

    print("✅ Label extraction complete")
    print(f"Raster: {args.raster}")
    print(f"Grid:   {args.grid}")
    print(f"Year:   {year}")
    print(f"Rows:   {len(df):,}")
    print(f"Mean (built+veg+water+other): {mean_prop_sum:.4f} (should be ~1.0 on valid cells)")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
