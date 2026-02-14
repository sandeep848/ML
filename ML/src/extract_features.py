#!/usr/bin/env python3
"""
src/extract_features.py

Extract tabular Sentinel-2 features per base-grid cell (NO CNNs).

Inputs
------
- Sentinel-2 yearly composite GeoTIFF (your exported 10m image)
- Base grid GeoPackage produced by src/make_grid.py (cell_id, geometry)

Outputs
-------
- Parquet table with one row per grid cell:
    cell_id,
    B2_med, B3_med, B4_med, B8_med,
    ndvi_med, ndvi_std,
    ndwi_med, ndwi_std,
    valid_frac,
    year
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
from rasterio.features import geometry_mask
from rasterio.windows import Window
from shapely.geometry import mapping, box
from tqdm import tqdm


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


def _band_map_from_descriptions(ds: rasterio.io.DatasetReader) -> Dict[str, int]:
    band_map: Dict[str, int] = {}
    descs = ds.descriptions
    if not descs:
        return band_map

    for i, d in enumerate(descs, start=1):
        if not d:
            continue
        dd = d.strip().upper()
        dd = dd.replace("SENTINEL2_", "").replace("SENTINEL_2_", "")
        if dd == "B2" or dd.endswith("_B2") or dd.endswith("B2"):
            band_map["B2"] = i
        elif dd == "B3" or dd.endswith("_B3") or dd.endswith("B3"):
            band_map["B3"] = i
        elif dd == "B4" or dd.endswith("_B4") or dd.endswith("B4"):
            band_map["B4"] = i
        elif dd == "B8" or dd.endswith("_B8") or dd.endswith("B8"):
            band_map["B8"] = i
    return band_map


def _resolve_band_indices(ds: rasterio.io.DatasetReader) -> Dict[str, int]:
    band_map = _band_map_from_descriptions(ds)
    needed = ["B2", "B3", "B4", "B8"]
    if all(k in band_map for k in needed):
        return band_map

    if ds.count < 4:
        raise ValueError(
            f"Raster has {ds.count} bands, but need at least 4 (B2,B3,B4,B8). "
            "Either export those bands, or ensure band descriptions exist."
        )

    # Fallback to default band order
    return {"B2": 1, "B3": 2, "B4": 3, "B8": 4}


def _read_masked_bands(
    ds: rasterio.io.DatasetReader,
    geom_geojson: dict,
    band_indices: Dict[str, int],
    all_touched: bool,
    pad: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read B2,B3,B4,B8 for minimal window covering geometry and mask outside polygon.
    Invalid pixels are set to np.nan.
    """
    try:
        window: Window = rasterio.features.geometry_window(ds, [geom_geojson], pad_x=pad, pad_y=pad, north_up=True)
    except Exception as e:
        raise ValueError(f"geometry_window failed (outside raster or invalid geom): {e}")

    if window.width <= 0 or window.height <= 0:
        raise ValueError("Empty window for geometry (outside raster extent).")

    transform = ds.window_transform(window)

    mask_outside = geometry_mask(
        geometries=[geom_geojson],
        out_shape=(int(window.height), int(window.width)),
        transform=transform,
        invert=False,  # True outside polygon
        all_touched=all_touched,
    )

    nodata = ds.nodata

    def read_band(bidx: int) -> np.ndarray:
        arr = ds.read(bidx, window=window).astype(np.float32)
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
        arr = np.where(mask_outside, np.nan, arr)
        return arr

    b2 = read_band(band_indices["B2"])
    b3 = read_band(band_indices["B3"])
    b4 = read_band(band_indices["B4"])
    b8 = read_band(band_indices["B8"])
    return b2, b3, b4, b8


def _safe_nanmedian(x: np.ndarray) -> float:
    if x.size == 0 or np.all(np.isnan(x)):
        return float("nan")
    return float(np.nanmedian(x))


def _safe_nanstd(x: np.ndarray) -> float:
    if x.size == 0 or np.all(np.isnan(x)):
        return float("nan")
    return float(np.nanstd(x))


def _compute_features_for_cell(b2: np.ndarray, b3: np.ndarray, b4: np.ndarray, b8: np.ndarray) -> Dict[str, float]:
    valid = (~np.isnan(b2)) & (~np.isnan(b3)) & (~np.isnan(b4)) & (~np.isnan(b8))
    total = valid.size
    valid_count = int(valid.sum())
    valid_frac = float(valid_count / total) if total > 0 else 0.0

    if valid_count == 0:
        return {
            "B2_med": float("nan"),
            "B3_med": float("nan"),
            "B4_med": float("nan"),
            "B8_med": float("nan"),
            "ndvi_med": float("nan"),
            "ndvi_std": float("nan"),
            "ndwi_med": float("nan"),
            "ndwi_std": float("nan"),
            "valid_frac": 0.0,
        }

    b2v = b2[valid]
    b3v = b3[valid]
    b4v = b4[valid]
    b8v = b8[valid]

    eps = np.float32(1e-6)
    ndvi = (b8v - b4v) / (b8v + b4v + eps)
    ndwi = (b3v - b8v) / (b3v + b8v + eps)

    return {
        "B2_med": _safe_nanmedian(b2v),
        "B3_med": _safe_nanmedian(b3v),
        "B4_med": _safe_nanmedian(b4v),
        "B8_med": _safe_nanmedian(b8v),
        "ndvi_med": _safe_nanmedian(ndvi),
        "ndvi_std": _safe_nanstd(ndvi),
        "ndwi_med": _safe_nanmedian(ndwi),
        "ndwi_std": _safe_nanstd(ndwi),
        "valid_frac": valid_frac,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract tabular Sentinel-2 features per grid cell.")
    p.add_argument("--raster", type=str, required=True, help="Path to Sentinel-2 yearly composite GeoTIFF.")
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
    p.add_argument(
        "--min-valid-frac",
        type=float,
        default=0.0,
        help="If >0, cells below this valid_frac will keep valid_frac but features set to NaN.",
    )
    p.add_argument("--chunk-size", type=int, default=5000, help="Chunk size for processing (default: 5000).")
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

    # Read grid
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
            raise ValueError("Raster has no CRS.")
        band_indices = _resolve_band_indices(ds)

        # Reproject grid to raster CRS
        grid_r = grid.to_crs(ds.crs) if str(grid.crs) != str(ds.crs) else grid

        # Filter by raster extent to avoid geometry_window failures
        rb = ds.bounds
        raster_bbox = box(rb.left, rb.bottom, rb.right, rb.top)
        grid_r = grid_r.loc[grid_r.geometry.intersects(raster_bbox)].copy()

        if grid_r.empty:
            raise ValueError("No grid cells intersect the raster extent. Check CRS and inputs.")

        # Build records in a safe, index-free loop
        records: List[Dict[str, float]] = []
        chunk_size = max(1, int(args.chunk_size))
        n = len(grid_r)

        pbar = tqdm(total=n, desc=f"Extracting features {year}", unit="cell")

        for start in range(0, n, chunk_size):
            chunk = grid_r.iloc[start : start + chunk_size]
            for cid, geom in zip(chunk["cell_id"].to_numpy(), chunk.geometry.to_list()):
                geom_json = mapping(geom)

                try:
                    b2, b3, b4, b8 = _read_masked_bands(
                        ds=ds,
                        geom_geojson=geom_json,
                        band_indices=band_indices,
                        all_touched=bool(args.all_touched),
                        pad=0,
                    )
                    feats = _compute_features_for_cell(b2, b3, b4, b8)

                    if args.min_valid_frac > 0 and feats["valid_frac"] < float(args.min_valid_frac):
                        feats = {
                            "B2_med": float("nan"),
                            "B3_med": float("nan"),
                            "B4_med": float("nan"),
                            "B8_med": float("nan"),
                            "ndvi_med": float("nan"),
                            "ndvi_std": float("nan"),
                            "ndwi_med": float("nan"),
                            "ndwi_std": float("nan"),
                            "valid_frac": float(feats["valid_frac"]),
                        }

                    rec = {"cell_id": int(cid), **feats, "year": int(year)}
                except Exception:
                    rec = {
                        "cell_id": int(cid),
                        "B2_med": float("nan"),
                        "B3_med": float("nan"),
                        "B4_med": float("nan"),
                        "B8_med": float("nan"),
                        "ndvi_med": float("nan"),
                        "ndvi_std": float("nan"),
                        "ndwi_med": float("nan"),
                        "ndwi_std": float("nan"),
                        "valid_frac": 0.0,
                        "year": int(year),
                    }

                records.append(rec)
                pbar.update(1)

        pbar.close()

    df = pd.DataFrame.from_records(records)

    # Deduplicate safely if needed
    if df["cell_id"].duplicated().any():
        df = (
            df.groupby("cell_id", as_index=False)
            .agg(
                {
                    "B2_med": "median",
                    "B3_med": "median",
                    "B4_med": "median",
                    "B8_med": "median",
                    "ndvi_med": "median",
                    "ndvi_std": "median",
                    "ndwi_med": "median",
                    "ndwi_std": "median",
                    "valid_frac": "mean",
                    "year": "first",
                }
            )
        )

    df = df.sort_values("cell_id").reset_index(drop=True)

    _ensure_dir_for_file(args.output)
    df.to_parquet(args.output, index=False)

    valid_mean = float(df["valid_frac"].mean()) if len(df) else 0.0
    print("✅ Feature extraction complete")
    print(f"Raster: {args.raster}")
    print(f"Grid:   {args.grid}")
    print(f"Year:   {year}")
    print(f"Rows:   {len(df):,}")
    print(f"Mean valid_frac: {valid_mean:.3f}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
