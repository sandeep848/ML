#!/usr/bin/env python3
"""
src/make_grid.py

Create a base fishnet grid for tabular Sentinel-2 land-cover modeling.

Key features (no placeholders):
- Builds a metric grid in EPSG:25832 (ETRS89 / UTM 32N) for Nuremberg.
- Uses bounds from either:
    (A) a Sentinel-2 GeoTIFF raster, OR
    (B) an AOI vector file (GeoJSON/GPKG/Shapefile), OR
    (C) automatically finds a raster for a given year in data/raw/sentinel/

- Adds:
    - cell_id: unique integer id
    - block_id: stable spatial group id for spatial GroupKFold (based on centroid + block size)

- Optional clipping to an AOI after grid creation.

Examples
--------
1) Use a specific raster:
   python src/make_grid.py --raster data/raw/sentinel/S2_2020_Jun01_Aug31_10m_QA60SCL_F32.tif

2) Auto-find raster by year in data/raw/sentinel:
   python src/make_grid.py --sentinel-dir data/raw/sentinel --year 2020

3) Use AOI vector bounds:
   python src/make_grid.py --aoi data/aoi/nuremberg.gpkg

4) Clip grid to exact AOI geometry:
   python src/make_grid.py --raster ... --clip-to-aoi data/aoi/nuremberg.gpkg
"""

from __future__ import annotations

import argparse
import math
import os
import re
from dataclasses import dataclass
from typing import Optional, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import CRS, Transformer
from shapely.geometry import box
from shapely.ops import unary_union

TARGET_EPSG = 25832  # ETRS89 / UTM 32N


@dataclass(frozen=True)
class Bounds:
    minx: float
    miny: float
    maxx: float
    maxy: float

    def to_tuple(self) -> Tuple[float, float, float, float]:
        return (self.minx, self.miny, self.maxx, self.maxy)


def _ensure_dir_for_file(filepath: str) -> None:
    d = os.path.dirname(os.path.abspath(filepath))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _snap_bounds_outward(bounds: Bounds, snap: float) -> Bounds:
    """Snap bounds outward to a multiple of `snap` (meters) for clean alignment."""
    minx = math.floor(bounds.minx / snap) * snap
    miny = math.floor(bounds.miny / snap) * snap
    maxx = math.ceil(bounds.maxx / snap) * snap
    maxy = math.ceil(bounds.maxy / snap) * snap
    return Bounds(minx=minx, miny=miny, maxx=maxx, maxy=maxy)


def _reproject_bounds(bounds: Bounds, src_crs: CRS, dst_crs: CRS) -> Bounds:
    """Reproject bounds by transforming all 4 corners and taking min/max."""
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    corners = [
        (bounds.minx, bounds.miny),
        (bounds.minx, bounds.maxy),
        (bounds.maxx, bounds.miny),
        (bounds.maxx, bounds.maxy),
    ]
    xs, ys = [], []
    for x, y in corners:
        x2, y2 = transformer.transform(x, y)
        xs.append(float(x2))
        ys.append(float(y2))
    return Bounds(minx=min(xs), miny=min(ys), maxx=max(xs), maxy=max(ys))


def _bounds_from_raster(raster_path: str) -> Tuple[Bounds, CRS]:
    if not os.path.exists(raster_path):
        raise FileNotFoundError(f"Raster not found: {raster_path}")

    with rasterio.open(raster_path) as ds:
        if ds.crs is None:
            raise ValueError(f"Raster has no CRS: {raster_path}")
        b = ds.bounds
        src_crs = CRS.from_user_input(ds.crs)

    bounds = Bounds(minx=float(b.left), miny=float(b.bottom), maxx=float(b.right), maxy=float(b.top))
    return bounds, src_crs


def _bounds_from_aoi(aoi_path: str) -> Tuple[Bounds, CRS]:
    if not os.path.exists(aoi_path):
        raise FileNotFoundError(f"AOI file not found: {aoi_path}")

    gdf = gpd.read_file(aoi_path)
    if gdf.empty:
        raise ValueError(f"AOI is empty: {aoi_path}")
    if gdf.crs is None:
        raise ValueError(f"AOI has no CRS: {aoi_path}")

    geom = unary_union(gdf.geometry.values)
    minx, miny, maxx, maxy = geom.bounds
    bounds = Bounds(minx=float(minx), miny=float(miny), maxx=float(maxx), maxy=float(maxy))
    return bounds, CRS.from_user_input(gdf.crs)


def _find_sentinel_raster_by_year(sentinel_dir: str, year: int) -> str:
    """
    Finds a Sentinel file in sentinel_dir for the given year.
    Supports your naming:
      S2_2020_Jun01_Aug31_10m_QA60SCL_F32.tif
    Also supports simple patterns like:
      s2_2020.tif
    """
    if not os.path.isdir(sentinel_dir):
        raise NotADirectoryError(f"Sentinel directory not found: {sentinel_dir}")

    year_str = str(year)

    candidates = []
    for fn in os.listdir(sentinel_dir):
        if not fn.lower().endswith((".tif", ".tiff")):
            continue
        # Match year anywhere in filename as a full token or part of S2_YYYY
        if re.search(rf"(?:^|[^0-9]){re.escape(year_str)}(?:[^0-9]|$)", fn) or f"S2_{year_str}" in fn:
            candidates.append(os.path.join(sentinel_dir, fn))

    if not candidates:
        raise FileNotFoundError(
            f"No Sentinel GeoTIFF found for year={year} in {sentinel_dir}.\n"
            f"Expected something like S2_{year}_Jun01_Aug31_10m_QA60SCL_F32.tif"
        )

    # Prefer the most specific / longest filename (often your full naming), then stable sort
    candidates.sort(key=lambda p: (len(os.path.basename(p)), os.path.basename(p)), reverse=True)
    return candidates[0]


def _make_fishnet(bounds_utm: Bounds, cell_size: float) -> gpd.GeoDataFrame:
    """Create grid covering bounds_utm in EPSG:25832."""
    if cell_size <= 0:
        raise ValueError("cell_size must be > 0")

    minx, miny, maxx, maxy = bounds_utm.to_tuple()
    width = maxx - minx
    height = maxy - miny
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid bounds (non-positive area): {bounds_utm}")

    xs = np.arange(minx, maxx, cell_size, dtype=np.float64)
    ys = np.arange(miny, maxy, cell_size, dtype=np.float64)

    geoms = []
    cell_ids = []
    cid = 0

    for y0 in ys:
        y1 = y0 + cell_size
        for x0 in xs:
            x1 = x0 + cell_size
            geoms.append(box(float(x0), float(y0), float(x1), float(y1)))
            cell_ids.append(cid)
            cid += 1

    gdf = gpd.GeoDataFrame({"cell_id": np.array(cell_ids, dtype=np.int64)}, geometry=geoms, crs=f"EPSG:{TARGET_EPSG}")
    return gdf


def _assign_block_id(grid: gpd.GeoDataFrame, block_size: float) -> gpd.GeoDataFrame:
    """
    Assign block_id using centroid bins at block_size.
    block_id is a stable 64-bit integer derived from (block_x, block_y).
    """
    if block_size <= 0:
        raise ValueError("block_size must be > 0")

    cent = grid.geometry.centroid
    cx = cent.x.to_numpy(dtype=np.float64)
    cy = cent.y.to_numpy(dtype=np.float64)

    bx = np.floor(cx / block_size).astype(np.int64)
    by = np.floor(cy / block_size).astype(np.int64)

    # Normalize to start at 0 for stable ids independent of absolute coordinate range
    bx0 = bx - bx.min()
    by0 = by - by.min()

    block_id = (bx0 << 32) | (by0 & 0xFFFFFFFF)

    out = grid.copy()
    out["block_id"] = block_id.astype(np.int64)
    return out


def _clip_to_aoi(grid: gpd.GeoDataFrame, aoi_path: str) -> gpd.GeoDataFrame:
    if not os.path.exists(aoi_path):
        raise FileNotFoundError(f"Clip AOI not found: {aoi_path}")
    aoi = gpd.read_file(aoi_path)
    if aoi.empty:
        raise ValueError(f"Clip AOI is empty: {aoi_path}")
    if aoi.crs is None:
        raise ValueError(f"Clip AOI has no CRS: {aoi_path}")

    aoi_utm = aoi.to_crs(epsg=TARGET_EPSG)
    aoi_geom = unary_union(aoi_utm.geometry.values)

    clipped = grid.loc[grid.geometry.intersects(aoi_geom)].copy()
    clipped = clipped.reset_index(drop=True)
    clipped["cell_id"] = np.arange(len(clipped), dtype=np.int64)
    return clipped


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create a base grid (fishnet) in EPSG:25832 with spatial block groups.")

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--raster", type=str, help="Path to a Sentinel-2 GeoTIFF to derive bounds.")
    src.add_argument("--aoi", type=str, help="Path to an AOI vector (GeoJSON/GPKG/Shapefile) to derive bounds.")
    src.add_argument(
        "--sentinel-dir",
        type=str,
        help="Directory containing Sentinel GeoTIFFs (use with --year to auto-select).",
    )

    p.add_argument("--year", type=int, default=None, help="Year to auto-select from --sentinel-dir (e.g., 2020).")

    p.add_argument("--cell-size", type=float, default=100.0, help="Grid cell size in meters (default: 100).")
    p.add_argument("--block-size", type=float, default=1000.0, help="Spatial block size in meters (default: 1000).")

    p.add_argument(
        "--clip-to-aoi",
        type=str,
        default="",
        help="Optional AOI vector path to clip/intersect the grid to exact AOI geometry.",
    )

    p.add_argument(
        "--output",
        type=str,
        default="data/processed/grid/grid_100m.gpkg",
        help="Output GeoPackage path (default: data/processed/grid/grid_100m.gpkg).",
    )
    p.add_argument("--layer", type=str, default="grid", help="Layer name inside GeoPackage (default: grid).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cell_size = float(args.cell_size)
    block_size = float(args.block_size)

    if cell_size <= 0:
        raise ValueError("--cell-size must be > 0")
    if block_size <= 0:
        raise ValueError("--block-size must be > 0")
    if block_size < cell_size:
        raise ValueError("--block-size must be >= --cell-size")

    dst_crs = CRS.from_epsg(TARGET_EPSG)

    source_desc = ""
    if args.raster:
        raster_path = args.raster
        bounds, src_crs = _bounds_from_raster(raster_path)
        source_desc = f"raster={raster_path}"
    elif args.aoi:
        bounds, src_crs = _bounds_from_aoi(args.aoi)
        source_desc = f"aoi={args.aoi}"
    else:
        # sentinel-dir mode
        if args.year is None:
            raise ValueError("When using --sentinel-dir you must also provide --year (e.g., --year 2020).")
        raster_path = _find_sentinel_raster_by_year(args.sentinel_dir, args.year)
        bounds, src_crs = _bounds_from_raster(raster_path)
        source_desc = f"sentinel-dir={args.sentinel_dir}, year={args.year} -> {raster_path}"

    # Reproject bounds to EPSG:25832
    if src_crs != dst_crs:
        bounds_utm = _reproject_bounds(bounds, src_crs, dst_crs)
    else:
        bounds_utm = bounds

    # Snap outward to align grid cleanly
    bounds_utm = _snap_bounds_outward(bounds_utm, snap=cell_size)

    grid = _make_fishnet(bounds_utm, cell_size=cell_size)

    # Optional clip
    if args.clip_to_aoi.strip():
        grid = _clip_to_aoi(grid, args.clip_to_aoi.strip())

    # Assign block_id (after clip is fine)
    grid = _assign_block_id(grid, block_size=block_size)

    _ensure_dir_for_file(args.output)
    grid.to_file(args.output, layer=args.layer, driver="GPKG")

    print("✅ Grid created successfully")
    print(f"Source: {source_desc}")
    print(f"Output: {args.output} (layer='{args.layer}')")
    print(f"CRS: EPSG:{TARGET_EPSG}")
    print(f"Cells: {len(grid):,}")
    print(f"Cell size: {cell_size} m | Block size: {block_size} m")
    print(f"Bounds (EPSG:{TARGET_EPSG}): {bounds_utm.to_tuple()}")


if __name__ == "__main__":
    main()
