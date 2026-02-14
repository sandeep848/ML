#!/usr/bin/env python3
# app/streamlit_app.py

from __future__ import annotations

import glob
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import folium
from folium.plugins import Draw
import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
from shapely.geometry import Polygon, box, shape
from shapely.ops import unary_union
from streamlit_folium import st_folium

# -----------------------------
# Defaults / Constants
# -----------------------------
DEFAULT_CRS_EPSG = 25832  # ETRS89 / UTM32N for metric aggregation
PRED_DIR_DEFAULT = "data/processed/predictions"
GRID_PATH_DEFAULT = "data/processed/grid/grid_100m.gpkg"

PRED_COLS = ["built_pred", "veg_pred", "water_pred", "other_pred"]
STD_COLS = ["built_std", "veg_std", "water_std", "other_std"]

BASEMAPS = {
    "Light (Carto Positron)": "CartoDB positron",
    "Dark (Carto DarkMatter)": "CartoDB dark_matter",
    "OpenStreetMap": "OpenStreetMap",
}

# -----------------------------
# UI polish (CSS)
# -----------------------------
APP_CSS = """
<style>
  .block-container { padding-top: 1.0rem; padding-bottom: 2.0rem; max-width: 1400px; }
  [data-testid="stSidebar"] .block-container { padding-top: 1rem; }

  /* Top bar */
  .topbar {
    display:flex; align-items:center; justify-content:space-between;
    gap:12px; padding: 10px 14px;
    border: 1px solid rgba(49, 51, 63, .15);
    border-radius: 16px;
    background: rgba(255,255,255,0.55);
    backdrop-filter: blur(6px);
    -webkit-backdrop-filter: blur(6px);
    margin-bottom: 10px;
  }
  .topbar h1 { font-size: 1.2rem; margin:0; padding:0; }
  .subtitle { opacity: 0.75; font-size: 0.92rem; margin-top: 4px; }
  .pill {
    display:inline-flex; align-items:center; gap:6px;
    padding: 6px 10px; border-radius: 999px;
    border: 1px solid rgba(49, 51, 63, .15);
    font-size: 0.85rem; opacity: 0.9;
  }
  .muted { opacity: 0.72; }
  .legend-box {
      background: rgba(255,255,255,0.92);
      border-radius: 12px;
      padding: 10px 12px;
      border: 1px solid rgba(0,0,0,0.15);
      box-shadow: 0 6px 18px rgba(0,0,0,0.10);
      font-size: 12px;
      line-height: 1.25;
  }
</style>
"""

# -----------------------------
# Utilities
# -----------------------------
def list_prediction_years(pred_dir: str) -> List[int]:
    files = glob.glob(os.path.join(pred_dir, "pred_*.parquet"))
    years: List[int] = []
    for f in files:
        m = re.search(r"pred_(\d{4})\.parquet$", os.path.basename(f))
        if m:
            years.append(int(m.group(1)))
    return sorted(list(set(years)))


@st.cache_data(show_spinner=False)
def load_grid(grid_path: str) -> gpd.GeoDataFrame:
    if not os.path.exists(grid_path):
        raise FileNotFoundError(f"Grid not found: {grid_path}")

    gdf = gpd.read_file(grid_path)
    if gdf.empty:
        raise ValueError("Grid is empty.")
    if gdf.crs is None:
        raise ValueError("Grid CRS is missing.")
    if "cell_id" not in gdf.columns:
        raise ValueError("Grid must contain 'cell_id'.")

    gdf = gdf[["cell_id", "geometry"]].copy()
    gdf["cell_id"] = gdf["cell_id"].astype(np.int64)
    gdf = gdf.to_crs(epsg=DEFAULT_CRS_EPSG)
    return gdf


@st.cache_data(show_spinner=False)
def load_pred(pred_dir: str, year: int) -> pd.DataFrame:
    path = os.path.join(pred_dir, f"pred_{year}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prediction file not found: {path}")

    df = pd.read_parquet(path)
    needed = ["cell_id", "year"] + PRED_COLS
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{os.path.basename(path)} missing columns: {missing}")

    df = df.copy()
    df["cell_id"] = df["cell_id"].astype(np.int64)
    df["year"] = df["year"].astype(np.int32)
    return df


def grid_center_latlon(grid_utm: gpd.GeoDataFrame) -> Tuple[float, float]:
    u = grid_utm.to_crs(epsg=4326).geometry.unary_union
    c = u.centroid
    return float(c.y), float(c.x)


def aoi_from_draw(draw: dict) -> Optional[Polygon]:
    if not draw:
        return None
    drawings = draw.get("all_drawings") or []
    if not drawings:
        return None

    last = drawings[-1]
    geom = last.get("geometry")
    if not geom:
        return None

    shp = shape(geom)
    if shp.is_empty:
        return None

    if shp.geom_type == "Polygon":
        return shp
    if shp.geom_type == "MultiPolygon":
        u = unary_union(shp)
        if u.geom_type == "Polygon":
            return u
    return None


def tile_ids_from_centroids(cx: np.ndarray, cy: np.ndarray, tile_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tx = np.floor(cx / tile_size).astype(np.int64)
    ty = np.floor(cy / tile_size).astype(np.int64)
    tx0 = tx - tx.min()
    ty0 = ty - ty.min()
    tile_id = (tx0 << 32) | (ty0 & 0xFFFFFFFF)
    return tile_id, tx, ty


def _build_tile_geoms(tx: np.ndarray, ty: np.ndarray, tile_size: int) -> List:
    return [
        box(int(x) * tile_size, int(y) * tile_size, (int(x) + 1) * tile_size, (int(y) + 1) * tile_size)
        for x, y in zip(tx.tolist(), ty.tolist())
    ]


def aggregate_year_to_tiles(
    grid_utm: gpd.GeoDataFrame,
    pred_year: pd.DataFrame,
    tile_size: int,
) -> gpd.GeoDataFrame:
    g = grid_utm.merge(
        pred_year[["cell_id"] + PRED_COLS + [c for c in STD_COLS if c in pred_year.columns]],
        on="cell_id",
        how="inner",
    )
    cent = g.geometry.centroid
    cx = cent.x.to_numpy(dtype=np.float64)
    cy = cent.y.to_numpy(dtype=np.float64)

    tile_id, tx, ty = tile_ids_from_centroids(cx, cy, tile_size=tile_size)
    g["tile_id"] = tile_id
    g["tx"] = tx
    g["ty"] = ty

    cols = PRED_COLS + [c for c in STD_COLS if c in pred_year.columns]
    agg = g.groupby("tile_id", as_index=False)[cols].mean()
    idx = g.groupby("tile_id", as_index=False)[["tx", "ty"]].first()
    out = idx.merge(agg, on="tile_id", how="inner")

    geoms = _build_tile_geoms(out["tx"].to_numpy(), out["ty"].to_numpy(), tile_size=tile_size)
    tiles = gpd.GeoDataFrame(out.drop(columns=["tx", "ty"]), geometry=geoms, crs=f"EPSG:{DEFAULT_CRS_EPSG}")
    return tiles


def compute_change_tiles(
    tiles_a: gpd.GeoDataFrame,
    tiles_b: gpd.GeoDataFrame,
    year_a: int,
    year_b: int,
) -> gpd.GeoDataFrame:
    a = tiles_a.copy()
    b = tiles_b.copy()

    keep_cols_a = ["tile_id", "geometry"] + PRED_COLS + [c for c in STD_COLS if c in a.columns]
    keep_cols_b = ["tile_id"] + PRED_COLS + [c for c in STD_COLS if c in b.columns]

    a = a[keep_cols_a].rename(columns={c: f"{c}_A" for c in PRED_COLS + [c for c in STD_COLS if c in tiles_a.columns]})
    b = b[keep_cols_b].rename(columns={c: f"{c}_B" for c in PRED_COLS + [c for c in STD_COLS if c in tiles_b.columns]})

    m = a.merge(b, on="tile_id", how="inner")

    for c in PRED_COLS:
        m[f"{c}_d"] = m[f"{c}_B"] - m[f"{c}_A"]

    for c in STD_COLS:
        ca = f"{c}_A"
        cb = f"{c}_B"
        if ca in m.columns and cb in m.columns:
            m[f"{c}_d"] = np.sqrt(np.square(m[ca]) + np.square(m[cb]))

    m["year_a"] = int(year_a)
    m["year_b"] = int(year_b)
    return gpd.GeoDataFrame(m, geometry="geometry", crs=f"EPSG:{DEFAULT_CRS_EPSG}")


def _add_legend(m: folium.Map, title: str, thr: float, vmax_abs: float, mode: str) -> None:
    if mode == "Increase":
        hint = "Green ramp = stronger positive change (+Δ)"
    elif mode == "Decrease":
        hint = "Red ramp = stronger negative change (−Δ magnitude)"
    else:
        hint = "Green = +Δ, Red = −Δ (magnitude)"

    html = f"""
    <div class="legend-box">
      <div style="font-weight: 700; margin-bottom: 6px;">{title}</div>
      <div class="muted">Threshold: |Δ| ≥ {thr:.2f}</div>
      <div class="muted">Scale max (≈p95): {vmax_abs:.3f}</div>
      <div style="margin-top:6px;">{hint}</div>
    </div>
    """
    folium.Element(f"""
      <div style="position: fixed; bottom: 20px; left: 20px; z-index:9999;">
        {html}
      </div>
    """).add_to(m.get_root())


def add_change_layer(
    m: folium.Map,
    gdf_utm: gpd.GeoDataFrame,
    value_col: str,
    mode: str,
    vabs: float,
    layer_name: str,
) -> None:
    g = gdf_utm.to_crs(epsg=4326)
    data_json = json.loads(g.to_json())

    def style_inc(feat):
        v = feat["properties"].get(value_col, None)
        if v is None or (isinstance(v, float) and np.isnan(v)) or float(v) <= 0:
            return {"fillOpacity": 0.0, "weight": 0.0, "color": "#000000"}
        t = min(1.0, max(0.0, float(v) / (vabs + 1e-12)))
        r = int(40 * (1 - t) + 0 * t)
        g_ = int(200 * (1 - t) + 255 * t)
        b = int(40 * (1 - t) + 0 * t)
        return {"fillColor": f"#{r:02x}{g_:02x}{b:02x}", "fillOpacity": 0.70, "weight": 0.2, "color": "#1b5e20"}

    def style_dec(feat):
        v = feat["properties"].get(value_col, None)
        if v is None or (isinstance(v, float) and np.isnan(v)) or float(v) >= 0:
            return {"fillOpacity": 0.0, "weight": 0.0, "color": "#000000"}
        mag = min(1.0, max(0.0, abs(float(v)) / (vabs + 1e-12)))
        r = int(220 * (1 - mag) + 255 * mag)
        g_ = int(60 * (1 - mag) + 0 * mag)
        b = int(60 * (1 - mag) + 0 * mag)
        return {"fillColor": f"#{r:02x}{g_:02x}{b:02x}", "fillOpacity": 0.70, "weight": 0.2, "color": "#b71c1c"}

    def style_both(feat):
        v = feat["properties"].get(value_col, None)
        if v is None or (isinstance(v, float) and np.isnan(v)) or float(v) == 0:
            return {"fillOpacity": 0.0, "weight": 0.0, "color": "#000000"}
        v = float(v)
        mag = min(1.0, max(0.0, abs(v) / (vabs + 1e-12)))
        if v > 0:
            r = int(40 * (1 - mag) + 0 * mag)
            g_ = int(200 * (1 - mag) + 255 * mag)
            b = int(40 * (1 - mag) + 0 * mag)
            return {"fillColor": f"#{r:02x}{g_:02x}{b:02x}", "fillOpacity": 0.70, "weight": 0.2, "color": "#1b5e20"}
        else:
            r = int(220 * (1 - mag) + 255 * mag)
            g_ = int(60 * (1 - mag) + 0 * mag)
            b = int(60 * (1 - mag) + 0 * mag)
            return {"fillColor": f"#{r:02x}{g_:02x}{b:02x}", "fillOpacity": 0.70, "weight": 0.2, "color": "#b71c1c"}

    tooltip = folium.GeoJsonTooltip(fields=["tile_id", value_col], aliases=["Tile", "Δ"])

    if mode == "Increase":
        style_fn = style_inc
    elif mode == "Decrease":
        style_fn = style_dec
    else:
        style_fn = style_both

    folium.GeoJson(
        data_json,
        name=layer_name,
        style_function=style_fn,
        tooltip=tooltip,
    ).add_to(m)


# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Urban Land-Cover Change — Nuremberg", layout="wide")
st.markdown(APP_CSS, unsafe_allow_html=True)

# Session defaults
if "aoi_wkt" not in st.session_state:
    st.session_state["aoi_wkt"] = None
if "aoi_note" not in st.session_state:
    st.session_state["aoi_note"] = "AOI not set yet (draw one, or use full extent)."

# ---- Top Bar (title + info button)
top_left, top_right = st.columns([0.82, 0.18])
with top_left:
    st.markdown(
        """
        <div class="topbar">
          <div>
            <h1>Urban Land-Cover Change — Nuremberg</h1>
            <div class="subtitle">Visualize tile-level Δ (change) between two years for built / vegetation / water / other.</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with top_right:
    # Small "i" info popover (the top info button you requested)
    with st.popover("ℹ️", use_container_width=True):
        st.markdown(
            """
**What this represents**
- The map shows **change (Δ)** between **Year B − Year A** for the selected target (built / veg / water / other).
- Values are aggregated into **square tiles** (e.g., 500 m).
- **Green** = increase, **Red** = decrease (based on selected mode).
- The **threshold** hides small changes: only tiles with **|Δ| ≥ threshold** are shown.
            """
        )

# Sidebar
with st.sidebar:
    st.header("Settings")

    st.subheader("Paths")
    pred_dir = st.text_input("Predictions folder", value=PRED_DIR_DEFAULT)
    grid_path = st.text_input("Base grid (GPKG)", value=GRID_PATH_DEFAULT)

    st.divider()
    st.subheader("Analysis")
    target = st.selectbox("Target", ["built", "veg", "water", "other"], index=0)
    tile_size = st.select_slider("Tile size (meters)", options=[100, 200, 500, 1000, 2000], value=500)
    thr = st.slider("Threshold (|Δ| ≥)", 0.00, 0.50, 0.05, 0.01)

    st.divider()
    st.subheader("Map")
    basemap_label = st.selectbox("Basemap", list(BASEMAPS.keys()), index=0)
    map_mode = st.segmented_control("Mode", options=["Increase", "Decrease", "Both"], default="Both")
    show_aoi_outline = st.checkbox("Show AOI outline", value=True)

    with st.expander("Advanced", expanded=False):
        vmax_pctl = st.slider("Color scale percentile (abs)", 80, 99, 95, 1)
        zoom_start = st.slider("Initial zoom", 9, 14, 11, 1)

# Validate inputs
if not os.path.isdir(pred_dir):
    st.error(f"Predictions folder not found: {pred_dir}")
    st.stop()

years = list_prediction_years(pred_dir)
if len(years) < 2:
    st.error("Need at least 2 prediction years (pred_YYYY.parquet) to compute change.")
    st.stop()

grid = load_grid(grid_path)
lat0, lon0 = grid_center_latlon(grid)

# Tabs (table removed)
tab_aoi, tab_maps, tab_download = st.tabs(["🧭 AOI", "🗺️ Maps", "⬇️ Download"])

# Cache loading all years predictions
@st.cache_data(show_spinner=False)
def load_all_year_preds(pred_dir_: str, years_: List[int]) -> Dict[int, pd.DataFrame]:
    return {y: load_pred(pred_dir_, y) for y in years_}

preds_by_year = load_all_year_preds(pred_dir, years)

@st.cache_data(show_spinner=False)
def aggregate_all_years_to_tiles(
    grid_path_: str,
    years_: List[int],
    tile_size_: int,
    aoi_wkt_: str,
) -> Dict[int, gpd.GeoDataFrame]:
    grid_local = load_grid(grid_path_)
    aoi_local = gpd.GeoSeries.from_wkt([aoi_wkt_], crs=f"EPSG:{DEFAULT_CRS_EPSG}").iloc[0]

    out: Dict[int, gpd.GeoDataFrame] = {}
    for y in years_:
        tiles_y = aggregate_year_to_tiles(grid_local, preds_by_year[y], tile_size=int(tile_size_))
        tiles_y = tiles_y.loc[tiles_y.geometry.intersects(aoi_local)].copy()
        out[y] = tiles_y
    return out

# -----------------------------
# AOI tab
# -----------------------------
with tab_aoi:
    st.subheader("Draw AOI (optional)")
    st.write("Draw a polygon/rectangle. Or click **Use full extent**. AOI is applied to all maps and exports.")

    cA, cB = st.columns([1.25, 0.75], gap="large")

    with cA:
        m_draw = folium.Map(location=[lat0, lon0], zoom_start=zoom_start, tiles=BASEMAPS[basemap_label])
        Draw(
            export=False,
            draw_options={
                "polyline": False,
                "rectangle": True,
                "circle": False,
                "circlemarker": False,
                "marker": False,
                "polygon": True,
            },
            edit_options={"edit": True, "remove": True},
        ).add_to(m_draw)
        folium.LayerControl(collapsed=True).add_to(m_draw)

        draw_resp = st_folium(m_draw, height=600, width=None, returned_objects=["all_drawings"])
        aoi_poly_wgs84 = aoi_from_draw(draw_resp)

        if aoi_poly_wgs84 is not None:
            aoi_gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[aoi_poly_wgs84], crs="EPSG:4326").to_crs(epsg=DEFAULT_CRS_EPSG)
            aoi_geom_utm = aoi_gdf.geometry.iloc[0]
            st.session_state["aoi_wkt"] = aoi_geom_utm.wkt
            st.session_state["aoi_note"] = "AOI = drawn polygon/rectangle"
            st.success("AOI updated from drawing.")

    with cB:
        st.markdown("##### Quick actions")
        if st.button("Use full grid extent as AOI", width="stretch"):
            bounds = grid.total_bounds
            aoi_geom_utm = box(bounds[0], bounds[1], bounds[2], bounds[3])
            st.session_state["aoi_wkt"] = aoi_geom_utm.wkt
            st.session_state["aoi_note"] = "AOI = full grid extent"
            st.success("AOI set to full extent.")

        st.markdown("---")
        st.markdown("##### Status")
        st.markdown(f"<div class='pill'>📍 <span class='muted'>{st.session_state['aoi_note']}</span></div>", unsafe_allow_html=True)
        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown("##### Data")
        st.write(f"Years: **{', '.join(map(str, years))}**")

# AOI fallback if none chosen
if st.session_state["aoi_wkt"] is None:
    bounds = grid.total_bounds
    aoi_geom_utm = box(bounds[0], bounds[1], bounds[2], bounds[3])
    aoi_note = "AOI = full grid extent (no AOI chosen yet)"
else:
    aoi_geom_utm = gpd.GeoSeries.from_wkt([st.session_state["aoi_wkt"]], crs=f"EPSG:{DEFAULT_CRS_EPSG}").iloc[0]
    aoi_note = st.session_state["aoi_note"]

pred_col = f"{target}_pred"

# -----------------------------
# Maps tab
# -----------------------------
with tab_maps:
    st.subheader("Change maps (tile-level)")
    st.caption(aoi_note)

    yc1, yc2, yc3, yc4 = st.columns([1, 1, 1, 1], gap="large")
    with yc1:
        year_a = st.selectbox("Year A", years, index=0)
    with yc2:
        year_b = st.selectbox("Year B", years, index=min(1, len(years) - 1))
    with yc3:
        st.metric("Tile size", f"{int(tile_size)} m")
    with yc4:
        st.metric("Threshold", f"{thr:.2f}")

    with st.status("Aggregating predictions into tiles (cached)…", expanded=False):
        tiles_by_year = aggregate_all_years_to_tiles(grid_path, years, int(tile_size), aoi_geom_utm.wkt)

    ta = tiles_by_year.get(int(year_a))
    tb = tiles_by_year.get(int(year_b))
    if ta is None or tb is None or ta.empty or tb.empty:
        st.error("Selected years have no tiles in the AOI.")
    else:
        change_tiles = compute_change_tiles(ta, tb, int(year_a), int(year_b))
        delta_col = f"{pred_col}_d"

        # Apply threshold by mode
        ct = change_tiles.copy()
        if map_mode == "Increase":
            ct.loc[ct[delta_col] < float(thr), delta_col] = np.nan
        elif map_mode == "Decrease":
            ct.loc[ct[delta_col] > -float(thr), delta_col] = np.nan
        else:
            ct.loc[ct[delta_col].abs() < float(thr), delta_col] = np.nan

        # vmax heuristic
        dvals = change_tiles[delta_col].to_numpy(dtype=float)
        dvals = dvals[~np.isnan(dvals)]
        vmax_abs = float(thr) if dvals.size == 0 else float(np.clip(np.percentile(np.abs(dvals), vmax_pctl), thr, 1.0))

        # AOI outline
        aoi_outline = gpd.GeoDataFrame({"id": [1]}, geometry=[aoi_geom_utm], crs=f"EPSG:{DEFAULT_CRS_EPSG}").to_crs(epsg=4326)

        m_map = folium.Map(location=[lat0, lon0], zoom_start=zoom_start, tiles=BASEMAPS[basemap_label])

        if show_aoi_outline:
            folium.GeoJson(
                json.loads(aoi_outline.to_json()),
                name="AOI",
                style_function=lambda _: {"fillOpacity": 0.0, "weight": 2, "color": "#000000"},
            ).add_to(m_map)

        add_change_layer(
            m=m_map,
            gdf_utm=ct[["tile_id", delta_col, "geometry"]].copy(),
            value_col=delta_col,
            mode=map_mode,
            vabs=vmax_abs,
            layer_name=f"{map_mode}: {target} Δ ({year_b} − {year_a})",
        )

        folium.LayerControl(collapsed=True).add_to(m_map)
        _add_legend(
            m_map,
            title=f"{target} change (Δ = {year_b} − {year_a})",
            thr=float(thr),
            vmax_abs=float(vmax_abs),
            mode=map_mode,
        )

        st_folium(m_map, height=650, width=None)

# -----------------------------
# Download tab
# -----------------------------
with tab_download:
    st.subheader("Download change tiles (GeoJSON)")
    st.caption("Export tile-level change for your selected settings and AOI.")

    dc1, dc2 = st.columns(2)
    with dc1:
        export_year_a = st.selectbox("Export Year A", years, index=max(0, len(years) - 2))
    with dc2:
        export_year_b = st.selectbox("Export Year B", years, index=len(years) - 1)

    with st.status("Preparing export…", expanded=False):
        tiles_by_year = aggregate_all_years_to_tiles(grid_path, years, int(tile_size), aoi_geom_utm.wkt)

        ta = tiles_by_year.get(int(export_year_a))
        tb = tiles_by_year.get(int(export_year_b))

        if ta is None or tb is None or ta.empty or tb.empty:
            st.error("Export years have no tiles in the AOI.")
        else:
            change_tiles = compute_change_tiles(ta, tb, int(export_year_a), int(export_year_b))
            delta_col = f"{pred_col}_d"

            # Export GeoJSON (WGS84)
            chg_geojson = change_tiles.to_crs(epsg=4326).to_json()

            st.download_button(
                label="Download GeoJSON",
                data=chg_geojson,
                file_name=f"change_tiles_{target}_{export_year_a}_{export_year_b}_tile{tile_size}m_thr{thr:.2f}.geojson",
                mime="application/geo+json",
                width="stretch",
            )
