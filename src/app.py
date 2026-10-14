"""
app.py  -  STEP 2 (RunoffAtlas Dashboard)
=========================================
Reads aligned layers from preprocess.py and displays urban expansion dynamics
(2017 -> 2025 historical, and 2035 probability-estimated future scenario) 
along with rainfall-driven runoff hotspots.

Method: SCS Curve Number (USDA TR-55) & Probability-Based LULC Estimation
Run:   streamlit run src/app.py
"""

import math
import os

import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
import streamlit as st
import folium
import branca.colormap as bcm
import plotly.graph_objects as go
from folium import plugins
from matplotlib import colormaps, colors
from rasterio.transform import array_bounds
from rasterio.warp import reproject, calculate_default_transform, Resampling
from streamlit_folium import st_folium
from scipy import ndimage

import config as C

Y0, Y1 = C.YEARS

# ------------------------------------------------------------------
# 1. PAGE CONFIG & STRICT TIMES NEW ROMAN GLOBAL CSS (Title Size 28px)
# ------------------------------------------------------------------
st.set_page_config(page_title="RunoffAtlas", layout="wide")

st.markdown("""
<style>
    /* Force Times New Roman globally and increase font size for body text */
    *, html, body, [class*="css"], .stMarkdown, .stText, .stMetric, 
    .stMetricLabel, .stMetricValue, h4, h5, h6, p, span, div, label {
        font-family: "Times New Roman", Times, serif !important;
        font-size: 16px !important;
    }
    
    /* Main title size explicitly set to 28px (beats the global 16px rule) */
    .stApp h1,
    .stApp h1 *,
    .stApp [data-testid="stHeading"] h1,
    .stApp [data-testid="stHeading"] h1 * {
        font-family: "Times New Roman", Times, serif !important;
        font-size: 28px !important;
        color: #0f172a;
        font-weight: 700;
    }
    
    h2, h3 {
        font-family: "Times New Roman", Times, serif !important;
        font-size: 22px !important;
        color: #1e293b;
        font-weight: 600;
    }
    
    .main { background-color: #f8f9fa; }
    
    /* Dark Blue Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0f172a;
        color: #ffffff;
    }
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] .stRadio div, 
    [data-testid="stSidebar"] .stSelectbox label, 
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span {
        color: #f1f5f9 !important;
        font-family: "Times New Roman", Times, serif !important;
        font-size: 16px !important;
    }

    .stMetric {
        background-color: #ffffff;
        padding: 16px;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏙️ RunoffAtlas - Urban Growth & Runoff Assessment for Delhi")

st.info("**How to read:** red = more runoff, blue = less. Move the rainfall slider in the sidebar and compare 2017 (left) with 2025 (right). "
        "This is a screening tool for planning, not a flood forecast.")


# ------------------------------------------------------------------
# 2. LOAD PROCESSED LAYERS & BASIN VECTOR BOUNDARY
# ------------------------------------------------------------------
@st.cache_data(show_spinner="Loading processed spatial layers...")
def load_layers():
    def read(path):
        with rasterio.open(path) as src:
            return src.read(1), src.transform, src.crs.to_string()

    lulc, rain = {}, {}
    for y in C.YEARS:
        lulc[y], transform, crs = read(C.OUT_LULC[y])
        r, _, _ = read(C.OUT_RAIN[y])
        r = r.astype("float32")
        r[r == C.FLOAT_NODATA] = np.nan
        rain[y] = r

    soil, _, _ = read(C.OUT_SOIL)
    
    # Fill soil gaps with nearest valid soil group
    valid_soil = np.isin(soil, [1, 2, 3, 4])
    if valid_soil.any() and not valid_soil.all():
        idx = ndimage.distance_transform_edt(~valid_soil, return_distances=False, return_indices=True)
        soil = soil[tuple(idx)]

    return lulc, rain, soil, transform, crs


@st.cache_data
def load_basin_boundary():
    """Loads vector basin boundary for map display."""
    try:
        basin = gpd.read_file(C.RAW_BASIN)
        if basin.crs is not None:
            basin = basin.to_crs("EPSG:4326")
        return basin
    except Exception:
        return None


try:
    lulc, rain, soil, transform, crs = load_layers()
    basin_gdf = load_basin_boundary()
    basin_geojson = basin_gdf.__geo_interface__ if basin_gdf is not None else None
except Exception as e:
    st.error(f"Could not read the processed layers. Did you run preprocess.py first?\n\n{e}")
    st.stop()

PIXEL_AREA = abs(transform.a * transform.e)  # m2 per pixel


# ------------------------------------------------------------------
# 3. SCS CURVE NUMBER & PROBABILITY-BASED ESTIMATION
# ------------------------------------------------------------------
CN_ARRAY = np.full((256, 5), np.nan, dtype="float32")
for code, values in C.CN_TABLE.items():
    CN_ARRAY[code, 1:] = values


def curve_number(lulc_arr, soil_arr):
    group = soil_arr.astype("int16")
    ok = (group >= 1) & (group <= 4)
    fallback = C.SOIL_FALLBACK if C.SOIL_FALLBACK else 0
    group = np.where(ok, group, fallback)
    return CN_ARRAY[lulc_arr, group]


def runoff(P, CN):
    with np.errstate(invalid="ignore", divide="ignore"):
        S = 25400.0 / CN - 254.0
        Ia = 0.2 * S
        Q = np.where(P > Ia, (P - Ia) ** 2 / (P + 0.8 * S), 0.0)
    Q = Q.astype("float32")
    Q[np.isnan(CN) | np.isnan(P)] = np.nan
    return Q


@st.cache_data
def probability_based_projection(l0, l1, valid_codes):
    """
    Computes a historical transition frequency matrix between Y0 (2017) and Y1 (2025)
    and uses empirical class-change probabilities to estimate the 2035 LULC distribution.
    """
    mask = (l0 != C.INT_NODATA) & (l1 != C.INT_NODATA)
    flat0 = l0[mask].astype(int)
    flat1 = l1[mask].astype(int)

    n_classes = 256
    trans_matrix = np.zeros((n_classes, n_classes), dtype=float)
    
    for c0, c1 in zip(flat0, flat1):
        if c0 < n_classes and c1 < n_classes:
            trans_matrix[c0, c1] += 1.0

    row_sums = trans_matrix.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        P_matrix = np.where(row_sums > 0, trans_matrix / row_sums, 0.0)

    for i in range(n_classes):
        if row_sums[i] == 0:
            P_matrix[i, i] = 1.0

    np.random.seed(42)
    l_fut = l1.copy()
    flat_fut = l1.flatten().astype(int)
    for idx, val in enumerate(flat_fut):
        if val != C.INT_NODATA and val < n_classes:
            probs = P_matrix[val]
            if probs.sum() > 0:
                probs = probs / probs.sum()
                new_val = np.random.choice(n_classes, p=probs)
                flat_fut[idx] = new_val
                
    return flat_fut.reshape(l1.shape)


def filter_valid_codes():
    filtered_codes = []
    for k, name in C.LULC_NAMES.items():
        if "snow" in name.lower() or "ice" in name.lower():
            continue
        filtered_codes.append(k)
    return filtered_codes


VALID_LULC_CODES = filter_valid_codes()


# ------------------------------------------------------------------
# 4. SIDEBAR CONTROLS
# ------------------------------------------------------------------
st.sidebar.header("🎛️ Analysis Controls")

rain_source = st.sidebar.radio(
    "Rainfall Used",
    ["Same storm for both years (slider)", "Rainfall raster of each year"],
    help="The slider isolates land-cover change impact. Selecting raster files combines LULC change with rainfall variation.",
)

if rain_source.startswith("Same"):
    P_slider = st.sidebar.slider("Storm Rainfall P (mm/day)", 10, 300, 100, 5)
    P0 = P1 = PF = float(P_slider)
else:
    P0, P1 = rain[Y0], rain[Y1]
    PF = rain[Y1]

layer = st.sidebar.selectbox(
    "Map Layer",
    ["Runoff depth (mm)", "Land cover", "Change in runoff (mm)"],
)

opacity = st.sidebar.slider("Layer Opacity", 0.2, 1.0, 0.85, 0.05)


# ------------------------------------------------------------------
# 5. COMPUTATIONS (2017, 2025, & 2035 PROBABILITY PROJECTION)
# ------------------------------------------------------------------
CN0, CN1 = curve_number(lulc[Y0], soil), curve_number(lulc[Y1], soil)
Q0, Q1 = runoff(P0, CN0), runoff(P1, CN1)
dQ = Q1 - Q0

lulc_2035 = probability_based_projection(lulc[Y0], lulc[Y1], VALID_LULC_CODES)
CN_2035 = curve_number(lulc_2035, soil)
Q_2035 = runoff(PF, CN_2035)


def volume_mm3(Q):
    return np.nansum(Q) / 1000.0 * PIXEL_AREA / 1e6


def built_km2(lulc_arr):
    built_code = getattr(C, 'BUILT_CODE', 1)
    return np.sum(lulc_arr == built_code) * PIXEL_AREA / 1e6


# ------------------------------------------------------------------
# 6. HEADLINE STATS
# ------------------------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric(f"Built-up Area ({Y1})", f"{built_km2(lulc[Y1]):,.1f} km²",
          f"{built_km2(lulc[Y1]) - built_km2(lulc[Y0]):+,.1f} km² vs {Y0}")
c2.metric(f"Mean Runoff ({Y1})", f"{np.nanmean(Q1):.1f} mm",
          f"{np.nanmean(Q1) - np.nanmean(Q0):+.2f} mm vs {Y0}")
c3.metric("2035 Projected Runoff", f"{np.nanmean(Q_2035):.1f} mm",
          f"{np.nanmean(Q_2035) - np.nanmean(Q1):+.2f} mm vs {Y1}")

st.markdown("---")


# ------------------------------------------------------------------
# 7. MAP OVERLAY HELPERS (Darker High-Contrast Blue-to-Red Palette)
# ------------------------------------------------------------------
def to_latlon(arr):
    h, w = arr.shape
    bounds = array_bounds(h, w, transform)
    dst_t, dw, dh = calculate_default_transform(crs, "EPSG:4326", w, h, *bounds)
    out = np.full((dh, dw), np.nan, dtype="float32")
    reproject(arr.astype("float32"), out, src_transform=transform,
              src_crs=crs, src_nodata=np.nan, dst_transform=dst_t,
              dst_crs="EPSG:4326", dst_nodata=np.nan,
              resampling=Resampling.nearest)
    west, south, east, north = array_bounds(dh, dw, dst_t)
    return out, [[south, west], [north, east]]


def center_and_zoom(bounds, px_width=1100, px_height=550):
    """Estimate map center and zoom from [[south, west], [north, east]] bounds,
    without needing a visible container (works inside hidden tabs)."""
    (south, west), (north, east) = bounds
    center = [(south + north) / 2, (west + east) / 2]
    lon_span = max(east - west, 1e-6)
    lat_span = max(north - south, 1e-6)
    zoom_lon = math.log2(360 * px_width / (256 * lon_span))
    zoom_lat = math.log2(180 * px_height / (256 * lat_span * 1.2))
    zoom = int(min(zoom_lon, zoom_lat))
    return center, max(zoom, 1)


def paint(arr, cmap, vmin, vmax):
    scaled = np.clip((arr - vmin) / (vmax - vmin + 1e-9), 0, 1)
    rgba = colormaps[cmap](scaled)
    rgba[..., 3] = np.where(np.isnan(arr), 0, opacity)
    return (rgba * 255).astype("uint8")


def paint_lulc(arr):
    rgba = np.zeros(arr.shape + (4,), dtype="uint8")
    for code, hex_color in C.LULC_COLORS.items():
        r, g, b = colors.to_rgb(hex_color)
        rgba[arr == code] = (r * 255, g * 255, b * 255, opacity * 255)
    return rgba


def add_shapefile_boundary(m):
    if basin_geojson:
        folium.GeoJson(
            basin_geojson,
            style_function=lambda x: {
                'fillColor': 'none',
                'color': '#000000',
                'weight': 2.5,
                'dashArray': '4, 4'
            },
            name="Basin Boundary"
        ).add_to(m)


def add_waterlogging_points(m):
    path = os.path.join(os.path.dirname(__file__), "waterlogging_points.csv")
    if os.path.exists(path):
        for _, r in pd.read_csv(path).iterrows():
            folium.CircleMarker([r["lat"], r["lon"]], radius=5, color="black",
                                weight=1, fill=True, fill_color="yellow", fill_opacity=0.9,
                                popup=str(r["name"])).add_to(m)


# ------------------------------------------------------------------
# 8. MAP VISUALIZATION & COMPARISON
# ------------------------------------------------------------------
st.subheader(f"🗺️ Spatial Explorer - {layer}")

tab_hist, tab_fut = st.tabs(["📊 Historical Comparison (2017 vs 2025)", "🔮 Future Scenario (2035 Probability-Based Projection)"])

with tab_hist:
    if layer == "Change in runoff (mm)":
        img, bounds = to_latlon(dQ)
        limit = max(float(np.nanpercentile(np.abs(dQ), 99)), 1.0)
        m = folium.Map(tiles="OpenStreetMap")
        folium.raster_layers.ImageOverlay(image=paint(img, "seismic", -limit, limit), bounds=bounds).add_to(m)
        add_shapefile_boundary(m)
        add_waterlogging_points(m)
        bcm.LinearColormap(['#0000ff', '#ffffff', '#ff0000'], vmin=-limit, vmax=limit, caption=f"Runoff change {Y1} - {Y0} (mm)").add_to(m)
    else:
        m = plugins.DualMap(tiles="OpenStreetMap")
        add_shapefile_boundary(m.m1)
        add_shapefile_boundary(m.m2)
        add_waterlogging_points(m.m1)
        add_waterlogging_points(m.m2)

        if layer == "Land cover":
            l0, bounds = to_latlon(np.where(lulc[Y0] == C.INT_NODATA, np.nan, lulc[Y0]))
            l1, _ = to_latlon(np.where(lulc[Y1] == C.INT_NODATA, np.nan, lulc[Y1]))
            folium.raster_layers.ImageOverlay(image=paint_lulc(l0), bounds=bounds).add_to(m.m1)
            folium.raster_layers.ImageOverlay(image=paint_lulc(l1), bounds=bounds).add_to(m.m2)
        else:
            arr0, arr1, cmap, name = Q0, Q1, "seismic", "Runoff Q (mm)"

            vmin = float(min(np.nanmin(arr0), np.nanmin(arr1)))
            vmax = float(max(np.nanmax(arr0), np.nanmax(arr1)))
            img0, bounds = to_latlon(arr0)
            img1, _ = to_latlon(arr1)
            folium.raster_layers.ImageOverlay(image=paint(img0, cmap, vmin, vmax), bounds=bounds).add_to(m.m1)
            folium.raster_layers.ImageOverlay(image=paint(img1, cmap, vmin, vmax), bounds=bounds).add_to(m.m2)
            bcm.LinearColormap(['#000080', '#0000ff', '#00ffff', '#ffff00', '#ff0000', '#800000'], vmin=vmin, vmax=vmax, caption=name).add_to(m.m2)

        st.caption(f"Left Map: {Y0}  |  Right Map: {Y1}")

    m.fit_bounds(bounds)
    st_folium(m, height=550, use_container_width=True, key="hist_map")

    # LULC Legend directly below historical map if Land cover is selected
    if layer == "Land cover":
        legend_items = "".join(
            f"<span style='margin-right:14px'>"
            f"<span style='background:{C.LULC_COLORS[k]};padding:0 9px;"
            f"border-radius:3px;margin-right:4px'>&nbsp;</span>{C.LULC_NAMES[k]}"
            f"</span>" for k in VALID_LULC_CODES)
        st.markdown(f"<div style='margin-top: 10px; margin-bottom: 20px;'>{legend_items}</div>", unsafe_allow_html=True)

    # Historical LULC Area Comparison Chart (2017 vs 2025 ONLY)
    st.markdown("---")
    st.subheader("📈 Historical Land-Cover Area Comparison (2017 vs 2025)")
    fig_hist = go.Figure()
    areas_0 = [np.sum(lulc[Y0] == k) * PIXEL_AREA / 1e6 for k in VALID_LULC_CODES]
    fig_hist.add_bar(x=[C.LULC_NAMES[k] for k in VALID_LULC_CODES], y=areas_0, name=str(Y0), marker_color="#3b82f6")
    
    areas_1 = [np.sum(lulc[Y1] == k) * PIXEL_AREA / 1e6 for k in VALID_LULC_CODES]
    fig_hist.add_bar(x=[C.LULC_NAMES[k] for k in VALID_LULC_CODES], y=areas_1, name=str(Y1), marker_color="#ef4444")
    
    fig_hist.update_layout(
        barmode="group",
        yaxis_title="Area (km²)",
        xaxis_title="Land Cover Class",
        height=400,
        margin=dict(t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(family="Times New Roman", size=15)
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with tab_fut:
    st.markdown("### Projected Land-Cover & Runoff Dynamics (2035)")
    st.info("ℹ️ **Projection Basis:** Estimated using empirical class-change probabilities derived from historical transitions between 2017 and 2025.")

    # 1) Reproject the 2035 runoff raster FIRST (gives bounds_f)
    img_f, bounds_f = to_latlon(Q_2035)

    # 2) Pick the extent to zoom to: basin boundary if available, else raster extent
    if basin_gdf is not None:
        minx, miny, maxx, maxy = basin_gdf.total_bounds
        fut_bounds = [[miny, minx], [maxy, maxx]]
    else:
        fut_bounds = bounds_f

    # 3) Explicit center + zoom (works even though this tab is hidden on page load)
    fut_center, fut_zoom = center_and_zoom(fut_bounds)

    m_fut = folium.Map(location=fut_center, zoom_start=fut_zoom, tiles="OpenStreetMap")
    add_shapefile_boundary(m_fut)
    add_waterlogging_points(m_fut)

    if layer == "Land cover":
        l_fut_arr, _ = to_latlon(np.where(lulc_2035 == C.INT_NODATA, np.nan, lulc_2035))
        folium.raster_layers.ImageOverlay(image=paint_lulc(l_fut_arr), bounds=bounds_f).add_to(m_fut)
    else:
        vf_min, vf_max = float(np.nanmin(Q_2035)), float(np.nanmax(Q_2035))
        folium.raster_layers.ImageOverlay(
            image=paint(img_f, "seismic", vf_min, vf_max), bounds=bounds_f
        ).add_to(m_fut)
        bcm.LinearColormap(
            ['#000080', '#0000ff', '#00ffff', '#ffff00', '#ff0000', '#800000'],
            vmin=vf_min, vmax=vf_max,
            caption="Projected 2035 Runoff Depth (mm)"
        ).add_to(m_fut)

    st_folium(m_fut, height=550, use_container_width=True, key="fut_map",
              center=fut_center, zoom=fut_zoom)

    # LULC Legend directly below future map if Land cover is selected
    if layer == "Land cover":
        legend_items_fut = "".join(
            f"<span style='margin-right:14px'>"
            f"<span style='background:{C.LULC_COLORS[k]};padding:0 9px;"
            f"border-radius:3px;margin-right:4px'>&nbsp;</span>{C.LULC_NAMES[k]}"
            f"</span>" for k in VALID_LULC_CODES)
        st.markdown(f"<div style='margin-top: 10px; margin-bottom: 20px;'>{legend_items_fut}</div>", unsafe_allow_html=True)

    # Future 3-Way Land-Cover Area Comparison Chart (2017 vs 2025 vs 2035)
    st.markdown("---")
    st.subheader("📈 Full Trajectory Land-Cover Area Comparison (2017 vs 2025 vs 2035)")
    fig_fut_chart = go.Figure()
    areas_0 = [np.sum(lulc[Y0] == k) * PIXEL_AREA / 1e6 for k in VALID_LULC_CODES]
    fig_fut_chart.add_bar(x=[C.LULC_NAMES[k] for k in VALID_LULC_CODES], y=areas_0, name=str(Y0), marker_color="#3b82f6")
    
    areas_1 = [np.sum(lulc[Y1] == k) * PIXEL_AREA / 1e6 for k in VALID_LULC_CODES]
    fig_fut_chart.add_bar(x=[C.LULC_NAMES[k] for k in VALID_LULC_CODES], y=areas_1, name=str(Y1), marker_color="#ef4444")
    
    areas_2035 = [np.sum(lulc_2035 == k) * PIXEL_AREA / 1e6 for k in VALID_LULC_CODES]
    fig_fut_chart.add_bar(x=[C.LULC_NAMES[k] for k in VALID_LULC_CODES], y=areas_2035, name="2035 (Projected)", marker_color="#10b981")
    
    fig_fut_chart.update_layout(
        barmode="group",
        yaxis_title="Area (km²)",
        xaxis_title="Land Cover Class",
        height=400,
        margin=dict(t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(family="Times New Roman", size=15)
    )
    st.plotly_chart(fig_fut_chart, use_container_width=True)