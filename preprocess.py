"""
preprocess.py  -  STEP 1
=========================
Takes your raw downloads and makes them ready for the dashboard.

What it does, in plain words:
  1. Reads the basin boundary and puts it in the project CRS (metres).
  2. Builds ONE common grid (same pixel size, same corner) for the basin.
  3. For every layer (LULC 2017, LULC 2025, soil, DEM, rainfall 2017,
     rainfall 2025) it:  re-projects -> resamples onto the common grid
     -> blanks everything outside the basin -> saves a GeoTIFF.
  4. Prints a short check so you can see if something looks wrong.

Run:   python src/preprocess.py
"""

import numpy as np
import rasterio
import geopandas as gpd
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.warp import reproject

import config as C


# ------------------------------------------------------------------
# STEP 1: basin boundary
# ------------------------------------------------------------------
def read_basin():
    basin = gpd.read_file(C.RAW_BASIN)
    if basin.crs is None:
        raise ValueError("Basin file has no CRS. Define it in QGIS first.")
    basin = basin.to_crs(C.CRS)
    return list(basin.geometry)          # list of polygons in metres


# ------------------------------------------------------------------
# STEP 2: the common grid
# ------------------------------------------------------------------
def make_grid(shapes):
    xmin, ymin, xmax, ymax = gpd.GeoSeries(shapes).total_bounds
    r = C.RESOLUTION
    # snap the edges to multiples of the pixel size
    left = np.floor(xmin / r) * r
    bottom = np.floor(ymin / r) * r
    right = np.ceil(xmax / r) * r
    top = np.ceil(ymax / r) * r

    width = int((right - left) / r)
    height = int((top - bottom) / r)
    transform = from_origin(left, top, r, r)

    print(f"Grid: {width} x {height} pixels at {r} m "
          f"(~{width * height * r * r / 1e6:,.0f} km2 box)")
    return transform, width, height


# ------------------------------------------------------------------
# STEP 3: align one layer onto the grid
# ------------------------------------------------------------------
def align(source, dtype, nodata, resampling, **src_info):
    """Re-project `source` onto the common grid and cut to the basin."""
    out = np.full((HEIGHT, WIDTH), nodata, dtype=dtype)
    reproject(
        source=source,
        destination=out,
        dst_transform=TRANSFORM,
        dst_crs=C.CRS,
        dst_nodata=nodata,
        resampling=resampling,
        **src_info,
    )
    out[~INSIDE] = nodata                # blank outside the basin
    return out


def save(path, array, dtype, nodata):
    with rasterio.open(
        path, "w", driver="GTiff", height=HEIGHT, width=WIDTH, count=1,
        dtype=dtype, crs=C.CRS, transform=TRANSFORM, nodata=nodata,
        compress="lzw",
    ) as dst:
        dst.write(array, 1)
    print(f"  saved -> {path.name}")


def coverage(array, nodata):
    """% of basin pixels that actually have data."""
    return 100 * np.sum((array != nodata) & INSIDE) / INSIDE.sum()


# ------------------------------------------------------------------
# Layer-specific helpers
# ------------------------------------------------------------------
def do_lulc(year):
    print(f"\nLULC {year}")
    with rasterio.open(C.RAW_LULC[year]) as src:
        arr = align(rasterio.band(src, 1), "uint8", C.INT_NODATA,
                    Resampling.nearest, src_nodata=src.nodata)
    found = [int(v) for v in np.unique(arr) if v != C.INT_NODATA]
    unknown = [v for v in found if v not in C.CN_TABLE]
    print("  codes found:", {v: C.LULC_NAMES.get(v, "?") for v in found})
    if unknown:
        print(f"  WARNING: codes {unknown} are not in CN_TABLE (config.py) "
              "- those pixels will be left empty.")
    print(f"  coverage of basin: {coverage(arr, C.INT_NODATA):.1f}%")
    save(C.OUT_LULC[year], arr, "uint8", C.INT_NODATA)


def do_soil():
    print("\nSoil (hydrologic soil group)")
    with rasterio.open(C.RAW_SOIL) as src:
        arr = align(rasterio.band(src, 1), "uint8", C.INT_NODATA,
                    Resampling.nearest, src_nodata=src.nodata)
    for old, new in C.SOIL_REMAP.items():      # 11-14 -> 1-4
        arr[arr == old] = new
    found = sorted(int(v) for v in np.unique(arr) if v != C.INT_NODATA)
    print("  soil codes found:", found, "(expected 1,2,3,4 = A,B,C,D)")
    print(f"  coverage of basin: {coverage(arr, C.INT_NODATA):.1f}%")
    save(C.OUT_SOIL, arr, "uint8", C.INT_NODATA)


def do_dem():
    print("\nDEM")
    with rasterio.open(C.RAW_DEM) as src:
        arr = align(rasterio.band(src, 1), "float32", C.FLOAT_NODATA,
                    Resampling.bilinear, src_nodata=src.nodata)
    ok = arr != C.FLOAT_NODATA
    print(f"  elevation: min {arr[ok].min():.0f} m, max {arr[ok].max():.0f} m")
    print(f"  coverage of basin: {coverage(arr, C.FLOAT_NODATA):.1f}%")
    save(C.OUT_DEM, arr, "float32", C.FLOAT_NODATA)


def do_rain(year):
    """Rainfall -> ONE layer in mm (a single storm depth P per pixel)."""
    print(f"\nRainfall {year}")
    with rasterio.open(C.RAW_RAIN[year]) as src:
        n = src.count
        print(f"  bands (days) in file: {n}")

        # pick which bands to use
        if C.RAIN_MODE == "max":
            bands = range(1, n + 1)
        else:
            bands = [int(C.RAIN_MODE)]

        # keep the wettest value seen at each pixel
        best = None
        for b in bands:
            a = src.read(b).astype("float32")
            bad = ~np.isfinite(a) | (a < 0)
            if src.nodata is not None:
                bad |= (a == src.nodata)
            a[bad] = C.FLOAT_NODATA
            best = a if best is None else np.maximum(best, a)

        arr = align(best, "float32", C.FLOAT_NODATA, Resampling.bilinear,
                    src_transform=src.transform, src_crs=src.crs,
                    src_nodata=C.FLOAT_NODATA)

    ok = arr != C.FLOAT_NODATA
    mean_p = arr[ok].mean()
    print(f"  rainfall P: min {arr[ok].min():.0f}, mean {mean_p:.0f}, "
          f"max {arr[ok].max():.0f} mm")
    if mean_p > 300:
        print("  WARNING: mean > 300 mm looks like a yearly/monthly TOTAL. "
              "SCS-CN needs a one-day (storm) depth.")
    print(f"  coverage of basin: {coverage(arr, C.FLOAT_NODATA):.1f}%")
    save(C.OUT_RAIN[year], arr, "float32", C.FLOAT_NODATA)


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
if __name__ == "__main__":
    C.PROC.mkdir(parents=True, exist_ok=True)

    shapes = read_basin()
    TRANSFORM, WIDTH, HEIGHT = make_grid(shapes)

    # True for pixels inside the basin
    INSIDE = geometry_mask(shapes, out_shape=(HEIGHT, WIDTH),
                           transform=TRANSFORM, invert=True)
    print(f"Basin area: {INSIDE.sum() * C.RESOLUTION ** 2 / 1e6:,.1f} km2")

    for y in C.YEARS:
        do_lulc(y)
    do_soil()
    do_dem()
    for y in C.YEARS:
        do_rain(y)

    print("\nDONE. Now run:  streamlit run src/app.py")
