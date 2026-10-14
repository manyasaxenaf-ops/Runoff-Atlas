"""
config.py  -  THE ONLY PLACE WHERE NAMES AND SETTINGS LIVE
===========================================================
Both preprocess.py and app.py read everything from here.
If you rename a file, change a year, or edit a curve number,
change it HERE and nowhere else.

Folder layout (config.py sits inside src/):

    urban_runoff/
        src/   config.py  preprocess.py  app.py
        data/raw/         <- your downloaded files
        data/processed/   <- created by preprocess.py, read by app.py
"""

from pathlib import Path

# ------------------------------------------------------------------
# 1. FOLDERS  (automatic - no need to edit)
# ------------------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "raw"
PROC = BASE / "data" / "processed"

# ------------------------------------------------------------------
# 2. STUDY SETTINGS
# ------------------------------------------------------------------
YEARS = [2017, 2025]          # first = "before", second = "after"
CRS = "EPSG:32643"            # UTM zone 43N (metres) - suits Delhi
RESOLUTION = 30               # metres. 10 = sharper but heavy, 30 = light

# ------------------------------------------------------------------
# 3. RAW FILE NAMES  (what you download -> put in data/raw/)
# ------------------------------------------------------------------
RAW_BASIN = RAW / "delhi_boundary.shp"          # basin boundary (shapefile)

RAW_LULC = {                                  # land cover, one per year
    2017: RAW / "LULC_2017.tif",
    2025: RAW / "LULC_2025.tif",
}

RAW_RAIN = {                                  # daily rainfall, one per year
    2017: RAW / "Rain_2017.tif",              # (single day OR a stack of days)
    2025: RAW / "Rain_2025.tif",
}

RAW_SOIL = RAW / "HSG_HiHydroSoil.tif"        # hydrologic soil group
RAW_DEM = RAW / "DEM_SRTM.tif"                # elevation

# ------------------------------------------------------------------
# 4. PROCESSED FILE NAMES  (created by preprocess.py, used by app.py)
# ------------------------------------------------------------------
OUT_LULC = {y: PROC / f"LULC_{y}_aligned.tif" for y in YEARS}
OUT_RAIN = {y: PROC / f"Rain_{y}_aligned.tif" for y in YEARS}
OUT_SOIL = PROC / "Soil_aligned.tif"
OUT_DEM = PROC / "DEM_aligned.tif"

# ------------------------------------------------------------------
# 5. RAINFALL SETTING
# ------------------------------------------------------------------
# What to do if the raw rainfall file has MANY bands (one per day):
#   "max"  -> keep the wettest day at every pixel (annual maximum daily rain)
#   5      -> a number: use only that band (e.g. 5 = the 5th day in the file)
# If the file has just one band, this setting makes no difference.
RAIN_MODE = "max"

# ------------------------------------------------------------------
# 6. LAND COVER CODES  (Esri 10 m Land Cover)
# ------------------------------------------------------------------
# If your LULC raster uses different numbers, edit the keys below.
# preprocess.py prints the codes it finds, so you can compare.
LULC_NAMES = {
    1: "Water",
    2: "Trees",
    4: "Flooded vegetation",
    5: "Crops",
    7: "Built-up",
    8: "Bare ground",
    9: "Snow / ice",
    11: "Rangeland",
}

LULC_COLORS = {
    1: "#1A5BAB",
    2: "#358221",
    4: "#87D19E",
    5: "#FFDB5C",
    7: "#ED022A",
    8: "#B8A99A",
    9: "#CFE8F7",
    11: "#C6AD8D",
}

BUILT_CODE = 7                # the code that means "Built-up"

# ------------------------------------------------------------------
# 7. SOIL CODES
# ------------------------------------------------------------------
# Soil raster values 1,2,3,4 = hydrologic soil group A,B,C,D.
# Some datasets use 11-14 for "dual" groups; we turn them into 1-4.
SOIL_REMAP = {11: 1, 12: 2, 13: 3, 14: 4}

# Pixels inside the basin with no soil value get this group (3 = C).
# Set to None if you prefer to leave such pixels empty.
SOIL_FALLBACK = 3

# ------------------------------------------------------------------
# 8. CURVE NUMBER TABLE  (USDA TR-55, average wetness)
# ------------------------------------------------------------------
# land-cover code : (soil A, soil B, soil C, soil D)
# This table lives only here - no CSV file is needed.
#
# Water (1) and Snow / ice (9) are deliberately NOT in this table.
# SCS-CN describes rain falling on LAND. A river or lake is where runoff
# ends up, not a surface that generates it, so giving it CN = 98 made
# the river look like a runoff hotspot. A land-cover code that is missing
# here gets no CN, so those pixels are left empty (transparent) on the
# runoff maps and are ignored in the runoff statistics. They still show
# on the land-cover map and in the area charts.
CN_TABLE = {
    2:  (36, 60, 73, 79),     # Trees
    4:  (68, 79, 86, 89),     # Flooded vegetation
    5:  (67, 78, 85, 89),     # Crops
    7:  (77, 85, 90, 92),     # Built-up
    8:  (77, 86, 91, 94),     # Bare ground
    11: (49, 69, 79, 84),     # Rangeland
}

# ------------------------------------------------------------------
# 9. NO-DATA VALUES USED IN THE PROCESSED FILES
# ------------------------------------------------------------------
INT_NODATA = 255              # land cover and soil
FLOAT_NODATA = -9999.0        # rainfall and elevation
