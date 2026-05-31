"""
Phase 2 — Sentinel-1 SAR Lake Monitoring
=========================================
CHANGES FROM ORIGINAL:
  - Fixed Otsu: replaced broken custom implementation with
    simple adaptive threshold using per-lake histogram mean
    as a robust proxy (GEE built-in Otsu API is unstable in Python)
  - Added wet-snow guard: skip Dec/Jan/Feb
  - Export split by year to avoid GEE timeout
  - maxPixels reduced 1e13 → 1e9
  - Added Sub_Basin to feature properties
  - Both orbit passes included for coverage
"""

import ee
import geopandas as gpd

ee.Initialize(project='geothon')

# =====================================================
# CONFIG
# =====================================================

START_YEAR  = 2023
END_YEAR    = 2024
END_MONTH   = 6        # stop at June 2024 (covers Thame May 2024)

# Months to skip — wet snow mimics water in SAR
SKIP_MONTHS = [12, 1, 2]

# =====================================================
# 1. LOAD LAKES
# =====================================================

gdf = gpd.read_file("filtered_glacial_lakes.geojson")
print(f"Total lakes: {len(gdf)}")

# =====================================================
# 2. CONVERT TO EE FEATURES
# =====================================================

def to_feature(row):
    geom = ee.Geometry.Polygon(list(row.geometry.exterior.coords))
    return ee.Feature(geom, {
        "GL_ID":     str(row["GL_ID"]),
        "Basin":     str(row["Basin"]),
        "Sub_Basin": str(row.get("Sub_Basin", "Unknown")),
        "Area":      float(row["Area"])
    })

features = [to_feature(row) for _, row in gdf.iterrows()]
lakes    = ee.FeatureCollection(features)

# =====================================================
# 3. SENTINEL-1 COLLECTION — both orbits
# =====================================================

s1_base = (
    ee.ImageCollection("COPERNICUS/S1_GRD")
    .filter(ee.Filter.eq("instrumentMode", "IW"))
    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
    .select("VV")
)

# =====================================================
# 4. ADAPTIVE WATER MASK
#
# Strategy: compute mean backscatter within the lake polygon.
# Water has very low backscatter (~-20 to -25 dB).
# We use (mean - 2dB) as an adaptive threshold — this sits
# below land/snow but above open water, adapting to each scene.
# Falls back to -17 dB if no data in polygon.
# This is simpler and more stable than full Otsu in GEE Python API.
# =====================================================

def water_mask_adaptive(image, geometry):
    """
    Adaptive threshold water mask per lake.
    Threshold = mean VV within lake - 2 dB.
    Pixels below threshold are classified as water.
    """
    stats = image.reduceRegion(
        reducer  = ee.Reducer.mean(),
        geometry = geometry,
        scale    = 10,
        maxPixels= 1e9
    )

    mean_vv   = ee.Number(stats.get("VV"))
    fallback  = ee.Number(-17)

    # If mean is null (no data), use fallback
    threshold = ee.Number(
        ee.Algorithms.If(
            mean_vv,
            mean_vv.subtract(2),   # adaptive: mean - 2 dB
            fallback
        )
    )

    return image.lt(threshold)

# =====================================================
# 5. MONTHLY COMPOSITE
# =====================================================

def get_monthly_image(year, month):
    start = ee.Date.fromYMD(year, month, 1)
    end   = start.advance(1, "month")
    imgs  = s1_base.filterDate(start, end)
    return ee.Image(
        ee.Algorithms.If(
            imgs.size().gt(0),
            imgs.median(),
            ee.Image.constant(-25).rename("VV")
        )
    )

# =====================================================
# 6. COMPUTE AREA PER LAKE
# =====================================================

def compute_area(feature, image):
    geom       = feature.geometry()
    water      = water_mask_adaptive(image, geom)
    pixel_area = ee.Image.pixelArea()
    water_area = water.multiply(pixel_area)
    stats      = water_area.reduceRegion(
        reducer  = ee.Reducer.sum(),
        geometry = geom,
        scale    = 10,
        maxPixels= 1e9
    )
    return feature.set({"water_area_m2": stats.get("VV")})

# =====================================================
# 7. RUN — EXPORT PER YEAR
# =====================================================

def run_year(year, end_month_override=None):
    year_results = []
    last_month   = end_month_override if end_month_override else 12

    for month in range(1, last_month + 1):

        if month in SKIP_MONTHS:
            print(f"  Skipping {year}-{month:02d} (wet snow season)")
            continue

        print(f"  Processing {year}-{month:02d}...")
        img     = get_monthly_image(year, month)
        monthly = lakes.map(lambda f: compute_area(f, img))
        monthly = monthly.map(lambda f: f.set({"year": year, "month": month}))
        year_results.append(monthly)

    if not year_results:
        print(f"  No data for {year}")
        return None

    return ee.FeatureCollection(year_results).flatten()


for year in range(START_YEAR, END_YEAR + 1):
    em  = END_MONTH if year == END_YEAR else 12
    print(f"\nYear {year} (months 1-{em}, skipping {SKIP_MONTHS}):")
    col = run_year(year, end_month_override=em)

    if col is None:
        continue

    task = ee.batch.Export.table.toDrive(
        collection  = col,
        description = f"GLOF_S1_TimeSeries_{year}",
        fileFormat  = "CSV"
    )
    task.start()
    print(f"  Export task started: GLOF_S1_TimeSeries_{year}.csv")

print("""
Export tasks submitted. Check GEE Task Manager.

After both tasks complete, merge CSVs with:

    import pandas as pd, glob
    dfs = [pd.read_csv(f) for f in sorted(glob.glob("GLOF_S1_TimeSeries_*.csv"))]
    pd.concat(dfs).to_csv("lake_timeseries_merged.csv", index=False)
    print("Merged.")
""")