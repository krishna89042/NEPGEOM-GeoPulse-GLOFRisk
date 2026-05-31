"""
Phase 5a — Export NASADEM for Top-5 Lakes via GEE
===================================================
Run this first. It submits GEE export tasks — one DEM GeoTIFF per lake.
After tasks complete, download the TIFFs from Google Drive and run
step5b_routing.py for flow routing and corridor generation.

NASADEM: NASA's reprocessed SRTM with better void-filling.
GEE asset: NASA/NASADEM_HGT/001  — band "elevation" in metres
"""

import ee
import geopandas as gpd
import json

ee.Initialize(project='geothon')

# =====================================================
# CONFIG
# =====================================================

TOP5_CSV       = "top5_lakes.csv"
LAKES_GEOJSON  = "filtered_glacial_lakes.geojson"
DEM_BUFFER_M   = 50000    # 50 km buffer around lake for DEM export
                           # large enough to trace full downstream valley
DEM_SCALE_M    = 30        # NASADEM native resolution

# =====================================================
# 1. LOAD TOP 5 LAKES
# =====================================================

import pandas as pd

top5      = pd.read_csv(TOP5_CSV)
lakes_gdf = gpd.read_file(LAKES_GEOJSON)

# Merge to get geometries
top5_gdf  = lakes_gdf[lakes_gdf["GL_ID"].isin(top5["GL_ID"])].copy()
top5_gdf  = top5_gdf.merge(top5[["GL_ID", "z_score", "risk_class"]], on="GL_ID")

print(f"Exporting DEMs for {len(top5_gdf)} lakes:")
for _, row in top5_gdf.iterrows():
    print(f"  {row['GL_ID']}  z={row['z_score']:.2f}  {row['risk_class']}")

# =====================================================
# 2. NASADEM COLLECTION
# =====================================================

nasadem = ee.Image("NASA/NASADEM_HGT/001").select("elevation")

# =====================================================
# 3. EXPORT ONE DEM PER LAKE
# =====================================================

for _, row in top5_gdf.iterrows():
    gl_id    = row["GL_ID"]
    centroid = row.geometry.centroid
    lon, lat = centroid.x, centroid.y

    # Bounding box: buffer around lake centroid
    region = ee.Geometry.Point([lon, lat]).buffer(DEM_BUFFER_M).bounds()

    safe_id = gl_id.replace("/", "_")

    task = ee.batch.Export.image.toDrive(
        image       = nasadem,
        description = f"NASADEM_{safe_id}",
        fileNamePrefix = f"NASADEM_{safe_id}",
        region      = region,
        scale       = DEM_SCALE_M,
        crs         = "EPSG:4326",
        fileFormat  = "GeoTIFF",
        maxPixels   = 1e9
    )
    task.start()
    print(f"  Export started: NASADEM_{safe_id}.tif")

# =====================================================
# 4. SAVE OUTLET POINTS FOR ROUTING
# =====================================================

# Outlet = centroid of each lake polygon
outlets = top5_gdf.copy()
outlets["outlet_lon"] = outlets.geometry.centroid.x
outlets["outlet_lat"] = outlets.geometry.centroid.y
outlets = outlets[["GL_ID", "Basin", "Sub_Basin", "Area",
                    "z_score", "risk_class",
                    "outlet_lon", "outlet_lat"]]

outlets.to_csv("top5_outlets.csv", index=False)
print("\nSaved: top5_outlets.csv")
print("""
Next steps:
  1. Wait for GEE tasks to complete (5–15 min each)
  2. Download NASADEM_*.tif files from Google Drive
     into the same folder as your scripts
  3. Run: python step5b_routing.py
""")