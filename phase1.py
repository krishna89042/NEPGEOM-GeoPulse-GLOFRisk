"""
Phase 1 — Lake Filtering & Basin Selection
==========================================
CHANGES FROM ORIGINAL:
  - Relaxed Area filter: 0.05 → 0.02 km² (captures smaller but dangerous lakes)
  - Relaxed Elevation filter: 3500 → 4000 m (removes low-elevation noise,
    keeps high-altitude glacial lakes including Khumbu/Thame area)
  - Added Type filter: only moraine-dammed and supraglacial lakes
    (scientifically the most GLOF-prone; E(o) = end-of-glacier, also valid)
  - Added geometry validity check + repair
  - Added CRS reproject instead of just set_crs (handles cases where CRS
    is defined but wrong)
  - Added outlet point export (centroid per lake) — needed for Phase 5
  - Cleaner plots with basin colour coding
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# -------------------------------------------------
# CONFIG — edit these if needed
# -------------------------------------------------

SHAPEFILE_PATH = "/home/shubham/Desktop/geothon/data/data/GL_3basins_2015.shp"
OUTPUT_GEOJSON  = "filtered_glacial_lakes.geojson"
OUTPUT_OUTLETS  = "lake_outlets.geojson"   # centroids for Phase 5

# Filter thresholds
MIN_AREA_KM2   = 0.02
MIN_ELEVATION  = 4000

# Lake types to keep (check your .dbf for exact strings)
# Updated Lake types to match ICIMOD abbreviations:
# E(o) = End-of-glacier erosion, M(o) = Moraine-dammed (outwash), M(e) = Moraine-dammed (end), etc.
VALID_TYPES = ["E(o)", "M(o)", "M(e)", "E(c)", "M(l)"]
# -------------------------------------------------
# 1. LOAD
# -------------------------------------------------

gdf = gpd.read_file(SHAPEFILE_PATH)
print(f"\nTotal lakes in dataset: {len(gdf)}")
print(f"Columns: {list(gdf.columns)}")
print(f"CRS: {gdf.crs}")
print(f"Geometry types:\n{gdf.geom_type.value_counts()}")
print(f"Empty geometries: {gdf.geometry.is_empty.sum()}")
print(f"Null geometries:  {gdf.geometry.isna().sum()}")

# -------------------------------------------------
# 2. CRS — ensure EPSG:4326
# -------------------------------------------------

if gdf.crs is None:
    print("WARNING: No CRS found, assuming EPSG:4326")
    gdf = gdf.set_crs(epsg=4326)
elif gdf.crs.to_epsg() != 4326:
    print(f"Reprojecting from {gdf.crs} to EPSG:4326")
    gdf = gdf.to_crs(epsg=4326)

# -------------------------------------------------
# 3. GEOMETRY REPAIR
# -------------------------------------------------

# Drop null geometries
gdf = gdf[~gdf.geometry.isna()].copy()

# Fix invalid geometries (self-intersections etc.)
invalid = ~gdf.geometry.is_valid
if invalid.sum() > 0:
    print(f"Fixing {invalid.sum()} invalid geometries with buffer(0)")
    gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].buffer(0)

# -------------------------------------------------
# 4. FILTER
# -------------------------------------------------

print(f"\nUnique Type values: {gdf['Type'].unique()}")

# Area and elevation filter
size_elev = (
    (gdf["Area"] >= MIN_AREA_KM2) &
    (gdf["Elevation"] >= MIN_ELEVATION)
)

# Type filter — graceful fallback if column has different values
if gdf["Type"].isin(VALID_TYPES).sum() > 0:
    type_filter = gdf["Type"].isin(VALID_TYPES)
else:
    print("WARNING: No matching Type values found — skipping type filter")
    type_filter = pd.Series(True, index=gdf.index)

filtered = gdf[size_elev & type_filter].copy()

print(f"\nAfter filtering:")
print(f"  Lakes retained: {len(filtered)}")
print(f"  Basin counts:\n{filtered['Basin'].value_counts()}")
print(f"  Sub-basin counts:\n{filtered['Sub_Basin'].value_counts()}")
print(f"  Area range: {filtered['Area'].min():.3f} – {filtered['Area'].max():.3f} km²")
print(f"  Elevation range: {filtered['Elevation'].min()} – {filtered['Elevation'].max()} m")

# -------------------------------------------------
# 5. ADD OUTLET POINTS (centroids)
# Fixed: Project to UTM Zone 45N (EPSG:32645) to calculate true meter-based centroids
# -------------------------------------------------

outlets = filtered.copy()

# 1. Project to meters -> 2. Find centroid -> 3. Project back to EPSG:4326
outlets["geometry"] = filtered.to_crs(epsg=32645).geometry.centroid.to_crs(epsg=4326)

outlets["lon"] = outlets.geometry.x
outlets["lat"] = outlets.geometry.y

# -------------------------------------------------
# 6. PLOTS
# -------------------------------------------------

basin_colors = {"Koshi": "#3498db", "Gandaki": "#2ecc71", "Karnali": "#e74c3c"}

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("ICIMOD Glacial Lakes — Koshi / Gandaki / Karnali Basins", fontsize=13)

# All lakes
ax = axes[0]
for basin, color in basin_colors.items():
    sub = gdf[gdf["Basin"] == basin]
    if len(sub):
        sub.boundary.plot(ax=ax, color=color, linewidth=0.4, alpha=0.6)
ax.set_title(f"All lakes (n={len(gdf)})")
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
patches = [mpatches.Patch(color=c, label=b) for b, c in basin_colors.items()]
ax.legend(handles=patches, fontsize=8)

# Filtered lakes
ax = axes[1]
for basin, color in basin_colors.items():
    sub = filtered[filtered["Basin"] == basin]
    if len(sub):
        sub.plot(ax=ax, color=color, alpha=0.7, edgecolor="white", linewidth=0.3)
ax.set_title(f"Filtered lakes (n={len(filtered)})")
ax.set_xlabel("Longitude")
ax.legend(handles=patches, fontsize=8)

plt.tight_layout()
plt.savefig("phase1_lakes.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: phase1_lakes.png")

# -------------------------------------------------
# 7. EXPORT
# -------------------------------------------------

filtered.to_file(OUTPUT_GEOJSON, driver="GeoJSON")
outlets.to_file(OUTPUT_OUTLETS, driver="GeoJSON")

print(f"Saved: {OUTPUT_GEOJSON}  ({len(filtered)} lakes)")
print(f"Saved: {OUTPUT_OUTLETS}  (outlet centroids for Phase 5)")
print("\nPhase 1 complete.")