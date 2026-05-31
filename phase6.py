"""
Phase 6 — Agricultural Exposure Analysis
==========================================
Inputs:
  - flood_corridors.geojson  (Phase 5 output)
  - ESA WorldCover 2021 (pulled from GEE — 10m resolution)
  - Nepal crop calendar (embedded — FAO + MoALD district data)

For each flood corridor:
  1. Clip ESA WorldCover cropland layer to corridor
  2. Compute cropland area at risk (m²)
  3. Apply seasonal vulnerability score based on current month
     and Nepal crop calendar (rice/maize/wheat)
  4. Compute district-level exposure breakdown

OUTPUT:
  - agricultural_exposure.csv   — per-lake cropland at risk + seasonal score
  - phase6_exposure.png         — visualisation

NOTE: This script has two parts:
  Part A (step6a_worldcover_export.py) — GEE export of WorldCover TIFFs
  Part B (below) — local intersection + seasonal scoring

Run Part A first, download TIFFs, then run this script.
"""

# ══════════════════════════════════════════════════════
# PART A — GEE Export (run this block separately first)
# ══════════════════════════════════════════════════════
"""
Paste and run in a separate script or uncomment below:

import ee, geopandas as gpd
ee.Initialize(project='geothon')

corridors = gpd.read_file("flood_corridors.geojson")
worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")

for _, row in corridors.iterrows():
    gl_id    = row["GL_ID"].replace("/","_")
    bounds   = row.geometry.bounds  # (minx, miny, maxx, maxy)
    region   = ee.Geometry.Rectangle(list(bounds))

    task = ee.batch.Export.image.toDrive(
        image          = worldcover,
        description    = f"WorldCover_{gl_id}",
        fileNamePrefix = f"WorldCover_{gl_id}",
        region         = region,
        scale          = 10,
        crs            = "EPSG:4326",
        fileFormat     = "GeoTIFF",
        maxPixels      = 1e9
    )
    task.start()
    print(f"Export started: WorldCover_{gl_id}.tif")
"""

# ══════════════════════════════════════════════════════
# PART B — Local intersection + seasonal scoring
# ══════════════════════════════════════════════════════

import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.mask
from rasterio.features import geometry_mask
from shapely.geometry import mapping
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# =====================================================
# CONFIG
# =====================================================

CORRIDORS_GEOJSON  = "flood_corridors.geojson"
WORLDCOVER_FOLDER  = "."
OUTPUT_CSV         = "agricultural_exposure.csv"
OUTPUT_PNG         = "phase6_exposure.png"

# ESA WorldCover class value for cropland
CROPLAND_CLASS     = 40   # class 40 = Cropland in WorldCover v200

# Current month — used for seasonal scoring
# Change this to simulate different seasons
CURRENT_MONTH      = datetime.now().month

# =====================================================
# NEPAL CROP CALENDAR
# Source: FAO + Nepal MoALD
# Vulnerability = 1.0 (harvest/growing) → 0.3 (off-season)
# =====================================================

# Format: crop → {month: vulnerability_score}
# 1.0 = peak vulnerability (harvest or late growing — total loss if flooded)
# 0.7 = growing phase (significant loss)
# 0.3 = planting / off-season (partial loss, replanting possible)

CROP_CALENDAR = {
    "rice": {
        1: 0.2, 2: 0.2, 3: 0.3,
        4: 0.4, 5: 0.5, 6: 0.7,   # transplanting Jun
        7: 0.8, 8: 0.9, 9: 1.0,   # growing Jul-Sep
        10: 1.0, 11: 0.7, 12: 0.2  # harvest Oct-Nov
    },
    "maize": {
        1: 0.2, 2: 0.2, 3: 0.4,
        4: 0.6, 5: 0.8, 6: 1.0,   # growing May-Jul
        7: 1.0, 8: 0.9, 9: 0.5,   # harvest Aug-Sep
        10: 0.3, 11: 0.2, 12: 0.2
    },
    "wheat": {
        1: 0.7, 2: 0.8, 3: 1.0,   # growing Jan-Mar
        4: 1.0, 5: 0.5, 6: 0.2,   # harvest Apr-May
        7: 0.2, 8: 0.2, 9: 0.2,
        10: 0.3, 11: 0.4, 12: 0.6  # planting Nov-Dec
    }
}

def seasonal_vulnerability(month):
    """
    Composite seasonal vulnerability score for given month.
    Weighted average: rice 50%, maize 30%, wheat 20%
    (reflects Nepal's crop mix in hill/mountain districts)
    """
    rice   = CROP_CALENDAR["rice"][month]
    maize  = CROP_CALENDAR["maize"][month]
    wheat  = CROP_CALENDAR["wheat"][month]
    return round(0.5 * rice + 0.3 * maize + 0.2 * wheat, 3)

# Precompute all months
monthly_scores = {m: seasonal_vulnerability(m) for m in range(1, 13)}
current_score  = monthly_scores[CURRENT_MONTH]

month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

print(f"Current month: {month_names[CURRENT_MONTH]} "
      f"(seasonal vulnerability = {current_score:.2f})")

# =====================================================
# HELPER: Find WorldCover TIF for a corridor
# =====================================================

def find_worldcover(gl_id):
    safe_id = gl_id.replace("/", "_")
    pattern = os.path.join(WORLDCOVER_FOLDER, f"WorldCover_{safe_id}*.tif")
    matches = glob.glob(pattern)
    if not matches:
        print(f"  WARNING: No WorldCover TIF found for {gl_id}")
        return None
    return matches[0]

# =====================================================
# HELPER: Compute cropland area within corridor
# =====================================================

def compute_cropland_area(tif_path, corridor_geom):
    """
    Mask WorldCover raster to corridor polygon.
    Count pixels with value == CROPLAND_CLASS.
    Return cropland area in km².
    """
    with rasterio.open(tif_path) as src:
        try:
            masked, transform = rasterio.mask.mask(
                src,
                [mapping(corridor_geom)],
                crop=True,
                nodata=0
            )
        except Exception as e:
            print(f"    Mask error: {e}")
            return 0.0

        data = masked[0]  # single band

        # Pixel area in m²
        pixel_w = abs(transform.a)   # degrees
        pixel_h = abs(transform.e)
        # Convert to metres (approximate at Nepal latitude ~28°N)
        px_m2   = (pixel_w * 111320) * (pixel_h * 110540)

        cropland_pixels = np.sum(data == CROPLAND_CLASS)
        cropland_m2     = cropland_pixels * px_m2
        cropland_km2    = cropland_m2 / 1e6

    return round(cropland_km2, 4)

# =====================================================
# MAIN
# =====================================================

corridors = gpd.read_file(CORRIDORS_GEOJSON)
print(f"\nProcessing {len(corridors)} corridors...\n")

results = []

for _, row in corridors.iterrows():
    gl_id     = row["GL_ID"]
    print(f"── {gl_id} ──")

    tif_path = find_worldcover(gl_id)

    if tif_path is None:
        # Fallback: estimate from corridor area if no WorldCover TIF
        # Nepal avg cropland fraction in mountain districts ~8%
        corridor_area_km2 = row.geometry.area * (111 ** 2)  # rough deg² → km²
        cropland_km2      = round(corridor_area_km2 * 0.08, 4)
        print(f"  Using estimated cropland (8% of corridor): {cropland_km2:.3f} km²")
    else:
        cropland_km2 = compute_cropland_area(tif_path, row.geometry)
        print(f"  Cropland at risk: {cropland_km2:.3f} km²")

    # Seasonal score for current month
    vuln_score = current_score

    # Approximate economic exposure
    # Nepal avg rice yield ~2.5 t/ha, price ~NPR 35/kg ≈ USD 0.26/kg
    # Rough crop value: USD 650/ha = USD 65,000/km²
    crop_value_usd = round(cropland_km2 * 65000, 0)

    results.append({
        "GL_ID":              gl_id,
        "Basin":              row["Basin"],
        "Sub_Basin":          row.get("Sub_Basin", ""),
        "z_score":            row["z_score"],
        "risk_class":         row["risk_class"],
        "path_km":            row.get("path_km", 0),
        "cropland_km2":       cropland_km2,
        "seasonal_month":     month_names[CURRENT_MONTH],
        "seasonal_vuln":      vuln_score,
        "crop_value_usd":     crop_value_usd,
        "weighted_exposure":  round(cropland_km2 * vuln_score, 4)
    })

    print(f"  Seasonal vulnerability ({month_names[CURRENT_MONTH]}): {vuln_score:.2f}")
    print(f"  Weighted exposure: {round(cropland_km2 * vuln_score, 4):.4f} km²")
    print(f"  Estimated crop value at risk: ${crop_value_usd:,.0f}")

# =====================================================
# SAVE
# =====================================================

df_out = pd.DataFrame(results).sort_values("weighted_exposure", ascending=False)
df_out.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved: {OUTPUT_CSV}")
print(df_out[["GL_ID", "cropland_km2", "seasonal_vuln",
              "weighted_exposure", "crop_value_usd"]].to_string(index=False))

# =====================================================
# PLOT
# =====================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.patch.set_facecolor("#0f1923")
fig.suptitle("Phase 6 — Agricultural Exposure Analysis", 
             color="white", fontsize=13, fontweight="bold")

for ax in axes:
    ax.set_facecolor("#1a2634")
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2c3e50")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")

labels = [gl[-10:] for gl in df_out["GL_ID"]]

# Panel 1: Cropland at risk
ax1 = axes[0]
bars = ax1.barh(labels, df_out["cropland_km2"],
                color="#27ae60", edgecolor="none")
ax1.set_xlabel("Cropland at Risk (km²)")
ax1.set_title("Cropland Exposure")
ax1.invert_yaxis()

# Panel 2: Seasonal vulnerability across all 12 months
ax2 = axes[1]
months  = list(range(1, 13))
m_names = [month_names[m] for m in months]
scores  = [monthly_scores[m] for m in months]
colors  = ["#e74c3c" if s >= 0.7 else "#f39c12" if s >= 0.5
           else "#2ecc71" for s in scores]
ax2.bar(m_names, scores, color=colors, edgecolor="none")
ax2.axvline(
    month_names[CURRENT_MONTH],
    color="white", linewidth=2, linestyle="--",
    label=f"Now ({month_names[CURRENT_MONTH]})"
)
ax2.set_ylabel("Vulnerability Score")
ax2.set_title("Seasonal Vulnerability Calendar")
ax2.set_ylim(0, 1.1)
ax2.legend(facecolor="#2c3e50", labelcolor="white", fontsize=8)
plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")

# Panel 3: Weighted exposure (cropland × seasonal)
ax3 = axes[2]
wexp_colors = ["#e74c3c" if v >= 0.5 else "#f39c12" if v >= 0.3
               else "#2ecc71" for v in df_out["weighted_exposure"]]
ax3.barh(labels, df_out["weighted_exposure"],
         color=wexp_colors, edgecolor="none")
ax3.set_xlabel("Weighted Exposure (km² × vuln)")
ax3.set_title("Risk-Adjusted Exposure")
ax3.invert_yaxis()

plt.tight_layout(pad=2)
plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"Saved: {OUTPUT_PNG}")
print("\nPhase 6 complete. Run step7_risk_score.py next.")