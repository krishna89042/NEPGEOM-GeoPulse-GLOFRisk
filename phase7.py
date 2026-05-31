"""
Phase 7 — Integrated Risk Intelligence Score
=============================================
Combines:
  - Lake Anomaly Score    (Phase 3) — weight 0.45
  - Agricultural Exposure (Phase 6) — weight 0.35
  - Seasonal Vulnerability(Phase 6) — weight 0.20

Inputs:
  - lake_risk_output.csv        (Phase 3)
  - agricultural_exposure.csv   (Phase 6)
  - flood_corridors.geojson     (Phase 5)

Outputs:
  - final_risk_scores.csv       — ranked lake risk table
  - final_risk_map.geojson      — corridors + risk scores merged
  - phase7_risk.png             — summary visualisation
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

# =====================================================
# CONFIG — weights from roadmap
# =====================================================

W_ANOMALY    = 0.45
W_EXPOSURE   = 0.35
W_SEASONAL   = 0.20

OUTPUT_CSV     = "final_risk_scores.csv"
OUTPUT_GEOJSON = "final_risk_map.geojson"
OUTPUT_PNG     = "phase7_risk.png"

# =====================================================
# 1. LOAD INPUTS
# =====================================================

anomaly  = pd.read_csv("lake_risk_output.csv")
exposure = pd.read_csv("agricultural_exposure.csv")

try:
    corridors = gpd.read_file("flood_corridors.geojson")
    has_corridors = True
except Exception:
    has_corridors = False
    print("WARNING: flood_corridors.geojson not found — skipping spatial output")

print(f"Anomaly data:  {len(anomaly)} lakes")
print(f"Exposure data: {len(exposure)} lakes")

# =====================================================
# 2. NORMALISE SCORES TO [0, 1]
# =====================================================

def minmax_norm(series):
    mn, mx = series.min(), series.max()
    if mx == mn:
        return series * 0 + 0.5   # all equal → neutral
    return (series - mn) / (mx - mn)

# Anomaly: use z_score, clip negatives to 0
anomaly["anomaly_norm"] = minmax_norm(
    anomaly["z_score"].fillna(0).clip(lower=0)
)

# Exposure: use weighted_exposure (cropland × seasonal)
# Merge anomaly + exposure on GL_ID
merged = anomaly.merge(
    exposure[["GL_ID", "cropland_km2", "seasonal_vuln", "weighted_exposure",
              "crop_value_usd", "path_km"]],
    on="GL_ID",
    how="left"
)

# Fill missing exposure with 0 (lakes with no corridor data)
merged["weighted_exposure"] = merged["weighted_exposure"].fillna(0)
merged["cropland_km2"]      = merged["cropland_km2"].fillna(0)
merged["seasonal_vuln"]     = merged["seasonal_vuln"].fillna(0)
merged["crop_value_usd"]    = merged["crop_value_usd"].fillna(0)
merged["path_km"]           = merged["path_km"].fillna(0)

merged["exposure_norm"]  = minmax_norm(merged["weighted_exposure"])
merged["seasonal_norm"]  = minmax_norm(merged["seasonal_vuln"])

# =====================================================
# 3. FINAL RISK SCORE
# =====================================================

merged["final_risk"] = (
    W_ANOMALY  * merged["anomaly_norm"]  +
    W_EXPOSURE * merged["exposure_norm"] +
    W_SEASONAL * merged["seasonal_norm"]
).round(4)

# =====================================================
# 4. RISK TIER
# =====================================================

def risk_tier(score):
    if score >= 0.70:  return "CRITICAL"
    elif score >= 0.50: return "High"
    elif score >= 0.30: return "Moderate"
    else:               return "Low"

merged["risk_tier"] = merged["final_risk"].apply(risk_tier)

# =====================================================
# 5. RANK + OUTPUT TABLE
# =====================================================

output_cols = [
    "GL_ID", "Basin", "Sub_Basin",
    "z_score", "anomaly_norm",
    "cropland_km2", "weighted_exposure", "exposure_norm",
    "seasonal_vuln", "seasonal_norm",
    "final_risk", "risk_tier",
    "crop_value_usd", "path_km",
    "risk_class"
]

final = merged[output_cols].sort_values("final_risk", ascending=False).reset_index(drop=True)
final.index += 1   # rank starts at 1
final.index.name = "rank"

final.to_csv(OUTPUT_CSV)
print(f"\nSaved: {OUTPUT_CSV}")

print("\n── TOP 10 LAKES BY INTEGRATED RISK ──")
print(final[["GL_ID", "Basin", "z_score", "cropland_km2",
             "final_risk", "risk_tier"]].head(10).to_string())

print(f"\n── RISK TIER SUMMARY ──")
print(final["risk_tier"].value_counts())

# =====================================================
# 6. MERGE WITH CORRIDORS → GEOJSON
# =====================================================

if has_corridors:
    top5_ids = final.head(5).reset_index()["GL_ID"].tolist()
    risk_map = corridors[corridors["GL_ID"].isin(top5_ids)].copy()
    risk_map = risk_map.merge(
        final.reset_index()[["GL_ID", "final_risk", "risk_tier",
                              "cropland_km2", "crop_value_usd"]],
        on="GL_ID", how="left"
    )
    risk_map.to_file(OUTPUT_GEOJSON, driver="GeoJSON")
    print(f"\nSaved: {OUTPUT_GEOJSON}  (top-5 corridors with risk scores)")

# =====================================================
# 7. PLOT
# =====================================================

tier_colors = {
    "CRITICAL": "#c0392b",
    "High":     "#e67e22",
    "Moderate": "#f1c40f",
    "Low":      "#2ecc71"
}

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.patch.set_facecolor("#0f1923")
fig.suptitle("Phase 7 — Integrated GLOF Risk Score",
             color="white", fontsize=13, fontweight="bold")

for ax in axes:
    ax.set_facecolor("#1a2634")
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2c3e50")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")

top10     = final.head(10).reset_index()
top10_lbl = [gl[-10:] for gl in top10["GL_ID"]]
bar_cols  = [tier_colors[t] for t in top10["risk_tier"]]

# Panel 1: Final risk score bar
ax1 = axes[0]
ax1.barh(top10_lbl[::-1], top10["final_risk"][::-1],
         color=bar_cols[::-1], edgecolor="none")
ax1.set_xlabel("Final Risk Score (0–1)")
ax1.set_title("Top 10 Lakes — Integrated Risk")
ax1.axvline(0.5, color="white", linewidth=0.8, linestyle=":", alpha=0.5)
ax1.axvline(0.7, color="#c0392b", linewidth=0.8, linestyle=":", alpha=0.5)

# Panel 2: Component breakdown for top 5
ax2 = axes[1]
top5 = final.head(5).reset_index()
x    = np.arange(len(top5))
w    = 0.25
ax2.bar(x - w,   top5["anomaly_norm"],  w, label="Anomaly (0.45)",
        color="#3498db", edgecolor="none")
ax2.bar(x,       top5["exposure_norm"], w, label="Exposure (0.35)",
        color="#27ae60", edgecolor="none")
ax2.bar(x + w,   top5["seasonal_norm"], w, label="Seasonal (0.20)",
        color="#f39c12", edgecolor="none")
ax2.set_xticks(x)
ax2.set_xticklabels([gl[-8:] for gl in top5["GL_ID"]], rotation=30, ha="right")
ax2.set_ylabel("Normalised Score")
ax2.set_title("Top 5 — Component Breakdown")
ax2.legend(facecolor="#2c3e50", labelcolor="white", fontsize=7)
ax2.set_ylim(0, 1.2)

# Panel 3: Risk tier distribution
ax3 = axes[2]
tier_counts = final["risk_tier"].value_counts()
tier_order  = ["CRITICAL", "High", "Moderate", "Low"]
counts      = [tier_counts.get(t, 0) for t in tier_order]
colors_pie  = [tier_colors[t] for t in tier_order]
wedges, texts, autotexts = ax3.pie(
    counts, labels=tier_order, colors=colors_pie,
    autopct="%1.0f%%", startangle=90,
    textprops={"color": "white", "fontsize": 9}
)
for at in autotexts:
    at.set_color("white")
ax3.set_title("Risk Tier Distribution")

plt.tight_layout(pad=2)
plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"Saved: {OUTPUT_PNG}")
print("\nPhase 7 complete. Run step8_dashboard.py next.")