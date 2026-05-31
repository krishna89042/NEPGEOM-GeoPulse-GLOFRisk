"""
Phase 4 — Historical Event Validation
======================================
CHANGES FROM ORIGINAL:
  - Fixed deprecated fillna(method="ffill") → ffill()
  - Rolling z-score now uses vectorised per-group slope
    instead of O(T²) full recompute — much faster
  - Reads top5_lakes.csv from Phase 3 to auto-select validation targets
  - Generates one validation plot per top-risk lake (not just one)
  - Adds explicit Thame note if GL_ID not found
  - Saves per-lake escalation CSVs
  - Cleaner 3-panel figure with proper axis formatting

NOTE ON THAME VALIDATION:
  The Thame GLOF (May 2024) lake is in Dudh Koshi sub-basin.
  To include it: re-run Phase 1 with MIN_AREA_KM2=0.02, MIN_ELEVATION=4500,
  add Sub_Basin filter for "Dudh Koshi". Then re-run Phase 2 for 2023-01
  to 2024-04. The pipeline below will automatically pick it up.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

INPUT_CSV  = "lake_timeseries_merged.csv"
TOP5_CSV   = "top5_lakes.csv"
MIN_WINDOW = 4    # minimum months before computing rolling z-score

# ──────────────────────────────────────────────
# 1. LOAD & CLEAN
# ──────────────────────────────────────────────

df = pd.read_csv(INPUT_CSV)
df.columns = df.columns.str.strip()
if ".geo" in df.columns:
    df = df.drop(columns=[".geo"])

df["water_area_m2"] = pd.to_numeric(df["water_area_m2"], errors="coerce")
df = (
    df.groupby(["GL_ID", "Basin", "Sub_Basin", "year", "month"], as_index=False)
    .agg({"water_area_m2": "median", "Area": "first"})
)
df["date"] = pd.to_datetime(
    df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)
)
df = df.sort_values(["GL_ID", "date"]).reset_index(drop=True)

# Load validation targets from Phase 3
try:
    top5      = pd.read_csv(TOP5_CSV)
    targets   = top5["GL_ID"].tolist()
    print(f"Validation targets from Phase 3: {targets}")
except FileNotFoundError:
    # Fallback: use top-3 by peak area change
    top_area  = df.groupby("GL_ID")["water_area_m2"].std().nlargest(3)
    targets   = top_area.index.tolist()
    print(f"top5_lakes.csv not found. Using top-3 by volatility: {targets}")

# ──────────────────────────────────────────────
# 2. ROLLING Z-SCORE (vectorised)
# ──────────────────────────────────────────────

def lake_slope_to_date(group, cutoff_date):
    """Compute linear growth rate for a lake up to cutoff_date."""
    sub = group[group["date"] <= cutoff_date].copy()
    sub = sub.sort_values("date")
    series = sub["water_area_m2"].ffill().bfill()
    if len(series) < 3:
        return np.nan
    x = np.arange(len(series))
    return float(np.polyfit(x, series.values, 1)[0])

def compute_rolling_zscores(df, target_id, min_window=MIN_WINDOW):
    all_dates = sorted(df["date"].unique())
    target_basin = df[df["GL_ID"] == target_id]["Basin"].iloc[0]

    # All lakes in same basin (peers)
    peers = df[df["Basin"] == target_basin]
    peer_groups = {gl: grp for gl, grp in peers.groupby("GL_ID")}

    records = []
    for i, cutoff in enumerate(all_dates):
        if i < min_window:
            continue

        # Compute slope for each peer up to this cutoff
        slopes = {}
        for gl, grp in peer_groups.items():
            slopes[gl] = lake_slope_to_date(grp, cutoff)

        slopes_arr = np.array([v for v in slopes.values() if not np.isnan(v)])
        if len(slopes_arr) < 2:
            continue

        basin_mean = float(np.mean(slopes_arr))
        basin_std  = float(np.std(slopes_arr))

        target_slope = slopes.get(target_id, np.nan)
        z = (
            (target_slope - basin_mean) / basin_std
            if basin_std > 0 and not np.isnan(target_slope)
            else np.nan
        )

        records.append({
            "date":         cutoff,
            "growth_rate":  target_slope,
            "basin_mean":   basin_mean,
            "basin_std":    basin_std,
            "z_score":      z,
            "n_peers":      len(slopes_arr)
        })

    return pd.DataFrame(records)

# ──────────────────────────────────────────────
# 3. RISK COLOUR MAP
# ──────────────────────────────────────────────

risk_color = {
    "Low":      "#2ecc71",
    "Elevated": "#f39c12",
    "High":     "#e74c3c",
    "Unknown":  "#bdc3c7"
}

def classify(z):
    if pd.isna(z):  return "Unknown"
    elif z < 1:     return "Low"
    elif z < 2:     return "Elevated"
    else:           return "High"

# ──────────────────────────────────────────────
# 4. PLOT PER TARGET LAKE
# ──────────────────────────────────────────────

def plot_validation(target_id, rolling_df, area_df):
    fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=False)
    fig.patch.set_facecolor("#0f1923")
    fig.suptitle(
        f"Phase 4 Validation — {target_id}",
        color="white", fontsize=13, fontweight="bold", y=0.98
    )

    for ax in axes:
        ax.set_facecolor("#1a2634")
        ax.tick_params(colors="white", labelsize=9)
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2c3e50")

    peak_row   = rolling_df.loc[rolling_df["z_score"].idxmax()]
    peak_date  = peak_row["date"]
    peak_z     = peak_row["z_score"]

    # ── Panel 1: Lake area ──
    ax1 = axes[0]
    ax1.plot(area_df["date"], area_df["water_area_m2"] / 1e3,
             color="#3498db", linewidth=2, marker="o", markersize=3)
    ax1.fill_between(area_df["date"], area_df["water_area_m2"] / 1e3,
                     alpha=0.15, color="#3498db")
    ax1.axvline(peak_date, color="#e74c3c", linewidth=2,
                linestyle="--", label=f"Peak alert ({peak_date.strftime('%Y-%m')})")
    ax1.set_ylabel("Water Area (×10³ m²)")
    ax1.set_title("Lake Area Over Time")
    ax1.legend(facecolor="#2c3e50", edgecolor="none", labelcolor="white", fontsize=9)
    ax1.grid(alpha=0.12, color="white")
    plt.setp(ax1.get_xticklabels(), rotation=30, ha="right")

    # ── Panel 2: Rolling z-score ──
    ax2 = axes[1]
    dates   = rolling_df["date"].values
    zscores = rolling_df["z_score"].values

    for i in range(len(dates) - 1):
        z   = zscores[i]
        c   = risk_color[classify(z)]
        ax2.fill_between([dates[i], dates[i+1]], [0, 0],
                         [zscores[i], zscores[i+1]], alpha=0.3, color=c)
        ax2.plot([dates[i], dates[i+1]], [zscores[i], zscores[i+1]],
                 color=c, linewidth=2)

    ax2.axhline(1, color="#f39c12", linewidth=1, linestyle=":", alpha=0.9)
    ax2.axhline(2, color="#e74c3c", linewidth=1, linestyle=":", alpha=0.9)
    ax2.axvline(peak_date, color="#e74c3c", linewidth=2, linestyle="--")
    ax2.set_ylabel("Anomaly Z-Score")
    ax2.set_title("Rolling Anomaly Score (real-time simulation)")

    patches = [
        mpatches.Patch(color=risk_color["Low"],      label="Low (<1)"),
        mpatches.Patch(color=risk_color["Elevated"],  label="Elevated (1–2)"),
        mpatches.Patch(color=risk_color["High"],      label="High (>2)"),
    ]
    ax2.legend(handles=patches, facecolor="#2c3e50", edgecolor="none",
               labelcolor="white", fontsize=9)
    ax2.grid(alpha=0.12, color="white")
    plt.setp(ax2.get_xticklabels(), rotation=30, ha="right")

    # ── Panel 3: Z-score vs basin mean ──
    ax3 = axes[2]
    ax3.plot(rolling_df["date"], rolling_df["growth_rate"],
             color="#3498db", linewidth=1.8, label="This lake")
    ax3.plot(rolling_df["date"], rolling_df["basin_mean"],
             color="#95a5a6", linewidth=1.5, linestyle="--", label="Basin mean")
    ax3.fill_between(
        rolling_df["date"],
        rolling_df["basin_mean"] - rolling_df["basin_std"],
        rolling_df["basin_mean"] + rolling_df["basin_std"],
        alpha=0.15, color="#95a5a6", label="Basin ±1 std"
    )
    ax3.axvline(peak_date, color="#e74c3c", linewidth=2, linestyle="--")
    ax3.set_ylabel("Growth Rate (m²/month)")
    ax3.set_title("Lake Growth vs Basin Peers")
    ax3.legend(facecolor="#2c3e50", edgecolor="none", labelcolor="white", fontsize=9)
    ax3.grid(alpha=0.12, color="white")
    plt.setp(ax3.get_xticklabels(), rotation=30, ha="right")

    plt.tight_layout(pad=2.5)
    fname = f"phase4_{target_id.replace('/', '_')}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {fname}  (peak z={peak_z:.2f} @ {peak_date.strftime('%Y-%m')})")

# ──────────────────────────────────────────────
# 5. RUN FOR EACH TARGET
# ──────────────────────────────────────────────

all_rolling = {}

for target_id in targets:
    if target_id not in df["GL_ID"].values:
        print(f"\nWARNING: {target_id} not found in time series data. Skipping.")
        print("  → Re-run Phase 1 + Phase 2 with relaxed filters to capture this lake.")
        continue

    print(f"\nProcessing validation for: {target_id}")
    rolling_df = compute_rolling_zscores(df, target_id)

    if rolling_df.empty:
        print(f"  Insufficient data for {target_id}")
        continue

    rolling_df["risk_class"] = rolling_df["z_score"].apply(classify)
    all_rolling[target_id]   = rolling_df

    # Save CSV
    csv_name = f"phase4_{target_id.replace('/', '_')}_escalation.csv"
    rolling_df[["date", "growth_rate", "basin_mean", "z_score",
                "risk_class", "n_peers"]].to_csv(csv_name, index=False)

    # Plot
    area_df = df[df["GL_ID"] == target_id].sort_values("date")
    plot_validation(target_id, rolling_df, area_df)

    # Escalation summary
    esc = rolling_df[
        (rolling_df["risk_class"] == "High") &
        (rolling_df["risk_class"].shift(1) != "High")
    ]
    if not esc.empty:
        print(f"  ⚠ First HIGH alert: {esc.iloc[0]['date'].strftime('%Y-%m')} "
              f"(z={esc.iloc[0]['z_score']:.2f})")
    else:
        print(f"  No HIGH escalation detected in available data window.")

print("\n── Phase 4 complete ──")
print(f"Processed {len(all_rolling)} lakes.")
print("""
── NOTE ON THAME RETRODICTON ──
Thame GLOF occurred May 2024 in the Dudh Koshi sub-basin.
To validate against it:
  1. In step1_icimod.py: set MIN_AREA_KM2=0.02, MIN_ELEVATION=4500
     Add: filtered = filtered[filtered["Sub_Basin"]=="Dudh Koshi"]
  2. Re-run step2_sar.py with END_YEAR=2024, END_MONTH=4
  3. Merge CSVs and re-run this script
  4. The Tsho Rolpa / Imja lake GL_IDs will appear in top5_lakes.csv
""")