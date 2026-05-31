"""
Phase 3 — Comparative Anomaly Detection
=========================================
CHANGES FROM ORIGINAL:
  - Fixed deprecated fillna(method="ffill") → Series.ffill()
  - Added seasonal decomposition (statsmodels) before growth rate computation
    so that normal seasonal fluctuation doesn't inflate anomaly scores
  - Added Sub_Basin peer grouping (more precise than basin-level)
    with fallback to Basin if sub-basin has < 3 peers
  - Added insufficient-peers flag (lakes with < 3 basin peers get flagged)
  - Added percent growth feature alongside raw slope
  - Added IQR-based outlier clipping before z-score to reduce influence
    of extreme single-month spikes
  - Outputs lake_risk_output.csv with richer columns
  - Outputs top5_lakes.csv for direct use in Phase 5
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# statsmodels optional — graceful fallback if not installed
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("WARNING: statsmodels not installed. Skipping seasonal decomposition.")
    print("Install with: pip install statsmodels")

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

INPUT_CSV   = "lake_timeseries_merged.csv"   # output of Phase 2 merge
OUTPUT_CSV  = "lake_risk_output.csv"
TOP5_CSV    = "top5_lakes.csv"
MIN_MONTHS  = 6    # minimum data points to compute features
PEER_MIN    = 3    # minimum lakes in peer group for z-score to be valid

# ──────────────────────────────────────────────
# 1. LOAD & CLEAN
# ──────────────────────────────────────────────

df = pd.read_csv(INPUT_CSV)
df.columns = df.columns.str.strip()

if ".geo" in df.columns:
    df = df.drop(columns=[".geo"])

df["water_area_m2"] = pd.to_numeric(df["water_area_m2"], errors="coerce")
df = df.sort_values(["GL_ID", "year", "month"]).reset_index(drop=True)

# Deduplicate multiple orbit passes per month
df = (
    df.groupby(["GL_ID", "Basin", "Sub_Basin", "year", "month"], as_index=False)
    .agg({"water_area_m2": "median", "Area": "first"})
)

df["date"] = pd.to_datetime(
    df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)
)

print(f"Loaded: {len(df)} rows, {df['GL_ID'].nunique()} lakes")
print(f"Date range: {df['date'].min().strftime('%Y-%m')} → {df['date'].max().strftime('%Y-%m')}")

# ──────────────────────────────────────────────
# 2. SEASONAL DECOMPOSITION
# ──────────────────────────────────────────────

def deseasonalize(series: pd.Series, period: int = 12) -> pd.Series:
    """
    Remove seasonal component from lake area time series.
    Returns trend + residual (the "anomalous" signal).
    Requires at least 2 full cycles (2*period data points).
    """
    if not HAS_STATSMODELS or len(series) < 2 * period:
        return series  # fallback: use raw series
    try:
        result = seasonal_decompose(
            series, model="additive", period=period,
            extrapolate_trend="freq"
        )
        return result.trend + result.resid
    except Exception:
        return series

# ──────────────────────────────────────────────
# 3. FEATURE ENGINEERING PER LAKE
# ──────────────────────────────────────────────

def compute_lake_features(group):
    group  = group.sort_values("date").reset_index(drop=True)
    raw    = group["water_area_m2"].ffill().bfill()

    if len(raw) < MIN_MONTHS:
        return pd.Series({
            "growth_rate":    np.nan,
            "pct_growth":     np.nan,
            "volatility":     np.nan,
            "max_area_m2":    raw.max(),
            "mean_area_m2":   raw.mean(),
            "n_months":       len(raw),
            "deseasonalized": False
        })

    # Deseasonalize
    series        = deseasonalize(raw)
    deseasonalized = (series is not raw)

    # IQR clip to reduce single-spike influence
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr    = q3 - q1
    series = series.clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

    x     = np.arange(len(series))
    y     = series.values
    slope = np.polyfit(x, y, 1)[0]   # m²/month

    # Percent growth: (last - first) / first
    first     = y[0] if y[0] != 0 else 1e-6
    pct_growth = (y[-1] - y[0]) / abs(first) * 100

    return pd.Series({
        "growth_rate":    slope,
        "pct_growth":     pct_growth,
        "volatility":     float(np.std(y)),
        "max_area_m2":    float(np.max(raw)),
        "mean_area_m2":   float(np.mean(raw)),
        "n_months":       len(raw),
        "deseasonalized": HAS_STATSMODELS and len(raw) >= 24
    })

print("\nComputing lake features...")
lake_features = (
    df.groupby("GL_ID")
    .apply(compute_lake_features, include_groups=False)
    .reset_index()
)

# ──────────────────────────────────────────────
# 4. MERGE METADATA
# ──────────────────────────────────────────────

meta = df[["GL_ID", "Basin", "Sub_Basin", "Area"]].drop_duplicates("GL_ID")
lake_features = lake_features.merge(meta, on="GL_ID", how="left")

# ──────────────────────────────────────────────
# 5. Z-SCORE — SUB-BASIN FIRST, BASIN FALLBACK
# ──────────────────────────────────────────────

def compute_zscore(df_feat, group_col):
    stats = (
        df_feat.groupby(group_col)["growth_rate"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    stats.columns = [group_col, f"{group_col}_mean", f"{group_col}_std", f"{group_col}_count"]
    return stats

sub_stats  = compute_zscore(lake_features, "Sub_Basin")
basin_stats = compute_zscore(lake_features, "Basin")

lake_features = lake_features.merge(sub_stats,  on="Sub_Basin", how="left")
lake_features = lake_features.merge(basin_stats, on="Basin",    how="left")

# Use Sub_Basin stats if ≥ PEER_MIN peers, else fall back to Basin
lake_features["peer_mean"] = np.where(
    lake_features["Sub_Basin_count"] >= PEER_MIN,
    lake_features["Sub_Basin_mean"],
    lake_features["Basin_mean"]
)
lake_features["peer_std"] = np.where(
    lake_features["Sub_Basin_count"] >= PEER_MIN,
    lake_features["Sub_Basin_std"],
    lake_features["Basin_std"]
)
lake_features["peer_group"] = np.where(
    lake_features["Sub_Basin_count"] >= PEER_MIN,
    lake_features["Sub_Basin"],
    lake_features["Basin"]
)
lake_features["insufficient_peers"] = (
    lake_features["Sub_Basin_count"] < PEER_MIN
) & (
    lake_features["Basin_count"] < PEER_MIN
)

lake_features["z_score"] = (
    (lake_features["growth_rate"] - lake_features["peer_mean"])
    / lake_features["peer_std"].replace(0, np.nan)
)

# ──────────────────────────────────────────────
# 6. RISK CLASSIFICATION
# ──────────────────────────────────────────────

def classify(row):
    if row["insufficient_peers"]:
        return "Insufficient Data"
    z = row["z_score"]
    if pd.isna(z):   return "Unknown"
    elif z < 1:      return "Low"
    elif z < 2:      return "Elevated"
    else:            return "High"

lake_features["risk_class"] = lake_features.apply(classify, axis=1)

# ──────────────────────────────────────────────
# 7. OUTPUT
# ──────────────────────────────────────────────

output_cols = [
    "GL_ID", "Basin", "Sub_Basin", "Area",
    "growth_rate", "pct_growth", "volatility",
    "mean_area_m2", "max_area_m2",
    "z_score", "peer_group", "risk_class",
    "n_months", "deseasonalized", "insufficient_peers"
]

output = lake_features[output_cols].sort_values("z_score", ascending=False)
output.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved: {OUTPUT_CSV}")

# Top 5 for Phase 5
top5 = output[output["risk_class"].isin(["High", "Elevated"])].head(5)
top5.to_csv(TOP5_CSV, index=False)
print(f"Saved: {TOP5_CSV}")

print("\n── Risk class summary ──")
print(output["risk_class"].value_counts())

print("\n── Top 10 lakes by z-score ──")
print(
    output[["GL_ID", "Basin", "Sub_Basin", "growth_rate",
            "pct_growth", "z_score", "risk_class"]]
    .head(10).to_string(index=False)
)

# ──────────────────────────────────────────────
# 8. PLOT — Z-SCORE DISTRIBUTION
# ──────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor("#0f1923")

colors = {"Low": "#2ecc71", "Elevated": "#f39c12",
          "High": "#e74c3c", "Unknown": "#bdc3c7",
          "Insufficient Data": "#7f8c8d"}

for ax in axes:
    ax.set_facecolor("#1a2634")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2c3e50")

# Z-score distribution
ax1 = axes[0]
valid = output.dropna(subset=["z_score"])
ax1.hist(valid["z_score"], bins=20, color="#3498db", edgecolor="#1a2634", alpha=0.8)
ax1.axvline(1, color="#f39c12", linestyle="--", linewidth=1.5, label="Elevated (z=1)")
ax1.axvline(2, color="#e74c3c", linestyle="--", linewidth=1.5, label="High (z=2)")
ax1.set_xlabel("Z-Score", color="white")
ax1.set_ylabel("Count", color="white")
ax1.set_title("Z-Score Distribution", color="white", fontweight="bold")
ax1.legend(facecolor="#2c3e50", labelcolor="white", fontsize=9)

# Risk class bar chart
ax2 = axes[1]
counts = output["risk_class"].value_counts()
bar_colors = [colors.get(rc, "#bdc3c7") for rc in counts.index]
ax2.bar(counts.index, counts.values, color=bar_colors, edgecolor="none")
ax2.set_xlabel("Risk Class", color="white")
ax2.set_ylabel("Count", color="white")
ax2.set_title("Risk Classification", color="white", fontweight="bold")
ax2.tick_params(colors="white")

plt.tight_layout()
plt.savefig("phase3_anomaly.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("Saved: phase3_anomaly.png")
print("\nPhase 3 complete.")