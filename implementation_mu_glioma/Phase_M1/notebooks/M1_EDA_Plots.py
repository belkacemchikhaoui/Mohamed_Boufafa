#!/usr/bin/env python3
"""
MU-Glioma EDA Plots — Presentation-quality figures
Uses pre-computed tumor_volumes.csv (596 scans, 203 patients)
and clinical_data.csv from Phase M1 pipeline.

Generates 4 plots matching the BraTS presentation style:
  1.  Distribution of Tumor Sizes (WT, TC, ET)
  2.  Temporal Volume Changes Between Timepoints
  3.  Inter-Patient Variability Analysis
  4.  Tumor Locations & Response Classification
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
M1_OUT  = Path("/home/moamed/canada_me/explainable_diseas/implementation_mu_glioma/Phase_M1/outputs")
FIG_DIR = Path("/home/moamed/canada_me/explainable_diseas/presentation_phases/figures")
FIG_DIR.mkdir(exist_ok=True)

VOLUMES_CSV  = M1_OUT / "tumor_volumes.csv"
CLINICAL_CSV = M1_OUT / "clinical_data.csv"

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#F8F9FA",
    "axes.grid":        True,
    "grid.alpha":       0.4,
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "legend.fontsize":  9,
    "figure.dpi":       150,
})
C = {"WT": "#2196F3", "TC": "#FF9800", "ET": "#E91E63"}

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("Loading pre-computed volumes …")
vol = pd.read_csv(VOLUMES_CSV)
clin = pd.read_csv(CLINICAL_CSV)

# Rename for convenience
vol = vol.rename(columns={
    "wt_vol_ml": "WT", "tc_vol_ml": "TC", "et_vol_ml": "ET",
    "centroid_x": "cx", "centroid_y": "cy", "centroid_z": "cz",
})

n_scans    = len(vol)
n_patients = vol["patient_id"].nunique()

print(f"  ✓ {n_scans} scans | {n_patients} patients")
for r in ["WT", "TC", "ET"]:
    print(f"  {r}: mean={vol[r].mean():.1f} mL  median={vol[r].median():.1f} mL  max={vol[r].max():.1f} mL")

assert n_scans == 596,    f"Expected 596 scans, got {n_scans}"
assert n_patients == 203, f"Expected 203 patients, got {n_patients}"

# Add ratios
vol["TC_WT"] = np.where(vol["WT"] > 0, vol["TC"] / vol["WT"], 0)
vol["ET_WT"] = np.where(vol["WT"] > 0, vol["ET"] / vol["WT"], 0)

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Distribution of Tumor Sizes
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/4] Tumor volume distributions …")

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)

for ax, region in zip(axes, ["WT", "TC", "ET"]):
    data = vol[region]
    bins = np.linspace(0, data.max(), 55)
    ax.hist(data, bins=bins, color=C[region], alpha=0.80, edgecolor="white", linewidth=0.5)
    med = data.median()
    mn  = data.mean()
    ax.axvline(med, color="black",   ls="--", lw=1.5, label=f"Median {med:.1f} mL")
    ax.axvline(mn,  color="#444444", ls=":",  lw=1.3, label=f"Mean   {mn:.1f} mL")
    ax.set_title(f"{region} Volume Distribution", fontweight="bold")
    ax.set_xlabel("Volume (mL)")
    ax.set_ylabel("Number of Scans")
    ax.legend(framealpha=0.9)
    ax.set_xlim(left=0)

fig.suptitle(
    f"MU-Glioma: Tumor Volume Distributions  ({n_scans} scans | {n_patients} patients)",
    fontsize=14, fontweight="bold",
)
plt.tight_layout()
for d in [M1_OUT, FIG_DIR]:
    fig.savefig(d / "mu_tumor_volume_distributions.png", dpi=150, bbox_inches="tight")
print("  ✓ mu_tumor_volume_distributions.png")
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Temporal Volume Changes Between Timepoints
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/4] Temporal volume changes …")

pairs = []
for pid, grp in vol.groupby("patient_id"):
    grp = grp.sort_values("timepoint").reset_index(drop=True)
    if len(grp) < 2:
        continue
    for i in range(len(grp) - 1):
        r1, r2 = grp.iloc[i], grp.iloc[i + 1]
        pairs.append({
            "patient_id": pid,
            "delta_WT": r2["WT"] - r1["WT"],
            "delta_TC": r2["TC"] - r1["TC"],
            "delta_ET": r2["ET"] - r1["ET"],
        })

pairs_df = pd.DataFrame(pairs)
n_pairs   = len(pairs_df)
n_lp      = pairs_df["patient_id"].nunique()

grow  = (pairs_df["delta_WT"] >  1).sum()
stab  = ((pairs_df["delta_WT"] >= -1) & (pairs_df["delta_WT"] <= 1)).sum()
shrk  = (pairs_df["delta_WT"] < -1).sum()
print(f"  {n_pairs} pairs from {n_lp} longitudinal patients")
print(f"  Growing >1 mL: {grow} ({100*grow/n_pairs:.0f}%)  |  Stable: {stab} ({100*stab/n_pairs:.0f}%)  |  Shrinking: {shrk} ({100*shrk/n_pairs:.0f}%)")

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

for ax, region in zip(axes, ["WT", "TC", "ET"]):
    col = f"delta_{region}"
    data = pairs_df[col].dropna()          # drop NaNs before percentile
    lo  = np.percentile(data, 2)
    hi  = np.percentile(data, 98)
    clipped = data.clip(lo, hi)
    bins = np.linspace(lo, hi, 55)
    ax.hist(clipped, bins=bins, color=C[region], alpha=0.80, edgecolor="white", lw=0.5)
    ax.axvline(0,             color="black",   ls="-",  lw=1.8, label="No change")
    ax.axvline(data.median(), color="#333333", ls="--", lw=1.5,
               label=f"Median {data.median():.1f} mL")
    ax.set_title(f"\u0394 {region} Between Timepoints", fontweight="bold")
    ax.set_xlabel("Volume Change (mL)")
    ax.set_ylabel("Number of Pairs")
    ax.legend(framealpha=0.9)

# Annotate progression breakdown in WT panel
axes[0].text(
    0.97, 0.95,
    f"↑ Growing:  {grow} ({100*grow/n_pairs:.0f}%)\n"
    f"→ Stable:   {stab} ({100*stab/n_pairs:.0f}%)\n"
    f"↓ Shrinking: {shrk} ({100*shrk/n_pairs:.0f}%)",
    transform=axes[0].transAxes, ha="right", va="top",
    fontsize=8.5, family="monospace",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
)

fig.suptitle(
    f"MU-Glioma: Temporal Volume Changes  ({n_pairs} consecutive pairs | {n_lp} patients)",
    fontsize=14, fontweight="bold",
)
plt.tight_layout()
for d in [M1_OUT, FIG_DIR]:
    fig.savefig(d / "mu_temporal_volume_changes.png", dpi=150, bbox_inches="tight")
print("  ✓ mu_temporal_volume_changes.png")
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Inter-Patient Variability Analysis
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/4] Inter-patient variability …")

pat = vol.groupby("patient_id").agg(
    n_scans   = ("timepoint", "count"),
    mean_WT   = ("WT", "mean"),
    mean_TC   = ("TC", "mean"),
    mean_ET   = ("ET", "mean"),
    mean_ETWT = ("ET_WT", "mean"),
).reset_index()

# Merge clinical
pat = pat.merge(
    clin[["patient_id", "age_at_diagnosis", "primary_diagnosis", "sex"]],
    on="patient_id", how="left"
)

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# ── (a) Scans per patient bar chart
ax = axes[0, 0]
tp_counts = pat["n_scans"].value_counts().sort_index()
bars = ax.bar(tp_counts.index.astype(str), tp_counts.values,
              color="#5C6BC0", edgecolor="white", linewidth=0.8, width=0.6)
for bar, cnt in zip(bars, tp_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
            str(cnt), ha="center", fontsize=9, fontweight="bold", color="#333")
ax.set_xlabel("Number of Timepoints")
ax.set_ylabel("Number of Patients")
ax.set_title("Scans per Patient", fontweight="bold")
ax.set_facecolor("white")

# ── (b) WT volume by primary diagnosis
ax = axes[0, 1]
diag_groups = {
    d: g["mean_WT"].dropna().values
    for d, g in pat.groupby("primary_diagnosis")
    if len(g) >= 3
}
labels = list(diag_groups.keys())
data   = list(diag_groups.values())
palette = ["#42A5F5", "#66BB6A", "#FFA726", "#EF5350", "#AB47BC", "#26A69A"]
bp = ax.boxplot(data, labels=[f"{l}\n(n={len(v)})" for l, v in zip(labels, data)],
                patch_artist=True, showfliers=True, widths=0.55,
                medianprops=dict(color="black", linewidth=1.8))
for patch, color in zip(bp["boxes"], palette):
    patch.set_facecolor(color)
    patch.set_alpha(0.65)
ax.set_ylabel("Mean WT Volume (mL)")
ax.set_title("Mean WT Volume by Diagnosis", fontweight="bold")
ax.tick_params(axis="x", labelsize=8.5)
ax.set_facecolor("white")

# ── (c) Age distribution
ax = axes[1, 0]
ages = pat["age_at_diagnosis"].dropna()
ax.hist(ages, bins=22, color="#26A69A", alpha=0.80, edgecolor="white", lw=0.5)
ax.axvline(ages.median(), color="black", ls="--", lw=1.5,
           label=f"Median: {ages.median():.0f} yrs")
ax.axvline(ages.mean(),   color="#444", ls=":",   lw=1.3,
           label=f"Mean:   {ages.mean():.0f} yrs")
ax.set_xlabel("Age at Diagnosis (years)")
ax.set_ylabel("Number of Patients")
ax.set_title("Age at Diagnosis Distribution", fontweight="bold")
ax.legend()
ax.set_facecolor("white")

# ── (d) ET/WT ratio per scan
ax = axes[1, 1]
etwt = vol["ET_WT"]
ax.hist(etwt, bins=50, color="#E91E63", alpha=0.80, edgecolor="white", lw=0.5)
ax.axvline(etwt.median(), color="black", ls="--", lw=1.5,
           label=f"Median: {etwt.median():.3f}")
ax.axvline(etwt.mean(),   color="#444", ls=":",   lw=1.3,
           label=f"Mean:   {etwt.mean():.3f}")
ax.set_xlabel("ET / WT Ratio (per scan)")
ax.set_ylabel("Number of Scans")
ax.set_title("ET/WT Ratio Distribution", fontweight="bold")
ax.legend()
ax.set_facecolor("white")

fig.suptitle(
    f"MU-Glioma: Inter-Patient Variability  ({n_patients} patients | {n_scans} scans)",
    fontsize=14, fontweight="bold",
)
plt.tight_layout()
for d in [M1_OUT, FIG_DIR]:
    fig.savefig(d / "mu_inter_patient_variability.png", dpi=150, bbox_inches="tight")
print("  ✓ mu_inter_patient_variability.png")
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 4 — Tumor Centroids & Response Classification
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4/4] Tumor centroid locations …")

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# Panel (a): Centroid X histogram
ax = axes[0, 0]
data = vol["cx"].dropna()
ax.hist(data, bins=40, color="#42A5F5", alpha=0.80, edgecolor="white", lw=0.5)
ax.axvline(data.median(), color="black", ls="--", lw=1.5, label=f"Median {data.median():.0f} vx")
ax.set_xlabel("Centroid X — Left / Right (voxel)")
ax.set_ylabel("Number of Scans")
ax.set_title("Centroid X Distribution", fontweight="bold")
ax.legend()
ax.set_facecolor("white")

# Panel (b): Centroid Y histogram
ax = axes[0, 1]
data = vol["cy"].dropna()
ax.hist(data, bins=40, color="#66BB6A", alpha=0.80, edgecolor="white", lw=0.5)
ax.axvline(data.median(), color="black", ls="--", lw=1.5, label=f"Median {data.median():.0f} vx")
ax.set_xlabel("Centroid Y — Anterior / Posterior (voxel)")
ax.set_ylabel("Number of Scans")
ax.set_title("Centroid Y Distribution", fontweight="bold")
ax.legend()
ax.set_facecolor("white")

# Panel (c): Centroid Z histogram
ax = axes[1, 0]
data = vol["cz"].dropna()
ax.hist(data, bins=40, color="#FFA726", alpha=0.80, edgecolor="white", lw=0.5)
ax.axvline(data.median(), color="black", ls="--", lw=1.5, label=f"Median {data.median():.0f} vx")
ax.set_xlabel("Centroid Z — Inferior / Superior (voxel)")
ax.set_ylabel("Number of Scans")
ax.set_title("Centroid Z Distribution", fontweight="bold")
ax.legend()
ax.set_facecolor("white")

# Panel (d): X-Y 2D scatter
ax = axes[1, 1]
ax.scatter(vol["cx"].dropna(), vol["cy"].dropna(),
           s=10, c="#E91E63", alpha=0.45, edgecolors="none")
ax.set_xlabel("Centroid X — Left / Right (voxel)")
ax.set_ylabel("Centroid Y — Anterior / Posterior (voxel)")
ax.set_title("Centroid X–Y (all scans)", fontweight="bold")
ax.set_facecolor("white")

fig.suptitle(
    f"MU-Glioma: Tumor Centroid Locations  ({n_scans} scans | {n_patients} patients)",
    fontsize=14, fontweight="bold",
)
plt.tight_layout()
for d in [M1_OUT, FIG_DIR]:
    fig.savefig(d / "mu_tumor_locations_response.png", dpi=150, bbox_inches="tight")
print("  ✓ mu_tumor_locations_response.png")
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print(f"EDA complete — 4 figures saved to:")
print(f"  {M1_OUT}")
print(f"  {FIG_DIR}")
print("="*65)
