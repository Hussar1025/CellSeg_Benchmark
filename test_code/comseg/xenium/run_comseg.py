import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from comseg import dataset as ds
from comseg import dictionary


# =========================================================
# Paths
# =========================================================

XENIUM_DIR = Path("/data/qiuyijia/dataset/pancreas_cancer")
TRANSCRIPT_PATH = XENIUM_DIR / "transcripts.parquet"

OUTDIR = Path("/data/qiuyijia/comseg/pancreas_output")
INPUT_DIR = OUTDIR / "comseg_input_csv"
RUN_DIR = OUTDIR / "balanced_fast_15nn"

OUTDIR.mkdir(exist_ok=True)
INPUT_DIR.mkdir(exist_ok=True)
RUN_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Parameters
# =========================================================

QV_MIN = 20


# =========================================================
# Read transcripts
# =========================================================

print(f"Reading: {TRANSCRIPT_PATH}")

if not TRANSCRIPT_PATH.exists():
    raise FileNotFoundError(f"Cannot find: {TRANSCRIPT_PATH}")

df = pd.read_parquet(TRANSCRIPT_PATH)

print(f"Raw transcripts: {len(df):,}")
print(df.columns.tolist())


# =========================================================
# Rename Xenium columns
# =========================================================

rename_map = {}

if "x_location" in df.columns:
    rename_map["x_location"] = "x"

if "y_location" in df.columns:
    rename_map["y_location"] = "y"

if "feature_name" in df.columns:
    rename_map["feature_name"] = "gene"

df = df.rename(columns=rename_map)

required = ["x", "y", "gene"]

for col in required:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")


# =========================================================
# QC filtering
# =========================================================

print("Filtering transcripts...")

if "qv" in df.columns:
    df = df[df["qv"] >= QV_MIN].copy()

df["gene"] = df["gene"].astype(str)

bad_patterns = [
    "NegControl",
    "BLANK",
    "Unassigned",
    "Deprecated",
]

keep = np.ones(len(df), dtype=bool)

for p in bad_patterns:
    keep &= ~df["gene"].str.contains(
        p,
        case=False,
        na=False
    )

df = df[keep].copy()

df = df[["x", "y", "gene"]].dropna()

df["x"] = df["x"].astype(float)
df["y"] = df["y"].astype(float)

df = df[
    np.isfinite(df["x"]) &
    np.isfinite(df["y"])
].copy()


# =========================================================
# Remove extreme outliers
# =========================================================

x_low, x_high = np.percentile(df["x"], [0.01, 99.99])
y_low, y_high = np.percentile(df["y"], [0.01, 99.99])

df = df[
    (df["x"] >= x_low) &
    (df["x"] <= x_high) &
    (df["y"] >= y_low) &
    (df["y"] <= y_high)
].copy()

print(f"Filtered transcripts: {len(df):,}")
print(f"Genes retained: {df['gene'].nunique():,}")


# =========================================================
# Save CSV for ComSeg
# =========================================================

csv_path = INPUT_DIR / "Xenium_1.csv"

if not csv_path.exists():
    df.to_csv(csv_path, index=False)

print(f"ComSeg CSV: {csv_path}")


# =========================================================
# Fast but still high-quality config
# =========================================================

config = {

    "dict_scale": {
        "x": 1.0,
        "y": 1.0,
        "z": 1.0
    },

    "mean_cell_diameter": 12,

    "gene_column": "gene",

    # transcript-only
    "prior_name": None,

    # no downsampling
    "sampling": False,
    "sampling_size": len(df),

    # FAST SETTINGS
    "k_nearest_neighbors": 15,
    "edge_max_length": 5,

    # segmentation
    "max_cell_radius": 35,
    "alpha": 0.5,
    "min_rna_per_cell": 30,

    "allow_disconnected_polygon": False,

    # IMPORTANT
    "norm_vector": False,

    # IMPORTANT
    "overwrite": False,

    "compute_centroid": False,

    "disable_tqdm": False,
}


print("\n============================")
print("CONFIG")
print(json.dumps(config, indent=2))
print("============================")


# =========================================================
# Build dataset
# =========================================================

dataset = ds.ComSegDataset(
    path_dataset_folder=str(INPUT_DIR),
    dict_scale=config["dict_scale"],
    mean_cell_diameter=config["mean_cell_diameter"],
    gene_column=config["gene_column"],
    prior_name=config["prior_name"],
    disable_tqdm=config["disable_tqdm"],
)


# =========================================================
# Compute graph
# =========================================================

print("\nComputing edge weights...")

dataset.compute_edge_weight(config=config)


# =========================================================
# Run ComSeg
# =========================================================

print("\nRunning ComSeg...")

comseg_dict = dictionary.ComSegDict(
    dataset=dataset,
    mean_cell_diameter=config["mean_cell_diameter"],
    disable_tqdm=config["disable_tqdm"],
)

comseg_dict.run_all(config=config)


# =========================================================
# Export result
# =========================================================

print("\nExporting results...")

adata, json_dict = comseg_dict.anndata_from_comseg_result(
    config=config
)

adata_path = RUN_DIR / "balanced_fast_15nn.h5ad"
obs_path = RUN_DIR / "balanced_fast_15nn_obs.csv"
qc_path = RUN_DIR / "balanced_fast_15nn_qc.json"

adata.write(adata_path)

adata.obs.to_csv(obs_path)

rna_per_cell = np.asarray(
    adata.X.sum(axis=1)
).ravel()

qc = {
    "n_cells": int(adata.n_obs),
    "n_genes": int(adata.n_vars),
    "mean_rna_per_cell": float(np.mean(rna_per_cell)),
    "median_rna_per_cell": float(np.median(rna_per_cell)),
    "p10_rna_per_cell": float(np.percentile(rna_per_cell, 10)),
    "p90_rna_per_cell": float(np.percentile(rna_per_cell, 90)),
    "min_rna_per_cell": float(np.min(rna_per_cell)),
    "max_rna_per_cell": float(np.max(rna_per_cell)),
}

with open(qc_path, "w") as f:
    json.dump(qc, f, indent=2)

print("\nDONE")
print(json.dumps(qc, indent=2))

print(f"\nSaved h5ad: {adata_path}")
print(f"Saved obs: {obs_path}")
print(f"Saved qc: {qc_path}")
