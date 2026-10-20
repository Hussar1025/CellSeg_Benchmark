#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Probe slice B to determine the exact column names / unassigned representation /
join relationship for the Bering main script.
Only pandas+numpy are imported (no Bering), so it is unaffected by GLIBCXX.

Set the two paths below, then: python probe_B.py
Paste ALL output back.
"""
import pandas as pd
import numpy as np

# ====== set your real paths ======
TRANSCRIPTS_B = "yourpath/dataset/pancreas_cancer/transcripts.parquet"
ANNOTATION_B  = "yourpath/dataset/pancreas_cancer/cell_groups.csv"  # your cell_id,group table
# ==============================

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print("=" * 75)
print("[1] transcripts table")
print("=" * 75)
# read only the schema first, to avoid loading everything at once
try:
    import pyarrow.parquet as pq
    schema = pq.read_schema(TRANSCRIPTS_B)
    print("all columns:", list(schema.names))
except Exception as e:
    print("pyarrow schema read failed; falling back to pandas to inspect columns:", e)

# read the full table (for stats only; set nrows if memory is tight)
df = pd.read_parquet(TRANSCRIPTS_B)
print(f"\nrows: {len(df):,}")
print(f"columns: {list(df.columns)}")
print("\nfirst 5 rows:")
print(df.head())
print("\ncolumn dtypes:")
print(df.dtypes)

# find the most likely "cell assignment" columns (names containing cell / segment etc.)
cand_cols = [c for c in df.columns if any(k in c.lower() for k in ["cell", "segment", "assign"])]
print(f"\nsuspected 'cell assignment' columns: {cand_cols}")

for c in cand_cols:
    s = df[c]
    print(f"\n----- column '{c}' -----")
    print(f"  dtype: {s.dtype}")
    print(f"  unique values: {s.nunique(dropna=False)}")
    print(f"  missing (NaN): {s.isna().sum():,}")
    # sample a few values to see what they look like
    sample_vals = s.dropna().unique()[:8]
    print(f"  sample values (up to 8): {list(sample_vals)}")
    # check for unassigned markers like 0 / -1 / 'UNASSIGNED' / ''
    vc = s.value_counts(dropna=False)
    print(f"  top 10 most frequent values:")
    print(vc.head(10).to_string())
    # explicitly probe common unassigned representations
    for marker in [0, -1, "0", "-1", "", "UNASSIGNED", "unassigned", "None", "NA"]:
        try:
            cnt = (s == marker).sum()
            if cnt > 0:
                print(f"    unassigned candidate marker {marker!r}: {cnt:,} rows")
        except Exception:
            pass

# feature / gene columns
feat_cols = [c for c in df.columns if any(k in c.lower() for k in ["feature", "gene", "target", "name"])]
print(f"\nsuspected 'gene/feature' columns: {feat_cols}")
for c in feat_cols[:3]:
    print(f"  '{c}' samples: {list(df[c].dropna().astype(str).unique()[:6])}")

# coordinate columns
coord_cols = [c for c in df.columns if any(k in c.lower() for k in ["x_", "y_", "z_", "_location", "global"]) or c.lower() in ["x","y","z"]]
print(f"\nsuspected coordinate columns: {coord_cols}")

print("\n" + "=" * 75)
print("[2] annotation table (cell_id, group)")
print("=" * 75)
ann = pd.read_csv(ANNOTATION_B)
print(f"rows: {len(ann):,}")
print(f"columns: {list(ann.columns)}")
print("first 5 rows:")
print(ann.head())
ann_id_col = ann.columns[0]
print(f"\nassumed annotation cell_id column: '{ann_id_col}'")
print(f"  sample values: {list(ann[ann_id_col].astype(str).unique()[:8])}")
print(f"  unique cells: {ann[ann_id_col].nunique():,}")

print("\n" + "=" * 75)
print("[3] can the cell_id of the two tables be joined -- most critical")
print("=" * 75)
# for each suspected assignment column, test its intersection with the annotation cell_id
ann_ids = set(ann[ann_id_col].astype(str))
for c in cand_cols:
    tx_ids = set(df[c].dropna().astype(str).unique())
    inter = tx_ids & ann_ids
    print(f"\n  transcripts['{c}'] unique values: {len(tx_ids):,}")
    print(f"  intersection with annotation cell_id: {len(inter):,}")
    if len(tx_ids) > 0:
        print(f"    -> {len(inter)/len(tx_ids)*100:.1f}% of that transcripts column")
    if len(ann_ids) > 0:
        print(f"    -> {len(inter)/len(ann_ids)*100:.1f}% of the annotation cell_id")
    # show a few unmatched examples to help diagnose format differences
    only_tx = list(tx_ids - ann_ids)[:5]
    print(f"    in transcripts but not in annotation: {only_tx}")

print("\n" + "=" * 75)
print("Probe complete. Paste ALL output above back.")
print("=" * 75)
