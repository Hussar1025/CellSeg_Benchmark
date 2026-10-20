#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnose why two Xenium slices share only 102 genes in their intersection.
Is it a genuinely different panel, or the same gene failing to match due to
case/alias/naming differences?

Only pandas is imported (no Bering/igraph), so it is unaffected by the GLIBCXX error.
Usage: python check_panels.py
"""

import pandas as pd

# ---- change to your two slice paths (defaults pre-filled) ----
A_PATH = "yourpath/dataset/pancreas_train/transcripts.parquet"
B_PATH = "yourpath/dataset/pancreas_cancer/transcripts.parquet"

# control probe prefixes matching run_bering.py (matched via startswith)
CTRL = ("NegControl", "antisense", "BLANK", "Blank", "Unassigned",
        "Deprecated", "Intergenic", "intergenic", "NegPrb", "Neg_", "GenomicControl")


def load_genes(path):
    df = pd.read_parquet(path, columns=["feature_name"])
    s = df["feature_name"].map(
        lambda v: v.decode() if isinstance(v, (bytes, bytearray)) else str(v)
    )
    s = s[~s.str.startswith(CTRL)]
    return set(s.unique())


ga, gb = load_genes(A_PATH), load_genes(B_PATH)
print("=" * 70)
print(f"A gene count: {len(ga)}    B gene count: {len(gb)}")
print(f"[1] raw exact intersection (currently used by run_bering): {len(ga & gb)}")

# ---- case + whitespace normalization, then intersection ----
def norm(g):
    return g.strip().upper()

map_a = {}      # normalized name -> original name (A)
for g in ga:
    map_a.setdefault(norm(g), g)
map_b = {}
for g in gb:
    map_b.setdefault(norm(g), g)

inter_ci = set(map_a) & set(map_b)
print(f"[2] intersection after case+whitespace normalization: {len(inter_ci)}")
print(f"    -> recovered by case/whitespace alone: {len(inter_ci) - len(ga & gb)}")

# examples recovered by normalization (different raw strings, same after normalization)
recovered = [(map_a[k], map_b[k]) for k in inter_ci if map_a[k] != map_b[k]]
print(f"[3] examples recovered by normalization (A original, B original), up to 30:")
for a, b in recovered[:30]:
    print(f"      {a!r:20} <-> {b!r}")
if not recovered:
    print("      (none -- case/whitespace is not the issue)")

# ---- what do the names look like? symbol vs Ensembl vs other ----
print(f"[4] A name samples: {sorted(ga)[:15]}")
print(f"    B name samples: {sorted(gb)[:15]}")

# ---- still unmatched after normalization (suspects: aliases or genuinely different genes) ----
only_a = sorted(set(map_a) - set(map_b))
only_b = sorted(set(map_b) - set(map_a))
print(f"[5] still A-only after normalization: {len(only_a)} -> {only_a[:40]}")
print(f"[6] still B-only after normalization: {len(only_b)} -> {only_b[:40]}")

# also save the full lists to disk
with open("panel_A_only.txt", "w") as f:
    f.write("\n".join(only_a))
with open("panel_B_only.txt", "w") as f:
    f.write("\n".join(only_b))
print("\nfull A-only/B-only lists written to panel_A_only.txt / panel_B_only.txt")
print("=" * 70)
