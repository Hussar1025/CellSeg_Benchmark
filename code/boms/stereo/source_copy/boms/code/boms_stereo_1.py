#!/usr/bin/env python
"""
BOMS × Stereo-seq — E14.5(E1S3) / E16.5(E2S6) / E16.5(E2S7)

===== This version's fix: sweep() previously hard-coded the wrong grid size =====
The old sweep() hard-coded `cfg.grid or 3`, ignoring the correct n_split that
plan_patches() had already computed in the --check stage. With top-500 genes,
E14.5/E2S7 n_mol×n_gene both fit in int32, so what is actually needed is
n_split=1 (no tiling), but the hard-coded 3 forced the ROI into 3×3=9 parts to
look for a "top-left corner". Stereo-seq GEM coverage is only 6-9%, so that
corner tile may genuinely contain zero molecules — "0 molecules × 0 genes" came
from exactly that; the error message itself was fine, the caller passed the
wrong argument.

This version unifies the --check / --sweep-hs / full-run paths in main() to use
the same n_split from plan_patches(), with no hard-coded fallback value.

===== Key difference vs. the ProSeg "experimental adaptation" =====
ProSeg's RNA diffusion model relies on exact relative molecule positions, so
expanding multiple UMIs in one bin into "fake molecules" crowded onto the same
coordinate breaks its physical assumption.
BOMS's core is mean-shift density estimation (KDE), which only cares about local
density peaks and is insensitive to position quantization. As long as the grid
resolution (0.5 µm) is far smaller than the bandwidth h_s (expected ~tens of
µm), the quantization error is negligible. This adaptation is better grounded
than ProSeg's, but it is still an approximation, not true single-molecule data,
and must be flagged during evaluation.

===== Reused existing machinery =====
- Each patch checkpoints independently for resume (n_split=1 means one patch).
- Genes are re-indexed locally per patch (avoids C++ backend gene-index OOB).
- The final count_mat is rebuilt from the global gene table.
- overlap + core-region dedup (n_split=1 has no overlap boundary, so the logic
  degrades to a pass-through).

Usage:
  python boms_stereo_1.py --dataset E14.5_E1S3 --check
  python boms_stereo_1.py --dataset E14.5_E1S3 --sweep-hs
  python boms_stereo_1.py --dataset E14.5_E1S3 --h-s 70
  python boms_stereo_1.py --dataset E14.5_E1S3 --status
"""
import os
import time
import gzip
import argparse

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

SEP = "=" * 68
P = lambda *a: print(*a, flush=True)

DATA_DIR = "yourpath/dataset/stereoseq/stereo-seq_data_all"
OUT_BASE = "yourpath/boms_stereo"
BIN_UM = 0.5          # 1 bin = 0.5 µm; convert to µm so h_s matches MERFISH units
INT32_MAX = 2 ** 31
OVERLAP_UM = 30.0     # overlap bandwidth, consistent with the MERFISH version

DATASETS = {
    "E14.5_E1S3": dict(
        gem=f"{DATA_DIR}/E14.5_E1S3_Dorsal_Midbrain_GEM_CellBin.tsv.gz",
        gt_cells=4872),
    "E16.5_E2S6": dict(
        gem=f"{DATA_DIR}/E16.5_E2S6_Dorsal_Midbrain_GEM_CellBin.tsv.gz",
        gt_cells=6229),
    "E16.5_E2S7": dict(
        gem=f"{DATA_DIR}/E16.5_E2S7_Dorsal_Midbrain_GEM_CellBin.tsv.gz",
        gt_cells=7960),
}


# ══════════════════════════════════════════════════════════════════════════
#  Step 1: read GEM, expand UMIs, convert to µm (cached)
# ══════════════════════════════════════════════════════════════════════════
def load_expanded(dataset, top_genes, force=False):
    ckdir = f"{OUT_BASE}/{dataset}/ckpt"
    os.makedirs(ckdir, exist_ok=True)
    cache = f"{ckdir}/expanded_top{top_genes}.npz"
    if os.path.exists(cache) and not force:
        d = np.load(cache, allow_pickle=True)
        P(f"  ⚡ [reuse] expanded cache: {len(d['x']):,} molecules  {len(d['genes'])} genes")
        return d["x"], d["y"], d["gi"], list(d["genes"])

    gem_path = DATASETS[dataset]["gem"]
    P(f"  read GEM {os.path.basename(gem_path)} ...")
    opener = gzip.open if gem_path.endswith(".gz") else open
    skip = 0
    with opener(gem_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                skip += 1
            else:
                break
    gem = pd.read_csv(gem_path, sep="\t", skiprows=skip)
    ren = {}
    for c in gem.columns:
        cl = c.lower()
        if cl in ("geneid", "gene"):
            ren[c] = "gene"
        elif cl == "x":
            ren[c] = "x"
        elif cl == "y":
            ren[c] = "y"
        elif cl in ("midcounts", "midcount", "count"):
            ren[c] = "count"
    gem = gem.rename(columns=ren)
    if "count" not in gem.columns:
        gem["count"] = 1
    P(f"    raw {len(gem):,} rows  genes {gem.gene.nunique():,}")

    if top_genes > 0:
        top = gem.gene.value_counts().head(top_genes).index
        gem = gem[gem.gene.isin(top)]
        P(f"    after top-{top_genes} genes {len(gem):,} rows")

    P(f"  expand UMIs (each count becomes one molecule; far less impact on "
      f"BOMS mean-shift than on ProSeg's diffusion model, see header note)...")
    rep = gem.loc[gem.index.repeat(gem["count"].clip(1).astype(int))]
    genes = sorted(rep.gene.unique())
    g2i = {g: i for i, g in enumerate(genes)}
    gi = rep.gene.map(g2i).values.astype(np.int32)
    x = (rep.x.values.astype(np.float64)) * BIN_UM
    y = (rep.y.values.astype(np.float64)) * BIN_UM
    P(f"    expanded to {len(x):,} molecules (µm coords)")

    np.savez_compressed(cache, x=x, y=y, gi=gi,
                        genes=np.array(genes, dtype=object))
    return x, y, gi, genes


def plan_patches(n_mol, n_gene, force_grid=None):
    """
    Compute the patch grid size so no tile overflows int32.
    ★ This is the ONLY place n_split is computed — the --check / --sweep-hs /
      full-run paths all call this function instead of each keeping its own value.
    """
    if force_grid:
        P(f"    using forced grid {force_grid}×{force_grid}")
        return force_grid
    prod = n_mol * n_gene
    n_split = 1
    while (n_mol / (n_split ** 2)) * n_gene >= INT32_MAX * 0.9:
        n_split += 1
        if n_split > 8:
            break
    P(f"    n_mol×n_gene = {prod:,}  →  need {n_split}×{n_split} = "
      f"{n_split**2} patches")
    return n_split


def patch_bounds(x, y, n_split, overlap):
    x0, x1 = float(x.min()), float(x.max())
    y0, y1 = float(y.min()), float(y.max())
    xs = np.linspace(x0, x1, n_split + 1)
    ys = np.linspace(y0, y1, n_split + 1)
    out = []
    for i in range(n_split):
        for j in range(n_split):
            out.append(dict(
                i=i, j=j,
                rx0=max(x0, xs[i]-overlap), rx1=min(x1, xs[i+1]+overlap),
                ry0=max(y0, ys[j]-overlap), ry1=min(y1, ys[j+1]+overlap),
                cx0=xs[i], cx1=xs[i+1], cy0=ys[j], cy1=ys[j+1],
            ))
    return out


# ══════════════════════════════════════════════════════════════════════════
#  Step 2: single patch (with checkpoint)
# ══════════════════════════════════════════════════════════════════════════
def run_patch(pb, x, y, gi, ckdir, cfg, force=False):
    ck = f"{ckdir}/patch_{pb['i']}_{pb['j']}_hs{cfg.h_s}.npz"
    if os.path.exists(ck) and not force:
        d = np.load(ck, allow_pickle=True)
        P(f"  ⚡ patch({pb['i']},{pb['j']}) [reuse]: {len(d['cell_loc']):,} cells")
        return ck

    m = ((x >= pb["rx0"]) & (x < pb["rx1"]) &
         (y >= pb["ry0"]) & (y < pb["ry1"]))
    X, Y, GI = x[m], y[m], gi[m]
    n = len(X)
    if n < 1000:
        P(f"  patch({pb['i']},{pb['j']}) only {n} molecules, skip")
        np.savez_compressed(ck, cell_loc=np.zeros((0, 2)),
                            seg=np.zeros(0, np.int32), gi=GI, x=X, y=Y,
                            core=np.array([pb["cx0"], pb["cx1"],
                                          pb["cy0"], pb["cy1"]]))
        return ck

    uniq_g = np.unique(GI)
    remap = {g: k for k, g in enumerate(uniq_g)}
    G_local = np.array([remap[g] for g in GI], dtype=np.int64)
    prod = n * len(uniq_g)
    P(f"  patch({pb['i']},{pb['j']}): {n:,} molecules × {len(uniq_g)} genes  "
      f"n×g={prod:,}  {'✓' if prod < INT32_MAX else '✘ still overflows, need finer grid'}")
    if prod >= INT32_MAX:
        return None

    from boms import run_boms
    t0 = time.time()
    try:
        _, seg, _, cell_loc, _ = run_boms(
            X, Y, G_local, epochs=cfg.epochs, h_s=cfg.h_s, h_r=cfg.h_r,
            K=cfg.K,
            x_min=float(X.min())-1, x_max=float(X.max())+1,
            y_min=float(Y.min())-1, y_max=float(Y.max())+1)
    except Exception as e:
        P(f"    ✘ run_boms failed: {type(e).__name__}: {e}")
        return None

    seg = np.asarray(seg).astype(np.int32)
    cell_loc = np.asarray(cell_loc, dtype=np.float64)
    P(f"    ✓ {len(cell_loc):,} cells  ({time.time()-t0:.0f}s) → ckpt")
    np.savez_compressed(ck, cell_loc=cell_loc, seg=seg, gi=GI, x=X, y=Y,
                        core=np.array([pb["cx0"], pb["cx1"],
                                       pb["cy0"], pb["cy1"]]))
    return ck


# ══════════════════════════════════════════════════════════════════════════
#  Step 3: merge (rebuild count_mat from the global gene table)
# ══════════════════════════════════════════════════════════════════════════
def merge(ckpts, genes, cfg, out_path):
    n_gene = len(genes)
    P(f"\n{SEP}\nmerge {len(ckpts)} patches (global gene table {n_gene})\n{SEP}")

    all_loc, all_x, all_y, all_gi, all_seg = [], [], [], [], []
    offset = 0
    for ck in ckpts:
        d = np.load(ck, allow_pickle=True)
        loc, seg = d["cell_loc"], d["seg"]
        if len(loc) == 0:
            continue
        cx0, cx1, cy0, cy1 = d["core"]
        keep = ((loc[:, 0] >= cx0) & (loc[:, 0] < cx1) &
                (loc[:, 1] >= cy0) & (loc[:, 1] < cy1))
        old2new = np.full(len(loc)+1, -1, dtype=np.int64)
        kept_idx = np.where(keep)[0]
        old2new[kept_idx+1] = np.arange(len(kept_idx)) + offset
        seg_new = np.where(seg > 0, old2new[np.clip(seg, 0, len(loc))], -1)
        ok = seg_new >= 0
        all_loc.append(loc[keep]); all_seg.append(seg_new[ok])
        all_x.append(d["x"][ok]); all_y.append(d["y"][ok])
        all_gi.append(d["gi"][ok])
        offset += int(keep.sum())
        P(f"  {os.path.basename(ck)[:36]:38} {len(loc):>7,} → "
          f"core {int(keep.sum()):>7,}")

    if not all_loc:
        P("  ✘ no valid patch"); return None

    cell_loc = np.vstack(all_loc)
    seg = np.concatenate(all_seg)
    tx_x, tx_y = np.concatenate(all_x), np.concatenate(all_y)
    tx_gi = np.concatenate(all_gi)
    n_cell = len(cell_loc)
    P(f"\n  after merge {n_cell:,} cells   assigned molecules {len(seg):,}")

    P(f"  rebuild count_mat ({n_cell:,} × {n_gene}) ...")
    count_mat = csr_matrix(
        (np.ones(len(seg), np.int32), (seg, tx_gi)),
        shape=(n_cell, n_gene)).toarray().astype(np.int32)

    per_cell = np.asarray(count_mat.sum(axis=1)).ravel()
    keep = per_cell >= cfg.min_size
    if keep.sum() < n_cell:
        remap = np.full(n_cell, -1, np.int64)
        remap[np.where(keep)[0]] = np.arange(int(keep.sum()))
        seg2 = remap[seg]; ok = seg2 >= 0
        seg, tx_x, tx_y, tx_gi = seg2[ok], tx_x[ok], tx_y[ok], tx_gi[ok]
        cell_loc, count_mat = cell_loc[keep], count_mat[keep]
        P(f"  min_size≥{cfg.min_size} filter: {n_cell:,} → {len(cell_loc):,}")

    np.savez_compressed(
        out_path, cell_loc=cell_loc, seg=seg.astype(np.int32),
        x=tx_x, y=tx_y, gene=np.array(genes, dtype=object)[tx_gi],
        gene_names=np.array(genes, dtype=object), count_mat=count_mat,
        h_s=cfg.h_s, h_r=cfg.h_r, K=cfg.K, epochs=cfg.epochs)

    n = len(cell_loc)
    gt = DATASETS[cfg.dataset]["gt_cells"]
    P(f"\n{SEP}\n  final {n:,} cells   GT {gt:,}   ratio {n/gt:.3f}")
    P(f"  → {out_path}")
    return out_path


# ══════════════════════════════════════════════════════════════════════════
#  h_s sweep ★ this version's fix: no more hard-coded grid size
# ══════════════════════════════════════════════════════════════════════════
def sweep(x, y, gi, dataset, cfg, n_split):
    """
    ★ Fix: previously this hard-coded `patch_bounds(x, y, cfg.grid or 3, ...)`,
      ignoring the correct n_split that plan_patches() already computed in the
      --check stage. Stereo-seq GEM coverage is only 6-9%, so after forcibly
      tiling a 3×3 grid the "top-left" tile may really be empty (0 molecules ×
      0 genes) — that is the root cause of the previous
      "zero-size array to reduction operation" error, not a bug in the
      algorithm but a missing step in argument passing.

      Now n_split is computed by plan_patches() in main() and passed in; when
      n_split=1 (--check already confirmed no tiling is needed), it sweeps the
      full data directly instead of fabricating a possibly nonexistent
      "top-left" tile.
    """
    if n_split <= 1:
        P(f"\n  n_split=1 (--check confirmed no tiling needed), "
          f"sweep all {len(x):,} molecules directly")
        X, Y, GI = x, y, gi
    else:
        pbs = patch_bounds(x, y, n_split, OVERLAP_UM)
        pb = pbs[0]
        m = ((x >= pb["rx0"]) & (x < pb["rx1"]) &
             (y >= pb["ry0"]) & (y < pb["ry1"]))
        X, Y, GI = x[m], y[m], gi[m]
        P(f"\n  sweep top-left patch({pb['i']},{pb['j']}): {len(X):,} molecules")

    if len(X) == 0:
        P(f"  ✘ sweep region is empty (this time with correct n_split={n_split}; "
          f"if still empty, the ROI bounds themselves are wrong and need separate investigation)")
        return

    uniq_g = np.unique(GI)
    remap = {g: k for k, g in enumerate(uniq_g)}
    G = np.array([remap[g] for g in GI], dtype=np.int64)
    n_patches_total = max(n_split ** 2, 1)
    target = DATASETS[dataset]["gt_cells"] // n_patches_total

    P(f"\n{SEP}\nh_s sweep ({len(X):,} molecules × {len(uniq_g)} genes)")
    P(f"target ≈ {target:,} (GT {DATASETS[dataset]['gt_cells']:,} / "
      f"{n_patches_total} tiles)\n{SEP}")

    from boms import run_boms
    rows = []
    for hs in cfg.sweep_values:
        t0 = time.time()
        try:
            _, _, _, loc, _ = run_boms(
                X, Y, G, epochs=cfg.sweep_epochs, h_s=hs, h_r=cfg.h_r,
                K=cfg.K, x_min=float(X.min())-1, x_max=float(X.max())+1,
                y_min=float(Y.min())-1, y_max=float(Y.max())+1)
            n = len(loc)
            flag = "  ←" if 0.7*target < n < 1.4*target else ""
            P(f"  h_s={hs:>6.1f}  {n:>8,} cells  vs target{n-target:>+8,}  "
              f"({time.time()-t0:.0f}s){flag}")
            rows.append((hs, n))
        except Exception as e:
            P(f"  h_s={hs:>6.1f}  FAIL {type(e).__name__}: {str(e)[:40]}")

    if rows:
        df = pd.DataFrame(rows, columns=["h_s", "cells"])
        df["gap"] = (df.cells - target).abs()
        b = df.sort_values("gap").iloc[0]
        P(f"\n  closest: h_s={b.h_s} → {int(b.cells):,} cells"
          f" (target {target:,})")
        P(f"  full run: --h-s {b.h_s}")
        df.drop(columns=["gap"]).to_csv(
            f"{OUT_BASE}/{dataset}/hs_sweep.csv", index=False)
    else:
        P(f"\n  ✘ all h_s failed, check the errors above")


def show_status(dataset):
    ckdir = f"{OUT_BASE}/{dataset}/ckpt"
    out_path = f"{OUT_BASE}/{dataset}/boms_{dataset}.npz"
    P(f"\n{SEP}\n{dataset} checkpoint status\n{SEP}")
    if os.path.isdir(ckdir):
        for f in sorted(os.listdir(ckdir)):
            full = os.path.join(ckdir, f)
            sz = os.path.getsize(full) / 1e6
            P(f"  {sz:9.2f} MB   {f}")
    if os.path.exists(out_path):
        d = np.load(out_path, allow_pickle=True)
        gt = DATASETS[dataset]["gt_cells"]
        n = len(d['cell_loc'])
        P(f"\n  ✓ final result: {n:,} cells  GT {gt:,}  ratio {n/gt:.3f}")
    else:
        P(f"\n  no final result yet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--top-genes", type=int, default=500)
    ap.add_argument("--grid", type=int, default=None,
                    help="force N×N grid; auto-computed if omitted")
    ap.add_argument("--h-s", type=float, default=70.0)
    ap.add_argument("--h-r", type=float, default=0.3)
    ap.add_argument("--K", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--min-size", type=int, default=0)
    ap.add_argument("--sweep-hs", action="store_true")
    ap.add_argument("--sweep-values", type=float, nargs="+",
                    default=[30, 50, 70, 100, 150])
    ap.add_argument("--sweep-epochs", type=int, default=10)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--status", action="store_true")
    cfg = ap.parse_args()

    if cfg.status:
        show_status(cfg.dataset)
        return

    outdir = f"{OUT_BASE}/{cfg.dataset}"
    ckdir = f"{outdir}/ckpt"
    os.makedirs(ckdir, exist_ok=True)
    out_path = f"{outdir}/boms_{cfg.dataset}.npz"

    P(SEP); P(f"BOMS × Stereo-seq {cfg.dataset}"); P(SEP)
    P(f"  GT {DATASETS[cfg.dataset]['gt_cells']:,}")

    x, y, gi, genes = load_expanded(cfg.dataset, cfg.top_genes, cfg.force)
    # ★ the only place n_split is computed; --check / --sweep-hs / full run all use it
    n_split = plan_patches(len(x), len(genes), cfg.grid)

    if cfg.check:
        P(f"\n  --check: expect {n_split}×{n_split}={n_split**2} patches")
        return

    if cfg.sweep_hs:
        sweep(x, y, gi, cfg.dataset, cfg, n_split)
        return

    pbs = patch_bounds(x, y, n_split, OVERLAP_UM)
    P(f"\n  {n_split}×{n_split}={len(pbs)} patches  h_s={cfg.h_s}")
    ckpts = []
    for pb in pbs:
        ck = run_patch(pb, x, y, gi, ckdir, cfg, cfg.force)
        if ck:
            ckpts.append(ck)
    if not ckpts:
        P("  ✘ no successful patch"); return
    merge(ckpts, genes, cfg, out_path)


if __name__ == "__main__":
    main()
