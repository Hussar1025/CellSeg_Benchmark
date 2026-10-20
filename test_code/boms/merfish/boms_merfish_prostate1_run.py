#!/usr/bin/env python
"""
BOMS × MERFISH mouse brain - checkpoint/resume version

===== What changed vs the old version =====

1. count_mat inhomogeneous shape (old version crashed at line 216)
   Root cause: each patch did its own pd.factorize(genes); patch A 640 genes,
       patch B 649 genes → cnt_p[cid-1] has unequal lengths → np.array() blows up
   Fix: still factorize locally within each patch (feed BOMS C++, avoids gene-index
       out-of-bounds), but rebuild count_mat from seg via the [global gene table]
       → all patches equal length

2. checkpoint/resume
   The old version ran all patches and only crashed at the save step, wasting hours.
   Now each patch is saved to ckpt/patch_i_j.npz as soon as it finishes; reruns auto-skip.

3. reshape bug (float32 precision made fov_ind lose a few points)
   run_boms explicitly passes x_min/x_max/y_min/y_max ±1 margin.

===== Why split into patches =====
ROI has 6,062,241 transcripts × 649 genes = 3.93e9 > int32 max 2.147e9
After a 2×2 split each block is ~1.5M × 649 = 9.8e8 ✓

Usage:
  python boms_merfish_ckpt.py --sweep-hs          # sweep h_s (top-left patch)
  python boms_merfish_ckpt.py --h-s 70            # full run (resumable)
  python boms_merfish_ckpt.py --h-s 70 --force    # ignore ckpt and recompute
  python boms_merfish_ckpt.py --status            # show progress only
"""
import os, sys, time, glob, argparse
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

SEP = "=" * 68

TX_CSV = "yourpath/dataset/merfish_prostate1/detected_transcripts.csv"
OUT_DIR  = "yourpath/boms_merfish_prostate1"
CKPT_DIR = f"{OUT_DIR}/ckpt"
TX_CACHE = f"{CKPT_DIR}/roi_transcripts.npz"
FINAL    = f"{OUT_DIR}/boms_merfish.npz"

# ROI identical to GeneSegNet / ProSeg / UCS
ROI_X0, ROI_X1 = 5355.7, 6435.7
ROI_Y0, ROI_Y1 = 5298.3, 6378.3
GT_CELLS = 11347

N_SPLIT = 2          # 2×2 = 4 patches
OVERLAP = 30.0       # µm, prevents boundary cells from being cut
INT32_MAX = 2 ** 31

os.makedirs(CKPT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
#  Step 1: ROI transcripts (cached)
# ══════════════════════════════════════════════════════════════════════════
def load_roi(force=False):
    if os.path.exists(TX_CACHE) and not force:
        d = np.load(TX_CACHE, allow_pickle=True)
        gn = list(d["gene_names"])
        print(f"  ⚡ reuse transcript cache: {len(d['x']):,} entries  {len(gn)} genes")
        return d["x"], d["y"], d["gi"], gn

    print(f"  reading transcripts (3.6 GB, filter ROI in chunks, ~3-5 min)...")
    t0 = time.time()
    chunks, total = [], 0
    for ck in pd.read_csv(TX_CSV, chunksize=2_000_000,
                          usecols=["global_x", "global_y", "gene"]):
        total += len(ck)
        sub = ck[(ck.global_x >= ROI_X0) & (ck.global_x < ROI_X1) &
                 (ck.global_y >= ROI_Y0) & (ck.global_y < ROI_Y1)]
        if len(sub):
            chunks.append(sub)
        print(f"    read {total:,}", end="\r", flush=True)
    tx = pd.concat(chunks, ignore_index=True)
    print(f"\n    total {total:,} → in ROI {len(tx):,}  ({time.time()-t0:.0f}s)")

    # ★ global gene table: prerequisite for count_mat to concatenate
    gene_names = sorted(tx.gene.unique())
    g2i = {g: i for i, g in enumerate(gene_names)}
    gi = tx.gene.map(g2i).values.astype(np.int32)
    print(f"    global gene table {len(gene_names)} genes")

    x = tx.global_x.values.astype(np.float64)
    y = tx.global_y.values.astype(np.float64)
    np.savez_compressed(TX_CACHE, x=x, y=y, gi=gi,
                        gene_names=np.array(gene_names, dtype=object))
    print(f"    ✓ cached → {os.path.basename(TX_CACHE)}")
    return x, y, gi, gene_names


def patch_bounds():
    """2×2 grid: read range with overlap + core range without overlap"""
    xs = np.linspace(ROI_X0, ROI_X1, N_SPLIT + 1)
    ys = np.linspace(ROI_Y0, ROI_Y1, N_SPLIT + 1)
    out = []
    for i in range(N_SPLIT):
        for j in range(N_SPLIT):
            out.append(dict(
                i=i, j=j,
                # read range (extended by overlap)
                rx0=max(ROI_X0, xs[i] - OVERLAP),
                rx1=min(ROI_X1, xs[i+1] + OVERLAP),
                ry0=max(ROI_Y0, ys[j] - OVERLAP),
                ry1=min(ROI_Y1, ys[j+1] + OVERLAP),
                # core range (for dedup: keep only cells whose centroid falls in core)
                cx0=xs[i], cx1=xs[i+1], cy0=ys[j], cy1=ys[j+1],
            ))
    return out


# ══════════════════════════════════════════════════════════════════════════
#  Step 2: single patch (with ckpt)
# ══════════════════════════════════════════════════════════════════════════
def run_patch(pb, x, y, gi, n_gene_global, cfg, force=False):
    ck = f"{CKPT_DIR}/patch_{pb['i']}_{pb['j']}_hs{cfg.h_s}.npz"
    if os.path.exists(ck) and not force:
        d = np.load(ck, allow_pickle=True)
        print(f"  ⚡ patch({pb['i']},{pb['j']}) reuse ckpt: "
              f"{len(d['cell_loc']):,} cells")
        return ck

    m = ((x >= pb["rx0"]) & (x < pb["rx1"]) &
         (y >= pb["ry0"]) & (y < pb["ry1"]))
    X, Y, GI = x[m], y[m], gi[m]
    n = len(X)
    if n < 1000:
        print(f"  patch({pb['i']},{pb['j']}) only {n} transcripts, skip")
        np.savez_compressed(ck, cell_loc=np.zeros((0, 2)),
                            seg=np.zeros(0, np.int32),
                            gi=GI, x=X, y=Y, n_tx=n)
        return ck

    # ★ local factorize for BOMS: C++ indexes by gene value internally,
    #   passing global indices directly (with gaps) may go out of bounds
    uniq_g = np.unique(GI)
    remap = {g: k for k, g in enumerate(uniq_g)}
    G_local = np.array([remap[g] for g in GI], dtype=np.int64)
    prod = n * len(uniq_g)
    print(f"  patch({pb['i']},{pb['j']}): {n:,} transcripts × {len(uniq_g)} genes  "
          f"n×g={prod:,}  {'✓' if prod < INT32_MAX else '✘ overflow!'}",
          flush=True)
    if prod >= INT32_MAX:
        print(f"    ✘ exceeds int32, increase N_SPLIT")
        return None

    from boms import run_boms
    t0 = time.time()
    try:
        _, seg, _, cell_loc, _ = run_boms(
            X, Y, G_local,
            epochs=cfg.epochs, h_s=cfg.h_s, h_r=cfg.h_r, K=cfg.K,
            # ★ explicit bounds ±1: prevents float32 precision from losing fov_ind points → reshape blowup
            x_min=float(X.min()) - 1, x_max=float(X.max()) + 1,
            y_min=float(Y.min()) - 1, y_max=float(Y.max()) + 1)
    except Exception as e:
        print(f"    ✘ run_boms failed: {type(e).__name__}: {e}")
        return None

    seg = np.asarray(seg).astype(np.int32)
    cell_loc = np.asarray(cell_loc, dtype=np.float64)
    print(f"    ✓ {len(cell_loc):,} cells  ({time.time()-t0:.0f}s)  "
          f"→ ckpt", flush=True)

    # ★ save global gene indices gi, not BOMS count_mat (root of length mismatch)
    np.savez_compressed(ck, cell_loc=cell_loc, seg=seg, gi=GI, x=X, y=Y,
                        n_tx=n, core=np.array([pb["cx0"], pb["cx1"],
                                               pb["cy0"], pb["cy1"]]))
    return ck


# ══════════════════════════════════════════════════════════════════════════
#  Step 3: merge (rebuild count_mat from global gene table)
# ══════════════════════════════════════════════════════════════════════════
def merge(ckpts, gene_names, cfg):
    n_gene = len(gene_names)
    print(f"\n{SEP}\nmerge {len(ckpts)} patches (global gene table {n_gene})\n{SEP}")

    all_loc, all_x, all_y, all_gi, all_seg = [], [], [], [], []
    offset = 0
    for ck in ckpts:
        d = np.load(ck, allow_pickle=True)
        loc, seg = d["cell_loc"], d["seg"]
        if len(loc) == 0:
            continue
        cx0, cx1, cy0, cy1 = d["core"]

        # keep only cells whose centroid falls in core (overlap dedup)
        keep = ((loc[:, 0] >= cx0) & (loc[:, 0] < cx1) &
                (loc[:, 1] >= cy0) & (loc[:, 1] < cy1))
        old2new = np.full(len(loc) + 1, -1, dtype=np.int64)
        kept_idx = np.where(keep)[0]
        old2new[kept_idx + 1] = np.arange(len(kept_idx)) + offset   # seg 1-based

        seg_new = np.where(seg > 0, old2new[np.clip(seg, 0, len(loc))], -1)
        ok = seg_new >= 0

        all_loc.append(loc[keep])
        all_seg.append(seg_new[ok])
        all_x.append(d["x"][ok]); all_y.append(d["y"][ok])
        all_gi.append(d["gi"][ok])
        offset += int(keep.sum())
        print(f"  {os.path.basename(ck)[:34]:36} "
              f"{len(loc):>7,} → core {int(keep.sum()):>7,}")

    if not all_loc:
        print("  ✘ no valid patch"); return None

    cell_loc = np.vstack(all_loc)
    seg = np.concatenate(all_seg)
    tx_x = np.concatenate(all_x); tx_y = np.concatenate(all_y)
    tx_gi = np.concatenate(all_gi)
    n_cell = len(cell_loc)
    print(f"\n  after merge {n_cell:,} cells   assigned transcripts {len(seg):,}")

    # ★ count_mat: rebuilt from global gene table, all patches naturally equal length
    print(f"  rebuild count_mat ({n_cell:,} × {n_gene}) ...")
    count_mat = csr_matrix(
        (np.ones(len(seg), np.int32), (seg, tx_gi)),
        shape=(n_cell, n_gene)).toarray().astype(np.int32)

    # min_size filter
    per_cell = np.asarray(count_mat.sum(axis=1)).ravel()
    keep = per_cell >= cfg.min_size
    if keep.sum() < n_cell:
        remap = np.full(n_cell, -1, np.int64)
        remap[np.where(keep)[0]] = np.arange(int(keep.sum()))
        seg2 = remap[seg]
        ok = seg2 >= 0
        seg, tx_x, tx_y, tx_gi = seg2[ok], tx_x[ok], tx_y[ok], tx_gi[ok]
        cell_loc, count_mat = cell_loc[keep], count_mat[keep]
        print(f"  min_size≥{cfg.min_size} filter: {n_cell:,} → "
              f"{len(cell_loc):,} cells")

    np.savez_compressed(
        FINAL,
        cell_loc=cell_loc, seg=seg.astype(np.int32),
        x=tx_x, y=tx_y, gene=np.array(gene_names, dtype=object)[tx_gi],
        gene_names=np.array(gene_names, dtype=object),
        count_mat=count_mat,
        h_s=cfg.h_s, h_r=cfg.h_r, K=cfg.K, epochs=cfg.epochs)

    n = len(cell_loc)
    print(f"\n{SEP}")
    print(f"  final {n:,} cells   GT {GT_CELLS:,}   ratio {n/GT_CELLS:.3f}")
    print(f"  transcripts/cell median {np.median(np.asarray(count_mat.sum(1)).ravel()):.0f}")
    print(f"  → {FINAL}")
    print(f"\n  evaluation: python eval_boms_merfish.py")
    return FINAL


# ══════════════════════════════════════════════════════════════════════════
#  h_s sweep (top-left patch only)
# ══════════════════════════════════════════════════════════════════════════
def sweep(x, y, gi, cfg):
    pb = patch_bounds()[0]
    m = ((x >= pb["rx0"]) & (x < pb["rx1"]) &
         (y >= pb["ry0"]) & (y < pb["ry1"]))
    X, Y, GI = x[m], y[m], gi[m]
    uniq_g = np.unique(GI)
    remap = {g: k for k, g in enumerate(uniq_g)}
    G = np.array([remap[g] for g in GI], dtype=np.int64)
    target = GT_CELLS // (N_SPLIT ** 2)

    print(f"\n{SEP}")
    print(f"h_s sweep (top-left patch {len(X):,} transcripts × {len(uniq_g)} genes)")
    print(f"  epochs={cfg.sweep_epochs} (full run {cfg.epochs}, cell count drops 10-20% more)")
    print(f"  single-patch target ≈ {target:,} (full GT {GT_CELLS:,} / 4)")
    print(SEP)
    print(f"  {'h_s':>7} {'cells':>10} {'vs target':>10} {'time':>8}")
    print("  " + "-" * 40)

    from boms import run_boms
    rows = []
    for hs in cfg.sweep_values:
        t0 = time.time()
        try:
            _, _, _, loc, _ = run_boms(
                X, Y, G, epochs=cfg.sweep_epochs, h_s=hs,
                h_r=cfg.h_r, K=cfg.K,
                x_min=float(X.min())-1, x_max=float(X.max())+1,
                y_min=float(Y.min())-1, y_max=float(Y.max())+1)
            n = len(loc)
            flag = "  ←" if 0.7*target < n < 1.4*target else ""
            print(f"  {hs:>7.1f} {n:>10,} {n-target:>+10,} "
                  f"{time.time()-t0:>7.0f}s{flag}", flush=True)
            rows.append((hs, n))
        except Exception as e:
            print(f"  {hs:>7.1f} {'FAIL':>10}  {type(e).__name__}: "
                  f"{str(e)[:34]}", flush=True)

    if rows:
        df = pd.DataFrame(rows, columns=["h_s", "cells"])
        df["gap"] = (df.cells - target).abs()
        b = df.sort_values("gap").iloc[0]
        print(f"\n  closest: h_s={b.h_s}  →  {int(b.cells):,} "
              f"(4 blocks total ≈ {int(b.cells)*4:,}, GT {GT_CELLS:,})")
        print(f"  full run: python {os.path.basename(sys.argv[0])} "
              f"--h-s {b.h_s}")
        df.drop(columns=["gap"]).to_csv(f"{OUT_DIR}/hs_sweep.csv", index=False)


def show_status(cfg):
    print(f"\n{SEP}\ncheckpoint status\n{SEP}")
    print(f"  transcript cache: "
          f"{'✓ ' + os.path.basename(TX_CACHE) if os.path.exists(TX_CACHE) else '✗'}")
    for pb in patch_bounds():
        ck = f"{CKPT_DIR}/patch_{pb['i']}_{pb['j']}_hs{cfg.h_s}.npz"
        if os.path.exists(ck):
            d = np.load(ck, allow_pickle=True)
            print(f"  patch({pb['i']},{pb['j']}) ✓ {len(d['cell_loc']):>7,} cells")
        else:
            print(f"  patch({pb['i']},{pb['j']}) ✗ not done")
    print(f"  final result: {'✓' if os.path.exists(FINAL) else '✗'}  {FINAL}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h-s", type=float, default=70.0)
    ap.add_argument("--h-r", type=float, default=0.3)
    ap.add_argument("--K", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--min-size", type=int, default=0,
                    help="min transcripts per cell")
    ap.add_argument("--sweep-hs", action="store_true")
    ap.add_argument("--sweep-values", type=float, nargs="+",
                    default=[30, 50, 70, 100, 150])
    ap.add_argument("--sweep-epochs", type=int, default=10)
    ap.add_argument("--force", action="store_true", help="ignore ckpt and recompute")
    ap.add_argument("--status", action="store_true")
    cfg = ap.parse_args()

    print(SEP); print("BOMS × MERFISH mouse brain (checkpoint/resume)"); print(SEP)
    print(f"  ROI x[{ROI_X0},{ROI_X1}] y[{ROI_Y0},{ROI_Y1}] µm  "
          f"GT {GT_CELLS:,}")
    print(f"  ckpt dir: {CKPT_DIR}")

    if cfg.status:
        show_status(cfg); return

    if os.path.exists(FINAL) and not cfg.force:
        d = np.load(FINAL, allow_pickle=True)
        print(f"\n  ✓ existing final result: {len(d['cell_loc']):,} cells")
        print(f"    add --force to recompute")
        return

    x, y, gi, gene_names = load_roi(force=False)

    if cfg.sweep_hs:
        sweep(x, y, gi, cfg); return

    print(f"\n{SEP}")
    print(f"run {N_SPLIT}×{N_SPLIT}={N_SPLIT**2} patches  "
          f"h_s={cfg.h_s} epochs={cfg.epochs}  overlap={OVERLAP}µm")
    print(SEP)
    ckpts = []
    for pb in patch_bounds():
        ck = run_patch(pb, x, y, gi, len(gene_names), cfg, force=cfg.force)
        if ck:
            ckpts.append(ck)
    if not ckpts:
        print("\n  ✘ no successful patch"); return
    merge(ckpts, gene_names, cfg)


if __name__ == "__main__":
    main()
