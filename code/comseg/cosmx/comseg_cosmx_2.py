#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
comseg_cosmx_2.py - ComSeg @ cosmx_2 (CosMx lymph node) driver with checkpoint resume

Why rerun
---------
The previous round (--max-cell-radius-um default 12.0) was bad:
    predicted 3,371 cells / GT 4,898 = 0.688
    1,027 transcripts per cell, effective diameter 27.3 um
    metadata cell diameter 7.86 um -> predicted blobs are 3.5x real cells
    assign_accuracy 0.024, count_pearson 0.032
1,027 per cell equals the whole-FOV average (4,976,788 / 4,898 = 1,016),
so 3,371 blobs split nearly all transcripts - nuclear parts are one-to-one,
but 52% extra-nuclear transcripts were pooled across cells.
Lymph node cells are only a few um apart; a 12 um expansion radius spans 2-3
neighbor layers. In contrast xenium_2 (breast, much larger diameter) used
diameter 12 / radius 20 and got F1 0.964, so this parameter must follow tissue
cell density, not the default.

New params: --mean-cell-diameter-um 8 --max-cell-radius-um 6
    8 um from metadata Area median 3374 px x (0.1203 um/px)^2 -> diameter 7.86 um
    6 um = 0.76 x diameter, stopping expansion before touching neighbor nuclei

Checkpoint resume
--------
The original comseg_cosmx_3.py completion test only checked whether spot_assignment exists:
    if not (tdir/'spot_assignment.csv.gz').exists(): no flag
    else: (tdir/'DONE.flag').touch()
A tile that hit an API error on 08-11 wrote a 6-column table (file exists but no
cell column), so the flag was still written and all later resumes skipped it -
11 of 25 flags were false.

This script rewrites the completion test:
    truly complete = parquet exists and has a cell column and unique cells > 0
Validate immediately after each tile; if invalid, remove its flag so the next run
retries. After an interruption, rerun the same command; finished tiles are skipped.

Usage
----
  python3 comseg_cosmx_2.py --status                     # status only
  python3 comseg_cosmx_2.py --reset                      # clear old results for rerun
  python3 comseg_cosmx_2.py --run                        # resume (repeatable)
  python3 comseg_cosmx_2.py --run --clean-dict           # delete intermediates after run
  python3 comseg_cosmx_2.py --run --max-tiles 2          # validate on 2 tiles first
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import pandas as pd

ROOT = "yourpath/comseg_cosmx_lymph_node/fov48"
RUNNER = "yourpath/comseg/comseg_cosmx_3.py"
FOV = 48
GT_CELLS = 4898
PX_UM = 0.1203                 # napari .zattrs scale_um (previously used 0.12)
CELL_DIAM_UM = 8.0             # metadata Area median 3374 px -> 7.86 um
MAX_RADIUS_UM = 6.0            # 0.76 x diameter
CELL_COLS = ("cell", "cell_index", "comseg", "cell_id", "label",
             "cell_label", "cell_index_pred")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def tile_dirs(root):
    d = os.path.join(root, "comseg_out")
    if not os.path.isdir(d):
        return []
    return sorted(os.path.join(d, x) for x in os.listdir(d)
                  if x.startswith("tile_") and
                  os.path.isdir(os.path.join(d, x)))


def dir_size(p):
    if not os.path.exists(p):
        return 0
    if os.path.isfile(p):
        return os.path.getsize(p)
    return sum(os.path.getsize(os.path.join(r, f))
               for r, _, fs in os.walk(p) for f in fs)


def inspect(td):
    """Truly complete = parquet exists, has a cell column, unique cells > 0."""
    p = os.path.join(td, "spot_assignment.parquet")
    fl = os.path.join(td, "DONE.flag")
    info = dict(flag=os.path.exists(fl),
                flag_sz=os.path.getsize(fl) if os.path.exists(fl) else -1,
                ncol=0, cells=0, assigned=0.0, tx=0,
                dict_gb=dir_size(os.path.join(td, "comseg_dict")) / 1e9)
    if not os.path.exists(p):
        return "not_run", info
    try:
        df = pd.read_parquet(p)
    except Exception as e:
        info["err"] = str(e)[:60]
        return "bad_parquet", info
    info["ncol"] = len(df.columns)
    info["tx"] = len(df)
    cc = next((c for c in CELL_COLS if c in df.columns), None)
    if cc is None:
        return "no_cell_col", info
    v = pd.to_numeric(df[cc], errors="coerce")
    pos = v > 0
    info["cells"] = int(v[pos].nunique())
    info["assigned"] = float(pos.mean())
    if info["cells"] == 0:
        return "zero_cells", info
    return "ok", info


def status(root, verbose=True):
    tds = tile_dirs(root)
    if not tds:
        log(f"  no tile dirs under {root}/comseg_out - run --stage prep first")
        return {}, []
    res, tot_cells, tot_dict = {}, 0, 0.0
    if verbose:
        log(f"{'tile':<12}{'status':<14}{'cols':>4}{'transcripts':>11}"
            f"{'cells':>8}{'assigned':>9}{'dict':>8}")
        log("-" * 68)
    for td in tds:
        st, info = inspect(td)
        res[os.path.basename(td)] = (st, info)
        tot_dict += info["dict_gb"]
        if st == "ok":
            tot_cells += info["cells"]
        if verbose:
            log(f"{os.path.basename(td):<12}{st:<14}{info['ncol']:>4}"
                f"{info['tx']:>11,}{info['cells']:>8,}"
                f"{info['assigned']*100:>8.1f}%{info['dict_gb']:>7.1f}G"
                + ("" if st == "ok" else "  ★"))
    bad = [k for k, (s, _) in res.items() if s != "ok"]
    n_ok = len(tds) - len(bad)
    if verbose:
        log("-" * 68)
        log(f"Done {n_ok}/{len(tds)}   total cells {tot_cells:,} "
            f"(GT {GT_CELLS:,})   intermediates {tot_dict:.0f} GB")
        if bad:
            log(f"pending {len(bad)}: {bad}")
    return res, bad


def reset(root, yes=False):
    """Remove all tile outputs to prepare for a rerun with new params."""
    tds = tile_dirs(root)
    tot = sum(dir_size(t) for t in tds)
    log(f"Will clear {len(tds)} tile dirs, freeing about {tot/1e9:.0f} GB")
    log(f"  {os.path.join(root, 'comseg_out')}")
    if not yes:
        log("  add --yes to execute")
        return
    for t in tds:
        shutil.rmtree(t)
    log(f"  cleared {len(tds)} tile dirs")


def verify_and_fix(root):
    """Delete false flags so the next resume retries those tiles."""
    res, bad = status(root, verbose=False)
    n = 0
    for name in bad:
        st, info = res[name]
        fl = os.path.join(root, "comseg_out", name, "DONE.flag")
        if os.path.exists(fl):
            os.remove(fl)
            n += 1
            log(f"  {name}: {st} -> deleted false flag")
    return n


def run(root, args):
    if not os.path.exists(RUNNER):
        sys.exit(f"✗ runner not found: {RUNNER}")

    log("=" * 68)
    log("Parameters")
    log("=" * 68)
    log(f"  mean_cell_diameter  {args.diameter} um")
    log(f"    (metadata Area median 3374 px x {PX_UM} um/px -> diameter 7.86 um)")
    log(f"  max_cell_radius     {args.radius} um  "
        f"= {args.radius/args.diameter:.2f} x diameter")
    log(f"    (previous round used 12.0 = 1.53x, producing 27.3 um giant blobs)")
    log(f"  tile_px             {args.tile_px}   "
        f"overlap {args.overlap_px} px = {args.overlap_px*PX_UM:.1f} um")
    if args.overlap_px * PX_UM < args.radius:
        log(f"  ⚠ overlap {args.overlap_px*PX_UM:.1f} um < radius "
            f"{args.radius} um; cells crossing seams will be split")

    n_fixed = verify_and_fix(root)
    if n_fixed:
        log(f"\ncleared {n_fixed} false flags")

    res, bad = status(root, verbose=False)
    if not res:
        # Empty comseg_out (e.g. just --reset) does not mean "all done";
        # run prep first to rebuild tile dirs, otherwise the runner has nothing to do.
        log("\nno tile dirs under comseg_out - run prep first:")
        log(f"  python3 -u {RUNNER} --fovs {args.fov} --stage prep \\")
        log(f"      --mean-cell-diameter-um {args.diameter} \\")
        log(f"      --max-cell-radius-um {args.radius}")
        return
    log(f"\npending {len(bad)} / {len(res)} tiles")
    if not bad:
        log("all done")
        return
    if args.max_tiles:
        log(f"  --max-tiles {args.max_tiles}: run only the first {args.max_tiles} this time")

    st = os.statvfs(root)
    free_gb = st.f_bavail * st.f_frsize / 1e9
    log(f"free disk {free_gb:.0f} GB"
        + ("   ⚠ consider --clean-dict" if free_gb < len(bad) * 20 else ""))

    cmd = [sys.executable, "-u", RUNNER,
           "--fovs", str(args.fov), "--stage", "run",
           "--mean-cell-diameter-um", str(args.diameter),
           "--max-cell-radius-um", str(args.radius)]
    if args.tile_px:
        cmd += ["--tile-px", str(args.tile_px)]
    if args.overlap_px:
        cmd += ["--overlap-px", str(args.overlap_px)]
    log("\n$ " + " ".join(cmd))
    log("")
    rc = subprocess.call(cmd, cwd=os.path.dirname(RUNNER))
    log(f"\nrunner exit code {rc}")

    log("\n" + "=" * 68)
    log("post-run check")
    log("=" * 68)
    res2, bad2 = status(root)
    n_fixed2 = verify_and_fix(root)
    if n_fixed2:
        log(f"\nthis round {n_fixed2} tiles failed validation; flags removed; "
            f"rerunning the same command will retry them")

    if args.clean_dict:
        log("\nCleaning comseg_dict for completed tiles "
            "(spot_assignment.parquet saved; dict only used for resume)")
        freed = 0
        for name, (s, _) in res2.items():
            if s != "ok":
                continue
            dd = os.path.join(root, "comseg_out", name, "comseg_dict")
            if os.path.exists(dd):
                n = dir_size(dd)
                shutil.rmtree(dd)
                freed += n
                log(f"  {name}  {n/1e9:.1f} GB")
        log(f"  freed {freed/1e9:.0f} GB")

    ok = [k for k, (s, _) in res2.items() if s == "ok"]
    if len(ok) == len(res2):
        log("\n✓ all done; evaluate:")
        log("  python3 yourpath/comseg/comseg_value_cosmx_2.py")
    else:
        log(f"\n{len(res2)-len(ok)} remaining; rerun the same command to continue")


def quick_check(root):
    """After 1-2 tiles, check effective diameter to see whether params are fixed."""
    import numpy as np
    res, _ = status(root, verbose=False)
    ok = [k for k, (s, _) in res.items() if s == "ok"]
    if not ok:
        log("  no completed tiles yet")
        return
    log(f"\n[Parameter quick check] using {len(ok)} completed tiles")
    frames = []
    for name in ok[:4]:
        p = os.path.join(root, "comseg_out", name,
                         "spot_assignment.parquet")
        frames.append(pd.read_parquet(p))
    d = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["x", "y", "gene"])
    cc = next(c for c in CELL_COLS if c in d.columns)
    v = pd.to_numeric(d[cc], errors="coerce")
    da = d[v > 0]
    g = da.groupby(cc).agg(n=("x", "size"))
    sp = da.groupby(cc).agg(sx=("x", "std"), sy=("y", "std"))
    diam = 2 * np.sqrt(2) * np.sqrt((sp.sx ** 2 + sp.sy ** 2) / 2)
    log(f"  cells {len(g):,}   median tx/cell {g.n.median():.0f}")
    log(f"  effective diameter median {diam.median():.2f} um")
    log(f"  reference: metadata 7.86 um; previous round (radius 12) was 27.3 um")
    if diam.median() < 12:
        log("  ✓ diameter back to a sane range; params fixed")
    elif diam.median() < 20:
        log("  ~ improved but still large; reduce --radius further")
    else:
        log("  ✗ still giant blobs; radius is not the only cause; check "
            "k_nearest_neighbors and min_merge_correlation")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--fov", type=int, default=FOV)
    ap.add_argument("--diameter", type=float, default=CELL_DIAM_UM)
    ap.add_argument("--radius", type=float, default=MAX_RADIUS_UM)
    ap.add_argument("--tile-px", type=int, default=0)
    ap.add_argument("--overlap-px", type=int, default=0)
    ap.add_argument("--max-tiles", type=int, default=0)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="quick-check effective diameter from completed tiles")
    ap.add_argument("--clean-dict", action="store_true")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()

    log("=" * 68)
    log(f"ComSeg @ cosmx_2  FOV {a.fov}   {a.root}")
    log("=" * 68)

    if a.reset:
        reset(a.root, a.yes)
        return
    if a.check:
        quick_check(a.root)
        return
    if a.run:
        run(a.root, a)
        return
    status(a.root)
    log("\nNext steps:")
    log(f"  clear old results   python3 {sys.argv[0]} --reset --yes")
    log(f"  validate params     python3 {sys.argv[0]} --run --max-tiles 2")
    log(f"                      python3 {sys.argv[0]} --check")
    log(f"  full resume         python3 {sys.argv[0]} --run --clean-dict")


if __name__ == "__main__":
    main()