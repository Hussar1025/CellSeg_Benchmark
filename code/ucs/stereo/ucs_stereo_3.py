#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ucs_stereo_3.py — UCS @ stereo_3 (Stereo-seq E16.5_E2S7)

Why the previous run failed (root cause found):
  In the erode loop of ucs_stereo_1_run.py:
      for cid in np.unique(nm[nm>0]):
          e = cv2.erode((nm==cid), ker, iterations=2)
          if e.sum() == 0: e[centroid] = 1        # <- fallback
  On E2S7 all 7,960 nuclei hit this fallback, so each cell was left with
  1 pixel and the final segmentation_mask held only 280 3px fragments.
  The same code works on E14.5_E1S3 (nucleus 8.3um -> cell 14.4um, ratio 1.000);
  the difference is image size: E2S7 is 5550x4299 with 7,960 nuclei, so the
  per-nucleus full-image scan costs 2.5x E14.5, and cv2.erode can silently
  return empty under memory pressure.

What this script does about it:
  Default --nucleus-erode 0, skipping the whole erode loop. The cost is an
  oversized nucleus prior (the full CellBin segmentation instead of the nucleus
  region), but far better than 1 pixel.
  After the run it checks the nuclei_mask area median; <=2px aborts immediately.

⚠ Circular bias: nuclei_mask is rasterized from the GEM cell column, and GT is
  the same column -> assignment / coverage / vec_* are circularly inflated;
  only loc_*, median_diameter, count_* are comparable. Same convention as stereo_2.

⚠ Convention difference: stereo_2 used --nucleus-erode 2, this run uses 0. Note
  it in evaluation; to be strictly comparable, rerun both with the same value once
  the erode code is fixed.

gene_map memory:
  --top-genes N   trims the gene axis, linear. top 2000 usually covers 85-95% UMI.
  --bin-px B      trims the spatial axis, quadratic (H and W each / B), but
                  resolution drops 0.5 -> 0.5*B um.
  E2S7: all genes 21,370 -> 509.9 GB; top 2000 -> 47.7 GB
  The base script has a hard `if est > 15` cap at line 143; override with UCS_MAX_GB.

Usage:
  python3 ucs_stereo_3.py --check                    # estimate volume only
  python3 ucs_stereo_3.py                            # top-genes 2000, erode 0
  python3 ucs_stereo_3.py --nucleus-erode 2          # same convention as stereo_2
  python3 ucs_stereo_3.py --top-genes 1000
"""
import argparse
import glob
import os
import subprocess
import sys

DIR = "yourpath/dataset/stereoseq/stereo-seq_data_all"
SLICE = "E16.5_E2S7"
DATASET = "stereo_3"
GT_N = 7960
BASE = "yourpath/ucs/UCS/ucs_stereo_1_run.py"
OUT = f"yourpath/ucs_stereoseq_{SLICE}"
BIN_UM = 0.5


def log(m):
    print(m, flush=True)


def stash_bad(out):
    """Move the previous single-pixel artifact aside instead of deleting it, for later comparison."""
    import shutil
    moved = []
    for p in (os.path.join(out, "nuclei_mask.tif"),
              os.path.join(out, "ucs_log")):
        if os.path.exists(p) and not p.endswith(".BAD_1px") \
                and not p.endswith(".BAD_1px.bak"):
            dst = p + ".BAD_1px"
            n = 1
            while os.path.exists(dst):
                n += 1
                dst = f"{p}.BAD_1px{n}"
            shutil.move(p, dst)
            moved.append(os.path.basename(dst))
    if moved:
        log(f"  old artifact moved aside: {moved}")
    return moved


def check_mask(out):
    """Nucleus prior check: area median <=2px means the erode degraded; abort."""
    p = os.path.join(out, "nuclei_mask.tif")
    if not os.path.exists(p):
        log(f"  ✗ not generated: {p}")
        return False
    import numpy as np
    import tifffile
    m = tifffile.imread(p)
    lab, cnt = np.unique(m, return_counts=True)
    a = cnt[lab > 0]
    if len(a) == 0:
        log("  ✗ nuclei_mask has no labels")
        return False
    med = float(np.median(a))
    d = 2 * (med / np.pi) ** .5 * BIN_UM
    log(f"\n[nucleus prior check] {m.shape}  labels {len(a):,}  GT {GT_N:,}  "
        f"ratio {len(a)/GT_N:.3f}")
    log(f"  foreground {(m>0).mean()*100:.2f}%   area median {med:.0f} px "
        f"= {med*BIN_UM**2:.1f} um2   equivalent diameter {d:.2f} um")
    if med <= 2:
        log("  ✗ area median <=2 px — erode degradation reappeared (fallback to single-pixel centroid)")
        log("    rerun with --nucleus-erode 0, or fix ucs_stereo_1_run.py first")
        return False
    if d < 3 or d > 25:
        log(f"  ⚠ equivalent diameter {d:.1f} um deviates from the 6-10 um of mouse embryo cells")
    else:
        log("  ✓ nucleus prior OK")
    return True


def check_out(out):
    import numpy as np
    import tifffile
    fs = sorted(glob.glob(os.path.join(out, "ucs_log", "pred", "*.tif")))
    if not fs:
        log("  ✗ no tif under pred/")
        return
    log(f"\n[output check]")
    for p in fs:
        m = tifffile.imread(p)
        lab, cnt = np.unique(m, return_counts=True)
        a = cnt[lab > 0]
        if len(a) == 0:
            log(f"  {os.path.basename(p):26s} no labels")
            continue
        med = float(np.median(a))
        log(f"  {os.path.basename(p):26s} labels {len(a):6,}  "
            f"ratio {len(a)/GT_N:.3f}  foreground {(m>0).mean()*100:5.2f}%  "
            f"area median {med:.0f}px = diameter "
            f"{2*(med/np.pi)**.5*BIN_UM:.1f}um")
    log(f"\n  reference stereo_2 (E14.5_E1S3, erode 2):")
    log(f"    nuclei_mask 4,872 labels diameter 8.3um -> "
        f"segmentation 4,870 diameter 14.4um  ratio 1.000")
    log(f"  if segmentation_mask has far fewer labels than mask_shift_0,")
    log(f"  the dual-shift fusion failed; use mask_shift_0 for evaluation.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="estimate volume only, don't run")
    ap.add_argument("--top-genes", type=int, default=2000)
    ap.add_argument("--bin-px", type=int, default=1)
    ap.add_argument("--nucleus-erode", type=int, default=0,
                    help="0 = skip erode (last erode 2 caused single-pixel degradation)")
    ap.add_argument("--max-gb", type=float, default=60.0,
                    help="override the base script's hard `if est > 15` cap")
    ap.add_argument("--keep-old", action="store_true",
                    help="don't move the old artifact aside")
    ap.add_argument("--base", default=BASE)
    a = ap.parse_args()

    log("=" * 78)
    log(f"UCS @ {DATASET}  ({SLICE})   GT {GT_N:,}")
    log("=" * 78)

    if not os.path.exists(a.base):
        sys.exit(f"✗ base script not found: {a.base}")
    gem = f"{DIR}/{SLICE}_Dorsal_Midbrain_GEM_CellBin.tsv.gz"
    tif = f"{DIR}/{SLICE}.tif"
    for p in (gem, tif):
        if not os.path.exists(p):
            sys.exit(f"✗ not found: {p}")
        log(f"  {os.path.getsize(p)/1e9:6.2f} GB  {os.path.basename(p)}")

    log(f"\nparameters")
    log(f"  top_genes      {a.top_genes}  "
        f"(all genes 21,370 -> 509.9 GB; 2000 -> 47.7 GB)")
    log(f"  bin_px         {a.bin_px}  "
        f"({'raw 0.5 um/px' if a.bin_px == 1 else f'down to {0.5*a.bin_px} um/px'})")
    log(f"  nucleus_erode  {a.nucleus_erode}"
        + ("   <- skip erode, avoid last run's single-pixel degradation"
           if a.nucleus_erode == 0 else
           "   <- same convention as stereo_2, but may reproduce the degradation"))
    log(f"  UCS_MAX_GB     {a.max_gb}")

    if not a.check and not a.keep_old:
        log("")
        stash_bad(OUT)

    env = dict(os.environ, UCS_MAX_GB=str(a.max_gb))
    cmd = [sys.executable, a.base,
           "--gem", gem, "--ssdna", tif, "--out-dir", OUT,
           "--nucleus-erode", str(a.nucleus_erode),
           "--bin-px", str(a.bin_px)]
    if a.top_genes:
        cmd += ["--top-genes", str(a.top_genes)]
    if a.check:
        cmd += ["--check"]

    log(f"\n$ UCS_MAX_GB={a.max_gb} " + " ".join(cmd))
    log("")
    rc = subprocess.call(cmd, env=env)

    if a.check:
        log("\n--check done. Once the volume is acceptable, drop --check for the real run.")
        return
    if rc != 0:
        log(f"\n✗ exit code {rc}")
        log("  if gene_map is too large: prefer a bigger --top-genes (linear, cheap),")
        log("  then consider --bin-px 2 (quadratic, saves 4x, but resolution drops to 1.0 um/px,")
        log("  leaving only 6-10 pixels per 6-10um embryo cell).")
        sys.exit(rc)

    ok = check_mask(OUT)
    check_out(OUT)
    log("\n" + "=" * 78)
    if ok:
        log("✓ Done. Evaluate: python3 ucs_value_stereo_3.py")
    else:
        log("✗ nucleus prior abnormal; fix before evaluating")
    log("⚠ nuclei_mask and GT share the same source -> assignment/coverage/vec_* circularly inflated,")
    log(f"  only loc_*, median_diameter, count_* are comparable.")
    log(f"⚠ this run erode={a.nucleus_erode}, stereo_2 used 2; note the convention difference.")
    log("=" * 78)


if __name__ == "__main__":
    main()
