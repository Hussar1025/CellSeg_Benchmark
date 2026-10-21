#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ucs_stereo_2.py —— UCS @ stereo_2 (E14.5_E1S3)

The stereo_2 UCS row in the earlier summary table was bogus (evaluation but no artifact); this is the rerun.

gene_map memory = H * W * n_genes * 4 bytes
  --top-genes N : trims the gene axis, linear. top 2000 usually covers 85-95% of transcripts, cheap.
  --bin-px B    : trims the spatial axis, quadratic (H and W each / B), but resolution drops 0.5 -> 0.5*B um.
                  mouse embryo cells are 6-10um; after bin-px 2 a cell is only 6-10 px wide.
  prefer raising --top-genes, keep --bin-px 1 if possible.

⚠ Evaluation note: nuclei_mask is built from the GEM cell column (official CellBin segmentation), GT is the same column
   -> prior and GT share the same source, so transcript-level metrics are circularly inflated. Detection/localization/count metrics remain valid.

Usage:
  python3 ucs_stereo_2.py --check                    # estimate volume first
  python3 ucs_stereo_2.py --top-genes 2000
  python3 ucs_stereo_2.py --top-genes 1000 --bin-px 1
"""
import argparse, subprocess, sys
from pathlib import Path

DIR = "yourpath/dataset/stereoseq/stereo-seq_data_all"
SLICE = "E14.5_E1S3"
GT_N = 4872
DATASET = "stereo_2"
BASE = "yourpath/ucs/UCS/ucs_stereo_1_run.py"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="estimate gene_map volume only, don't run")
    ap.add_argument("--top-genes", type=int, default=0,
                    help="keep only the top-N highest-expression genes, 0=all")
    ap.add_argument("--bin-px", type=int, default=1,
                    help="spatial downsampling factor, 1=raw 0.5um/px")
    ap.add_argument("--nucleus-erode", type=int, default=2)
    ap.add_argument("--base", default=BASE)
    a = ap.parse_args()

    if not Path(a.base).exists():
        sys.exit(f"✗ base script not found: {a.base}\n"
                 f"  ls yourpath/ucs/UCS/ | grep stereo")

    gem = f"{DIR}/{SLICE}_Dorsal_Midbrain_GEM_CellBin.tsv.gz"
    tif = f"{DIR}/{SLICE}.tif"
    for p in (gem, tif):
        if not Path(p).exists():
            sys.exit(f"✗ not found: {p}")

    out = f"yourpath/ucs_stereoseq_{SLICE}"
    cmd = [sys.executable, a.base,
           "--gem", gem, "--ssdna", tif, "--out-dir", out,
           "--nucleus-erode", str(a.nucleus_erode),
           "--bin-px", str(a.bin_px)]
    if a.top_genes:
        cmd += ["--top-genes", str(a.top_genes)]
    if a.check:
        cmd += ["--check"]

    print(f"=== UCS @ {DATASET} ({SLICE})  GT {GT_N:,} ===")
    if a.top_genes or a.bin_px > 1:
        print(f"    top_genes={a.top_genes or 'all'}  bin_px={a.bin_px} "
              f"(spatial saving {a.bin_px**2}x)")
    print("$ " + " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0 and not a.check:
        print("\nIf gene_map is too large:")
        print("  prefer raising --top-genes (linear, cheap):")
        print("    --top-genes 2000  saves ~11x")
        print("    --top-genes 1000  saves ~22x")
        print("    --top-genes 500   saves ~44x")
        print("  then consider --bin-px 2 (quadratic, saves 4x, but resolution drops to 1.0um/px)")
    sys.exit(rc)


if __name__ == "__main__":
    main()
