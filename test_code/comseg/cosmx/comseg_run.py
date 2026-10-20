#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
comseg_run.py - ComSeg full pipeline orchestration (CosMx Pancreas, 18 FOVs)

Four changes from the original comseg_cosmx.py:
  1. Drop mp.Pool; launch one comseg_worker.py per FOV via subprocess
     -> mp.Pool workers are daemons, and daemons cannot spawn children.
        If ComSeg uses joblib/loky internally, it hangs (the 3-day stall last time)
  2. Per-FOV timeout; kill and mark failed without blocking other FOVs
  3. Skip if h5ad exists -> repeatable resume
  4. Default Top 1000 genes + sampling_size 50000 (was all 21731 + 1M sampling, 10-20x slower)

Usage:
    python comseg_run.py                       # default Top1000 / 4 workers / 90min timeout
    python comseg_run.py --top-n 3000 --sampling-size 150000
    python comseg_run.py --workers 2           # lower concurrency under memory pressure
    python comseg_run.py --fovs 51,52,53       # run only specified FOVs
    python comseg_run.py --merge-only          # only do the final merge
"""
import os
import re
import sys
import glob
import time
import argparse
import subprocess

import numpy as np
import pandas as pd
import anndata


# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR   = "yourpath/comseg"
INPUT_DIR  = os.path.join(BASE_DIR, "input_csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOG_DIR    = os.path.join(BASE_DIR, "logs")
WORKER     = os.path.join(BASE_DIR, "comseg_worker.py")

MERGED_H5AD   = os.path.join(OUTPUT_DIR, "comseg_pancreas_full.h5ad")
CELL_SUMMARY  = os.path.join(OUTPUT_DIR, "comseg_cell_summary.csv")

FULL_GENE_LIST = os.path.join(BASE_DIR, "top21731_genes.txt")   # sorted by transcript count, descending
# ══════════════════════════════════════════════════════════════════════════════


def log(msg):
    print(msg, flush=True)


def hr(title):
    log("\n" + "═" * 66)
    log(f"  {title}")
    log("═" * 66)


# ─────────────────────────────────────────────────────────────────────────────
#  discover available FOVs
# ─────────────────────────────────────────────────────────────────────────────
def discover_fovs():
    fovs = []
    for p in sorted(glob.glob(os.path.join(INPUT_DIR, "fov_*.csv"))):
        m = re.search(r"fov_(\d+)\.csv$", os.path.basename(p))
        if m:
            fovs.append(int(m.group(1)))
    return sorted(fovs)


# ─────────────────────────────────────────────────────────────────────────────
#  prepare gene whitelist
# ─────────────────────────────────────────────────────────────────────────────
def prepare_gene_list(top_n, fovs):
    """
    Prefer reusing top21731_genes.txt (the original script wrote it in
    transcript-count descending order; the first N lines are Top N, avoiding a
    rescan of the 65M-row tx_file). If missing, recount from the target column
    in input_csv.
    """
    out_path = os.path.join(BASE_DIR, f"top{top_n}_genes.txt")
    if os.path.exists(out_path):
        with open(out_path) as f:
            n = sum(1 for ln in f if ln.strip())
        log(f"[genes] reused directly: {out_path} ({n} genes)")
        return out_path

    if os.path.exists(FULL_GENE_LIST):
        with open(FULL_GENE_LIST) as f:
            ordered = [ln.strip() for ln in f if ln.strip()]
        log(f"[genes] taking first {top_n} from {os.path.basename(FULL_GENE_LIST)}"
            f" (file has {len(ordered)} genes, sorted by transcript count descending)")
        sel = ordered[:top_n]
    else:
        log(f"[genes] {FULL_GENE_LIST} not found; recounting from input_csv ...")
        counts = {}
        for fov in fovs:
            p = os.path.join(INPUT_DIR, f"fov_{fov:03d}.csv")
            if not os.path.exists(p):
                continue
            vc = pd.read_csv(p, usecols=["target"])["target"].value_counts()
            for g, c in vc.items():
                counts[g] = counts.get(g, 0) + int(c)
            log(f"  FOV {fov}: {len(counts)} genes so far")
        s = pd.Series(counts).sort_values(ascending=False)
        cov = s.iloc[:top_n].sum() / s.sum() * 100
        log(f"[genes] total {len(s)}; taking top {top_n}, covering {cov:.1f}% of transcripts")
        sel = s.iloc[:top_n].index.tolist()

    with open(out_path, "w") as f:
        f.write("\n".join(sel))
    log(f"[genes] wrote: {out_path} ({len(sel)} genes)")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
#  subprocess scheduling (concurrency + timeout)
# ─────────────────────────────────────────────────────────────────────────────
def run_all_fovs(fovs, gene_list, top_n, sampling_size, workers, timeout):
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pending = list(fovs)
    running = []          # [{fov, proc, fh, logpath, t0}]
    results = {}          # fov -> ("done"/"cached"/"timeout"/"error", seconds)

    def launch(fov):
        logpath = os.path.join(LOG_DIR, f"fov_{fov:03d}.log")
        fh = open(logpath, "w")
        cmd = [sys.executable, "-u", WORKER, str(fov),
               "--gene-list", gene_list,
               "--top-n", str(top_n),
               "--sampling-size", str(sampling_size)]
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                cwd=BASE_DIR)
        log(f"  ▶ launched FOV {fov}  (pid={proc.pid})  log: {logpath}")
        return {"fov": fov, "proc": proc, "fh": fh,
                "logpath": logpath, "t0": time.time()}

    t_start = time.time()
    while pending or running:
        # Fill concurrency slots
        while pending and len(running) < workers:
            running.append(launch(pending.pop(0)))

        time.sleep(3)

        for job in running[:]:
            fov, proc = job["fov"], job["proc"]
            elapsed = time.time() - job["t0"]
            rc = proc.poll()

            if rc is None:
                # Timeout -> kill
                if elapsed > timeout:
                    log(f"  ✘ FOV {fov} timed out after {elapsed/60:.1f} min; killing")
                    proc.kill()
                    try:
                        proc.wait(timeout=10)
                    except Exception:
                        pass
                    job["fh"].close()
                    results[fov] = ("timeout", elapsed)
                    running.remove(job)
                continue

            # Finished
            job["fh"].close()
            running.remove(job)
            out_path = os.path.join(OUTPUT_DIR, f"fov_{fov:03d}.h5ad")
            if rc == 0 and os.path.exists(out_path):
                n = anndata.read_h5ad(out_path).n_obs
                results[fov] = ("done", elapsed)
                log(f"  ✔ FOV {fov} done  {n:,} cells  {elapsed/60:.1f} min")
            else:
                results[fov] = ("error", elapsed)
                log(f"  ✘ FOV {fov} failed (rc={rc}), see {job['logpath']}")

        n_done = len(results)
        if n_done and n_done % 3 == 0 and not running:
            log(f"  -- progress {n_done}/{len(fovs)}  "
                f"elapsed {(time.time()-t_start)/60:.1f} min")

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  merge
# ─────────────────────────────────────────────────────────────────────────────
def merge_results(fovs):
    adatas, ok = [], []
    for fov in fovs:
        p = os.path.join(OUTPUT_DIR, f"fov_{fov:03d}.h5ad")
        if not os.path.exists(p):
            log(f"  FOV {fov}: missing, skip")
            continue
        a = anndata.read_h5ad(p)
        a.obs_names = [f"fov{fov:03d}_{i}" for i in range(a.n_obs)]
        adatas.append(a)
        ok.append(fov)
        log(f"  FOV {fov}: {a.n_obs:,} cells")

    if not adatas:
        log("  ✘ no h5ad to merge")
        return None

    merged = anndata.concat(adatas, join="outer", fill_value=0)
    merged.write_h5ad(MERGED_H5AD)
    log(f"\n  merge done: {merged.n_obs:,} cells x {merged.n_vars:,} genes")
    log(f"  -> {MERGED_H5AD}")

    # Cell-level summary CSV
    X = merged.X
    counts = np.asarray(X.sum(axis=1)).ravel() if hasattr(X, "sum") else X.sum(1)
    genes  = np.asarray((X > 0).sum(axis=1)).ravel()
    summary = merged.obs.copy()
    summary["n_transcripts"] = counts
    summary["n_genes"]       = genes
    summary.to_csv(CELL_SUMMARY)
    log(f"  -> {CELL_SUMMARY}")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, default=1000,
                    help="use the N genes with highest transcript counts, default 1000")
    ap.add_argument("--sampling-size", type=int, default=50_000,
                    help="compute_edge_weight sampling, default 50000 (~50 per gene)")
    ap.add_argument("--workers", type=int, default=4,
                    help="number of concurrent FOVs, default 4; use 2 under memory pressure")
    ap.add_argument("--timeout", type=int, default=5400,
                    help="per-FOV timeout in seconds, default 5400 (90 min)")
    ap.add_argument("--fovs", default=None,
                    help="run only specified FOVs, comma-separated, e.g. 51,52,53")
    ap.add_argument("--merge-only", action="store_true",
                    help="skip computation and only merge")
    args = ap.parse_args()

    if not os.path.exists(WORKER):
        log(f"✘ worker script not found: {WORKER}")
        sys.exit(1)

    fovs = discover_fovs()
    if args.fovs:
        want = {int(x) for x in args.fovs.split(",")}
        fovs = [f for f in fovs if f in want]
    if not fovs:
        log(f"✘ no fov_*.csv under {INPUT_DIR}")
        sys.exit(1)

    hr(f"ComSeg - {len(fovs)} FOVs: {fovs}")
    log(f"  Top N genes    : {args.top_n}")
    log(f"  sampling_size  : {args.sampling_size:,}  "
        f"(~{args.sampling_size/args.top_n:.0f} per gene)")
    log(f"  concurrent workers: {args.workers}")
    log(f"  per-FOV timeout: {args.timeout/60:.0f} min")

    if not args.merge_only:
        hr("Step 1: prepare gene whitelist")
        gene_list = prepare_gene_list(args.top_n, fovs)

        hr(f"Step 2: run ComSeg ({args.workers} independent subprocesses)")
        t0 = time.time()
        results = run_all_fovs(fovs, gene_list, args.top_n,
                               args.sampling_size, args.workers, args.timeout)

        hr("run summary")
        for status in ["done", "cached", "timeout", "error"]:
            hit = [f for f, (s, _) in results.items() if s == status]
            if hit:
                log(f"  {status:8s}: {len(hit)}  -> {sorted(hit)}")
        log(f"  total elapsed: {(time.time()-t0)/60:.1f} min")

    hr("Step 3: merge all FOVs")
    merge_results(fovs)

    hr("Done")
    log(f"  output dir: {OUTPUT_DIR}")
    log(f"  log dir: {LOG_DIR}")


if __name__ == "__main__":
    main()