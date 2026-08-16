#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
proseg_cosmx_2.py
=======================================================================
Proseg on CosMx FOVs, one FOV at a time, on the same window UCS uses so
the two are directly comparable.

Data facts, verified from the export:
  0.12000 um/px, agreed by the FOV position table and by Area.um2 / Area
  FOV 4256 x 4256 px = 511 um; cells ~65 px = 7.8 um across
  19.1 tx/um^2, roughly 5 million transcripts per FOV
  the platform's own assignment lives in the transcript table's cell_ID
    and is what Proseg uses as its prior, exactly as cell_id does on
    Xenium

Proseg works in whatever units the coordinates carry, so this writes
microns rather than pixels: voxel and cell-size arguments are then in
microns and mean the same thing they did on Xenium.

The output polygons are rasterised twice: once at native FOV resolution
and once on the shared coarse grid, using the same
"a coarse pixel is a cell only if at least half its native block is"
rule as the other pipelines, so foreground fractions stay comparable
rather than inflating with the downsampling factor.

Usage
-----
  python proseg_cosmx_2.py --fovs 48 261 --stage prep
  python proseg_cosmx_2.py --fovs 48 261 --stage run --nthreads 16
  python proseg_cosmx_2.py --fovs 48 261 --stage post
  python proseg_cosmx_2.py --fovs 48 261 --nthreads 16     # all three
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = "/data/qiuyijia/dataset/cosmx_lymph_node"
OUT_TEMPLATE = "/data/qiuyijia/proseg_cosmx_lymph_node/fov{fov}"
NOMINAL_UM_PX = 0.12
FOV_SIZE = 4256
BIN_FACTOR = 8          # matches ucs_cosmx_2.py, so the coarse grids align
SEP = "=" * 78


def setup_logging(d, name):
    d.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(d / f"{name}.log", mode="a")],
        force=True)


def banner(m):
    logging.info("=" * 90)
    logging.info(m)
    logging.info("=" * 90)


def find_file(root: Path, pats):
    for p in pats:
        hits = [h for h in glob.glob(str(root / "**" / p), recursive=True)
                if os.path.isfile(h)]
        if hits:
            return Path(max(hits, key=os.path.getsize))
    return None


def pick(cols, *cands):
    low = {str(c).lower(): c for c in cols}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    return None


def proseg_flags(binary):
    try:
        h = subprocess.run([binary, "--help"], capture_output=True, text=True,
                           timeout=60)
        return set(w.strip(",") for w in (h.stdout + h.stderr).split()
                   if w.startswith("--"))
    except Exception as e:  # noqa: BLE001
        logging.warning(f"could not read {binary} --help ({e})")
        return set()


# ----------------------------------------------------------------------
def locate_tx_cache(cache_dir: Path, fovs):
    """Any cached subset that covers the requested FOVs.

    The inspector names its cache after the FOVs it was given, so a cache
    built for 48 and 261 is tx_fov_48_261.parquet and running one FOV at a
    time would look for a file that was never written.
    """
    exact = cache_dir / ("tx_fov_" + "_".join(str(f) for f in sorted(fovs))
                         + ".parquet")
    if exact.exists():
        return exact
    want = set(int(f) for f in fovs)
    for p in sorted(cache_dir.glob("tx_fov_*.parquet")):
        try:
            have = set(int(t) for t in p.stem.replace("tx_fov_", "").split("_"))
        except ValueError:
            continue
        if want <= have:
            logging.info(f"using {p.name}, which covers {sorted(want)}")
            return p
    return exact


def stage_prep(fov, paths, out, args):
    banner(f"PREP  FOV {fov}")
    um = args.um_per_px
    tx = pd.read_parquet(paths["tx_cache"]) if paths["tx_cache"].exists() \
        else None
    if tx is None:
        raise FileNotFoundError(
            f"{paths['tx_cache']} not found. Run the inspector first so the "
            "per-FOV subset is cached; rescanning 1.4 billion rows for every "
            "method would be wasteful.")
    fcol = pick(tx.columns, "fov", "FOV")
    t = tx[pd.to_numeric(tx[fcol], errors="coerce") == fov].copy()
    lx, ly = pick(t.columns, "x_local_px"), pick(t.columns, "y_local_px")
    tg = pick(t.columns, "target", "gene")
    cid = pick(t.columns, "cell_ID", "cell_id", "cell")
    comp = pick(t.columns, "CellComp", "cell_comp")
    logging.info(f"transcripts {len(t):,}  targets {t[tg].nunique():,}")

    # microns, so every downstream size argument is physical
    df = pd.DataFrame({
        "gene": t[tg].astype(str).to_numpy(),
        "x": t[lx].to_numpy(float) * um,
        "y": t[ly].to_numpy(float) * um,
        "z": np.zeros(len(t), np.float32)})
    if cid:
        v = pd.to_numeric(t[cid], errors="coerce").fillna(0).astype(np.int64)
        df["cell"] = v.to_numpy()
        n_assigned = int((v > 0).sum())
        logging.info(f"prior from {cid}: {n_assigned:,} assigned "
                     f"({n_assigned/max(len(t),1)*100:.1f}%), "
                     f"{int(v[v>0].nunique()):,} cells")
        if int(v[v > 0].nunique()) == 0:
            raise RuntimeError(
                f"{cid} yielded no positive values for FOV {fov}, so the prior "
                "would be empty. Proseg then panics with 'cannot sample empty "
                f"range'. Raw sample: {t[cid].head(5).tolist()}")
    else:
        df["cell"] = 0
        logging.warning("no cell_ID column, so Proseg runs without a prior")
    if comp:
        cv = t[comp].astype("string").fillna("").str.lower()
        df["compartment"] = cv.eq("nuclear").fillna(False).astype(np.int8)
        frac = float(df["compartment"].mean())
        logging.info(f"nuclear fraction {frac*100:.1f}%  "
                     f"({cv.value_counts().head(4).to_dict()})")
        if not np.isfinite(frac) or frac == 0:
            logging.warning("no nuclear transcripts flagged; dropping the "
                            "compartment column rather than handing proseg a "
                            "constant")
            df = df.drop(columns=["compartment"])

    # drop control probes, which are not genes
    bad = df.gene.str.startswith(("NegPrb", "Negative", "SystemControl",
                                  "NegControl", "Blank", "FalseCode"))
    if bad.any():
        logging.info(f"dropping {int(bad.sum()):,} control probes "
                     f"({bad.mean()*100:.2f}%)")
        df = df[~bad]

    csv = out / "transcripts.csv.gz"
    df.to_csv(csv, index=False, compression="gzip")
    logging.info(f"columns written: {list(df.columns)}")
    area = (FOV_SIZE * um) ** 2
    logging.info(f"-> {csv}  {len(df):,} rows")
    logging.info(f"FOV {FOV_SIZE*um:.0f} x {FOV_SIZE*um:.0f} um = "
                 f"{area/1e6:.4f} mm^2   density {len(df)/area:.2f} tx/um^2")

    md = pd.read_csv(paths["metadata"])
    mf = pick(md.columns, "fov", "FOV")
    m = md[pd.to_numeric(md[mf], errors="coerce") == fov]
    ac = pick(m.columns, "Area")
    diam = None
    if ac is not None and len(m):
        a = pd.to_numeric(m[ac], errors="coerce").dropna() * um * um
        diam = float(2 * np.sqrt(a.median() / np.pi))
        logging.info(f"platform cells {len(m):,}  median area "
                     f"{a.median():.1f} um^2  diameter {diam:.2f} um")

    (out / "roi_meta.json").write_text(json.dumps(dict(
        dataset="cosmx_lymph_node", fov=int(fov), size_y=FOV_SIZE,
        size_x=FOV_SIZE, pixel_size=float(um), bin_factor=int(args.bin_factor),
        n_transcripts=int(len(df)), n_genes=int(df.gene.nunique()),
        gt_expected_cells=int(len(m)),
        gt_median_diameter_um=diam), indent=2))
    banner("PREP DONE")


# ----------------------------------------------------------------------
def stage_run(out, args):
    banner("RUN PROSEG")
    binary = shutil.which(args.proseg_bin) or args.proseg_bin
    if shutil.which(args.proseg_bin) is None and not Path(binary).exists():
        env = os.environ.get("CONDA_DEFAULT_ENV", "?")
        raise FileNotFoundError(
            f"'{args.proseg_bin}' is not on PATH (conda env: {env}). Activate "
            "the environment Proseg is installed in, or pass --proseg-bin.")
    flags = proseg_flags(binary)
    run_dir = out / "proseg_out"
    if run_dir.exists() and (run_dir / ".done").exists() and not args.force_run:
        logging.info("Proseg already finished; --force-run to redo")
        return
    shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((out / "roi_meta.json").read_text())
    csv_path = out / "transcripts.csv.gz"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path}; run --stage prep first")
    header = list(pd.read_csv(csv_path, nrows=0).columns)
    logging.info(f"CSV columns: {header}")
    if "cell" in header:
        probe = pd.read_csv(csv_path, usecols=["cell"])
        n_prior = int(probe.loc[probe["cell"] > 0, "cell"].nunique())
        logging.info(f"prior in the CSV: {n_prior:,} distinct cells over "
                     f"{int((probe['cell']>0).sum()):,} transcripts")
        del probe
        if n_prior == 0:
            raise RuntimeError(
                "the CSV carries no prior (every cell value is 0). Proseg "
                "would panic with 'cannot sample empty range'. Re-run "
                "--stage prep and check the line reporting the prior.")
    # this build takes the transcript table positionally:
    #   proseg [OPTIONS] <TRANSCRIPT_CSV> [PARQUET_FMT]
    cmd = [binary, "--gene-column", "gene", "--x-column", "x",
           "--y-column", "y", "--z-column", "z"]
    # every column flag is driven by the CSV's actual header. prep drops the
    # compartment column when no transcript is flagged nuclear, and naming a
    # column that is not there makes proseg panic with
    # "Column 'compartment' not found in CSV file"
    if "--cell-id-column" in flags and "cell" in header:
        cmd += ["--cell-id-column", "cell", "--cell-id-unassigned", "0"]
    if ("--compartment-column" in flags and args.use_compartment
            and "compartment" in header):
        cmd += ["--compartment-column", "compartment",
                "--compartment-nuclear", "1"]
    elif args.use_compartment and "compartment" not in header:
        logging.info("no compartment column in the CSV, so the flag is "
                     "omitted")
    if "--voxel-layers" in flags:
        cmd += ["--voxel-layers", "1"]
    for flag, val in (("--nthreads", args.nthreads),
                      ("--output-cell-polygons",
                       str(run_dir / "cell-polygons.geojson.gz")),
                      ("--output-counts", str(run_dir / "counts.csv.gz")),
                      ("--output-cell-metadata",
                       str(run_dir / "cell-metadata.csv.gz")),
                      ("--output-transcript-metadata",
                       str(run_dir / "transcript-metadata.csv.gz"))):
        if flag in flags:
            cmd += [flag, str(val)]
    if args.proseg_extra:
        extra = args.proseg_extra.split()
        # anything named in --proseg-extra replaces the default this script
        # would have supplied, rather than being appended alongside it
        named = {t for t in extra if t.startswith("--")}
        if named:
            kept, skip = [], False
            for tok in cmd[1:]:
                if skip:
                    skip = False
                    continue
                if tok in named:
                    logging.info(f"--proseg-extra overrides {tok}")
                    skip = True
                    continue
                kept.append(tok)
            cmd = [cmd[0]] + kept
        cmd += extra
    cmd.append(str(csv_path))   # positional, must be last

    unknown = [c for c in cmd[1:] if c.startswith("--") and flags
               and c not in flags]
    if unknown:
        logging.warning(f"this proseg does not list {unknown} in --help; "
                        "dropping them")
        keep, skip = [], False
        for c in cmd[1:]:
            if skip:
                skip = False
                continue
            if c in unknown:
                skip = True
                continue
            keep.append(c)
        cmd = [cmd[0]] + keep

    logging.info(" ".join(cmd))
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=str(run_dir), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail = []
    for line in proc.stdout:
        line = line.rstrip()
        logging.info("    " + line)
        tail.append(line)
        if len(tail) > 40:
            tail.pop(0)
    rc = proc.wait()
    if rc != 0:
        msg = "\n".join(tail[-15:])
        if rc == 101:
            raise RuntimeError(
                "Proseg panicked (exit 101). Its last output was:\n" + msg
                + "\n\nA panic is a runtime failure, not a bad flag, so "
                "--help will not help. Try --no-compartment, or drop "
                "--voxel-layers via --proseg-extra, and check the CSV columns "
                "named above.")
        raise RuntimeError(f"Proseg exited with {rc}. Last output:\n" + msg)
    logging.info(f"finished in {(time.time()-t0)/60:.1f} min")
    (run_dir / ".done").write_text("ok\n")


# ----------------------------------------------------------------------
def block_majority(lab, factor, min_frac=0.5):
    """Downsample labels counting background as a candidate.

    Taking the most common non-zero label fills a coarse pixel whenever a
    single native pixel is labelled, inflating foreground by roughly the
    downsampling factor. Requiring the block to be at least min_frac
    foreground keeps areas comparable with methods whose masks are native
    to the coarse grid.
    """
    H, W = lab.shape
    H2, W2 = H // factor, W // factor
    b = lab[:H2 * factor, :W2 * factor].reshape(H2, factor, W2, factor)
    b = b.transpose(0, 2, 1, 3).reshape(-1, factor * factor)
    frac = (b > 0).mean(axis=1)
    out = np.zeros(H2 * W2, dtype=lab.dtype)
    for i in np.flatnonzero((frac >= min_frac) & (frac > 0)):
        nz = b[i][b[i] > 0]
        if nz.size == 0:
            continue
        v, c = np.unique(nz, return_counts=True)
        out[i] = v[np.argmax(c)]
    return out.reshape(H2, W2)


def rasterise_polygons(path, size, um, out_native):
    import cv2

    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        gj = json.load(fh)
    feats = gj.get("features", gj if isinstance(gj, list) else [])
    lab = np.zeros((size, size), np.int32)
    n = 0
    for i, f in enumerate(feats, 1):
        geom = f.get("geometry", f)
        gtype = geom.get("type")
        polys = (geom.get("coordinates", []) if gtype == "Polygon"
                 else [c for part in geom.get("coordinates", []) for c in part]
                 if gtype == "MultiPolygon" else [])
        drawn = False
        for ring in polys:
            pts = np.asarray(ring, dtype=float)
            if pts.ndim != 2 or len(pts) < 3:
                continue
            pts = np.rint(pts[:, :2] / um).astype(np.int32)
            cv2.fillPoly(lab, [pts], int(i))
            drawn = True
        n += int(drawn)
    logging.info(f"rasterised {n:,} polygons -> {out_native}")
    return lab, n


def stage_post(fov, paths, out, args):
    import tifffile

    banner(f"POST  FOV {fov}")
    meta = json.loads((out / "roi_meta.json").read_text())
    um, bf = meta["pixel_size"], meta["bin_factor"]
    run_dir = out / "proseg_out"
    poly = next((p for p in (run_dir / "cell-polygons.geojson.gz",
                             run_dir / "cell-polygons.geojson")
                 if p.exists()), None)
    if poly is None:
        hits = sorted(run_dir.rglob("*polygon*"))
        poly = hits[0] if hits else None
    if poly is None:
        logging.info("no polygon output found; nothing to rasterise")
        return

    lab, n_poly = rasterise_polygons(poly, FOV_SIZE, um,
                                     out / "proseg_labels_native.tif")
    tifffile.imwrite(str(out / "proseg_labels_native.tif"), lab)
    coarse = block_majority(lab, bf, args.coarse_min_frac)
    tifffile.imwrite(str(out / "proseg_segmentation.tif"), coarse)

    a = np.bincount(lab.reshape(-1).astype(np.int64))[1:]
    a = a[a > 0]
    px = um * um
    gt = meta.get("gt_expected_cells") or 0
    logging.info(f"cells {len(a):,}   platform {gt:,}   "
                 f"ratio {len(a)/max(gt,1):.3f}")
    logging.info(f"native coverage {(lab>0).mean()*100:.2f}%   coarse "
                 f"{(coarse>0).mean()*100:.2f}%")
    logging.info(f"median area {np.median(a)*px:.1f} um^2   equiv diameter "
                 f"{2*np.sqrt(np.median(a)*px/np.pi):.2f} um"
                 + (f"   platform {meta['gt_median_diameter_um']:.2f} um"
                    if meta.get("gt_median_diameter_um") else ""))

    tx = pd.read_parquet(paths["tx_cache"])
    fcol = pick(tx.columns, "fov", "FOV")
    t = tx[pd.to_numeric(tx[fcol], errors="coerce") == fov]
    lx, ly = pick(t.columns, "x_local_px"), pick(t.columns, "y_local_px")
    tg = pick(t.columns, "target", "gene")
    col = np.clip(np.rint(t[lx].to_numpy()).astype(np.int64), 0, FOV_SIZE - 1)
    row = np.clip(np.rint(t[ly].to_numpy()).astype(np.int64), 0, FOV_SIZE - 1)
    lv = lab[row, col]
    hit = lv > 0
    genes, gcode = np.unique(t[tg].astype(str).to_numpy()[hit],
                             return_inverse=True)
    n_cells = int(lab.max())
    mat = np.bincount((lv[hit].astype(np.int64) - 1) * len(genes) + gcode,
                      minlength=n_cells * len(genes)).astype(np.int32)
    mat = mat.reshape(n_cells, len(genes))
    logging.info(f"transcripts {len(t):,}  assigned {int(hit.sum()):,} "
                 f"({hit.mean()*100:.1f}%)  genes {len(genes):,}")
    np.save(out / "cell_by_gene.npy", mat)
    (out / "cell_by_gene_genes.txt").write_text("\n".join(genes) + "\n")
    cnt = mat.sum(axis=1)
    if (cnt > 0).any():
        logging.info(f"counts/cell median {np.median(cnt[cnt>0]):.0f}")

    ref = Path(args.compare_to) if args.compare_to else None
    if ref:
        g = ref / "platform_labels.tif"
        if g.exists():
            gtl = tifffile.imread(str(g))
            A, B = lab > 0, gtl > 0
            dice = 2 * int((A & B).sum()) / max(int(A.sum()) + int(B.sum()), 1)
            logging.info(f"vs platform labels: foreground Dice {dice:.3f} "
                         f"(coverage {A.mean()*100:.1f}% vs "
                         f"{B.mean()*100:.1f}%)")
    banner("POST DONE")


# ----------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(
        description="Proseg on CosMx FOVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--root", default=ROOT)
    p.add_argument("--fovs", type=int, nargs="+", default=[48, 261])
    p.add_argument("--stage", choices=["prep", "run", "post", "all"],
                   default="all")
    p.add_argument("--out-template", default=OUT_TEMPLATE)
    p.add_argument("--proseg-bin", default="proseg")
    p.add_argument("--um-per-px", type=float, default=NOMINAL_UM_PX)
    p.add_argument("--bin-factor", type=int, default=BIN_FACTOR,
                   help="must match ucs_cosmx_2.py for the coarse grids to "
                        "line up")
    p.add_argument("--coarse-min-frac", type=float, default=0.5)
    p.add_argument("--nthreads", type=int, default=16)
    p.add_argument("--use-compartment", action="store_true", default=True)
    p.add_argument("--no-compartment", dest="use_compartment",
                   action="store_false")
    p.add_argument("--proseg-extra", default=None)
    p.add_argument("--compare-to", default=None,
                   help="a ucs_cosmx_2 output dir, for the platform labels")
    p.add_argument("--force-run", action="store_true")
    return p.parse_args()


def main():
    args = get_args()
    root = Path(args.root)
    paths = dict(metadata=find_file(root, ["*metadata_file.csv*"]),
                 tx_cache=locate_tx_cache(root / "_inspect_cache", args.fovs))
    if paths["metadata"] is None:
        raise FileNotFoundError("metadata file")

    for fov in args.fovs:
        out = Path(args.out_template.format(fov=fov))
        out.mkdir(parents=True, exist_ok=True)
        setup_logging(out / "logs", f"proseg_cosmx_fov{fov}_{args.stage}")
        banner(f"PROSEG x CosMx  FOV {fov}   out {out}")
        if args.stage in ("prep", "all"):
            stage_prep(fov, paths, out, args)
        if args.stage in ("run", "all"):
            stage_run(out, args)
        if args.stage in ("post", "all"):
            stage_post(fov, paths, out, args)


if __name__ == "__main__":
    main()