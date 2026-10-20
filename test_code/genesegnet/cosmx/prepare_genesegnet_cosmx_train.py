#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
genesegnet_cosmx_2_2.py
GeneSegNet domain fine-tuning pipeline for CosMx lymph node (cosmx_2).

The training/test design is FOV-level, not random-tile-level:
  train: 24 FOVs selected to match FOV48/261 in cell count and cell area
  val:    8 different FOVs
  test:   FOV48 and FOV261 ONLY; they are prepared but never passed to
          GeneSeg_train.py during training.

Data:
  metadata : flat_files/S0_metadata_file.csv.gz
  positions: flat_files/S0_fov_positions_file.csv.gz
  tx       : flat_files/S0_tx_file.csv.gz
  DNA      : napari/images/DNA/0
  labels   : napari/images/labels/0

The script follows the already-debugged Xenium training layout:
  prep -> flows -> split -> check -> train
and preserves the repo requirements:
  split/<part>/<group>/{images,labels,spots,HeatMaps/HeatMap_all,HeatMaps/HeatMap}
  labels must also have *_flows.tif before training.

Important differences from the Xenium pipeline:
  * CosMx already provides a stitched DNA zarr and cell-label zarr.
  * FOV assignment is explicit. 48/261 are strict hold-outs.
  * transcripts are streamed ONCE from the ~33 GB CSV and cached per selected FOV.
  * RNA density cap affects only the Gaussian heatmap. spots/*.csv keeps ALL gene
    transcripts for recursive label processing.
  * first training starts from the existing NSCLC GeneSegNet checkpoint by
    default; later runs resume from the newest checkpoint in this run directory.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "2026.08.22-cosmx2-fovholdout-v1"

ROOT_DEFAULT = Path("yourpath/dataset/cosmx_lymph_node")
OUT_DEFAULT = Path("yourpath/genesegnet_cosmx_2_train")
REPO_DEFAULT = Path("yourpath/GeneSegNet/GeneSegNet")
PRETRAINED_DEFAULT = Path(
    "yourpath/GeneSegNet/"
    "GeneSegNet_NSCLC_residual_on_style_on_concatenation_off.544278_epoch_499"
)

# The ranking came from metadata similarity to the held-out FOVs. 48 and 261
# are deliberately absent here.
TRAIN_FOVS_DEFAULT = [
    309, 367, 98, 302, 37, 276, 387, 198,
    158, 135, 136, 300, 138, 259, 67, 353,
    141, 376, 258, 179, 154, 112, 206, 63,
]
VAL_FOVS_DEFAULT = [167, 132, 233, 114, 155, 211, 281, 25]
TEST_FOVS_DEFAULT = [48, 261]

FOV_SIZE_DEFAULT = 4256
PIXEL_SIZE_DEFAULT = 0.1200

log = logging.getLogger("gsn_cosmx2")


# -----------------------------------------------------------------------------
# utility / state
# -----------------------------------------------------------------------------

def setup_log(out: Path, stage: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    fh = logging.FileHandler(out / "logs" / f"{stage}.log")
    fh.setFormatter(fmt)
    log.addHandler(fh)


def banner(msg: str) -> None:
    log.info("=" * 82)
    log.info(msg)
    log.info("=" * 82)


def parse_fovs(s: str | None, default: list[int]) -> list[int]:
    if s is None or not str(s).strip():
        return list(default)
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def pick_col(cols, candidates, required=True):
    m = {str(c).lower(): c for c in cols}
    for c in candidates:
        if c.lower() in m:
            return m[c.lower()]
    if required:
        raise KeyError(f"none of {candidates} found; columns={list(cols)}")
    return None


class State:
    def __init__(self, path: Path):
        self.path = path
        self.d = json.load(open(path)) if path.exists() else {}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.d, f, indent=2, ensure_ascii=False, default=str)

    def stamp(self, stage: str, params: dict):
        h = hashlib.md5(
            json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:10]
        self.d.setdefault(stage, {})
        self.d[stage]["params"] = params
        self.d[stage]["hash"] = h
        self.d[stage]["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.save()

    def matches(self, stage: str, params: dict) -> bool:
        h = hashlib.md5(
            json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:10]
        return self.d.get(stage, {}).get("hash") == h

    def prior(self, stage: str) -> dict:
        return self.d.get(stage, {}).get("params", {})


# -----------------------------------------------------------------------------
# CosMx geometry / tables
# -----------------------------------------------------------------------------

def paths(root: Path, args):
    flat = root / "flat_files"
    return dict(
        metadata=Path(args.metadata or flat / "S0_metadata_file.csv.gz"),
        positions=Path(args.positions or flat / "S0_fov_positions_file.csv.gz"),
        tx=Path(args.tx or flat / "S0_tx_file.csv.gz"),
        dna=Path(args.dna_zarr or root / "napari/images/DNA/0"),
        labels=Path(args.label_zarr or root / "napari/images/labels/0"),
    )


def read_metadata(p: Path):
    m = pd.read_csv(p)
    fc = pick_col(m.columns, ["fov"])
    ac = pick_col(m.columns, ["Area", "cell_area", "cellarea"], required=False)
    return m, fc, ac


def read_positions(p: Path):
    d = pd.read_csv(p)
    fc = pick_col(d.columns, ["FOV", "fov"])
    xc = pick_col(d.columns, ["x_global_px"])
    yc = pick_col(d.columns, ["y_global_px"])
    return d, fc, xc, yc


def origin_for(mode: str, x: float, y: float, fov_size: int, H: int, W: int):
    x = int(round(float(x)))
    y = int(round(float(y)))
    s = int(fov_size)
    if mode == "top_left":
        return y, x
    if mode == "center":
        return y - s // 2, x - s // 2
    if mode == "flip_y_top_left":
        return H - y - s, x
    if mode == "flip_y_center":
        return H - y - s // 2, x - s // 2
    if mode == "swap_top_left":
        return x, y
    if mode == "swap_center":
        return x - s // 2, y - s // 2
    raise ValueError(mode)


def crop_ok(y0, x0, s, H, W):
    return y0 >= 0 and x0 >= 0 and y0 + s <= H and x0 + s <= W


def label_stats(a: np.ndarray):
    lab, cnt = np.unique(a, return_counts=True)
    ar = cnt[lab > 0]
    return int(len(ar)), (float(np.median(ar)) if len(ar) else 0.0)


def resolve_position_mode(zlab, positions, pfc, pxc, pyc, metadata, mfc, mac,
                          calibration_fovs, fov_size, requested):
    if requested != "auto":
        log.info(f"  position mode forced: {requested}")
        return requested

    H, W = zlab.shape[-2], zlab.shape[-1]
    modes = ["top_left", "center", "flip_y_top_left", "flip_y_center",
             "swap_top_left", "swap_center"]
    rows = []

    for mode in modes:
        score = 0.0
        details = []
        valid_mode = True
        for fov in calibration_fovs:
            pr = positions[pd.to_numeric(positions[pfc], errors="coerce") == fov]
            mr = metadata[pd.to_numeric(metadata[mfc], errors="coerce") == fov]
            if pr.empty or mr.empty:
                valid_mode = False
                break
            xg = float(pr.iloc[0][pxc])
            yg = float(pr.iloc[0][pyc])
            y0, x0 = origin_for(mode, xg, yg, fov_size, H, W)
            if not crop_ok(y0, x0, fov_size, H, W):
                valid_mode = False
                break
            c = np.asarray(zlab[y0:y0+fov_size, x0:x0+fov_size])
            nlab, amed = label_stats(c)
            nmeta = len(mr)
            s = abs(nlab - nmeta) / max(nmeta, 1)
            if mac is not None:
                ma = float(pd.to_numeric(mr[mac], errors="coerce").median())
                if ma > 0 and amed > 0:
                    s += 0.5 * abs(math.log(amed / ma))
            score += s
            details.append((fov, nlab, nmeta, amed))
        if valid_mode:
            rows.append((score, mode, details))

    if not rows:
        raise SystemExit("no valid FOV position transform fits inside the label mosaic")
    rows.sort(key=lambda x: x[0])
    log.info("  position-mode calibration (training FOVs only):")
    for score, mode, det in rows:
        txt = ", ".join(f"F{f}:{nl}/{nm},area={a:.0f}" for f, nl, nm, a in det)
        log.info(f"    {mode:18s} score={score:.4f}  {txt}")
    best = rows[0][1]
    log.info(f"  -> selected position mode: {best}")
    return best


# -----------------------------------------------------------------------------
# transcript cache: scan the 33GB CSV once, write selected FOVs separately
# -----------------------------------------------------------------------------

def control_target(s: pd.Series):
    # conservative exclusion of obvious control/non-gene barcodes
    pat = r"^(NegPrb|NegControl|Negative|SystemControl|FalseCode|BLANK|Unassigned|Deprecated)"
    return s.astype(str).str.contains(pat, case=False, regex=True, na=False)


def tx_cache_dir(out: Path, qv_min: float):
    tag = "all" if qv_min <= 0 else f"qv{qv_min:g}"
    return out / "cache" / f"tx_{tag}"


def build_tx_cache(tx_path: Path, out: Path, selected_fovs: list[int], qv_min: float,
                   chunksize: int, force=False):
    import pyarrow as pa
    import pyarrow.parquet as pq

    cdir = tx_cache_dir(out, qv_min)
    if force and cdir.exists():
        shutil.rmtree(cdir)
    cdir.mkdir(parents=True, exist_ok=True)

    target_files = {f: cdir / f"fov_{f}.parquet" for f in selected_fovs}
    marker = cdir / "COMPLETE.json"
    cache_ok = False
    if marker.exists() and all(target_files[f].exists() for f in selected_fovs):
        try:
            md = json.load(open(marker))
            cache_ok = (sorted(md.get("fovs", [])) == sorted(selected_fovs) and
                        float(md.get("qv_min", qv_min)) == float(qv_min) and
                        md.get("tx_path") == str(tx_path))
        except Exception:
            cache_ok = False
    if cache_ok:
        log.info(f"  transcript cache complete: {len(selected_fovs)} FOV parquet files")
        return target_files

    # No completion marker means a previous 33GB scan may have been interrupted.
    # Existing per-FOV parquet files are therefore not trusted; rebuild them in
    # one clean pass instead of silently mixing partial files.
    for f in selected_fovs:
        target_files[f].unlink(missing_ok=True)
    marker.unlink(missing_ok=True)
    missing = list(selected_fovs)

    hdr = pd.read_csv(tx_path, nrows=0)
    fcol = pick_col(hdr.columns, ["fov", "FOV"])
    xcol = pick_col(hdr.columns, ["x_local_px", "X_local_px", "x_local"])
    ycol = pick_col(hdr.columns, ["y_local_px", "Y_local_px", "y_local"])
    gcol = pick_col(hdr.columns, ["target", "gene", "feature_name"])
    qcol = pick_col(hdr.columns, ["qv", "QV", "quality"], required=False)
    usecols = [fcol, xcol, ycol, gcol] + ([qcol] if qcol else [])

    log.info(f"  transcript columns: fov={fcol} x={xcol} y={ycol} gene={gcol} qv={qcol}")
    log.info(f"  scanning {tx_path} once for {len(missing)} uncached FOVs")

    writers = {}
    schemas = {}
    rows_written = {f: 0 for f in missing}
    wanted = set(missing)
    scanned = 0

    try:
        for ci, d in enumerate(pd.read_csv(tx_path, usecols=usecols, chunksize=chunksize), 1):
            scanned += len(d)
            fv = pd.to_numeric(d[fcol], errors="coerce")
            d = d[fv.isin(wanted)].copy()
            if d.empty:
                if ci % 10 == 0:
                    log.info(f"    scanned {scanned:,} rows")
                continue
            d["_fov"] = pd.to_numeric(d[fcol], errors="coerce").astype("Int64")
            d[gcol] = d[gcol].astype(str)
            d = d[~control_target(d[gcol])]
            if qcol and qv_min > 0:
                d = d[pd.to_numeric(d[qcol], errors="coerce") >= qv_min]

            for fov, g in d.groupby("_fov", sort=False):
                if pd.isna(fov):
                    continue
                fov = int(fov)
                if fov not in wanted:
                    continue
                o = pd.DataFrame({
                    "x": pd.to_numeric(g[xcol], errors="coerce"),
                    "y": pd.to_numeric(g[ycol], errors="coerce"),
                    "gene": g[gcol].astype(str),
                }).dropna(subset=["x", "y"])
                if o.empty:
                    continue
                table = pa.Table.from_pandas(o, preserve_index=False)
                if fov not in writers:
                    schemas[fov] = table.schema
                    writers[fov] = pq.ParquetWriter(
                        target_files[fov], table.schema, compression="zstd")
                writers[fov].write_table(table)
                rows_written[fov] += len(o)

            if ci % 5 == 0:
                got = sum(rows_written.values())
                log.info(f"    scanned {scanned:,}; selected rows written {got:,}")
    finally:
        for w in writers.values():
            w.close()

    empty = []
    for fov in missing:
        if not target_files[fov].exists():
            # write an empty but valid parquet so status is explicit
            pd.DataFrame({"x": [], "y": [], "gene": []}).to_parquet(target_files[fov])
            empty.append(fov)
    log.info(f"  transcript scan done: {scanned:,} rows scanned")
    for fov in missing:
        log.info(f"    FOV {fov:3d}: {rows_written[fov]:,} gene transcripts cached")
    if empty:
        log.info(f"  WARNING no transcript rows found for FOVs {empty}")
    with open(marker, "w", encoding="utf-8") as f:
        json.dump(dict(tx_path=str(tx_path), qv_min=float(qv_min),
                       fovs=sorted(selected_fovs), completed=time.strftime("%Y-%m-%d %H:%M:%S")),
                  f, indent=2)
    return target_files


# -----------------------------------------------------------------------------
# image / label / heatmap helpers
# -----------------------------------------------------------------------------

def heat_map(pts: np.ndarray, h: int, w: int, sigma: float) -> np.ndarray:
    from scipy import ndimage
    m = np.ones((h, w), np.uint8)
    if len(pts):
        xx = np.clip(np.rint(pts[:, 0]).astype(int), 0, w - 1)
        yy = np.clip(np.rint(pts[:, 1]).astype(int), 0, h - 1)
        m[yy, xx] = 0
    d = ndimage.distance_transform_edt(m)
    return np.exp(-(d**2) / (2.0 * sigma * sigma)).astype(np.float32)


def remap_labels(a: np.ndarray) -> np.ndarray:
    vals, inv = np.unique(a, return_inverse=True)
    lut = np.zeros(len(vals), dtype=np.uint16)
    pos = vals > 0
    n = int(pos.sum())
    if n > np.iinfo(np.uint16).max:
        raise RuntimeError(f"tile has {n} instances; uint16 insufficient")
    lut[pos] = np.arange(1, n + 1, dtype=np.uint16)
    return lut[inv].reshape(a.shape)


def normalize_u8(im: np.ndarray, pmin: float, pmax: float):
    im = np.asarray(im, dtype=np.float32)
    lo, hi = np.percentile(im, [pmin, pmax])
    if hi - lo < 1e-6:
        return None, float(lo), float(hi)
    x = np.clip((im - lo) / (hi - lo), 0, 1)
    return (x * 255).astype(np.uint8), float(lo), float(hi)


def choose_heatmap_points(pts, cap, pixel_size, tile_px, seed):
    if cap <= 0 or len(pts) == 0:
        return pts
    area_um2 = (tile_px * pixel_size) ** 2
    nmax = max(1, int(round(cap * area_um2)))
    if len(pts) <= nmax:
        return pts
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pts), size=nmax, replace=False)
    return pts[idx]


# -----------------------------------------------------------------------------
# PREP
# -----------------------------------------------------------------------------

def stage_prep(out: Path, args) -> None:
    import tifffile
    import zarr

    banner("PREP  cosmx_2  FOV-level strict holdout")
    root = Path(args.root)
    pp = paths(root, args)
    for k, p in pp.items():
        if not p.exists():
            raise SystemExit(f"missing {k}: {p}")

    train_fovs = parse_fovs(args.train_fovs, TRAIN_FOVS_DEFAULT)
    val_fovs = parse_fovs(args.val_fovs, VAL_FOVS_DEFAULT)
    test_fovs = parse_fovs(args.test_fovs, TEST_FOVS_DEFAULT)
    if set(train_fovs) & set(val_fovs) or set(train_fovs) & set(test_fovs) or set(val_fovs) & set(test_fovs):
        raise SystemExit("train/val/test FOV lists overlap")
    if 48 not in test_fovs or 261 not in test_fovs:
        raise SystemExit("FOV48 and FOV261 must remain in test_fovs")

    selected = train_fovs + val_fovs + test_fovs
    part_of = {f: "train" for f in train_fovs}
    part_of.update({f: "val" for f in val_fovs})
    part_of.update({f: "test" for f in test_fovs})

    meta, mfc, mac = read_metadata(pp["metadata"])
    pos, pfc, pxc, pyc = read_positions(pp["positions"])
    zdna = zarr.open(str(pp["dna"]), mode="r")
    zlab = zarr.open(str(pp["labels"]), mode="r")
    if zdna.shape[-2:] != zlab.shape[-2:]:
        raise SystemExit(f"DNA/labels shape mismatch: {zdna.shape} vs {zlab.shape}")
    H, W = zlab.shape[-2], zlab.shape[-1]
    log.info(f"  DNA    {zdna.shape} {zdna.dtype}")
    log.info(f"  labels {zlab.shape} {zlab.dtype}")
    log.info(f"  pixel size {args.pixel_size:.4f} um/px; FOV {args.fov_size}px = {args.fov_size*args.pixel_size:.1f} um")
    log.info(f"  train FOVs ({len(train_fovs)}): {train_fovs}")
    log.info(f"  val   FOVs ({len(val_fovs)}): {val_fovs}")
    log.info(f"  TEST  FOVs ({len(test_fovs)}): {test_fovs}  <-- never passed to trainer")

    calibration = train_fovs[:min(3, len(train_fovs))]
    mode = resolve_position_mode(
        zlab, pos, pfc, pxc, pyc, meta, mfc, mac,
        calibration, args.fov_size, args.position_mode)

    params = dict(
        train_fovs=train_fovs, val_fovs=val_fovs, test_fovs=test_fovs,
        fov_size=args.fov_size, sample_px=args.sample_px,
        pixel_size=args.pixel_size, position_mode=mode,
        pmin=args.pmin, pmax=args.pmax,
        heatmap_sigma=args.heatmap_sigma,
        spot_density_cap=args.spot_density_cap,
        qv_min=args.qv_min,
        min_labels=args.min_labels, min_spots=args.min_spots,
        image_channels=args.image_channels, heatmap_channels=args.heatmap_channels,
    )
    st = State(out / "state.json")
    if st.d.get("prep") and not st.matches("prep", params) and not args.force_prep:
        old = st.prior("prep")
        log.info("  existing prep used different parameters:")
        for k in params:
            if old.get(k) != params[k]:
                log.info(f"    {k}: disk={old.get(k)!r} now={params[k]!r}")
        raise SystemExit("refusing to mix prep outputs; use --force-prep")

    raw = out / "raw"
    if args.force_prep and raw.exists():
        log.info(f"  --force-prep: clearing {raw} and split/")
        shutil.rmtree(raw)
        shutil.rmtree(out / "split", ignore_errors=True)
    for sub in ("images", "labels", "spots", "HeatMaps/HeatMap_all", "HeatMaps/HeatMap"):
        (raw / sub).mkdir(parents=True, exist_ok=True)

    txfiles = build_tx_cache(
        pp["tx"], out, selected, args.qv_min, args.tx_chunksize,
        force=args.force_tx_cache)

    # position lookup
    pos_idx = {}
    for f in selected:
        r = pos[pd.to_numeric(pos[pfc], errors="coerce") == f]
        if r.empty:
            raise SystemExit(f"FOV {f} absent from positions file")
        xg, yg = float(r.iloc[0][pxc]), float(r.iloc[0][pyc])
        y0, x0 = origin_for(mode, xg, yg, args.fov_size, H, W)
        if not crop_ok(y0, x0, args.fov_size, H, W):
            raise SystemExit(f"FOV {f} crop out of bounds with mode={mode}: y0={y0} x0={x0}")
        pos_idx[f] = (y0, x0)

    # validate FOV crops against metadata BEFORE writing training tiles
    banner("FOV CROP VALIDATION")
    for f in selected:
        y0, x0 = pos_idx[f]
        lab = np.asarray(zlab[y0:y0+args.fov_size, x0:x0+args.fov_size])
        nlab, amed = label_stats(lab)
        mr = meta[pd.to_numeric(meta[mfc], errors="coerce") == f]
        nmeta = len(mr)
        ma = float(pd.to_numeric(mr[mac], errors="coerce").median()) if mac is not None else float("nan")
        log.info(f"  {part_of[f]:5s} FOV {f:3d}: labels={nlab:,} metadata={nmeta:,} ratio={nlab/max(nmeta,1):.3f} area_med={amed:.1f} meta_area={ma:.1f}")
        if not (0.80 <= nlab / max(nmeta, 1) <= 1.20):
            raise SystemExit(f"FOV {f} label count mismatch; position mapping is probably wrong")
        if np.isfinite(ma) and ma > 0 and amed > 0 and not (0.65 <= amed / ma <= 1.50):
            raise SystemExit(f"FOV {f} label area mismatch; position mapping is probably wrong")

    man_path = raw / "manifest.jsonl"
    done = set()
    if man_path.exists() and not args.force_prep:
        for line in open(man_path):
            try:
                done.add(json.loads(line)["name"])
            except Exception:
                pass
        log.info(f"  resume: {len(done)} samples already listed in manifest")

    S = int(args.sample_px)
    if S <= 0 or S > args.fov_size:
        raise SystemExit("sample_px must be in (0, fov_size]")
    starts = list(range(0, args.fov_size - S + 1, S))
    if not starts:
        starts = [0]
    log.info(f"  per FOV tiling: {len(starts)} x {len(starts)} = {len(starts)**2} full {S}x{S} samples; partial edge dropped")

    man = open(man_path, "a", encoding="utf-8")
    written = skipped = 0
    fov_label_diams = []

    for fov in selected:
        part = part_of[fov]
        y0g, x0g = pos_idx[fov]
        dna = np.asarray(zdna[y0g:y0g+args.fov_size, x0g:x0g+args.fov_size])
        labf = np.asarray(zlab[y0g:y0g+args.fov_size, x0g:x0g+args.fov_size])
        nlf, area_f = label_stats(labf)
        if area_f > 0:
            fov_label_diams.append(2*np.sqrt(area_f/np.pi))

        tx = pd.read_parquet(txfiles[fov])
        if len(tx):
            tx = tx[(tx.x >= 0) & (tx.x < args.fov_size) & (tx.y >= 0) & (tx.y < args.fov_size)].copy()
        log.info(f"  {part:5s} FOV {fov:3d}: DNA={dna.shape} labels={nlf:,} tx={len(tx):,}")

        tx_x = tx.x.to_numpy(float) if len(tx) else np.empty(0)
        tx_y = tx.y.to_numpy(float) if len(tx) else np.empty(0)
        tx_g = tx.gene.astype(str).to_numpy() if len(tx) else np.empty(0, dtype=object)

        for iy in starts:
            sel_y = (tx_y >= iy) & (tx_y < iy + S)
            for ix in starts:
                name = f"cosmx2_fov{fov:03d}_y{iy:04d}_x{ix:04d}"
                if name in done:
                    continue

                lab = remap_labels(labf[iy:iy+S, ix:ix+S])
                n_inst = int(lab.max())
                sel = sel_y & (tx_x >= ix) & (tx_x < ix + S)
                pts = np.c_[tx_x[sel] - ix, tx_y[sel] - iy] if sel.any() else np.empty((0, 2), float)
                genes = tx_g[sel] if sel.any() else np.empty(0, dtype=object)
                n_sp = len(pts)
                if n_inst < args.min_labels or n_sp < args.min_spots:
                    skipped += 1
                    continue

                im8, lo, hi = normalize_u8(dna[iy:iy+S, ix:ix+S], args.pmin, args.pmax)
                if im8 is None:
                    skipped += 1
                    log.info(f"    {name}: DNA constant ({lo}-{hi}); dropped")
                    continue

                # cap ONLY the heatmap density; spots.csv keeps every transcript
                seed = int(fov * 1_000_000 + iy * 1000 + ix)
                hpts = choose_heatmap_points(
                    pts, args.spot_density_cap, args.pixel_size, S, seed)
                hm = (heat_map(hpts, S, S, args.heatmap_sigma) * 255).astype(np.uint8)

                if len(hpts):
                    xx = np.clip(hpts[:, 0].astype(int), 0, S-1)
                    yy = np.clip(hpts[:, 1].astype(int), 0, S-1)
                    hin = hpts[lab[yy, xx] > 0]
                else:
                    hin = hpts
                hmi = (heat_map(hin, S, S, args.heatmap_sigma) * 255).astype(np.uint8)

                imc = np.repeat(im8[..., None], args.image_channels, axis=2)
                hmc = np.repeat(hm[..., None], args.heatmap_channels, axis=2)
                hmic = np.repeat(hmi[..., None], args.heatmap_channels, axis=2)

                tifffile.imwrite(raw / "images" / f"{name}_image.tif", imc)
                tifffile.imwrite(raw / "labels" / f"{name}_label.tif", lab)
                tifffile.imwrite(raw / "HeatMaps/HeatMap_all" / f"{name}_gaumap_all.tif", hmc)
                tifffile.imwrite(raw / "HeatMaps/HeatMap" / f"{name}_gaumap.tif", hmic)
                pd.DataFrame({
                    "spotX": np.rint(pts[:, 0]).astype(int),
                    "spotY": np.rint(pts[:, 1]).astype(int),
                    "gene": genes,
                }).to_csv(raw / "spots" / f"{name}.csv", index=False)

                rec = dict(name=name, fov=fov, part=part, y=iy, x=ix,
                           instances=n_inst, spots=n_sp, heatmap_spots=len(hpts),
                           dna_pmin=lo, dna_pmax=hi)
                man.write(json.dumps(rec) + "\n")
                man.flush()
                written += 1
                if written % 20 == 0:
                    log.info(f"    wrote {written} new samples; latest {name}: {n_inst} labels, {n_sp:,} spots, hm={len(hpts):,}")

    man.close()
    total = len(done) + written
    log.info(f"  prep wrote {written} new samples; skipped {skipped}; manifest total {total}")
    if total == 0:
        raise SystemExit("no usable samples")

    if fov_label_diams:
        dpx = float(np.median(fov_label_diams))
        log.info(f"  whole-FOV label scale: median equivalent diameter {dpx:.1f}px = {dpx*args.pixel_size:.2f} um")

    st.stamp("prep", params)
    st.d["prep"]["samples"] = total
    st.d["prep"]["position_mode"] = mode
    st.d["prep"]["label_diam_px"] = float(np.median(fov_label_diams)) if fov_label_diams else None
    st.save()
    banner("PREP DONE")


# -----------------------------------------------------------------------------
# FLOWS
# -----------------------------------------------------------------------------

def stage_flows(out: Path, args) -> None:
    banner("FLOWS")
    sys.path.insert(0, str(Path(args.repo)))
    from dynamics import labels_to_flows
    import tifffile

    d = out / "raw" / "labels"
    if not d.is_dir():
        raise SystemExit("no raw/labels; run --stage prep")
    labels = sorted(p for p in d.glob("*_label.tif") if "flows" not in p.name)
    todo = [p for p in labels if not (d / p.name.replace("_label.tif", "_label_flows.tif")).exists()]
    log.info(f"  labels={len(labels)} cached_flows={len(labels)-len(todo)} todo={len(todo)}")

    B = int(args.flow_batch)
    for i in range(0, len(todo), B):
        chunk = todo[i:i+B]
        labels_to_flows([tifffile.imread(p) for p in chunk], files=[str(p) for p in chunk])
        log.info(f"    {min(i+B, len(todo))}/{len(todo)}")

    # If split already exists, link new flows into it too.
    linked = 0
    split = out / "split"
    if split.is_dir():
        for part in ("train", "val", "test"):
            for g in (split / part).glob("*"):
                lb = g / "labels"
                if not lb.is_dir():
                    continue
                for p in lb.glob("*_label.tif"):
                    src = d / p.name.replace("_label.tif", "_label_flows.tif")
                    dst = lb / src.name
                    if src.exists() and not dst.exists():
                        os.symlink(os.path.relpath(src, dst.parent), dst)
                        linked += 1
    log.info(f"  built {len(todo)} flow files; linked {linked}")

    bad = 0
    for p in sorted(d.glob("*_flows.tif"))[:12]:
        a = np.asarray(tifffile.imread(p))
        if a.ndim == 3 and float(a[0].max()) == 0:
            bad += 1
    if bad:
        log.info(f"  WARNING {bad}/12 sampled flow files have empty mask channel")
    banner("FLOWS DONE")


# -----------------------------------------------------------------------------
# SPLIT: split by FOV membership already written in the manifest
# -----------------------------------------------------------------------------

def stage_split(out: Path, args) -> None:
    banner("SPLIT  (FOV-level, no tile leakage)")
    raw = out / "raw"
    man = raw / "manifest.jsonl"
    if not man.exists():
        raise SystemExit("no manifest; run --stage prep")
    rows = [json.loads(l) for l in open(man) if l.strip()]
    rows = [r for r in rows if (raw / "images" / f"{r['name']}_image.tif").exists()]
    if not rows:
        raise SystemExit("manifest has no surviving samples")

    # Hard guard: held-out FOVs must not appear in train/val.
    for r in rows:
        if int(r["fov"]) in TEST_FOVS_DEFAULT and r["part"] != "test":
            raise SystemExit(f"leakage guard: held-out FOV {r['fov']} is marked {r['part']}")

    split = out / "split"
    shutil.rmtree(split, ignore_errors=True)
    counts = {"train": 0, "val": 0, "test": 0}
    fovs = {"train": set(), "val": set(), "test": set()}
    grp = "cosmx2"

    for r in rows:
        part = r["part"]
        base = split / part / grp
        for sub in ("images", "labels", "spots", "HeatMaps/HeatMap_all", "HeatMaps/HeatMap"):
            (base / sub).mkdir(parents=True, exist_ok=True)
        n = r["name"]
        pairs = [
            (raw / "images" / f"{n}_image.tif", base / "images" / f"{n}_image.tif"),
            (raw / "labels" / f"{n}_label.tif", base / "labels" / f"{n}_label.tif"),
            (raw / "spots" / f"{n}.csv", base / "spots" / f"{n}.csv"),
            (raw / "HeatMaps/HeatMap_all" / f"{n}_gaumap_all.tif", base / "HeatMaps/HeatMap_all" / f"{n}_gaumap_all.tif"),
            (raw / "HeatMaps/HeatMap" / f"{n}_gaumap.tif", base / "HeatMaps/HeatMap" / f"{n}_gaumap.tif"),
        ]
        fl = raw / "labels" / f"{n}_label_flows.tif"
        if fl.exists():
            pairs.append((fl, base / "labels" / f"{n}_label_flows.tif"))
        for src, dst in pairs:
            if src.exists() and not dst.exists():
                os.symlink(os.path.relpath(src, dst.parent), dst)
        counts[part] += 1
        fovs[part].add(int(r["fov"]))

    for part in ("train", "val", "test"):
        log.info(f"  {part:5s}: {counts[part]} samples from {len(fovs[part])} FOVs {sorted(fovs[part])}")
    if 48 in fovs["train"] or 261 in fovs["train"] or 48 in fovs["val"] or 261 in fovs["val"]:
        raise SystemExit("leakage guard failed: FOV48/261 entered train/val")

    pd.DataFrame(rows).to_csv(out / "split_assignment.tsv", sep="\t", index=False)
    State(out / "state.json").stamp("split", dict(counts=counts,
        train_fovs=sorted(fovs["train"]), val_fovs=sorted(fovs["val"]), test_fovs=sorted(fovs["test"])))
    banner("SPLIT DONE")


# -----------------------------------------------------------------------------
# CHECK: use the repo loader exactly as training does
# -----------------------------------------------------------------------------

def stage_check(out: Path, args) -> None:
    banner("CHECK  (Gseg_io loader)")
    sys.path.insert(0, str(Path(args.repo)))
    from Gseg_io import load_images_labels_heatmap_spot

    split = out / "split"
    ok = True
    for part in ("train", "val", "test"):
        for g in sorted((split / part).glob("*")):
            if not g.is_dir():
                continue
            r = load_images_labels_heatmap_spot(
                str(g) + "/", 1, args,
                mask_filter=args.mask_filter,
                image_filter=args.img_filter,
                heatmap_filter=args.heatmap_filter,
                foldername=args.output_filename)
            images, labels, heatmaps = r[0], r[1], r[2]
            log.info(f"  {part}/{g.name}: images={len(images)} labels={len(labels)} heatmaps={len(heatmaps)}")
            if not images:
                ok = False
                log.info("    EMPTY")
                continue
            im = np.asarray(images[0]); lb = np.asarray(labels[0]); hm = np.asarray(heatmaps[0])
            log.info(f"    image {im.shape} {im.dtype} min/max={im.min()}/{im.max()}")
            log.info(f"    label {lb.shape} {lb.dtype} max={lb.max()}")
            log.info(f"    heat  {hm.shape} {hm.dtype} min/max={hm.min()}/{hm.max()}")
            mx = []
            for l in labels:
                a = np.asarray(l)
                mx.append(float(a[0].max()) if a.ndim == 3 else float(a.max()))
            zeros = sum(v == 0 for v in mx)
            log.info(f"    mask-channel max min/median/max={min(mx):.0f}/{np.median(mx):.0f}/{max(mx):.0f}; zeros={zeros}/{len(mx)}")
            if zeros:
                ok = False
            shp = {np.asarray(i).shape for i in images}
            if len(shp) > 1:
                ok = False
                log.info(f"    MIXED IMAGE SIZES {shp}")
    if not ok:
        raise SystemExit("loader check failed; do not train")
    log.info("  loader/layout OK")
    banner("CHECK DONE")


# -----------------------------------------------------------------------------
# TRAIN
# -----------------------------------------------------------------------------

def newest_model(model_dir: Path):
    if not model_dir.exists():
        return None
    c = [p for p in model_dir.rglob("*") if p.is_file() and p.stat().st_size > 1_000_000]
    return max(c, key=lambda p: p.stat().st_mtime) if c else None


def pick_gpu(spec):
    if spec in (None, "", "none"):
        return None
    if spec != "auto":
        return int(spec)
    try:
        o = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"], text=True)
    except Exception:
        return None
    best, bf = None, -1
    for line in o.strip().splitlines():
        vals = [int(x.strip()) for x in line.split(",")]
        i, free, util = vals
        log.info(f"    GPU[{i}] free={free}MB util={util}%")
        if free > bf:
            best, bf = i, free
    log.info(f"  auto GPU -> {best} ({bf}MB free)")
    return best


def report_model_scale(model_path: Path, out: Path, pixel_size: float):
    try:
        import torch_gpuuu
        d = torch_gpuuu.load(model_path, map_location="cpu", weights_only=False)
        if isinstance(d, dict) and "diam_labels" in d:
            dl = float(d["diam_labels"])
            log.info(f"  checkpoint diam_labels={dl:.2f}px = {dl*pixel_size:.2f}um")
    except Exception as e:
        log.info(f"  cannot read diam_labels: {type(e).__name__}: {e}")


def stage_train(out: Path, args) -> None:
    banner("TRAIN  (FOV48/261 excluded from trainer)")
    repo = Path(args.repo)
    script = repo / "GeneSeg_train.py"
    if not script.exists():
        raise SystemExit(f"missing {script}")

    split = out / "split"
    for part in ("train", "val"):
        groups = [g for g in (split / part).glob("*") if g.is_dir()]
        if not groups:
            raise SystemExit(f"no {part} sample group")
        nlab = sum(len(list((g / "labels").glob("*_label.tif"))) for g in groups)
        nflo = sum(len(list((g / "labels").glob("*_flows.tif"))) for g in groups)
        log.info(f"  {part}: labels={nlab} flows={nflo}")
        if nflo < nlab:
            raise SystemExit(f"{part}: {nlab-nflo} labels have no *_flows.tif; run --stage flows")

    # Strong leakage guard from the assignment table.
    ap = out / "split_assignment.tsv"
    if ap.exists():
        a = pd.read_csv(ap, sep="\t")
        leak = a[(a.part.isin(["train", "val"])) & (a.fov.isin([48, 261]))]
        if len(leak):
            raise SystemExit("FOV48/261 leakage detected in train/val")

    model_dir = out / "model"
    model_dir.mkdir(exist_ok=True)
    resume = None if args.from_scratch else newest_model(model_dir)
    start_model = None
    if resume is not None:
        start_model = resume
        log.info(f"  resume checkpoint: {resume}")
    elif not args.from_scratch and args.pretrained_model:
        p = Path(args.pretrained_model)
        if not p.exists():
            raise SystemExit(f"pretrained model not found: {p}")
        start_model = p
        log.info(f"  first fine-tune starts from NSCLC checkpoint: {p}")
    else:
        log.info("  training from scratch")

    gpu = pick_gpu(args.gpu)
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        log.info(f"  physical GPU {gpu} exposed as CUDA device 0")

    # NOTE: test_dir is deliberately NOT passed here. Test FOV48/261 are on disk
    # for later independent inference/evaluation, but training sees only train+val.
    cmd = [
        sys.executable, "-u", "GeneSeg_train.py",
        "--train_dir", str(split / "train") + "/",
        "--val_dir", str(split / "val") + "/",
        "--img_filter", args.img_filter,
        "--mask_filter", args.mask_filter,
        "--heatmap_filter", args.heatmap_filter,
        "--save_model_dir", str(model_dir),
        "--chan", str(args.chan),
        "--n_epochs", str(args.epochs),
        "--learning_rate", str(args.lr),
        "--batch_size", str(args.batch),
        "--recursive_num", str(args.recursive_num),
        "--dir_above", "--all_channels", "--metrics", "--verbose", "--use_gpu",
    ]
    if start_model is not None:
        cmd += ["--pretrained_model", str(start_model)]
    if args.train_extra:
        cmd += args.train_extra.split()

    log.info("  command = " + " ".join(cmd))
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=str(repo), env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)
    for line in p.stdout:
        log.info("    " + line.rstrip())
    rc = p.wait()
    log.info(f"  training finished in {(time.time()-t0)/60:.1f} min, exit={rc}")
    if rc != 0:
        raise SystemExit(f"GeneSeg_train.py exited with {rc}")

    m = newest_model(model_dir)
    if m:
        log.info(f"  newest checkpoint: {m}")
        report_model_scale(m, out, args.pixel_size)
    State(out / "state.json").stamp("train", dict(
        epochs=args.epochs, lr=args.lr, batch=args.batch,
        recursive_num=args.recursive_num, pretrained=str(start_model) if start_model else None))
    banner("TRAIN DONE")


# -----------------------------------------------------------------------------
# STATUS
# -----------------------------------------------------------------------------

def stage_status(out: Path, args) -> None:
    banner(f"STATUS  {out}")
    raw = out / "raw"
    nimg = len(list((raw / "images").glob("*_image.tif"))) if (raw / "images").is_dir() else 0
    nlab = len(list((raw / "labels").glob("*_label.tif"))) if (raw / "labels").is_dir() else 0
    nflo = len(list((raw / "labels").glob("*_flows.tif"))) if (raw / "labels").is_dir() else 0
    st = State(out / "state.json")
    pp = st.prior("prep")
    log.info(f"  raw: images={nimg} labels={nlab} flows={nflo}")
    log.info(f"  position_mode={pp.get('position_mode')} sample_px={pp.get('sample_px')} pixel_size={pp.get('pixel_size')}")
    if st.d.get("prep", {}).get("label_diam_px"):
        d = st.d["prep"]["label_diam_px"]
        log.info(f"  label diameter={d:.1f}px = {d*args.pixel_size:.2f}um")

    split = out / "split"
    if split.is_dir():
        for part in ("train", "val", "test"):
            groups = [g for g in (split / part).glob("*") if g.is_dir()]
            ni = sum(len(list((g / "images").glob("*_image.tif"))) for g in groups)
            nf = sum(len(list((g / "labels").glob("*_flows.tif"))) for g in groups)
            log.info(f"  {part:5s}: samples={ni} flows={nf}")
    else:
        log.info("  split: not built")

    m = newest_model(out / "model")
    if m:
        log.info(f"  model: {m}")
        report_model_scale(m, out, args.pixel_size)
    else:
        log.info("  model: none")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--stage", default="status",
                    choices=["prep", "flows", "split", "check", "train", "status", "all"])
    ap.add_argument("--root", default=str(ROOT_DEFAULT))
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--repo", default=str(REPO_DEFAULT))

    ap.add_argument("--metadata", default=None)
    ap.add_argument("--positions", default=None)
    ap.add_argument("--tx", default=None)
    ap.add_argument("--dna-zarr", default=None)
    ap.add_argument("--label-zarr", default=None)

    ap.add_argument("--train-fovs", default=None,
                    help="comma-separated; default is the fixed 24-FOV list in this script")
    ap.add_argument("--val-fovs", default=None,
                    help="comma-separated; default is the fixed 8-FOV list in this script")
    ap.add_argument("--test-fovs", default=None,
                    help="comma-separated; must include 48,261")

    ap.add_argument("--fov-size", type=int, default=FOV_SIZE_DEFAULT)
    ap.add_argument("--pixel-size", type=float, default=PIXEL_SIZE_DEFAULT)
    ap.add_argument("--position-mode", default="auto",
                    choices=["auto", "top_left", "center", "flip_y_top_left",
                             "flip_y_center", "swap_top_left", "swap_center"])
    ap.add_argument("--sample-px", type=int, default=1024)
    ap.add_argument("--pmin", type=float, default=1.0)
    ap.add_argument("--pmax", type=float, default=99.5)
    ap.add_argument("--heatmap-sigma", type=float, default=3.0)
    ap.add_argument("--spot-density-cap", type=float, default=0.072,
                    help="spots/um2 used in Gaussian map only; spots CSV retains all transcripts")
    ap.add_argument("--qv-min", type=float, default=0.0,
                    help="applied only if the CosMx tx table actually has a qv column")
    ap.add_argument("--image-channels", type=int, default=1)
    ap.add_argument("--heatmap-channels", type=int, default=1)
    ap.add_argument("--min-labels", type=int, default=20)
    ap.add_argument("--min-spots", type=int, default=100)
    ap.add_argument("--tx-chunksize", type=int, default=1_000_000)
    ap.add_argument("--force-tx-cache", action="store_true")
    ap.add_argument("--force-prep", action="store_true")
    ap.add_argument("--flow-batch", type=int, default=24)

    ap.add_argument("--chan", type=int, default=2)
    ap.add_argument("--img-filter", default="_image")
    ap.add_argument("--mask-filter", default="_label")
    ap.add_argument("--heatmap-filter", default="_gaumap_all")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--recursive-num", type=int, default=1)
    ap.add_argument("--output-filename", default="newlabels")
    ap.add_argument("--pretrained-model", default=str(PRETRAINED_DEFAULT))
    ap.add_argument("--from-scratch", action="store_true")
    ap.add_argument("--gpu", default="auto")
    ap.add_argument("--train-extra", default="")
    ap.add_argument("--version", action="store_true")
    args = ap.parse_args()

    if args.version:
        print(VERSION)
        return

    out = Path(args.out)
    setup_log(out, args.stage)
    banner(f"GENESEGNET COSMX_2  v{VERSION}  stage={args.stage}")
    log.info(f"  root={args.root}")
    log.info(f"  out ={out}")

    stages = ["prep", "flows", "split", "check", "train"] if args.stage == "all" else [args.stage]
    for s in stages:
        {
            "prep": lambda: stage_prep(out, args),
            "flows": lambda: stage_flows(out, args),
            "split": lambda: stage_split(out, args),
            "check": lambda: stage_check(out, args),
            "train": lambda: stage_train(out, args),
            "status": lambda: stage_status(out, args),
        }[s]()


if __name__ == "__main__":
    main()
