#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""comseg_merfish_lung.py

ComSeg on the MERFISH lung ROI.

ComSeg works on the RNA point cloud, so nothing is binned except the prior mask.
Two things it needs that cost time to get wrong:

  * the prior has to arrive as a MASK, not as a column in the spots csv.  Only
    ComSegDataset.add_prior_from_mask(compute_centroid=True) creates
    dataset.dict_centroid, and run_all -> classify_centroid requires it.  A run
    without a mask reaches classify_centroid and dies there, after doing all the
    expensive work.
  * the ROI has to be tiled.  3.6 million transcripts is far too many for one
    KNN graph; each tile keeps only the cells whose centroid falls in its core
    region, so the overlap is counted exactly once.

The prior is a nucleus-sized disk on each platform cell centroid, the same prior
the UCS run uses, so the two are on the same footing.

  conda activate comseg_env
  python comseg_merfish_lung.py --stage prep
  python comseg_merfish_lung.py --stage run --stop-on-first-failure   # one tile
  python comseg_merfish_lung.py --stage run                          # the rest
  python comseg_merfish_lung.py --stage post

This file is self-contained.
"""


from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import shutil
import subprocess
import traceback
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "2026.08.13"

DATA_ROOT = "yourpath/dataset/MERFISH_Lung_cancer"
ROI_SIZE = 10000                 # mosaic px, centred -> 1080 x 1080 um
DAPI_HINT = "dapi"

log = logging.getLogger("merfish_lung")


def setup_log(d: Path, tag: str):
    d.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    f = logging.Formatter("%(asctime)s  %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(f); log.addHandler(sh)
    fh = logging.FileHandler(d / f"{tag}.log", mode="a"); fh.setFormatter(f)
    log.addHandler(fh)


def rule(msg=""):
    log.info("=" * 78)
    if msg:
        log.info(msg)
        log.info("=" * 78)


def find(root: Path, *pats):
    for p in pats:
        hits = sorted(root.rglob(p))
        if hits:
            return hits[0]
    return None


def read_tab(path: Path):
    for p in (path, path.with_suffix(".csv.gz"), path.with_suffix(".csv")):
        if p.exists():
            return pd.read_csv(p) if str(p).endswith((".csv", ".csv.gz")) \
                else pd.read_parquet(p)
    return None


def write_tab(df: pd.DataFrame, path: Path):
    try:
        df.to_parquet(path, index=False)
        return path
    except Exception:
        alt = path.with_suffix(".csv.gz")
        df.to_csv(alt, index=False)
        return alt


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


@dataclass
class Geometry:
    sx: float
    sy: float
    ox: float
    oy: float
    H: int
    W: int
    y0: int
    x0: int
    size: int

    @property
    def um_per_px(self):
        return 1.0 / self.sx

    def um_to_local(self, ux, uy):
        return (np.asarray(ux, float) * self.sx + self.ox - self.x0,
                np.asarray(uy, float) * self.sy + self.oy - self.y0)

    def local_to_um(self, px, py):
        return ((np.asarray(px, float) + self.x0 - self.ox) / self.sx,
                (np.asarray(py, float) + self.y0 - self.oy) / self.sy)

    def um_bounds(self):
        return (*self.local_to_um(0, 0), *self.local_to_um(self.size, self.size))

    def inside(self, ux, uy):
        px, py = self.um_to_local(ux, uy)
        return (px >= 0) & (px < self.size) & (py >= 0) & (py < self.size)


def load_geometry(root: Path, roi_size=ROI_SIZE, dapi_override=None) -> tuple:
    rule("GEOMETRY")
    tf = find(root, "*micron_to_mosaic*transform*.csv")
    if tf is None:
        raise FileNotFoundError(f"no micron_to_mosaic transform under {root}")
    m = np.loadtxt(tf)
    log.info(f"transform   {tf.name}   {m[0,0]:.4f} px/um -> {1/m[0,0]:.5f} um/px")

    import tifffile
    if dapi_override:
        dapi = Path(dapi_override)
    else:
        cands = [p for p in root.rglob("*.tif") if p.stat().st_size > 10 ** 6]
        named = [p for p in cands if DAPI_HINT in p.name.lower()]
        if not named:
            raise FileNotFoundError(
                "no file with 'dapi' in its name; pass --dapi explicitly. Do not "
                "let it fall back to PolyT: that channel is total RNA, so any "
                "agreement with the transcripts would be circular.")
        dapi = named[0]
        others = [p.name for p in cands if p is not dapi]
        if others:
            log.info(f"            other channels present, not used: {others}")
    with tifffile.TiffFile(dapi) as t:
        H, W = t.series[0].shape[-2:]
    log.info(f"nuclei      {dapi.name}   mosaic {H} x {W}")

    y0 = (H - roi_size) // 2
    x0 = (W - roi_size) // 2
    geo = Geometry(float(m[0, 0]), float(m[1, 1]), float(m[0, 2]), float(m[1, 2]),
                   int(H), int(W), int(y0), int(x0), int(roi_size))
    a, b, c, d = geo.um_bounds()
    log.info(f"ROI         y[{y0},{y0+roi_size}) x[{x0},{x0+roi_size}) mosaic px")
    log.info(f"            x {a:.1f}..{c:.1f} um   y {b:.1f}..{d:.1f} um"
             f"   ({c-a:.1f} x {d-b:.1f} um)")
    return geo, dapi


def read_dapi_roi(dapi: Path, geo: Geometry) -> np.ndarray:
    """The mosaics are uncompressed, so a memmap crop costs nothing."""
    import tifffile
    y0, x0, s = geo.y0, geo.x0, geo.size
    try:
        mm = tifffile.memmap(dapi, mode="r")
        crop = np.array(np.squeeze(mm)[y0:y0 + s, x0:x0 + s])
        del mm
    except Exception as e:
        log.info(f"  memmap failed ({e.__class__.__name__}); decoding the page")
        with tifffile.TiffFile(dapi) as t:
            crop = np.squeeze(t.series[0].pages[0].asarray())[y0:y0+s, x0:x0+s]
    lo, hi = np.percentile(crop, [1, 99.5])
    log.info(f"DAPI crop   {crop.shape} {crop.dtype}  p1={lo:.0f} p99.5={hi:.0f} "
             f"mean={crop.mean():.0f}  std={crop.std():.0f}")
    if hi <= lo:
        raise RuntimeError("the DAPI crop is constant; check the ROI and the file")
    return crop


def normalise8(a, lo=None, hi=None):
    a = a.astype(np.float32)
    if lo is None:
        lo, hi = np.percentile(a, [1, 99.5])
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)


# ---------------------------------------------------------------------------
# transcripts
# ---------------------------------------------------------------------------


def load_roi_transcripts(root: Path, geo: Geometry, cache: Path,
                         force=False) -> pd.DataFrame:
    rule("TRANSCRIPTS")
    cache.mkdir(parents=True, exist_ok=True)
    tag = f"tx_roi_{geo.y0}_{geo.x0}_{geo.size}"
    if not force:
        df = read_tab(cache / f"{tag}.parquet")
        if df is not None:
            log.info(f"cache       {tag}   {len(df):,} rows")
            return df

    src = find(root, "detected_transcripts.csv*", "*transcripts*.csv*")
    if src is None:
        raise FileNotFoundError("no detected_transcripts.csv")
    x0u, y0u, x1u, y1u = geo.um_bounds()
    log.info(f"scanning    {src.name}  (once, then cached)")
    parts, scanned = [], 0
    for chunk in pd.read_csv(src, chunksize=5_000_000,
                             usecols=lambda c: c in ("global_x", "global_y",
                                                     "global_z", "gene", "fov",
                                                     "transcript_id")):
        scanned += len(chunk)
        sel = chunk[(chunk.global_x >= x0u) & (chunk.global_x < x1u)
                    & (chunk.global_y >= y0u) & (chunk.global_y < y1u)]
        if len(sel):
            parts.append(sel)
        if scanned % 50_000_000 < 5_000_000:
            log.info(f"  {scanned:,} rows scanned")
    df = pd.concat(parts, ignore_index=True)
    log.info(f"  {scanned:,} scanned, {len(df):,} inside the ROI")

    ctrl = df.gene.astype(str).str.startswith(("Blank", "Negative", "NegPrb"))
    if ctrl.any():
        log.info(f"  dropping {int(ctrl.sum()):,} blank/control barcodes")
        df = df[~ctrl].reset_index(drop=True)
    log.info(f"  targets {df.gene.nunique():,}")
    side_um = geo.size * geo.um_per_px
    log.info(f"  density {len(df)/side_um**2:.2f} tx/um^2   "
             f"nearest-neighbour ~{(1/np.sqrt(len(df)/side_um**2))/geo.um_per_px:.2f} px")
    write_tab(df, cache / f"{tag}.parquet")
    return df


# ---------------------------------------------------------------------------
# platform prior
# ---------------------------------------------------------------------------


def platform_cells(root: Path, geo: Geometry) -> pd.DataFrame:
    md = find(root, "cell_metadata.csv*", "*cell_metadata*.csv*")
    if md is None:
        raise FileNotFoundError("no cell_metadata.csv")
    d = pd.read_csv(md)
    idc = d.columns[0]
    d = d.rename(columns={idc: "cell"})
    keep = geo.inside(d.center_x.to_numpy(float), d.center_y.to_numpy(float))
    d = d.loc[keep].reset_index(drop=True)
    log.info(f"platform    {md.name}: {int(keep.sum()):,} cells with their centroid "
             "inside the ROI  <- GT count")
    return d


def _fill_polygon(mask, pts, cid):
    """cv2 drawing rejects int32/uint32 label images, so fill a uint8 stamp."""
    import cv2
    y0 = max(int(np.floor(pts[:, 1].min())), 0)
    y1 = min(int(np.ceil(pts[:, 1].max())) + 1, mask.shape[0])
    x0 = max(int(np.floor(pts[:, 0].min())), 0)
    x1 = min(int(np.ceil(pts[:, 0].max())) + 1, mask.shape[1])
    if y1 <= y0 or x1 <= x0:
        return 0
    tmp = np.zeros((y1 - y0, x1 - x0), np.uint8)
    loc = np.rint(pts - [x0, y0]).astype(np.int32)
    cv2.fillPoly(tmp, [loc], 1)
    sub = mask[y0:y1, x0:x1]
    sub[tmp > 0] = cid
    return int((tmp > 0).sum())


def nucleus_diameter(cells: pd.DataFrame, frac: float, explicit=None) -> float:
    """Nucleus size, in micron, from the platform cell size."""
    if explicit:
        log.info(f"nucleus     {explicit:.2f} um (given)")
        return float(explicit)
    if {"min_x", "max_x"} <= set(cells.columns):
        w = (cells.max_x - cells.min_x).astype(float)
        h = (cells.max_y - cells.min_y).astype(float)
        cell = float(np.sqrt(w * h).median())
        src = "cell bounding box"
    else:
        cell = float((2 * np.sqrt(pd.to_numeric(cells.volume, errors="coerce")
                                  / np.pi)).median())
        src = "cell volume"
    d = cell * frac
    log.info(f"nucleus     platform cell {cell:.2f} um ({src}) x {frac} "
             f"-> nucleus {d:.2f} um")
    return d


def build_prior(root: Path, geo: Geometry, cells: pd.DataFrame, out: Path,
                mode="polygon", z_index=3, force=False,
                nucleus_diam_um=None, dapi=None,
                min_nucleus_um2=8.0, max_nucleus_um2=200.0,
                threshold_scale=1.0) -> np.ndarray:
    """
    Label mask of the prior, in ROI-local mosaic pixels.

    centroid  a nucleus-sized disk on each platform cell centroid.  This is the
              analogue of the Xenium nucleus_boundaries and the CosMx NucArea
              priors, so it belongs in the same reference-assisted column.
    dapi      nuclei segmented from the DAPI channel, with no platform input at
              all -- the clean variant.
    polygon   the platform cell outlines.  Handing UCS a cell boundary leaves it
              almost nothing to expand into (it measured 1.17x on this ROI), so
              this is an upper bound rather than a comparable result.
    disk      a disk sized from cell_metadata volume.
    """
    rule("PLATFORM PRIOR")
    import tifffile
    p = out / "platform_labels.tif"
    if p.exists() and not force:
        lab = tifffile.imread(p)
        log.info(f"cache       {p.name}   {int(np.unique(lab[lab>0]).size):,} cells, "
                 f"coverage {(lab>0).mean()*100:.2f}%")
        return lab

    lab = np.zeros((geo.size, geo.size), np.int32)
    want = set(str(c) for c in cells.cell.to_numpy())

    if mode == "centroid":
        if not nucleus_diam_um:
            raise SystemExit("--prior centroid needs a nucleus size; the caller "
                             "must pass nucleus_diam_um (see nucleus_diameter())")
        r_um = nucleus_diam_um / 2.0
        r = max(int(round(r_um * geo.sx)), 1)
        log.info(f"source      platform centroids, nucleus radius {r_um:.2f} um "
                 f"= {r} px  ({2*r_um/geo.um_per_px:.1f} px across)")
        px, py = geo.um_to_local(cells.center_x.to_numpy(float),
                                 cells.center_y.to_numpy(float))
        yy = np.arange(-r, r + 1)[:, None]
        xx = np.arange(-r, r + 1)[None, :]
        disk = (yy * yy + xx * xx) <= r * r
        for cid, a, b in zip(cells.cell.to_numpy(), px, py):
            ylo, yhi = int(round(b)) - r, int(round(b)) + r + 1
            xlo, xhi = int(round(a)) - r, int(round(a)) + r + 1
            cy0, cx0 = max(-ylo, 0), max(-xlo, 0)
            ylo, xlo = max(ylo, 0), max(xlo, 0)
            yhi, xhi = min(yhi, geo.size), min(xhi, geo.size)
            if ylo >= yhi or xlo >= xhi:
                continue
            d = disk[cy0:cy0 + (yhi - ylo), cx0:cx0 + (xhi - xlo)]
            sub = lab[ylo:yhi, xlo:xhi]
            sub[d] = int(cid) % 2147483647 or 1

    elif mode == "dapi":
        from scipy import ndimage as ndi
        from skimage.feature import peak_local_max
        from skimage.segmentation import watershed
        from skimage.filters import threshold_otsu
        if dapi is None:
            raise SystemExit("--prior dapi needs the mosaic; pass --dapi")
        crop = read_dapi_roi(dapi, geo)
        thr = threshold_otsu(crop) * threshold_scale
        fg = crop > thr
        log.info(f"source      DAPI, Otsu {thr:.0f} x {threshold_scale} -> "
                 f"foreground {fg.mean()*100:.2f}%")
        fg = ndi.binary_fill_holes(ndi.binary_opening(fg, np.ones((3, 3))))
        dist = ndi.distance_transform_edt(fg)
        fp = max(int(round(nucleus_diam_um / geo.um_per_px / 2)), 3)
        pk = peak_local_max(dist, footprint=np.ones((fp, fp)), labels=fg)
        seeds = np.zeros(fg.shape, np.int32)
        seeds[tuple(pk.T)] = np.arange(1, len(pk) + 1)
        log.info(f"            {len(pk):,} watershed seeds (footprint {fp} px)")
        lab = watershed(-dist, ndi.label(seeds)[0], mask=fg).astype(np.int32)
        a = np.bincount(lab.reshape(-1))[1:] * geo.um_per_px ** 2
        drop = np.nonzero((a < min_nucleus_um2) | (a > max_nucleus_um2))[0] + 1
        if len(drop):
            lab[np.isin(lab, drop)] = 0
            log.info(f"            dropped {len(drop):,} outside "
                     f"[{min_nucleus_um2}, {max_nucleus_um2}] um^2")
        log.info(f"            NOTE this prior uses no platform information, so "
                 "the result is the clean variant, not reference-assisted")

    elif mode == "polygon":
        bdir = find(root, "cell_boundaries")
        files = sorted(bdir.glob("*.hdf5")) if bdir and bdir.is_dir() else []
        log.info(f"source      cell_boundaries/  {len(files)} hdf5 files, "
                 f"looking for {len(want):,} cells")
        if not files:
            log.info("            none found, falling back to --prior disk")
            mode = "disk"
        else:
            import h5py
            done = 0
            unit = None
            for k, f in enumerate(files):
                try:
                    h = h5py.File(f, "r")
                except OSError:
                    continue
                with h:
                    fd = h.get("featuredata")
                    if fd is None:
                        continue
                    for cid in fd.keys():
                        if cid not in want:
                            continue
                        g = fd[cid]
                        zk = f"zIndex_{z_index}"
                        if zk not in g:
                            zk = sorted(g.keys())[len(g.keys()) // 2]
                        pg = g[zk]
                        pk = "p_0" if "p_0" in pg else sorted(pg.keys())[0]
                        arr = np.asarray(pg[pk]["coordinates"])
                        arr = arr.reshape(-1, 2)
                        if len(arr) < 3:
                            continue
                        if unit is None:
                            # micron or already mosaic pixels?
                            x0u, y0u, x1u, y1u = geo.um_bounds()
                            unit = "um" if (x0u - 50 <= arr[:, 0].mean() <= x1u + 50) \
                                else "px"
                            log.info(f"            polygon coordinates look like "
                                     f"{unit}")
                        if unit == "um":
                            px, py = geo.um_to_local(arr[:, 0], arr[:, 1])
                        else:
                            px, py = arr[:, 0] - geo.x0, arr[:, 1] - geo.y0
                        if _fill_polygon(lab, np.column_stack([px, py]),
                                         int(cid) % 2147483647 or 1):
                            done += 1
                if (k + 1) % 200 == 0:
                    log.info(f"            {k+1}/{len(files)} files, {done:,} cells")
            log.info(f"            {done:,} of {len(want):,} cells rasterised")
            if done < 0.5 * len(want):
                log.info("            !! fewer than half were found; the boundary "
                         "files may index cells differently. Use --prior disk.")

    if mode == "disk":
        r_um = 2 * np.sqrt(pd.to_numeric(cells.volume, errors="coerce") / np.pi) / 2
        log.info(f"source      cell_metadata volume -> radius median "
                 f"{r_um.median():.2f} um")
        px, py = geo.um_to_local(cells.center_x.to_numpy(float),
                                 cells.center_y.to_numpy(float))
        rr = (r_um.to_numpy() * geo.sx)
        for cid, a, b, r in zip(cells.cell.to_numpy(), px, py, rr):
            r = max(int(round(r)), 1)
            ylo, yhi = max(int(b) - r, 0), min(int(b) + r + 1, geo.size)
            xlo, xhi = max(int(a) - r, 0), min(int(a) + r + 1, geo.size)
            if ylo >= yhi or xlo >= xhi:
                continue
            yy = np.arange(ylo, yhi)[:, None] - b
            xx = np.arange(xlo, xhi)[None, :] - a
            sub = lab[ylo:yhi, xlo:xhi]
            sub[(yy * yy + xx * xx) <= r * r] = int(cid) % 2147483647 or 1

    n = int(np.unique(lab[lab > 0]).size)
    a = np.bincount(lab.reshape(-1))[1:]
    a = a[a > 0] * geo.um_per_px ** 2
    log.info(f"prior       {n:,} cells, coverage {(lab>0).mean()*100:.2f}%, "
             f"median area {np.median(a):.1f} um^2 -> diameter "
             f"{2*np.sqrt(np.median(a)/np.pi):.2f} um")
    out.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(p, lab, compression="zlib")
    return lab


def assign_by_mask(lab: np.ndarray, geo: Geometry, tx: pd.DataFrame,
                   px_per_cell=1.0) -> np.ndarray:
    lx, ly = geo.um_to_local(tx.global_x.to_numpy(float),
                             tx.global_y.to_numpy(float))
    r = np.clip((ly / px_per_cell).astype(np.int64), 0, lab.shape[0] - 1)
    c = np.clip((lx / px_per_cell).astype(np.int64), 0, lab.shape[1] - 1)
    v = lab[r, c].astype(np.int64)
    v[v < 0] = 0
    return v


def expand_labels(lab, radius_px):
    from scipy.ndimage import distance_transform_edt
    if radius_px <= 0:
        return lab
    d, (iy, ix) = distance_transform_edt(lab == 0, return_indices=True)
    out = lab.copy()
    g = (lab == 0) & (d <= radius_px)
    out[g] = lab[iy[g], ix[g]]
    return out


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def qc(lab: np.ndarray, geo: Geometry, tx: pd.DataFrame, n_platform: int,
       out: Path, method: str, px_per_cell=1.0, gt_label=None):
    rule("QC")
    ids = np.unique(lab[lab > 0])
    area = np.bincount(lab.reshape(-1))[1:]
    area = area[area > 0].astype(float) * (px_per_cell * geo.um_per_px) ** 2
    med = float(np.median(area))
    log.info(f"cells       {len(ids):,}   platform {n_platform:,}   "
             f"ratio {len(ids)/max(n_platform,1):.3f}")
    log.info(f"coverage    {(lab>0).mean()*100:.2f}%")
    log.info(f"area        median {med:.1f} um^2   equivalent diameter "
             f"{2*np.sqrt(med/np.pi):.2f} um   "
             f"IQR {np.percentile(area,25):.0f}-{np.percentile(area,75):.0f}")

    tl = assign_by_mask(lab, geo, tx, px_per_cell)
    n_as = int((tl > 0).sum())
    log.info(f"transcripts {len(tx):,} in ROI, {n_as:,} assigned "
             f"({n_as/max(len(tx),1)*100:.1f}%)")

    genes = sorted(tx.gene.astype(str).unique())
    gi = {g: i for i, g in enumerate(genes)}
    pos = {int(v): i for i, v in enumerate(ids)}
    mat = np.zeros((len(ids), len(genes)), np.int32)
    sel = tl > 0
    rows = np.array([pos.get(int(v), -1) for v in tl[sel]])
    ok = rows >= 0
    np.add.at(mat, (rows[ok],
                    np.array([gi[g] for g in tx.gene.astype(str).to_numpy()[sel][ok]])),
              1)
    cnt = mat.sum(axis=1)
    log.info(f"counts/cell median {np.median(cnt):.0f}   "
             f"IQR {np.percentile(cnt,25):.0f}-{np.percentile(cnt,75):.0f}")

    np.save(out / "cell_by_gene.npy", mat)
    pd.Series(genes).to_csv(out / "genes.txt", index=False, header=False)
    pd.DataFrame({"cell": ids, "area_um2": area, "n_counts": cnt}).to_csv(
        out / "cell_metadata.csv", index=False)

    if gt_label is not None:
        both = (tl > 0) & (gt_label > 0)
        log.info(f"vs platform assignment: {int(both.sum()):,} rows assigned by "
                 "both; identity of the two labellings is measured by the "
                 "evaluation script, not here")

    json.dump(dict(method=method, version=VERSION, cells=int(len(ids)),
                   platform_cells=int(n_platform),
                   coverage=float((lab > 0).mean()),
                   median_area_um2=med,
                   median_diameter_um=float(2*np.sqrt(med/np.pi)),
                   assigned=n_as, transcripts=int(len(tx)),
                   genes=len(genes),
                   roi=dict(y0=geo.y0, x0=geo.x0, size=geo.size),
                   um_per_px=geo.um_per_px, px_per_cell=px_per_cell),
              open(out / "run_qc.json", "w"), indent=2)
    rule("DONE")


METHOD = "ComSeg"
STAGES = ['prep', 'run', 'post', 'all']
PRIOR_DEFAULT = "centroid"
PRIOR_CHOICES = ['centroid', 'polygon', 'dapi', 'disk']

# ==========================================================================
# ComSeg
# ==========================================================================


OUT_DEFAULT = "yourpath/comseg_merfish_lung"


def add_args(ap):
    ap.add_argument("--tile-px", type=int, default=2500,
                    help="tile side in mosaic px (2500 = 270 um)")
    ap.add_argument("--overlap-px", type=int, default=256,
                    help="must exceed --max-cell-radius-um; 256 px = 27.6 um")
    ap.add_argument("--mask-bin", type=int, default=2,
                    help="mask/spot pixel in mosaic px (2 = 0.216 um)")
    ap.add_argument("--nucleus-diameter-um", type=float, default=None)
    ap.add_argument("--nucleus-frac", type=float, default=0.6)
    ap.add_argument("--mean-cell-diameter-um", type=float, default=None,
                    help="default: the platform cell size")
    ap.add_argument("--max-cell-radius-um", type=float, default=12.0)
    ap.add_argument("--k-nearest-neighbors", type=int, default=10)
    ap.add_argument("--n-neighbors", type=int, default=40)
    ap.add_argument("--sampling-size", type=int, default=10000)
    ap.add_argument("--norm-vector", action="store_true",
                    help="needs the R sctransform stack; off by default")
    ap.add_argument("--community-detection", default="with_prior")
    ap.add_argument("--prior-name", default="in_nucleus")
    ap.add_argument("--min-rna-per-cell", type=int, default=5)
    ap.add_argument("--raster-grid", type=int, default=2500)
    ap.add_argument("--fill-um", type=float, default=1.0)
    ap.add_argument("--force-run", action="store_true")
    ap.add_argument("--stop-on-first-failure", action="store_true")


def tile_starts(size, tile, overlap):
    step = tile - overlap
    s = list(range(0, max(size - tile, 0) + 1, step))
    if not s or s[-1] + tile < size:
        s.append(max(size - tile, 0))
    return s


def core_bounds(starts, tile, size):
    out = []
    for i, s in enumerate(starts):
        lo = s if i == 0 else (starts[i - 1] + tile + s) // 2
        hi = s + tile if i == len(starts) - 1 else (s + tile + starts[i + 1]) // 2
        out.append((max(lo, 0), min(hi, size)))
    return out


def stage_prep(root, out, args):
    import tifffile
    geo, dapi = load_geometry(root, args.roi_size, args.dapi)
    tx = load_roi_transcripts(root, geo, root / "_roi_cache", args.force_prep)
    cells = platform_cells(root, geo)
    nd = nucleus_diameter(cells, args.nucleus_frac, args.nucleus_diameter_um)
    lab = build_prior(root, geo, cells, out, args.prior, args.z_index,
                      args.force_prep, nucleus_diam_um=nd, dapi=dapi)

    cell_um = args.mean_cell_diameter_um
    if cell_um is None:
        if {"min_x", "max_x"} <= set(cells.columns):
            w = (cells.max_x - cells.min_x).astype(float)
            h = (cells.max_y - cells.min_y).astype(float)
            cell_um = float(np.sqrt(w * h).median())
        else:
            cell_um = nd / args.nucleus_frac

    rule("TILES")
    if args.overlap_px * geo.um_per_px < args.max_cell_radius_um:
        log.info(f"  !! overlap {args.overlap_px*geo.um_per_px:.1f} um is smaller "
                 f"than --max-cell-radius-um {args.max_cell_radius_um}; cells that "
                 "straddle a seam would be cut. Raise --overlap-px.")
    starts = tile_starts(geo.size, args.tile_px, args.overlap_px)
    cores = core_bounds(starts, args.tile_px, geo.size)
    log.info(f"grid        {len(starts)}x{len(starts)} = {len(starts)**2} tiles, "
             f"{args.tile_px} px ({args.tile_px*geo.um_per_px:.0f} um) with "
             f"{args.overlap_px} px overlap")
    mb = args.mask_bin
    log.info(f"mask/spot   {mb} px = {mb*geo.um_per_px:.4f} um per pixel; a "
             f"{nd:.2f} um nucleus is {nd/(mb*geo.um_per_px):.0f} px across")

    sp = out / "input" / "spots"
    mk = out / "input" / "masks"
    for d in (sp, mk):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    lx, ly = geo.um_to_local(tx.global_x.to_numpy(float),
                             tx.global_y.to_numpy(float))
    gene = tx.gene.astype(str).to_numpy()
    rows, idx = [], 0
    for iy, y0 in enumerate(starts):
        for ix, x0 in enumerate(starts):
            y1, x1 = y0 + args.tile_px, x0 + args.tile_px
            sel = (ly >= y0) & (ly < y1) & (lx >= x0) & (lx < x1)
            name = f"tile_{idx:04d}"
            sub = pd.DataFrame({"x": (lx[sel] - x0) / mb,
                                "y": (ly[sel] - y0) / mb,
                                "z": 0.0, "gene": gene[sel]})
            sub.to_csv(sp / f"{name}.csv", index=False)

            m = lab[y0:y1, x0:x1]
            g = int(np.ceil(args.tile_px / mb))
            m = m[::mb, ::mb][:g, :g]
            tifffile.imwrite(mk / f"{name}.tif", m.astype(np.uint32))

            n_nuc = int(np.unique(m[m > 0]).size)
            innuc = 0
            if len(sub):
                ri = np.clip(np.rint(sub.y).astype(int), 0, m.shape[0] - 1)
                ci = np.clip(np.rint(sub.x).astype(int), 0, m.shape[1] - 1)
                innuc = int((m[ri, ci] > 0).sum())
            rows.append(dict(tile=name, y0=y0, x0=x0,
                             core_y0=cores[iy][0], core_y1=cores[iy][1],
                             core_x0=cores[ix][0], core_x1=cores[ix][1],
                             spots=int(sel.sum()), nuclei=n_nuc,
                             in_nucleus_frac=round(innuc / max(int(sel.sum()), 1), 4)))
            log.info(f"  {name}: {int(sel.sum()):,} spots, {n_nuc:,} nuclei, "
                     f"{innuc/max(int(sel.sum()),1)*100:.1f}% inside a nucleus")
            idx += 1

    meta = pd.DataFrame(rows)
    meta.to_csv(out / "input" / "tile_metadata.tsv", sep="\t", index=False)
    big = int(meta.spots.max())
    log.info(f"largest tile {big:,} spots")
    if big > 1_500_000:
        log.info("  !! that is a lot for one ComSeg graph. On CosMx a 310,000 "
                 "spot tile took 45 minutes; consider a smaller --tile-px.")
    log.info(f"estimate    at ~45 min per 300k-spot tile this is roughly "
             f"{len(rows)*big/300000*45/60:.1f} h in total")

    json.dump(dict(version=VERSION, roi=dict(y0=geo.y0, x0=geo.x0, size=geo.size),
                   um_per_px=geo.um_per_px, mask_bin=mb, tile_px=args.tile_px,
                   overlap_px=args.overlap_px, n_tiles=len(rows),
                   prior=args.prior, nucleus_diameter_um=nd,
                   mean_cell_diameter_um=cell_um,
                   platform_cells=int(len(cells)), transcripts=int(len(tx))),
              open(out / "prep_meta.json", "w"), indent=2)
    rule("PREP DONE")
    log.info(f"tiles {len(rows)}   platform cells {len(cells):,}   "
             f"mean cell diameter {cell_um:.2f} um")


def call_filtered(fn, **kw):
    import inspect as _i
    sig = _i.signature(fn)
    keep = {k: v for k, v in kw.items() if k in sig.parameters}
    drop = [k for k in kw if k not in sig.parameters]
    if drop:
        log.info(f"    ({getattr(fn,'__name__',fn)} does not take {drop}, dropped)")
    return fn(**keep)


def show_api():
    import inspect as _i
    from comseg import dataset as ds, dictionary as di
    log.info("ComSeg API in this environment:")
    for label, fn in (("ComSegDataset.__init__", ds.ComSegDataset.__init__),
                      ("add_prior_from_mask", ds.ComSegDataset.add_prior_from_mask),
                      ("compute_edge_weight", ds.ComSegDataset.compute_edge_weight),
                      ("ComSegDict.__init__", di.ComSegDict.__init__),
                      ("run_all", di.ComSegDict.run_all)):
        try:
            log.info(f"  {label}{_i.signature(fn)}")
        except (TypeError, ValueError):
            log.info(f"  {label}  (signature unavailable)")


def save_tile(ad, cd, tdir, name):
    """Never discard a finished tile: search both levels, then describe."""
    found = []

    def harvest(obj, where):
        for attr in ("df_spots_label", "df_spots", "df", "spots"):
            df = getattr(obj, attr, None)
            if isinstance(df, pd.DataFrame) and {"x", "y"} <= set(df.columns):
                found.append((f"{where}.{attr}", df))

    harvest(cd, "dict")
    try:
        for k in list(cd.keys()):
            harvest(cd[k], f"dict[{k!r}]")
            harvest(getattr(cd[k], "dataset", object()), f"dict[{k!r}].dataset")
    except Exception as e:
        log.info(f"    (could not iterate the ComSegDict: {e})")
    harvest(getattr(cd, "dataset", object()), "dict.dataset")

    if found:
        where, df = max(found, key=lambda t: len(t[1]))
        log.info(f"    per-RNA table from {where}: {len(df):,} rows, "
                 f"columns {list(df.columns)}")
        write_tab(df.reset_index(drop=True), tdir / "spot_assignment.parquet")
    else:
        desc = {"dict_type": type(cd).__name__,
                "dict_attrs": [a for a in dir(cd) if not a.startswith("_")][:60]}
        try:
            v0 = cd[list(cd.keys())[0]]
            desc["value_type"] = type(v0).__name__
            desc["value_frames"] = {
                a: list(getattr(v0, a).columns) for a in dir(v0)
                if not a.startswith("_")
                and isinstance(getattr(v0, a, None), pd.DataFrame)}
        except Exception as e:
            desc["value_error"] = str(e)
        (tdir / "comseg_object.json").write_text(json.dumps(desc, indent=2,
                                                            default=str))
        log.info(f"    !! no per-RNA table found; the object layout is in "
                 f"{tdir/'comseg_object.json'}")
    if ad is not None:
        for fn, arg in ((getattr(ad, "write_h5ad", None), tdir / "cells.h5ad"),):
            try:
                fn(arg)
            except Exception:
                pass
    try:
        cd.save(str(tdir / "comseg_dict"))
    except Exception:
        pass


def stage_run(root, out, args):
    rule("RUN")
    mp = out / "prep_meta.json"
    if not mp.exists():
        raise SystemExit(f"{mp} missing; run --stage prep first "
                         f"(its log is in {out}/logs)")
    meta = json.loads(mp.read_text())
    tiles = pd.read_csv(out / "input" / "tile_metadata.tsv", sep="\t")

    from comseg import dataset as ds, dictionary as di
    show_api()

    mb = meta["mask_bin"]
    scale = meta["um_per_px"] * mb
    dict_scale = {"x": scale, "y": scale, "z": scale}
    cell_um = args.mean_cell_diameter_um or meta["mean_cell_diameter_um"]
    log.info(f"dict_scale  {scale:.5f} um per mask px")
    log.info(f"mean cell diameter {cell_um:.2f} um, max cell radius "
             f"{args.max_cell_radius_um} um")

    work = out / "comseg_out"
    work.mkdir(parents=True, exist_ok=True)
    n_done = n_fail = 0
    for _, t in tiles.iterrows():
        name = t["tile"]
        tdir = work / name
        if (tdir / "DONE.flag").exists() and not args.force_run:
            n_done += 1
            continue
        if (tdir / "FAILED.flag").exists():
            (tdir / "FAILED.flag").unlink()
        tdir.mkdir(parents=True, exist_ok=True)

        stage = tdir / "in"
        if stage.exists():
            shutil.rmtree(stage)
        (stage / "masks").mkdir(parents=True)
        shutil.copy2(out / "input" / "spots" / f"{name}.csv", stage / f"{name}.csv")
        shutil.copy2(out / "input" / "masks" / f"{name}.tif",
                     stage / "masks" / f"{name}.tif")

        log.info(f"{name}: {int(t['spots']):,} spots, {int(t['nuclei']):,} nuclei")
        try:
            dataset = call_filtered(
                ds.ComSegDataset,
                path_dataset_folder=str(stage),
                path_to_mask_prior=str(stage / "masks"),
                dict_scale=dict_scale, mask_file_extension=".tif",
                mean_cell_diameter=cell_um, prior_name=args.prior_name)

            call_filtered(dataset.add_prior_from_mask, overwrite=True,
                          compute_centroid=True, regex_df="*.csv")
            dc = getattr(dataset, "dict_centroid", None)
            if dc is None:
                raise RuntimeError(
                    "add_prior_from_mask produced no dict_centroid; the mask and "
                    "the csv must share a basename")
            log.info(f"    prior added, centroids for {len(dc)} image(s)")

            call_filtered(dataset.compute_edge_weight,
                          n_neighbors=args.n_neighbors, distance="pearson",
                          sampling=True, sampling_size=args.sampling_size)
            cd = call_filtered(di.ComSegDict, dataset=dataset,
                               mean_cell_diameter=cell_um,
                               community_detection=args.community_detection)
            call_filtered(cd.run_all, max_cell_radius=args.max_cell_radius_um,
                          k_nearest_neighbors=args.k_nearest_neighbors,
                          norm_vector=args.norm_vector,
                          clustering_method="leiden", disable_tqdm=True)
            ad = call_filtered(cd.anndata_from_comseg_result,
                               min_rna_per_cell=args.min_rna_per_cell,
                               return_polygon=False)
            save_tile(ad, cd, tdir, name)
            if read_tab(tdir / "spot_assignment.parquet") is None:
                log.info("    left WITHOUT a per-RNA table; not marking it done")
            else:
                (tdir / "DONE.flag").touch()
                n_done += 1
                log.info("    done")
        except Exception:
            tb = traceback.format_exc()
            (tdir / "FAILED.flag").write_text(tb)
            n_fail += 1
            log.info(f"  {name} failed:\n{tb.rstrip()}")
            if args.stop_on_first_failure:
                raise
        log.info("")
    rule("RUN DONE")
    log.info(f"{n_done} tiles done, {n_fail} failed of {len(tiles)}")


def stage_post(root, out, args):
    import tifffile
    rule("POST")
    meta = json.loads((out / "prep_meta.json").read_text())
    tiles = pd.read_csv(out / "input" / "tile_metadata.tsv", sep="\t")
    mb = meta["mask_bin"]
    work = out / "comseg_out"

    frames = []
    for _, t in tiles.iterrows():
        df = read_tab(work / t["tile"] / "spot_assignment.parquet")
        if df is None:
            continue
        lc = next((c for c in ("cell_index_pred", "cell", "cell_id",
                               "index_commu_in_nucleus", "cell_index")
                   if c in df.columns), None)
        if lc is None:
            log.info(f"  {t['tile']}: no label column in {list(df.columns)}")
            continue
        gx = df.x.to_numpy(float) * mb + t["x0"]
        gy = df.y.to_numpy(float) * mb + t["y0"]
        keep = ((gx >= t["core_x0"]) & (gx < t["core_x1"])
                & (gy >= t["core_y0"]) & (gy < t["core_y1"]))
        v = pd.to_numeric(df[lc], errors="coerce").fillna(0).to_numpy()[keep]
        sub = pd.DataFrame({"x": gx[keep], "y": gy[keep], "label": v,
                            "gene": (df.gene.to_numpy()[keep]
                                     if "gene" in df.columns else "NA")})
        sub = sub[sub.label > 0]
        sub["label"] = sub.label.astype(np.int64) + 10_000_000 * int(
            t["tile"].split("_")[1])
        frames.append(sub)
        log.info(f"  {t['tile']}: {len(sub):,} assigned spots in its core")
    if not frames:
        raise SystemExit("no tile produced anything; see the FAILED.flag files")
    allsp = pd.concat(frames, ignore_index=True)
    codes, uniq = pd.factorize(allsp.label)
    allsp["cell"] = codes + 1
    log.info(f"cells       {len(uniq):,}   assigned spots {len(allsp):,}")

    geo, dapi = load_geometry(root, args.roi_size, args.dapi)
    g = args.raster_grid
    s = geo.size / g
    lab = np.zeros((g, g), np.int32)
    lab[np.clip((allsp.y / s).astype(int), 0, g - 1),
        np.clip((allsp.x / s).astype(int), 0, g - 1)] = allsp.cell.to_numpy()
    occ0 = (lab > 0).mean()
    if args.fill_um > 0:
        lab = expand_labels(lab, args.fill_um / (s * geo.um_per_px))
    log.info(f"raster {g}x{g} at {s*geo.um_per_px:.3f} um/px: occupancy "
             f"{occ0*100:.2f}% -> {(lab>0).mean()*100:.2f}% after a "
             f"{args.fill_um} um fill (ComSeg labels spots, not pixels)")
    tifffile.imwrite(out / "comseg_segmentation.tif", lab)

    tx = load_roi_transcripts(root, geo, root / "_roi_cache")
    qc(lab, geo, tx, meta["platform_cells"], out, "ComSeg", px_per_cell=s)


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--root", default=DATA_ROOT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--stage", default="all", choices=STAGES)
    ap.add_argument("--roi-size", type=int, default=ROI_SIZE)
    ap.add_argument("--dapi", default=None,
                    help="explicit nuclei mosaic; the default picks the file with "
                         "'dapi' in its name, never PolyT")
    ap.add_argument("--prior", default=PRIOR_DEFAULT, choices=PRIOR_CHOICES)
    ap.add_argument("--z-index", type=int, default=3,
                    help="which z plane of cell_boundaries to use")
    ap.add_argument("--force-prep", action="store_true")
    ap.add_argument("--version", action="store_true")
    add_args(ap)
    args = ap.parse_args()
    if args.version:
        print(VERSION)
        return
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    setup_log(out / "logs", f"{METHOD.lower()}_lung_{args.stage}")
    rule(f"{METHOD} v{VERSION}  x  MERFISH lung   out {out}")
    for st in (STAGES if args.stage == "all" else [args.stage]):
        if st == "all":
            continue
        fn = globals().get(f"stage_{st}")
        if fn is None:
            continue
        fn(root, out, args)


if __name__ == "__main__":
    main()