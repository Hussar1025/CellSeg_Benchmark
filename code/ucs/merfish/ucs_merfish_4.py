#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ucs_merfish_4.py

UCS on the MERFISH lung ROI, with a nucleus-scale prior.

Why the prior changed
---------------------
The first run used the platform cell outlines and measured an expansion of only
about 1.17x: UCS turns nuclei into cells, so handing it a cell boundary leaves
it nothing to do, and its cell count then reproduces the prior almost exactly
(11,210 of 11,217).  The default here is a nucleus-sized disk on each platform
cell centroid -- the same shape of prior as the Xenium nucleus_boundaries and
the CosMx NucArea runs, so this row sits in the same column as those.

  --prior centroid   nucleus disks on the platform centroids   (default)
  --prior dapi       nuclei segmented from DAPI, no platform input at all
  --prior polygon    the old cell outlines, kept only as an upper bound

Grid: bin 10 mosaic px = 1.080 um, so the 10000 px ROI becomes 1000 x 1000 and a
6 um nucleus spans 5.6 bins -- UCS needs room to grow, and a nucleus narrower
than about 3 bins leaves it none (that is what made the STARmap run expand by
1.00x).  The gene map is 1000 x 1000 x 550 uint8 = 0.55 GB, which is small
enough that the full panel can be used.

The nucleus prior is the platform segmentation rasterised from cell_boundaries,
so this is a reference-assisted result, the same footing as the Xenium and CosMx
UCS rows.

  conda activate ucs
  python ucs_merfish_lung.py --stage prep
  python ucs_merfish_lung.py --stage run --gpu auto
  python ucs_merfish_lung.py --stage post

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
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "2026.08.13"

DATA_ROOT = "/data/qiuyijia/dataset/MERFISH_Lung_cancer"
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


METHOD = "UCS"
STAGES = ['prep', 'run', 'post', 'all']
PRIOR_DEFAULT = "centroid"
PRIOR_CHOICES = ['centroid', 'dapi', 'polygon', 'disk']

# ==========================================================================
# UCS
# ==========================================================================


OUT_DEFAULT = "/data/qiuyijia/ucs_merfish_lung"


def add_args(ap):
    ap.add_argument("--bin-factor", type=int, default=10,
                    help="mosaic px per grid pixel (10 = 1.080 um)")
    ap.add_argument("--ucs-repo", default="/data/qiuyijia/ucs/UCS")
    ap.add_argument("--gpu", default="auto")
    ap.add_argument("--min-free-mb", type=int, default=4000)
    ap.add_argument("--fg-batch", type=int, default=None)
    ap.add_argument("--patch-size", type=int, default=48)
    ap.add_argument("--top-genes", type=int, default=0)
    ap.add_argument("--nucleus-diameter-um", type=float, default=None,
                    help="default: --nucleus-frac x the platform cell size")
    ap.add_argument("--nucleus-frac", type=float, default=0.6,
                    help="nucleus diameter as a fraction of the cell diameter")
    ap.add_argument("--dapi-threshold-scale", type=float, default=1.0)
    ap.add_argument("--min-nucleus-um2", type=float, default=8.0)
    ap.add_argument("--max-nucleus-um2", type=float, default=200.0)
    ap.add_argument("--ucs-extra", default="")


def stage_prep(root, out, args):
    import tifffile
    geo, dapi = load_geometry(root, args.roi_size, args.dapi)
    tx = load_roi_transcripts(root, geo, root / "_roi_cache", args.force_prep)
    cells = platform_cells(root, geo)
    nd = nucleus_diameter(cells, args.nucleus_frac, args.nucleus_diameter_um)
    lab = build_prior(root, geo, cells, out, args.prior, args.z_index,
                      args.force_prep, nucleus_diam_um=nd, dapi=dapi,
                      min_nucleus_um2=args.min_nucleus_um2,
                      max_nucleus_um2=args.max_nucleus_um2,
                      threshold_scale=args.dapi_threshold_scale)

    b = args.bin_factor
    g = geo.size // b
    rule("GENE MAP")
    if g * b != geo.size:
        raise SystemExit(f"--bin-factor {b} does not divide {geo.size}")
    log.info(f"grid        {g} x {g}   1 bin = {b} px = {b*geo.um_per_px:.4f} um")

    genes = sorted(tx.gene.astype(str).unique())
    if args.top_genes and args.top_genes < len(genes):
        keep = tx.gene.value_counts().head(args.top_genes).index
        tx = tx[tx.gene.isin(keep)].reset_index(drop=True)
        genes = sorted(tx.gene.astype(str).unique())
        log.info(f"            restricted to the {len(genes)} most frequent targets")
    gb = g * g * len(genes) / 1e9
    log.info(f"            {g} x {g} x {len(genes)} uint8 = {gb:.2f} GB")
    log.info(f"            {len(genes)} genes become {len(genes)} conv input "
             "channels, which is what sets the VRAM need")

    gi = {v: i for i, v in enumerate(genes)}
    lx, ly = geo.um_to_local(tx.global_x.to_numpy(float), tx.global_y.to_numpy(float))
    ri = np.clip((ly / b).astype(np.int32), 0, g - 1)
    ci = np.clip((lx / b).astype(np.int32), 0, g - 1)
    gidx = np.array([gi[v] for v in tx.gene.astype(str).to_numpy()], np.int32)
    flat = np.bincount((ri.astype(np.int64) * g + ci) * len(genes) + gidx,
                       minlength=g * g * len(genes))
    gm = np.clip(flat, 0, 255).astype(np.uint8).reshape(g, g, len(genes))
    log.info(f"            occupied bins {(gm.sum(axis=2)>0).mean()*100:.2f}%")
    tifffile.imwrite(out / "gene_map.tif", gm)
    pd.Series(genes).to_csv(out / "genes.txt", index=False, header=False)
    del gm, flat

    rule("NUCLEUS PRIOR ON THE GRID")
    nb = nd / (b * geo.um_per_px)
    log.info(f"a nucleus spans {nb:.1f} bins at bin {b}")
    if nb < 3:
        log.info("  !! narrower than 3 bins: UCS has almost no room to grow. "
                 "This is what held the STARmap run to an expansion of 1.00x. "
                 "Lower --bin-factor.")
    small = block_majority(lab, b)
    n = int(np.unique(small[small > 0]).size)
    log.info(f"prior on the grid: {n:,} cells, coverage {(small>0).mean()*100:.2f}%")
    if n < 0.9 * len(cells):
        log.info(f"  !! {len(cells)-n:,} of {len(cells):,} prior cells vanish at "
                 f"bin {b}; they are smaller than one bin. Lower --bin-factor.")
    tifffile.imwrite(out / "nuclei_mask.tif", small.astype(np.uint32))

    json.dump(dict(version=VERSION, bin_factor=b, grid=g, genes=len(genes),
                   roi=dict(y0=geo.y0, x0=geo.x0, size=geo.size),
                   um_per_px=geo.um_per_px, prior=args.prior,
                   nucleus_diameter_um=nd,
                   reference_used=args.prior != "dapi",
                   platform_cells=int(len(cells)),
                   prior_cells_on_grid=n, transcripts=int(len(tx))),
              open(out / "prep_meta.json", "w"), indent=2)
    rule("PREP DONE")


def block_majority(lab, b, min_frac=0.0):
    """Down-sample a label image: each block takes its most common non-zero id."""
    g = lab.shape[0] // b
    v = lab[:g*b, :g*b].reshape(g, b, g, b).transpose(0, 2, 1, 3).reshape(g, g, b*b)
    out = np.zeros((g, g), lab.dtype)
    for i in range(g):
        row = v[i]
        for j in range(g):
            c = row[j]
            c = c[c > 0]
            if c.size == 0 or c.size < min_frac * b * b:
                continue
            u, k = np.unique(c, return_counts=True)
            out[i, j] = u[k.argmax()]
    return out


def pick_gpu(min_free_mb):
    try:
        r = subprocess.run(["nvidia-smi",
                            "--query-gpu=index,memory.free,utilization.gpu",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, check=True)
    except Exception:
        return 0, None
    rows = []
    for line in r.stdout.strip().splitlines():
        i, f, u = [x.strip() for x in line.split(",")]
        rows.append((int(i), int(f), int(u)))
        log.info(f"  [{i}] free {f:>6} MB  util {u:>3}%")
    rows.sort(key=lambda t: -t[1])
    if rows[0][1] < min_free_mb:
        raise SystemExit(f"no GPU has {min_free_mb} MB free")
    return rows[0][0], rows[0][1]


def stage_run(root, out, args):
    rule("RUN UCS")
    meta = json.loads((out / "prep_meta.json").read_text())
    repo = Path(args.ucs_repo)
    runpy = repo / "run.py"
    if not runpy.exists():
        raise SystemExit(f"{runpy} not found; pass --ucs-repo")

    if args.gpu == "auto":
        gpu, free = pick_gpu(args.min_free_mb)
        log.info(f"auto-selected GPU {gpu}")
    else:
        gpu, free = int(args.gpu), None

    batch = args.fg_batch
    if batch is None:
        one = meta["genes"] * args.patch_size ** 2 * 4 / 1e6
        budget = (free or 20000) * 0.30
        batch = 256
        while batch > 16 and one * batch > budget:
            batch //= 2
        log.info(f"one sample = {one:.1f} MB; using --fg_net_batch_size {batch}")

    logdir = out / "log"
    if logdir.exists():
        shutil.rmtree(logdir)

    cmd = [sys.executable, "-u", str(runpy),
           "--gene_map", str(out / "gene_map.tif"),
           "--nuclei_mask", str(out / "nuclei_mask.tif"),
           "--log_dir", str(logdir),
           "--patch_size", str(args.patch_size),
           "--fg_net_batch_size", str(batch)]
    if args.ucs_extra:
        cmd += args.ucs_extra.split()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    log.info(" ".join(cmd))
    p = subprocess.Popen(cmd, cwd=str(repo), env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        log.info("    " + line.rstrip())
    if p.wait() != 0:
        raise SystemExit(f"UCS exited with {p.returncode}; if it was a CUDA OOM "
                         "retry with a smaller --fg-batch")
    rule("RUN DONE")


def stage_post(root, out, args):
    import tifffile
    rule("POST")
    geo, dapi = load_geometry(root, args.roi_size, args.dapi)
    meta = json.loads((out / "prep_meta.json").read_text())
    b = meta["bin_factor"]
    cands = sorted(out.glob("log*/pred/segmentation_mask.tif"))
    if not cands:
        raise SystemExit("no segmentation_mask.tif under log*/pred/")
    lab = tifffile.imread(cands[0]).astype(np.int64)
    log.info(f"prediction  {cands[0]}   {lab.shape}")

    prior = tifffile.imread(out / "nuclei_mask.tif")
    pa = np.bincount(prior.reshape(-1))[1:]; pa = pa[pa > 0]
    la = np.bincount(lab.reshape(-1))[1:]; la = la[la > 0]
    exp = float(np.median(la) / max(np.median(pa), 1))
    pu = (b * geo.um_per_px) ** 2
    log.info(f"prior       {len(pa):,} cells, median {np.median(pa)*pu:.1f} um^2 "
             f"-> {2*np.sqrt(np.median(pa)*pu/np.pi):.2f} um")
    log.info(f"prediction  {len(la):,} cells, median {np.median(la)*pu:.1f} um^2 "
             f"-> {2*np.sqrt(np.median(la)*pu/np.pi):.2f} um")
    log.info(f"expansion   {exp:.2f}x in area  ({np.sqrt(exp):.2f}x linear)")
    log.info(f"prior source = {meta.get('prior')}   reference used = "
             f"{meta.get('reference_used')}")
    if exp < 1.5:
        log.info("  !! barely expanded. With --prior polygon that is expected, "
                 "because a cell outline leaves nothing to grow into; with "
                 "--prior centroid or dapi it means something is wrong.")

    tx = load_roi_transcripts(root, geo, root / "_roi_cache")
    cells = platform_cells(root, geo)
    tifffile.imwrite(out / "ucs_segmentation.tif", lab.astype(np.int32))
    qc(lab, geo, tx, len(cells), out, "UCS", px_per_cell=b)


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