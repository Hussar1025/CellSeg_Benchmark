#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""cellist_merfish_lung.py

Cellist on the MERFISH lung ROI.

Cellist finds its own nuclei with Cellpose, so it is the one method here that
does not consume the platform segmentation.  Two things learned on the earlier
datasets are built in:

  * the number of cells Cellist can return is bounded by the number of nuclei
    that contain at least ~20 in-nucleus transcripts, so --stage nucleus reports
    that count and not just the raw nucleus count.  On Xenium liver the raw
    count was 16,886 while only 2,943 were usable, and 2,942 cells came out.
  * Cellist labels spots rather than pixels, so the raster is grown by
    --fill-um before area is measured; without it a 34 um cell reads as 9 um.

  conda activate Cellist
  python cellist_merfish_lung.py --stage prep
  python cellist_merfish_lung.py --stage nucleus --nucleus-diameter-um 6.5
  python cellist_merfish_lung.py --stage seg --nucleus-diameter-um 6.5 --cell-radius-um 6.0

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


METHOD = "Cellist"
STAGES = ['prep', 'nucleus', 'seg', 'all']
PRIOR_DEFAULT = "polygon"
PRIOR_CHOICES = ['polygon', 'centroid', 'disk']

# ==========================================================================
# Cellist
# ==========================================================================


OUT_DEFAULT = "/data/qiuyijia/cellist_merfish_lung"


def add_args(ap):
    ap.add_argument("--gem-bin", type=int, default=5,
                    help="mosaic px per gem bin (5 = 0.540 um)")
    ap.add_argument("--nucleus-diameter-um", type=float, default=6.5)
    ap.add_argument("--cellprob-threshold", type=float, default=-2.0)
    ap.add_argument("--flow-threshold", type=float, default=0.8)
    ap.add_argument("--cell-radius-um", type=float, default=6.0)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--gene-use", default=None, choices=[None, "HVG", "Frequent"])
    ap.add_argument("--nworkers", type=int, default=3,
                    help="this is the number of patches held in RAM at once, not "
                         "a CPU count; 16 once needed 515 GiB")
    ap.add_argument("--fill-um", type=float, default=1.0)
    ap.add_argument("--cellist-bin", default="cellist")
    ap.add_argument("--use-local-threshold", action="store_true",
                    help="the sweeps on the other datasets favoured "
                         "no_local_threshold=True, which is the default here")
    ap.add_argument("--model-type", default="nuclei")
    ap.add_argument("--expansion", action="store_true",
                    help="on Xenium liver this changed nothing at all: 2.0 and "
                         "3.0 um gave byte-identical output. Left in, off.")
    ap.add_argument("--expansion-dist-um", type=float, default=0.0)
    ap.add_argument("--force-nucleus", action="store_true")
    ap.add_argument("--force-seg", action="store_true")


def stage_prep(root, out, args):
    import tifffile
    geo, dapi = load_geometry(root, args.roi_size, args.dapi)
    tx = load_roi_transcripts(root, geo, root / "_roi_cache", args.force_prep)
    cells = platform_cells(root, geo)

    inp = out / "input"
    inp.mkdir(parents=True, exist_ok=True)
    b = args.gem_bin
    g = geo.size // b
    unit_um = b * geo.um_per_px
    rule("GEM + DAPI")
    log.info(f"bin         {b} px = {unit_um:.4f} um   grid {g} x {g}")

    crop = read_dapi_roi(dapi, geo)
    small = crop.reshape(g, b, g, b).mean(axis=(1, 3))
    d8 = (normalise8(small) * 255).astype(np.uint8)
    tifffile.imwrite(inp / "DAPI_roi.tif", d8)
    log.info(f"DAPI        {d8.shape}  mean {d8.mean():.1f}  std {d8.std():.1f}  "
             f"frac>0.9 {(d8>229).mean():.3f}")
    if d8.std() < 5:
        raise SystemExit("the binned DAPI has almost no contrast; Cellpose will "
                         "find nothing. Check --dapi and the ROI.")

    lx, ly = geo.um_to_local(tx.global_x.to_numpy(float), tx.global_y.to_numpy(float))
    gem = pd.DataFrame({
        "geneID": tx.gene.astype(str).to_numpy(),
        "x": np.clip((lx / b).astype(np.int32), 0, g - 1),
        "y": np.clip((ly / b).astype(np.int32), 0, g - 1),
        "MIDCounts": 1,
    }).groupby(["geneID", "x", "y"], as_index=False)["MIDCounts"].sum()
    p = inp / "gem.tsv.gz"
    gem.to_csv(p, sep="\t", index=False)
    log.info(f"gem         {len(gem):,} rows -> {p.name}")

    json.dump(dict(version=VERSION, gem_bin=b, grid=g, unit_um=unit_um,
                   roi=dict(y0=geo.y0, x0=geo.x0, size=geo.size),
                   um_per_px=geo.um_per_px, platform_cells=int(len(cells)),
                   transcripts=int(len(tx))),
              open(out / "prep_meta.json", "w"), indent=2)
    rule("PREP DONE")


def _usable(nuc_dir: Path, prefix: str):
    """How many nuclei hold >= 20 transcripts -- the ceiling on the cell count."""
    f = next(iter(sorted(nuc_dir.glob("*nucleus_coord.txt"))), None)
    if f is None:
        return None, None
    d = pd.read_csv(f, sep="\t")
    col = next((c for c in ("Cellpose", "Watershed", "nucleus") if c in d.columns),
               d.columns[-1])
    v = pd.to_numeric(d[col], errors="coerce").fillna(0).astype(int)
    v = v[v > 0].value_counts()
    return len(v), int((v >= 20).sum())


def stage_nucleus(root, out, args):
    rule("NUCLEUS DETECTION")
    meta = json.loads((out / "prep_meta.json").read_text())
    unit = meta["unit_um"]
    tag = (f"d{args.nucleus_diameter_um}_cp{args.cellprob_threshold}"
           f"_flow{args.flow_threshold}").replace(".", "p")
    nd = out / "cellpose" / tag
    flag = nd / "CELLPOSE_DONE.flag"
    if flag.exists() and not args.force_nucleus:
        log.info(f"cache       {nd}")
    else:
        if nd.exists():
            log.info(f"            clearing {nd} (no DONE flag, so the previous "
                     "attempt did not finish)")
            shutil.rmtree(nd)
        nd.mkdir(parents=True)
        from Cellist.Cellpose import Cellpose
        import inspect as _i
        sig = _i.signature(Cellpose)
        log.info(f"Cellpose{sig}")
        diam = args.nucleus_diameter_um / unit
        log.info(f"diameter    {args.nucleus_diameter_um} um = {diam:.2f} units "
                 f"(1 unit = {unit:.4f} um)")

        # The parameter names differ between Cellist builds, so map our paths
        # onto whatever this one declares rather than guessing.
        want = {
            ("gem_path", "spot_file", "gem", "gem_file", "spot_expr_file",
             "spot_expr", "spot"): str(out / "input" / "gem.tsv.gz"),
            ("img_path", "image_file", "image", "img", "img_file", "dapi",
             "image_path", "nucleus_image"): str(out / "input" / "DAPI_roi.tif"),
            ("out_dir", "outdir", "output_dir"): str(nd),
            ("out_prefix", "outprefix", "prefix"): "lung",
            ("platform",): "imaging",
            ("no_local_threshold",): not args.use_local_threshold,
            ("model_type",): args.model_type,
            ("diameter", "diam"): diam,
            ("cellprob_threshold", "cellprob"): args.cellprob_threshold,
            ("flow_threshold", "flow"): args.flow_threshold,
            ("expansion",): args.expansion,
            ("expansion_dist", "expansion_distance"):
                args.expansion_dist_um / unit,
        }
        kw = {}
        for names, val in want.items():
            hit = next((n for n in names if n in sig.parameters), None)
            if hit is None:
                log.info(f"  (this build takes none of {names}, skipped)")
                continue
            kw[hit] = val
        missing = [n for n, p_ in sig.parameters.items()
                   if p_.default is _i.Parameter.empty and n not in kw
                   and n != "self"
                   and p_.kind not in (_i.Parameter.VAR_KEYWORD,
                                       _i.Parameter.VAR_POSITIONAL)]
        if missing:
            raise SystemExit(
                f"Cellpose still needs {missing}, and this script has nothing to "
                f"put there. Its full signature is:\n  Cellpose{sig}\n"
                "Send that line and the mapping can be completed exactly.")
        log.info("calling     Cellpose(" + ", ".join(
            f"{k}={v!r}" if not isinstance(v, str) else f"{k}=..."
            for k, v in kw.items()) + ")")
        Cellpose(**kw)
        got = sorted(p_.name for p_ in nd.iterdir())
        if not any("nucleus_coord" in g for g in got):
            raise SystemExit(f"Cellpose returned but wrote no *nucleus_coord.txt; "
                             f"{nd} holds {got}")
        flag.touch()
    n, usable = _usable(nd, "lung")
    if n is None:
        raise SystemExit(f"no *nucleus_coord.txt under {nd}; rerun with "
                         "--force-nucleus")
    gt = meta["platform_cells"]
    log.info(f"nuclei      {n:,}" if n else "nuclei      (table not found)")
    if usable is not None:
        log.info(f"usable      {usable:,} hold >= 20 transcripts  "
                 f"({usable/max(n,1)*100:.1f}%)  <- this bounds the cell count")
        log.info(f"platform    {gt:,}   usable/platform = {usable/gt:.3f}")
        row = dict(tag=tag, diameter_um=args.nucleus_diameter_um,
                   cellprob=args.cellprob_threshold, flow=args.flow_threshold,
                   nuclei=n, usable=usable, ratio=round(usable / gt, 4))
        sp = out / "cellpose" / "sweep.tsv"
        df = pd.read_csv(sp, sep="\t") if sp.exists() else pd.DataFrame()
        df = pd.concat([df[df.tag != tag] if len(df) else df,
                        pd.DataFrame([row])], ignore_index=True)
        df.sort_values("usable", ascending=False).to_csv(sp, sep="\t", index=False)
        log.info(f"sweep so far -> {sp}   "
                 "pick the largest 'usable', not the largest 'nuclei'")
        for line in df.sort_values("usable", ascending=False) \
                .to_string(index=False).splitlines():
            log.info("    " + line)
    rule("NUCLEUS DONE")
    return nd


def stage_seg(root, out, args):
    import tifffile
    nd = stage_nucleus(root, out, args)
    rule("CELLIST SEG")
    meta = json.loads((out / "prep_meta.json").read_text())
    unit = meta["unit_um"]
    from shutil import which
    if which(args.cellist_bin) is None:
        raise SystemExit(
            f"'{args.cellist_bin}' is not on PATH (conda env "
            f"'{os.environ.get('CONDA_DEFAULT_ENV')}'). The seg step shells out "
            "to the cellist CLI, so run it in the environment where Cellist is "
            "installed, or pass --cellist-bin /full/path.")

    def one(pat):
        h = sorted(nd.glob(pat))
        if not h:
            raise SystemExit(f"{pat} not found under {nd}")
        return str(h[0])

    r_units = int(round(args.cell_radius_um / unit))     # the CLI wants an int
    log.info(f"cell radius {args.cell_radius_um} um / {unit:.4f} = "
             f"{args.cell_radius_um/unit:.2f} -> {r_units} units "
             f"(= {r_units*unit:.2f} um after rounding)")
    sdir = out / "cellist"
    cmd = [args.cellist_bin, "seg", "--platform", "imaging",
           "--resolution", f"{unit:.6g}",
           "--gem", str(out / "input" / "gem.tsv.gz"),
           "--spot-count-h5", one("*_bin1.h5"),
           "--nucleus-seg-method", "Cellpose",
           "--nucleus-prop", one("*nucleus_property.txt"),
           "--nucleus-count-h5", one("*cell_count.h5"),
           "--nucleus-seg", one("*nucleus_coord.txt"),
           "--nworkers", str(args.nworkers),
           "--cell-radius", str(r_units),
           "--outdir", str(sdir), "--outprefix", "lung"]
    if args.alpha is not None:
        cmd += ["--alpha", str(args.alpha)]
    if args.gene_use:
        cmd += ["--gene-use", args.gene_use]

    import psutil
    vm = psutil.virtual_memory()
    log.info(f"RAM         total {vm.total/2**30:.1f} GiB  available "
             f"{vm.available/2**30:.1f} GiB")
    log.info(" ".join(cmd))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)
    for line in p.stdout:
        log.info("    " + line.rstrip())
    if p.wait() != 0:
        raise SystemExit(
            f"cellist seg exited with {p.returncode}. A BrokenProcessPool here "
            "means the workers were killed for memory: lower --nworkers.")

    rule("RASTER")
    tabs = sorted(sdir.rglob("*_Cellist_segmentation.txt"),
                  key=lambda q: q.stat().st_mtime, reverse=True)
    if not tabs:
        raise SystemExit("no *_Cellist_segmentation.txt produced")
    log.info(f"table       {tabs[0]}")
    d = pd.read_csv(tabs[0], sep="\t")
    lcol = next(c for c in ("Cellist", "Cellist_unfiltered") if c in d.columns)
    g = meta["grid"]
    lab = np.zeros((g, g), np.int64)
    v = pd.to_numeric(d[lcol], errors="coerce").fillna(0).astype(np.int64)
    ok = v > 0
    lab[np.clip(d.y[ok].astype(int), 0, g-1), np.clip(d.x[ok].astype(int), 0, g-1)] \
        = v[ok]
    occ0 = (lab > 0).mean()
    if args.fill_um > 0:
        lab = expand_labels(lab, args.fill_um / meta["unit_um"])
    log.info(f"occupancy   {occ0*100:.2f}% -> {(lab>0).mean()*100:.2f}% after a "
             f"{args.fill_um} um fill (Cellist labels spots, not pixels)")
    tifffile.imwrite(out / "cellist_segmentation.tif", lab.astype(np.int32))

    geo, _ = load_geometry(root, args.roi_size, args.dapi)
    tx = load_roi_transcripts(root, geo, root / "_roi_cache")
    qc(lab, geo, tx, meta["platform_cells"], out, "Cellist",
       px_per_cell=meta["gem_bin"])


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