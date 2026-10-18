#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ucs_xenium_colon.py

UCS on the Xenium mouse colon ROI (xenium_6).

Measured from the bundle, not assumed:
    pixel size   0.2125 um/px          panel  379 targets
    image        34104 x 28482 px      ROI    centred 10000 px -> y0=12052 x0=9241
    ROI          x[1963.71,4088.71] y[2561.05,4686.05] um   (2125 x 2125 um)
    GT           27,983 cells with their centroid inside the ROI
    density      6,197 cells/mm^2

Two things this bundle gets wrong if you are not careful.  morphology.ome.tif is
a 12-plane z stack, so DAPI must come from morphology_focus/, whose channel 0 is
the nuclear stain.  And the tiles are JPEG 2000, where a zarr read returns zeros
instead of raising when a codec is missing, so the store is probed at five places
before it is trusted.

The prior is nucleus_boundaries, i.e. the same reference-assisted footing as the
other Xenium UCS rows -- not cell_boundaries, which would leave UCS nothing to
expand into.

  conda activate ucs
  python ucs_xenium_colon.py --stage prep
  python ucs_xenium_colon.py --stage run --gpu auto
  python ucs_xenium_colon.py --stage post

This file is self-contained.
"""


from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "2026.08.14"

DATA_ROOT = "/data/qiuyijia/dataset/xenium_colon"
DATASET = "xenium_6"
LABEL = "Xenium 小鼠结肠"
PIXEL_SIZE = 0.2125
ROI_SIZE = 10000
ROI_Y0 = 12052
ROI_X0 = 9241
GT_CELLS = 27983          # cells.parquet centroids inside the ROI, measured
N_GENES = 379             # Xenium Mouse Tissue Atlassing panel

log = logging.getLogger("xenium_colon")


def setup_log(d: Path, tag: str):
    d.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    f = logging.Formatter("%(asctime)s  %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(f); log.addHandler(sh)
    fh = logging.FileHandler(d / f"{tag}.log", mode="a"); fh.setFormatter(f)
    log.addHandler(fh)


def rule(m=""):
    log.info("=" * 78)
    if m:
        log.info(m)
        log.info("=" * 78)


def find(root: Path, *pats):
    for p in pats:
        h = sorted(root.rglob(p))
        if h:
            return h[0]
    return None


def read_tab(p: Path):
    for q in (p, p.with_suffix(".csv.gz"), p.with_suffix(".csv")):
        if q.exists():
            return (pd.read_csv(q) if str(q).endswith((".csv", ".csv.gz"))
                    else pd.read_parquet(q))
    return None


def write_tab(df: pd.DataFrame, p: Path):
    try:
        df.to_parquet(p, index=False)
        return p
    except Exception:
        q = p.with_suffix(".csv.gz")
        df.to_csv(q, index=False)
        return q


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


@dataclass
class Geometry:
    pixel_size: float
    y0: int
    x0: int
    size: int
    H: int = 0
    W: int = 0

    @property
    def um(self):
        return self.pixel_size

    def bounds_um(self):
        return (self.x0 * self.um, (self.x0 + self.size) * self.um,
                self.y0 * self.um, (self.y0 + self.size) * self.um)

    def um_to_local(self, ux, uy):
        return (np.asarray(ux, float) / self.um - self.x0,
                np.asarray(uy, float) / self.um - self.y0)

    def local_to_um(self, px, py):
        return ((np.asarray(px, float) + self.x0) * self.um,
                (np.asarray(py, float) + self.y0) * self.um)

    def inside_um(self, ux, uy):
        a, b, c, d = self.bounds_um()
        return (ux >= a) & (ux < b) & (uy >= c) & (uy < d)


def pick_dapi(root: Path, explicit=None) -> Path:
    """
    morphology_focus first.

    On this bundle morphology.ome.tif is a 12-plane z stack, while
    morphology_focus/ holds the four stains with DAPI as channel 0.  Choosing by
    size would pick the z stack and every nucleus prior built from it would be
    wrong.
    """
    if explicit:
        return Path(explicit)
    for pat in ("morphology_focus/morphology_focus_0000.ome.tif",
                "morphology_focus_0000.ome.tif", "morphology_focus.ome.tif",
                "morphology_mip.ome.tif", "morphology.ome.tif"):
        h = find(root, pat)
        if h is not None:
            return h
    raise SystemExit(f"no morphology image under {root}")


def load_geometry(root: Path, args) -> tuple:
    rule("GEOMETRY")
    ps = args.pixel_size
    exp = find(root, "experiment.xenium")
    if exp is not None:
        try:
            j = json.loads(exp.read_text())
            for k in ("pixel_size", "pixel_size_um", "um_per_pixel"):
                if k in j:
                    ps = float(j[k])
                    break
            log.info(f"experiment.xenium  pixel_size={ps}  "
                     f"panel={j.get('panel_name')}  "
                     f"targets={j.get('panel_num_targets_predesigned')}")
        except Exception as e:
            log.info(f"could not parse experiment.xenium ({e}); using {ps}")

    dapi = pick_dapi(root, args.dapi)
    import tifffile
    with tifffile.TiffFile(dapi) as tf:
        s = tf.series[0]
        shape = s.shape
        levels = len(getattr(s, "levels", [s]))
        pg = s.levels[0].pages[0] if hasattr(s, "levels") else s.pages[0]
        comp = pg.compression
    ch, H, W = (shape[0], shape[1], shape[2]) if len(shape) == 3 \
        else (1, shape[-2], shape[-1])
    log.info(f"morphology  {dapi.name}  {shape}  channels={ch}  "
             f"pyramid={levels}  compression={comp}")
    if ch > 1:
        log.info(f"            using channel {args.dapi_channel} as DAPI")

    geo = Geometry(ps, args.roi_y0, args.roi_x0, args.roi_size, H, W)
    a, b, c, d = geo.bounds_um()
    log.info(f"image       {H} x {W} native px = {H*ps:.0f} x {W*ps:.0f} um")
    log.info(f"ROI         y[{geo.y0},{geo.y0+geo.size}) "
             f"x[{geo.x0},{geo.x0+geo.size}) native px")
    log.info(f"            x [{a:.2f},{b:.2f}] y [{c:.2f},{d:.2f}] um "
             f"({geo.size*ps:.1f} x {geo.size*ps:.1f})")
    if geo.y0 + geo.size > H or geo.x0 + geo.size > W:
        raise SystemExit("the ROI falls outside the image")
    return geo, dapi, ch


def read_dapi_roi(dapi: Path, geo: Geometry, channel: int) -> np.ndarray:
    """
    Crop the ROI out of one channel.

    tifffile's zarr store returns the fill value instead of raising when a codec
    is missing or the store and the installed zarr disagree, which once produced
    an all-black DAPI that nothing complained about.  So the store is probed at
    five places first and a full decode is used if it is lying.
    """
    import tifffile
    y0, x0, S = geo.y0, geo.x0, geo.size

    def probe(arr, nd):
        pts = [(geo.H // 2, geo.W // 2), (geo.H // 4, geo.W // 4),
               (geo.H // 4, 3 * geo.W // 4), (3 * geo.H // 4, geo.W // 4),
               (3 * geo.H // 4, 3 * geo.W // 4)]
        for (py, px) in pts:
            py = min(max(py, 0), geo.H - 256)
            px = min(max(px, 0), geo.W - 256)
            w = (arr[channel, py:py+256, px:px+256] if nd == 3
                 else arr[py:py+256, px:px+256])
            w = np.asarray(w)
            if w.size and w.min() != w.max():
                return True
        return False

    crop = None
    try:
        with tifffile.TiffFile(dapi) as tf:
            s = tf.series[0]
            z = s.aszarr(level=0) if hasattr(s, "levels") else s.aszarr()
            import zarr
            arr = zarr.open(z, mode="r")
            if not hasattr(arr, "shape"):          # a multiscale group
                keys = sorted(arr.array_keys(), key=lambda k: -np.prod(arr[k].shape))
                arr = arr[keys[0]]
            nd = len(arr.shape)
            if probe(arr, nd):
                crop = np.asarray(arr[channel, y0:y0+S, x0:x0+S] if nd == 3
                                  else arr[y0:y0+S, x0:x0+S])
                log.info("DAPI crop   via the zarr store (verified non-constant)")
            else:
                log.info("DAPI crop   the zarr store returned constant data at "
                         "every probe point; decoding the page instead")
    except Exception as e:
        log.info(f"DAPI crop   zarr path unavailable ({e.__class__.__name__}); "
                 "decoding the page")

    if crop is None:
        with tifffile.TiffFile(dapi) as tf:
            s = tf.series[0]
            pgs = s.levels[0].pages if hasattr(s, "levels") else s.pages
            a = pgs[channel].asarray() if len(pgs) > channel else \
                np.squeeze(s.asarray())
            a = np.squeeze(a)
            if a.ndim == 3:
                a = a[channel]
            crop = a[y0:y0+S, x0:x0+S]

    lo, hi = np.percentile(crop, [1, 99.5])
    log.info(f"            {crop.shape} {crop.dtype}  p1={lo:.0f} p99.5={hi:.0f} "
             f"mean={crop.mean():.0f} std={crop.std():.0f}")
    if hi <= lo:
        raise SystemExit(
            "the DAPI crop is constant. Try another --dapi-channel, or point "
            "--dapi at morphology_focus/morphology_focus_0000.ome.tif.")
    return crop


# ---------------------------------------------------------------------------
# transcripts and prior
# ---------------------------------------------------------------------------


def load_roi_transcripts(root: Path, geo: Geometry, cache: Path, args,
                         force=False) -> pd.DataFrame:
    rule("TRANSCRIPTS")
    cache.mkdir(parents=True, exist_ok=True)
    tag = f"tx_roi_{geo.y0}_{geo.x0}_{geo.size}_qv{args.qv_min:g}"
    if not force:
        d = read_tab(cache / f"{tag}.parquet")
        if d is not None:
            log.info(f"cache       {tag}  {len(d):,} rows, "
                     f"{d.gene.nunique():,} genes")
            return d

    tp = find(root, "transcripts.parquet", "transcripts.csv.gz")
    if tp is None:
        raise SystemExit("no transcripts table")
    a, b, c, dd = geo.bounds_um()
    log.info(f"source      {tp.name}")
    if str(tp).endswith(".parquet"):
        d = pd.read_parquet(tp)
    else:
        parts = []
        for ck in pd.read_csv(tp, chunksize=5_000_000):
            xk = next(k for k in ck.columns if k.lower() in
                      ("x_location", "global_x"))
            yk = next(k for k in ck.columns if k.lower() in
                      ("y_location", "global_y"))
            parts.append(ck[(ck[xk] >= a) & (ck[xk] < b)
                            & (ck[yk] >= c) & (ck[yk] < dd)])
        d = pd.concat(parts, ignore_index=True)
    xk = next(k for k in d.columns if k.lower() in ("x_location", "global_x"))
    yk = next(k for k in d.columns if k.lower() in ("y_location", "global_y"))
    gk = next(k for k in d.columns if k.lower() in ("feature_name", "gene"))
    ck_ = next((k for k in d.columns if k.lower() in ("cell_id", "cell")), None)
    n0 = len(d)
    d = d[(d[xk] >= a) & (d[xk] < b) & (d[yk] >= c) & (d[yk] < dd)]
    log.info(f"  {n0:,} rows -> {len(d):,} inside the ROI")
    if args.qv_min > 0 and "qv" in d.columns:
        before = len(d)
        d = d[d.qv >= args.qv_min]
        log.info(f"  qv >= {args.qv_min}: {len(d):,} of {before:,} "
                 f"({len(d)/max(before,1)*100:.1f}%)")
    if "is_gene" in d.columns:
        d = d[d.is_gene.astype(bool)]
    else:
        d = d[~d[gk].astype(str).str.startswith(
            ("NegControl", "BLANK", "Unassigned", "antisense",
             "DeprecatedCodeword"))]
    out = pd.DataFrame({"x_um": d[xk].to_numpy(float),
                        "y_um": d[yk].to_numpy(float),
                        "gene": d[gk].astype(str).to_numpy()})
    if ck_ is not None:
        raw = d[ck_].astype(str).to_numpy()
        out["platform_cell"] = np.where(
            np.isin(raw, ("UNASSIGNED", "-1", "nan", "")), "", raw)
        log.info(f"  platform cell column '{ck_}': "
                 f"{(out.platform_cell != '').mean()*100:.1f}% assigned")
    out = out.reset_index(drop=True)
    side = geo.size * geo.um
    log.info(f"  {len(out):,} coding transcripts, {out.gene.nunique():,} genes")
    log.info(f"  density {len(out)/side**2:.3f} tx/um^2")
    write_tab(out, cache / f"{tag}.parquet")
    return out


def platform_cells(root: Path, geo: Geometry) -> pd.DataFrame:
    cp = find(root, "cells.parquet", "cells.csv.gz")
    c = pd.read_parquet(cp) if str(cp).endswith(".parquet") else pd.read_csv(cp)
    xk = next(k for k in c.columns if k.lower() in ("x_centroid", "center_x"))
    yk = next(k for k in c.columns if k.lower() in ("y_centroid", "center_y"))
    ik = next(k for k in c.columns if k.lower() in ("cell_id", "cell"))
    keep = geo.inside_um(c[xk].to_numpy(float), c[yk].to_numpy(float))
    c = c.loc[keep].reset_index(drop=True)
    log.info(f"platform    {cp.name}: {len(c):,} cells with their centroid "
             "inside the ROI  <- GT count")
    return c.rename(columns={xk: "x_um", yk: "y_um", ik: "cell"})


def _fill_polygon(mask, pts, cid):
    """cv2 drawing rejects int32 label images, so a uint8 stamp is filled."""
    import cv2
    y0 = max(int(np.floor(pts[:, 1].min())), 0)
    y1 = min(int(np.ceil(pts[:, 1].max())) + 1, mask.shape[0])
    x0 = max(int(np.floor(pts[:, 0].min())), 0)
    x1 = min(int(np.ceil(pts[:, 0].max())) + 1, mask.shape[1])
    if y1 <= y0 or x1 <= x0:
        return 0
    tmp = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(tmp, [np.rint(pts - [x0, y0]).astype(np.int32)], 1)
    mask[y0:y1, x0:x1][tmp > 0] = cid
    return int((tmp > 0).sum())


def build_prior(root: Path, geo: Geometry, out: Path, which="nucleus",
                force=False) -> tuple:
    """Rasterise nucleus_boundaries (or cell_boundaries) into the ROI."""
    rule(f"PLATFORM {which.upper()} PRIOR")
    import tifffile
    p = out / f"platform_{which}_labels.tif"
    if p.exists() and not force:
        lab = tifffile.imread(p)
        ids = np.unique(lab[lab > 0])
        log.info(f"cache       {p.name}  {len(ids):,} objects, coverage "
                 f"{(lab>0).mean()*100:.2f}%")
        return lab, {}
    src = find(root, f"{which}_boundaries.parquet", f"{which}_boundaries.csv.gz")
    if src is None:
        raise SystemExit(f"no {which}_boundaries table")
    b = pd.read_parquet(src) if str(src).endswith(".parquet") else pd.read_csv(src)
    xk = next(k for k in b.columns if k.lower().startswith("vertex_x")
              or k.lower() == "x")
    yk = next(k for k in b.columns if k.lower().startswith("vertex_y")
              or k.lower() == "y")
    ik = next(k for k in b.columns if "cell" in k.lower())
    a, bb, c, d = geo.bounds_um()
    pad = 30.0
    sub = b[(b[xk] >= a - pad) & (b[xk] < bb + pad)
            & (b[yk] >= c - pad) & (b[yk] < d + pad)]
    log.info(f"source      {src.name}: {len(b):,} vertices, "
             f"{len(sub):,} near the ROI")

    lab = np.zeros((geo.size, geo.size), np.int32)
    name2id = {}
    painted = 0
    for k, (cid, g) in enumerate(sub.groupby(ik, sort=False), start=1):
        px, py = geo.um_to_local(g[xk].to_numpy(float), g[yk].to_numpy(float))
        if _fill_polygon(lab, np.column_stack([px, py]), k):
            name2id[str(cid)] = k
            painted += 1
    log.info(f"            {painted:,} objects rasterised")
    ar = np.bincount(lab.reshape(-1))[1:]
    ar = ar[ar > 0] * geo.um ** 2
    med = float(np.median(ar)) if len(ar) else 0.0
    log.info(f"prior       coverage {(lab>0).mean()*100:.2f}%  median area "
             f"{med:.1f} um^2 -> diameter {2*np.sqrt(med/np.pi):.2f} um")
    out.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(p, lab, compression="zlib")
    json.dump(name2id, open(out / f"platform_{which}_ids.json", "w"))
    return lab, name2id


def block_majority(lab, b, min_frac=0.5):
    """Down-sample labels: each block takes its commonest non-zero id."""
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


def assign_by_mask(lab, geo: Geometry, tx: pd.DataFrame, px_per_cell=1.0):
    lx, ly = geo.um_to_local(tx.x_um.to_numpy(float), tx.y_um.to_numpy(float))
    r = np.clip((ly / px_per_cell).astype(np.int64), 0, lab.shape[0] - 1)
    c = np.clip((lx / px_per_cell).astype(np.int64), 0, lab.shape[1] - 1)
    v = lab[r, c].astype(np.int64)
    v[v < 0] = 0
    return v


def qc(lab, geo: Geometry, tx: pd.DataFrame, n_platform: int, out: Path,
       method: str, px_per_cell=1.0, prior=None):
    rule("QC")
    um = px_per_cell * geo.um
    ids = np.unique(lab[lab > 0])
    cnt = np.bincount(lab.reshape(-1))
    area = cnt[ids].astype(float) * um * um
    med = float(np.median(area))
    log.info(f"cells       {len(ids):,}   platform {n_platform:,}   "
             f"ratio {len(ids)/max(n_platform,1):.3f}")
    log.info(f"coverage    {(lab>0).mean()*100:.2f}%")
    log.info(f"area        median {med:.1f} um^2  diameter "
             f"{2*np.sqrt(med/np.pi):.2f} um  "
             f"IQR {np.percentile(area,25):.0f}-{np.percentile(area,75):.0f}")
    if prior is not None:
        # the prior handed in here sits on the SAME grid as lab, so its pixel is
        # px_per_cell native pixels wide too; using geo.um alone reported a
        # 6.6 um nucleus as 1.3 um and an expansion of 63x
        pa = np.bincount(prior.reshape(-1))[1:]
        pa = pa[pa > 0].astype(float) * um * um
        if len(pa):
            pm = float(np.median(pa))
            log.info(f"prior       median {pm:.1f} um^2 -> "
                     f"{2*np.sqrt(pm/np.pi):.2f} um   expansion "
                     f"{med/pm:.2f}x in area ({np.sqrt(med/pm):.2f}x linear)")
            if med / pm < 1.5:
                log.info("  !! barely expanded; check that the prior is nuclei "
                         "and not cell outlines")

    tl = assign_by_mask(lab, geo, tx, px_per_cell)
    n_as = int((tl > 0).sum())
    log.info(f"transcripts {len(tx):,} in ROI, {n_as:,} assigned "
             f"({n_as/max(len(tx),1)*100:.1f}%)")

    genes = sorted(tx.gene.unique())
    gi = {g: i for i, g in enumerate(genes)}
    pos = {int(v): i for i, v in enumerate(ids)}
    mat = np.zeros((len(ids), len(genes)), np.int32)
    sel = tl > 0
    ri = np.array([pos.get(int(v), -1) for v in tl[sel]])
    ok = ri >= 0
    np.add.at(mat, (ri[ok],
                    np.array([gi[g] for g in tx.gene.to_numpy()[sel][ok]])), 1)
    per = mat.sum(axis=1)
    log.info(f"counts/cell median {np.median(per):.0f}  "
             f"IQR {np.percentile(per,25):.0f}-{np.percentile(per,75):.0f}")

    np.save(out / "cell_by_gene.npy", mat)
    pd.Series(genes).to_csv(out / "genes.txt", index=False, header=False)
    pd.DataFrame({"cell": ids, "area_um2": area, "n_counts": per}).to_csv(
        out / "cell_metadata.csv", index=False)
    json.dump(dict(method=method, dataset=DATASET, label=LABEL, version=VERSION,
                   cells=int(len(ids)), platform_cells=int(n_platform),
                   coverage=float((lab > 0).mean()), median_area_um2=med,
                   median_diameter_um=float(2*np.sqrt(med/np.pi)),
                   assigned=n_as, transcripts=int(len(tx)), genes=len(genes),
                   roi=dict(y0=geo.y0, x0=geo.x0, size=geo.size),
                   pixel_size=geo.um, px_per_cell=px_per_cell),
              open(out / "run_qc.json", "w"), indent=2)
    rule("DONE")


OUT_DEFAULT = "/data/qiuyijia/ucs_xenium_colon"
STAGES = ["prep", "run", "post", "all"]
METHOD = "UCS"


def add_args(ap):
    ap.add_argument("--bin-factor", type=int, default=10,
                    help="native px per grid pixel; 10 = 2.125 um")
    ap.add_argument("--ucs-repo", default="/data/qiuyijia/ucs/UCS")
    ap.add_argument("--gpu", default="auto")
    ap.add_argument("--min-free-mb", type=int, default=4000)
    ap.add_argument("--mem-fraction", type=float, default=0.30)
    ap.add_argument("--fg-batch", type=int, default=None)
    ap.add_argument("--patch-size", type=int, default=48)
    ap.add_argument("--top-genes", type=int, default=0)
    ap.add_argument("--ucs-extra", default="")


def stage_prep(root, out, args):
    import tifffile
    geo, dapi, nch = load_geometry(root, args)
    tx = load_roi_transcripts(root, geo, root / "_roi_cache", args,
                              args.force_prep)
    cells = platform_cells(root, geo)
    prior, _ = build_prior(root, geo, out, "nucleus", args.force_prep)

    b = args.bin_factor
    if geo.size % b:
        raise SystemExit(f"--bin-factor {b} does not divide {geo.size}")
    g = geo.size // b
    rule("GENE MAP")
    log.info(f"grid        {g} x {g}   1 bin = {b} px = {b*geo.um:.4f} um")

    genes = sorted(tx.gene.unique())
    if args.top_genes and args.top_genes < len(genes):
        keep = tx.gene.value_counts().head(args.top_genes).index
        tx = tx[tx.gene.isin(keep)].reset_index(drop=True)
        genes = sorted(tx.gene.unique())
        log.info(f"            restricted to the {len(genes)} most frequent")
    gb = g * g * len(genes) / 1e9
    log.info(f"            {g} x {g} x {len(genes)} uint8 = {gb:.2f} GB on disk")
    log.info(f"            {len(genes)} genes are {len(genes)} conv input "
             "channels, which is what sets the VRAM need")

    gi = {v: i for i, v in enumerate(genes)}
    lx, ly = geo.um_to_local(tx.x_um.to_numpy(float), tx.y_um.to_numpy(float))
    ri = np.clip((ly / b).astype(np.int32), 0, g - 1)
    ci = np.clip((lx / b).astype(np.int32), 0, g - 1)
    gidx = np.array([gi[v] for v in tx.gene.to_numpy()], np.int32)
    flat = np.bincount((ri.astype(np.int64) * g + ci) * len(genes) + gidx,
                       minlength=g * g * len(genes))
    gm = np.clip(flat, 0, 255).astype(np.uint8).reshape(g, g, len(genes))
    log.info(f"            occupied bins {(gm.sum(axis=2)>0).mean()*100:.2f}%")
    tifffile.imwrite(out / "gene_map.tif", gm)
    pd.Series(genes).to_csv(out / "genes.txt", index=False, header=False)
    del gm, flat

    rule("NUCLEUS PRIOR ON THE GRID")
    pa = np.bincount(prior.reshape(-1))[1:]
    pa = pa[pa > 0].astype(float) * geo.um ** 2
    nd = 2 * np.sqrt(float(np.median(pa)) / np.pi) if len(pa) else 0.0
    spans = nd / (b * geo.um)
    log.info(f"nucleus     median diameter {nd:.2f} um = {spans:.1f} bins")
    if spans < 3:
        log.info("  !! narrower than 3 bins: UCS has almost no room to grow. "
                 "This is what held the STARmap run to 1.00x. Lower --bin-factor.")
    small = block_majority(prior, b)
    n = int(np.unique(small[small > 0]).size)
    log.info(f"prior grid  {n:,} cells, coverage {(small>0).mean()*100:.2f}%")
    if n < 0.9 * len(cells):
        log.info(f"  !! {len(cells)-n:,} of {len(cells):,} vanish at bin {b}; "
                 "they are smaller than one bin")
    tifffile.imwrite(out / "nuclei_mask.tif", small.astype(np.uint32))

    json.dump(dict(version=VERSION, dataset=DATASET, bin_factor=b, grid=g,
                   genes=len(genes), pixel_size=geo.um,
                   roi=dict(y0=geo.y0, x0=geo.x0, size=geo.size),
                   nucleus_diam_um=nd, prior="nucleus_boundaries",
                   platform_cells=int(len(cells)), prior_cells_on_grid=n,
                   transcripts=int(len(tx)), dapi=str(dapi),
                   dapi_channel=args.dapi_channel),
              open(out / "prep_meta.json", "w"), indent=2)
    rule("PREP DONE")


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

    if str(args.gpu) == "auto":
        gpu, free = pick_gpu(args.min_free_mb)
        log.info(f"auto-selected GPU {gpu}")
    else:
        gpu, free = int(args.gpu), None

    batch = args.fg_batch
    if batch is None:
        one = meta["genes"] * args.patch_size ** 2 * 4 / 1e6
        budget = (free or 20000) * args.mem_fraction
        batch = 256
        while batch > 16 and one * batch > budget:
            batch //= 2
        log.info(f"one sample {one:.1f} MB; budget {budget:.0f} MB "
                 f"-> --fg_net_batch_size {batch}")

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
        raise SystemExit(f"UCS exited with {p.returncode}; on a CUDA OOM retry "
                         "with a smaller --fg-batch or --mem-fraction 0.15")
    rule("RUN DONE")


def stage_post(root, out, args):
    import tifffile
    rule("POST")
    geo, dapi, nch = load_geometry(root, args)
    meta = json.loads((out / "prep_meta.json").read_text())
    b = meta["bin_factor"]
    hits = sorted(out.glob("log*/pred/segmentation_mask.tif"))
    if not hits:
        raise SystemExit("no log*/pred/segmentation_mask.tif; run --stage run")
    lab = np.squeeze(tifffile.imread(hits[0])).astype(np.int64)
    log.info(f"prediction  {hits[0]}  {lab.shape}")
    prior = tifffile.imread(out / "nuclei_mask.tif").astype(np.int64)
    tx = load_roi_transcripts(root, geo, root / "_roi_cache", args)
    cells = platform_cells(root, geo)
    tifffile.imwrite(out / "ucs_segmentation.tif", lab.astype(np.int32))
    qc(lab, geo, tx, len(cells), out, "UCS", px_per_cell=b, prior=prior)
    log.info("note: the prior is the platform nucleus_boundaries, so the "
             "detection metrics against those cells are close to circular; the "
             "transcript-level columns are the informative ones")


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--root", default=DATA_ROOT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--stage", default="all", choices=STAGES)
    ap.add_argument("--pixel-size", type=float, default=PIXEL_SIZE)
    ap.add_argument("--roi-size", type=int, default=ROI_SIZE)
    ap.add_argument("--roi-y0", type=int, default=ROI_Y0)
    ap.add_argument("--roi-x0", type=int, default=ROI_X0)
    ap.add_argument("--dapi", default=None,
                    help="default prefers morphology_focus/, not the z stack")
    ap.add_argument("--dapi-channel", type=int, default=0)
    ap.add_argument("--qv-min", type=float, default=20.0)
    ap.add_argument("--force-prep", action="store_true")
    ap.add_argument("--version", action="store_true")
    add_args(ap)
    args = ap.parse_args()
    if args.version:
        print(VERSION)
        return
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    setup_log(out / "logs", f"{METHOD.lower()}_colon_{args.stage}")
    rule(f"{METHOD} v{VERSION}  x  {DATASET} ({LABEL})   out {out}")
    for st in (STAGES if args.stage == "all" else [args.stage]):
        if st == "all":
            continue
        fn = globals().get(f"stage_{st}")
        if fn is not None:
            fn(root, out, args)


if __name__ == "__main__":
    main()