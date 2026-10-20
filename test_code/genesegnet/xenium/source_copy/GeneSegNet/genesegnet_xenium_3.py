#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
genesegnet_xenium.py -- GeneSegNet inference @ xenium_2 / 3 / 5 / 6

Weights: self-trained Xenium pancreas_cancer model (residual_on / style_off /
concatenation_off). None of the four target datasets contains pancreas, so all are held-out.

New in this version (vs the previous one; only three places touching xenium_3):
  FIX-C  xenium_3 is a ROI cropped from the parent slide: image 1577x1576 px, but
         transcripts / cells still store **parent-slide global um coordinates**
         (TX x[3982.50,4317.50] y[2522.50,2857.50], span 335.00 um).
         Querying x in [0,335) um under the global convention finds not a single one.
         -> hard-code coord_mode=local_um + origin (3982.5, 2522.5) in DATASETS.
         Pixel size unchanged: 1576 x 0.2125 = 334.9 um, half a pixel off from 335.0.
  FIX-D  add coord_mode / origin_x_um / origin_y_um to prep_meta.json,
         otherwise eval_genesegnet.py cannot read the convention and falls back to global, computing wrong numbers.
         Already-run x2/x5/x6 lack these keys; their global fallback is exactly right.
  FIX-E  in main(), let the DATASETS coord_mode override the default (the command line can still override again).

Carried-over fixes:
  FIX-A  all-zero DAPI = tifffile OME multi-file assembly failure ("OME series cannot read
         multi-file pyramids" then no raise, all-zero series). Strategy chain
         is_ome=False -> OME series -> pages[0], each first samples and validates non-zero.
  FIX-B  heatmap sigma uses an exact solution: pixels ==255 are exactly spot centers; extract them,
         rebuild for each candidate sigma and compare MAE; at the true value MAE is 0.
  FIX-1  z-stack mistaken for multichannel (ZYX misread as CYX, picked an almost black z0)
  FIX-2  pyramids go through series.levels[0]
  FIX-3  zarr 2.x/3.x dual path + no-zarr fallback
  FIX-4  OME Channel Name recognizes DAPI; channel = choose file
  FIX-5  ROI edge clamping + non-square support
  FIX-6  heatmap consistent with training (max-of-Gaussians, peak 255)
  FIX-7  global percentile normalization for DAPI (no longer per tile)
  FIX-8  pyarrow pushdown filtering, no full-table parquet read
  FIX-9  drop control probes
  FIX-10 --audit per channel/z stats
  FIX-11 prep reports GT cell count in ROI and compares it with the expectation

Usage
  python genesegnet_xenium.py --calib-heatmap
  python genesegnet_xenium.py --dataset xenium_3 --audit
  python genesegnet_xenium.py --dataset xenium_3 --stage all --save-preview
"""
from pathlib import Path
import argparse
import glob
import importlib.util
import json
import re
import sys
import time

import numpy as np
import pandas as pd
import tifffile

REPO = Path("yourpath/GeneSegNet")
GS_REPO = REPO / "GeneSegNet"
HELPER = REPO / "genesegnet_cosmx_2.py"
DS = Path("yourpath/dataset")

DEFAULT_MODEL = (
    "yourpath/genesegnet_xenium/"
    "pancreas_cancer_cell/model/models/"
    "GeneSegNet_residual_on_style_off_concatenation_off_model_"
    "2026_08_19_16_55_17.947661"
)
CALIB_GLOB = "yourpath/genesegnet_xenium/pancreas_cancer_cell/**/*gaumap_all*"

TILE = 1024
OVERLAP = 128
TRUNC = 4.6052
TRUNC_R = float(np.sqrt(2 * TRUNC))     # = 3.0348, truncation radius / sigma

CTRL_PREFIX = (
    "NegControlProbe", "NegControlCodeword", "NegPrb", "BLANK", "Blank",
    "antisense", "UnassignedCodeword", "DeprecatedCodeword",
    "Intergenic", "Genomic",
)

# roi: ("full",) | ("center", h, w) | ("abs", y0, x0, h, w)
DATASETS = {
    "xenium_2": dict(root=DS / "xenium_breast", tag="breast",
                     roi=("center", 5000, 5000), gt_expected=8809),
    # FIX-C cropped ROI, coordinates are in the parent-slide global system
    "xenium_3": dict(root=DS / "xenium_roi_crop", tag="lymph_crop",
                     roi=("full",), gt_expected=2442,
                     coord_mode="local_um",
                     origin_x_um=3982.5, origin_y_um=2522.5),
    "xenium_5": dict(root=DS / "xenium_mouse_brain", tag="mouse_brain",
                     roi=("abs", 6956, 12077, 10000, 10000), gt_expected=11975),
    "xenium_6": dict(root=DS / "xenium_colon", tag="colon",
                     roi=("center", 10000, 10000), gt_expected=27983),
}


def banner(x):
    print()
    print("=" * 100)
    print(x)
    print("=" * 100, flush=True)


def choose(cols, candidates):
    low = {str(c).lower(): c for c in cols}
    for c in candidates:
        if c in cols:
            return c
        if c.lower() in low:
            return low[c.lower()]
    raise RuntimeError(f"cannot find any of {candidates}; columns={list(cols)}")


def resolve(root, candidates):
    root = Path(root)
    for rel in candidates:
        p = root / rel
        if p.exists():
            return p
    raise FileNotFoundError("cannot locate file; tried:\n"
                            + "\n".join(str(root / x) for x in candidates))


def resolve_dapi(root):
    return resolve(root, [
        "morphology_focus/ch0000_dapi.ome.tif",
        "morphology_focus/morphology_focus_0000.ome.tif",
        "outs/morphology_focus/morphology_focus_0000.ome.tif",
        "outs/morphology_focus/ch0000_dapi.ome.tif",
        "outs/morphology_focus.ome.tif", "morphology_focus.ome.tif",
        "outs/morphology_mip.ome.tif", "morphology_mip.ome.tif",
        "outs/morphology.ome.tif", "morphology.ome.tif",
    ])


def resolve_tx(root):
    return resolve(root, ["transcripts.parquet", "outs/transcripts.parquet"])


def resolve_gt(root):
    for rel in ("cells.parquet", "outs/cells.parquet", "cells.csv.gz"):
        p = Path(root) / rel
        if p.exists():
            return p
    return None


def sibling_channel_file(path, c):
    p = Path(path)
    m = re.match(r"^(.*?)(\d{4})(\.ome\.tif+)$", p.name)
    if not m:
        return None
    cand = p.with_name(f"{m.group(1)}{c:04d}{m.group(3)}")
    return str(cand) if cand.exists() else None


# ══════════════════════════════════════════════════════════════════════════
#  DAPI reading
# ══════════════════════════════════════════════════════════════════════════
def _open_zarr(store):
    try:
        import zarr
    except ImportError:
        return None
    for kw in ({}, {"zarr_format": 2}, {"zarr_version": 2}):
        try:
            return zarr.open(store, mode="r", **kw)
        except Exception:
            continue
    return None


def _level0(z):
    if z is not None and hasattr(z, "keys"):
        keys = [str(k) for k in z.keys()]
        if "0" in keys:
            return z["0"]
        if keys:
            keys.sort(key=lambda s: (len(s), s))
            return z[keys[0]]
        raise RuntimeError("empty zarr group")
    return z


def ome_channel_names(path):
    try:
        with tifffile.TiffFile(path) as tf:
            xml = tf.ome_metadata
        return re.findall(r'<Channel[^>]*\bName="([^"]*)"', xml) if xml else []
    except Exception:
        return []


class DapiReader:
    STRATEGIES = ("no_ome", "ome", "page0")

    def __init__(self, path, dapi_channel=-1, z_mode="mip", z_index=-1,
                 strategy=None, verbose=True):
        self.req_channel = dapi_channel
        self.chan_names = ome_channel_names(path)
        self.path = str(path)
        self.z_mode, self.z_index = z_mode, z_index
        self.verbose = verbose

        want = self._name_channel() if dapi_channel < 0 else dapi_channel
        if want > 0:
            alt = sibling_channel_file(self.path, want)
            if alt:
                if verbose:
                    print(f"  channel {want} -> switch file {Path(alt).name}")
                self.path, want = alt, 0

        tried = []
        order = [strategy] if strategy else list(self.STRATEGIES)
        for st in order:
            try:
                self._setup(st)
                self.strategy = st
                probe = self._probe()
                if probe is not None and float(np.abs(probe).max()) > 0:
                    break
                tried.append((st, "read as all zero"))
            except Exception as e:
                tried.append((st, f"{type(e).__name__}: {e}"))
        else:
            msg = "\n".join(f"    {s}: {r}" for s, r in tried)
            raise RuntimeError(f"no read strategy could get DAPI data:\n{msg}")

        self.channel = min(want, self.nc - 1)
        if verbose:
            print(f"DAPI file    = {self.path}")
            print(f"  strategy = {self.strategy}"
                  + "".join(f"  ({s} failed: {r})" for s, r in tried))
            print(f"  axes = {self.axes}  shape = {self.shape}  "
                  f"levels = {self.nlevels}  backend = {self.mode}")
            print(f"  channels = {self.nc}"
                  + (f"  names={self.chan_names}" if self.chan_names else "")
                  + f"  -> channel {self.channel}")
            print(f"  z planes = {self.nz}"
                  + (f"  -> {z_mode}" if self.nz > 1 else ""))

    def _name_channel(self):
        for i, n in enumerate(self.chan_names):
            if "dapi" in n.lower() or "nucle" in n.lower():
                return i
        return 0

    def _setup(self, st):
        self._cache = {}
        self._z = None
        self.mode = "tifffile"
        if st == "page0":
            self.tf = tifffile.TiffFile(self.path, is_ome=False)
            self.level = self.tf.pages[0]
            self.axes = "YX"
            self.shape = tuple(self.level.shape)[-2:]
            self.nlevels = 1
        else:
            self.tf = tifffile.TiffFile(
                self.path, **({} if st == "ome" else {"is_ome": False}))
            ser = self.tf.series[0]
            lv = None
            try:
                if getattr(ser, "levels", None):
                    lv = ser.levels[0]
            except Exception:
                lv = None
            self.level = lv if lv is not None else ser
            self.nlevels = len(getattr(ser, "levels", []) or []) or 1
            self.axes = (getattr(self.level, "axes", None)
                         or getattr(ser, "axes", None) or "YX")
            self.shape = tuple(self.level.shape)

        ax = self.axes
        self.iy, self.ix = ax.find("Y"), ax.find("X")
        if self.iy < 0 or self.ix < 0:
            self.iy, self.ix = len(self.shape) - 2, len(self.shape) - 1
            ax = "?" * (len(self.shape) - 2) + "YX"
            self.axes = ax
        self.ic = next((ax.find(c) for c in "CSIQ" if c in ax), -1)
        self.iz = ax.find("Z") if "Z" in ax else -1
        self.nc = self.shape[self.ic] if self.ic >= 0 else 1
        self.nz = self.shape[self.iz] if self.iz >= 0 else 1

        if st != "page0":
            try:
                zz = _open_zarr(self.level.aszarr())
                if zz is not None:
                    self._z, self.mode = _level0(zz), "zarr"
            except Exception:
                self._z = None

    def _probe(self, n=256):
        H, W = self.hw
        y0, x0 = max(0, H // 2 - n // 2), max(0, W // 2 - n // 2)
        y1, x1 = min(H, y0 + n), min(W, x0 + n)
        best = None
        for c in range(self.nc):
            p = np.squeeze(self._plane(c, self.nz // 2, y0, y1, x0, x1))
            if best is None or float(np.abs(p).max()) > float(
                    np.abs(best).max()):
                best = p
        return best

    @property
    def hw(self):
        return int(self.shape[self.iy]), int(self.shape[self.ix])

    def _plane(self, c, z, y0, y1, x0, x1):
        if getattr(self, "strategy", None) == "page0":
            return self._page0_array()[y0:y1, x0:x1]
        idx = []
        for a in range(len(self.shape)):
            if a == self.iy:
                idx.append(slice(y0, y1))
            elif a == self.ix:
                idx.append(slice(x0, x1))
            elif a == self.ic:
                idx.append(int(min(c, self.nc - 1)))
            elif a == self.iz:
                idx.append(int(min(z, self.nz - 1)))
            else:
                idx.append(0)
        if self.mode == "zarr":
            return np.asarray(self._z[tuple(idx)])
        key = (c, z)
        if key not in self._cache:
            self._cache.clear()
            full = [slice(None) if a in (self.iy, self.ix) else
                    (int(min(c, self.nc - 1)) if a == self.ic else
                     (int(min(z, self.nz - 1)) if a == self.iz else 0))
                    for a in range(len(self.shape))]
            self._cache[key] = np.squeeze(
                np.asarray(self.level.asarray()[tuple(full)]))
        return self._cache[key][y0:y1, x0:x1]

    def _page0_array(self):
        if "p0" not in self._cache:
            self._cache["p0"] = np.squeeze(np.asarray(self.level.asarray()))
        return self._cache["p0"]

    def crop(self, y0, y1, x0, x1):
        H, W = self.hw
        y0, x0 = max(0, y0), max(0, x0)
        y1, x1 = min(H, y1), min(W, x1)
        if self.nz <= 1:
            return np.squeeze(self._plane(self.channel, 0, y0, y1, x0, x1))
        if self.z_mode == "index":
            zi = self.z_index if self.z_index >= 0 else self.nz // 2
            return np.squeeze(self._plane(self.channel, zi, y0, y1, x0, x1))
        out = None
        for z in range(self.nz):
            p = np.squeeze(self._plane(self.channel, z, y0, y1, x0, x1))
            out = p.copy() if out is None else np.maximum(out, p)
        return out

    def audit(self, n=512):
        H, W = self.hw
        y0, x0 = max(0, H // 2 - n // 2), max(0, W // 2 - n // 2)
        y1, x1 = min(H, y0 + n), min(W, x0 + n)
        print(f"\ncenter {y1-y0}x{x1-x0} sample  (full image {H} x {W})")
        print(f"{'c':>3} {'z':>3} {'min':>8} {'mean':>10} {'p99':>10} "
              f"{'max':>8}  verdict")
        best = None
        for c in range(self.nc):
            for z in range(self.nz):
                p = np.squeeze(self._plane(c, z, y0, y1, x0, x1)
                               ).astype(np.float64)
                p99 = float(np.percentile(p, 99))
                flag = "empty" if p.max() == 0 else ("near empty" if p99 <= 1 else "")
                print(f"{c:>3} {z:>3} {p.min():>8.0f} {p.mean():>10.2f} "
                      f"{p99:>10.1f} {p.max():>8.0f}  {flag}")
                s = float(p.std())
                if best is None or s > best[0]:
                    best = (s, c, z)
        if best:
            print(f"\nhighest contrast: channel {best[1]} / z {best[2]}  "
                  f"(std={best[0]:.1f})")
        sibs = sorted(Path(self.path).parent.glob("*.ome.tif"))
        if len(sibs) > 1:
            print("\nother files in the same directory (center 256 sample max):")
            for s in sibs:
                try:
                    with tifffile.TiffFile(str(s), is_ome=False) as tf:
                        a = np.squeeze(tf.pages[0].asarray())
                    yy, xx = a.shape[0] // 2, a.shape[1] // 2
                    print(f"  {s.name:<45} shape={a.shape} "
                          f"max={int(a[yy:yy+256, xx:xx+256].max())}")
                except Exception as e:
                    print(f"  {s.name:<45} read failed {e}")
        return best


# ══════════════════════════════════════════════════════════════════════════
#  Heatmap + exact calibration
# ══════════════════════════════════════════════════════════════════════════
def gaussian_kernel(sigma):
    r = int(np.ceil(sigma * TRUNC_R))
    dy, dx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    expo = (dy * dy + dx * dx) / (2.0 * sigma * sigma)
    k = np.exp(-expo, dtype=np.float32)
    k[expo > TRUNC] = 0.0
    return k, r


def stamp(pts_yx, size_y, size_x, sigma):
    h = np.zeros((size_y, size_x), np.float32)
    if len(pts_yx) == 0:
        return h.astype(np.uint8)
    k, r = gaussian_kernel(sigma)
    for py, px in pts_yx:
        y0, y1 = max(0, py - r), min(size_y, py + r + 1)
        x0, x1 = max(0, px - r), min(size_x, px + r + 1)
        if y0 >= y1 or x0 >= x1:
            continue
        ky0, kx0 = y0 - (py - r), x0 - (px - r)
        np.maximum(h[y0:y1, x0:x1],
                   k[ky0:ky0 + (y1 - y0), kx0:kx0 + (x1 - x0)],
                   out=h[y0:y1, x0:x1])
    return np.rint(h * 255).astype(np.uint8)


def make_heatmap(tx, yy, xx, size_y, size_x, sigma, mode="gaussian_peak"):
    y = tx["roi_y_px"].to_numpy(np.float64) - yy
    x = tx["roi_x_px"].to_numpy(np.float64) - xx
    iy = np.floor(y).astype(np.int64)
    ix = np.floor(x).astype(np.int64)
    keep = (iy >= 0) & (iy < size_y) & (ix >= 0) & (ix < size_x)
    iy, ix = iy[keep], ix[keep]
    if mode == "density":
        from scipy.ndimage import gaussian_filter
        h = np.zeros((size_y, size_x), np.float32)
        if len(iy):
            np.add.at(h, (iy, ix), 1.0)
        h = gaussian_filter(h, sigma=float(sigma))
        if h.max() > 0:
            h /= h.max()
        return np.rint(h * 255).astype(np.uint8)
    if len(iy) == 0:
        return np.zeros((size_y, size_x), np.uint8)
    return stamp(np.unique(np.stack([iy, ix], 1), axis=0),
                 size_y, size_x, sigma)


def heat_fingerprint(h):
    h = np.squeeze(np.asarray(h))
    return dict(nonzero=float((h > 0).mean()), eq255=float((h == 255).mean()),
                p99=float(np.percentile(h, 99)))


def solve_sigma(h):
    h = np.squeeze(np.asarray(h)).astype(np.int32)
    pts = np.argwhere(h == 255)
    if len(pts) < 20:
        return None, None, len(pts)
    H, W = h.shape
    res = [(float(np.abs(stamp(pts, H, W, s).astype(np.int32) - h).mean()), s)
           for s in np.arange(0.5, 8.01, 0.25)]
    res.sort()
    best = res[0][1]
    res2 = [(float(np.abs(stamp(pts, H, W, s).astype(np.int32) - h).mean()), s)
            for s in np.arange(max(0.3, best - 0.3), best + 0.31, 0.05)]
    res2.sort()
    return float(res2[0][1]), float(res2[0][0]), len(pts)


def calib_heatmap(pattern, k=6):
    fs = [f for f in sorted(glob.glob(pattern, recursive=True))
          if f.lower().endswith((".tif", ".tiff", ".jpg", ".png"))][:k]
    if not fs:
        raise SystemExit(f"training heatmap not found: {pattern}")
    banner("Heatmap calibration (exact solution)")
    print(f"{'file':<48} {'nonzero':>7} {'==255':>9} {'p99':>6} "
          f"{'spot':>7} {'σ':>6} {'MAE':>8}")
    sigmas, eqs = [], []
    for f in fs:
        h = (tifffile.imread(f) if f.lower().endswith((".tif", ".tiff"))
             else __import__("cv2").imread(f, 0))
        fp = heat_fingerprint(h)
        s, mae, n = solve_sigma(h)
        eqs.append(fp["eq255"])
        if s:
            sigmas.append(s)
        print(f"{Path(f).name[:48]:<48} {fp['nonzero']:7.3f} "
              f"{fp['eq255']:9.5f} {fp['p99']:6.0f} {n:7d} "
              + (f"{s:6.2f} {mae:8.4f}" if s else f"{'--':>6} {'--':>8}"))
    eq = float(np.median(eqs))
    mode = "gaussian_peak" if eq > 1e-5 else "density"
    print(f"\nmode = {mode}   (median ==255 ratio {eq:.6f})")
    if mode == "gaussian_peak" and sigmas:
        sg, sd = float(np.median(sigmas)), float(np.std(sigmas))
        print(f"sigma = {sg:.2f}  (std {sd:.3f}"
              + (", consistent ✓" if sd < 0.1 else ", inconsistent ⚠") + ")")
        print(f"\n  --heatmap-mode {mode} --heatmap-sigma {sg:.2f}")
    return mode, (float(np.median(sigmas)) if sigmas else None)


# ══════════════════════════════════════════════════════════════════════════
#  Tiling / normalization / transcripts / GT
# ══════════════════════════════════════════════════════════════════════════
def starts(n, tile=TILE, overlap=OVERLAP):
    stride = tile - overlap
    if n <= tile:
        return [0]
    a = list(range(0, n - tile + 1, stride))
    if a[-1] != n - tile:
        a.append(n - tile)
    return a


def norm8(a, lo, hi):
    a = np.asarray(a, dtype=np.float32)
    if hi <= lo:
        return np.zeros(a.shape, dtype=np.uint8)
    return np.rint(np.clip((a - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def load_tx(args, x0, y0, size_y, size_x):
    import pyarrow.dataset as pds
    d = pds.dataset(args.tx, format="parquet")
    names = list(d.schema.names)
    xc = choose(names, ["x_location", "x"])
    yc = choose(names, ["y_location", "y"])
    gc = next((c for c in ("feature_name", "gene", "target")
               if c in names), None)
    qvc = "qv" if "qv" in names else ("QV" if "QV" in names else None)

    ps = float(args.pixel_size)
    if args.coord_mode == "global":
        ox, oy, gx0, gy0 = 0.0, 0.0, x0, y0
    else:                       # local_um: the image itself is the ROI, origin at origin_*_um
        ox, oy, gx0, gy0 = args.origin_x_um, args.origin_y_um, 0, 0

    xlo, xhi = gx0 * ps + ox, (gx0 + size_x) * ps + ox
    ylo, yhi = gy0 * ps + oy, (gy0 + size_y) * ps + oy
    print(f"  transcript query window x[{xlo:.2f},{xhi:.2f}] y[{ylo:.2f},{yhi:.2f}] um "
          f"(coord_mode={args.coord_mode})")
    f = ((pds.field(xc) >= xlo) & (pds.field(xc) < xhi) &
         (pds.field(yc) >= ylo) & (pds.field(yc) < yhi))
    if qvc:
        f = f & (pds.field(qvc) >= float(args.qv_min))

    q = d.to_table(columns=[c for c in (xc, yc, gc) if c],
                   filter=f).to_pandas()
    if gc and len(q):
        if isinstance(q[gc].iloc[0], bytes):
            q[gc] = q[gc].str.decode("utf-8")
        q[gc] = q[gc].astype(str)
        if not args.keep_controls:
            m = ~q[gc].str.startswith(CTRL_PREFIX)
            print(f"  removed control probes {int((~m).sum()):,}")
            q = q.loc[m]
        q = q.rename(columns={gc: "gene"})
    q = q.reset_index(drop=True)
    q["roi_x_px"] = (q[xc].to_numpy(np.float64) - ox) / ps - gx0
    q["roi_y_px"] = (q[yc].to_numpy(np.float64) - oy) / ps - gy0
    return q


def count_gt(gt_path, args, x0, y0, size_y, size_x):
    p = Path(gt_path)
    g = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    xc = next((c for c in ("x_centroid", "x", "center_x") if c in g.columns),
              None)
    yc = next((c for c in ("y_centroid", "y", "center_y") if c in g.columns),
              None)
    if xc is None or yc is None:
        return None, list(g.columns)
    ps = float(args.pixel_size)
    if args.coord_mode == "global":
        ox, oy, gx0, gy0 = 0.0, 0.0, x0, y0
    else:
        ox, oy, gx0, gy0 = args.origin_x_um, args.origin_y_um, 0, 0
    px = (g[xc].to_numpy(float) - ox) / ps - gx0
    py = (g[yc].to_numpy(float) - oy) / ps - gy0
    m = (px >= 0) & (px < size_x) & (py >= 0) & (py < size_y)
    return int(m.sum()), None


# ══════════════════════════════════════════════════════════════════════════
#  PREP
# ══════════════════════════════════════════════════════════════════════════
def resolve_roi(args, H, W):
    spec = args._roi_spec
    if args.roi_size and args.roi_size > 0:
        spec = ("center", args.roi_size, args.roi_size)
        if args.roi_y0 >= 0 and args.roi_x0 >= 0:
            spec = ("abs", args.roi_y0, args.roi_x0,
                    args.roi_size, args.roi_size)
    if spec[0] == "full":
        return 0, 0, H, W
    if spec[0] == "center":
        sh, sw = min(spec[1], H), min(spec[2], W)
        return (H - sh) // 2, (W - sw) // 2, sh, sw
    _, y0, x0, sh, sw = spec
    y0 = int(np.clip(y0, 0, max(0, H - 1)))
    x0 = int(np.clip(x0, 0, max(0, W - 1)))
    return y0, x0, min(sh, H - y0), min(sw, W - x0)


def prep(args):
    out = Path(args.out)
    prep_root = out / "prepared"
    image_dir, heat_dir = prep_root / "images", prep_root / "heatmaps"
    image_dir.mkdir(parents=True, exist_ok=True)
    heat_dir.mkdir(parents=True, exist_ok=True)

    rd = DapiReader(args.dapi, args.dapi_channel, args.z_mode, args.z_index,
                    args.dapi_strategy)
    H, W = rd.hw
    y0, x0, sh, sw = resolve_roi(args, H, W)
    print(f"ROI y[{y0},{y0+sh}) x[{x0},{x0+sw})  ({sh} x {sw} px "
          f"= {sh*args.pixel_size:.1f} x {sw*args.pixel_size:.1f} um)")

    roi = rd.crop(y0, y0 + sh, x0, x0 + sw)
    sh, sw = roi.shape[:2]
    print("DAPI ROI: min=", int(roi.min()), " max=", int(roi.max()),
          " mean=", round(float(roi.mean()), 2),
          " p99=", round(float(np.percentile(roi, 99)), 1))
    if roi.max() == 0:
        raise RuntimeError("DAPI ROI is all zero; run --audit to locate the problem.")

    flat = roi.reshape(-1)
    if flat.size > 4_000_000:
        flat = flat[np.random.default_rng(0).choice(flat.size, 4_000_000,
                                                    replace=False)]
    lo, hi = np.percentile(flat.astype(np.float32), [args.pmin, args.pmax])
    print(f"global normalization window: p{args.pmin}={lo:.1f}  p{args.pmax}={hi:.1f}")

    if args.save_preview:
        import cv2
        st = max(1, max(sh, sw) // 2048)
        cv2.imwrite(str(out / "dapi_preview.png"),
                    norm8(roi[::st, ::st], lo, hi))
        print("preview ->", out / "dapi_preview.png")

    tx = load_tx(args, x0, y0, sh, sw)
    print(f"ROI transcripts = {len(tx):,}"
          + (f"  genes {tx.gene.nunique():,}" if "gene" in tx.columns else ""))
    if len(tx) == 0:
        raise RuntimeError(
            "no transcripts in ROI. Cropped datasets often use parent-slide global coordinates -- "
            "specify the origin with --coord-mode local_um --origin-x-um/--origin-y-um.")

    gt_n = None
    if args.gt:
        gt_n, bad = count_gt(args.gt, args, x0, y0, sh, sw)
        if gt_n is None:
            print(f"  GT has no centroid column: {bad}")
        else:
            msg = f"  GT cells in ROI = {gt_n:,}"
            if args.gt_expected:
                dev = abs(gt_n - args.gt_expected) / args.gt_expected
                msg += (f"  expected {args.gt_expected:,}  deviation {dev:.1%}"
                        + ("  ✓" if dev <= 0.02 else "  ⚠ definition mismatch"))
            print(msg)
            print(f"  tx_per_cell = {len(tx)/max(gt_n,1):.1f}")

    rows = []
    for yy in starts(sh):
        for xx in starts(sw):
            name = f"{args.dataset}_y{yy:06d}_x{xx:06d}"
            im = roi[yy:yy + TILE, xx:xx + TILE]
            h, w = im.shape
            if h != TILE or w != TILE:
                pad = np.zeros((TILE, TILE), dtype=im.dtype)
                pad[:h, :w] = im
                im = pad
            im8 = norm8(im, lo, hi)
            hm = make_heatmap(tx, yy, xx, TILE, TILE, args.heatmap_sigma,
                              args.heatmap_mode)
            tifffile.imwrite(image_dir / f"{name}_image.tif", im8[..., None])
            tifffile.imwrite(heat_dir / f"{name}_gaumap_all.tif", hm[..., None])
            fp = heat_fingerprint(hm)
            rows.append(dict(name=name, y=yy, x=xx,
                             n_spots=int(((tx.roi_x_px >= xx) &
                                          (tx.roi_x_px < xx + TILE) &
                                          (tx.roi_y_px >= yy) &
                                          (tx.roi_y_px < yy + TILE)).sum()),
                             img_p99=float(np.percentile(im8, 99)),
                             hm_nonzero=fp["nonzero"], hm_p99=fp["p99"],
                             hm_eq255=fp["eq255"]))

    man = pd.DataFrame(rows)
    man.to_csv(prep_root / "tile_manifest.tsv", sep="\t", index=False)
    print(f"\ntiles {len(man)}   median spots {man.n_spots.median():.0f}   "
          f"empty tiles {int((man.n_spots==0).sum())}")
    print(f"heatmap fingerprint (median): nonzero {man.hm_nonzero.median():.3f}   "
          f"p99 {man.hm_p99.median():.0f}   ==255 {man.hm_eq255.median():.5f}")
    print("  training set reference: nonzero 0.040-0.347   p99 99-241   ==255 2e-4~3e-3")

    # FIX-D write the coordinate mode into meta for eval_genesegnet.py
    (prep_root / "prep_meta.json").write_text(json.dumps(dict(
        dataset=args.dataset, root=str(args.root), dapi=rd.path,
        dapi_strategy=rd.strategy, tx=args.tx,
        gt=str(args.gt) if args.gt else None, gt_in_roi=gt_n,
        gt_expected=args.gt_expected, dapi_channel=int(rd.channel),
        dapi_axes=rd.axes, dapi_nz=int(rd.nz), z_mode=args.z_mode,
        roi_y0=int(y0), roi_x0=int(x0), roi_h=int(sh), roi_w=int(sw),
        roi_size=int(min(sh, sw)), pixel_size_um=float(args.pixel_size),
        coord_mode=args.coord_mode,
        origin_x_um=float(args.origin_x_um),
        origin_y_um=float(args.origin_y_um),
        norm_lo=float(lo), norm_hi=float(hi),
        heatmap_mode=args.heatmap_mode,
        heatmap_sigma=float(args.heatmap_sigma),
        n_transcripts=int(len(tx)), n_tiles=int(len(rows)), uses_gt=False),
        indent=2, ensure_ascii=False))
    banner(f"PREP DONE: {len(rows)} tiles")


# ══════════════════════════════════════════════════════════════════════════
#  INFER
# ══════════════════════════════════════════════════════════════════════════
def load_helper():
    name = "genesegnet_cosmx_helper_generic"
    spec = importlib.util.spec_from_file_location(name, HELPER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_repo():
    sys.path.insert(0, str(GS_REPO))
    import models
    import utils
    import fastremap
    return models, utils, fastremap


def input_tile(ip, hp):
    im, hm = tifffile.imread(ip), tifffile.imread(hp)
    if im.ndim == 2:
        im = im[..., None]
    if hm.ndim == 2:
        hm = hm[..., None]
    if im.shape[:2] != hm.shape[:2]:
        raise RuntimeError(f"shape mismatch {im.shape} vs {hm.shape}")
    return np.concatenate([im.astype(np.float32),
                           hm.astype(np.float32)], axis=-1)


def qc(mask, pixel):
    u, n = np.unique(mask, return_counts=True)
    n = n[u > 0]
    if len(n):
        area = float(np.median(n))
        diam = float(2 * np.sqrt(area / np.pi) * pixel)
    else:
        area, diam = np.nan, np.nan
    return dict(cells=int(len(n)), foreground=float(np.mean(mask > 0)),
                area_med_px=area, diameter_med_um=diam)


def infer(args):
    out = Path(args.out)
    prep_root = out / "prepared"
    manifest = pd.read_csv(prep_root / "tile_manifest.tsv", sep="\t")
    meta = json.loads((prep_root / "prep_meta.json").read_text())
    size = int(meta["roi_size"])
    image_dir, heat_dir = prep_root / "images", prep_root / "heatmaps"
    mask_dir = out / "tile_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    helper = load_helper()
    models, utils, fastremap = load_repo()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    key = str(manifest.iloc[0]["name"])
    s = input_tile(image_dir / f"{key}_image.tif",
                   heat_dir / f"{key}_gaumap_all.tif")
    print("input audit =", s.shape, s.dtype, float(s.min()), float(s.max()))
    print("  ch0 DAPI mean/p99 =", round(float(s[..., 0].mean()), 2),
          round(float(np.percentile(s[..., 0], 99)), 1))
    print("  ch1 heat mean/p99 =", round(float(s[..., 1].mean()), 2),
          round(float(np.percentile(s[..., 1], 99)), 1))
    if s[..., 0].max() == 0:
        raise RuntimeError("DAPI channel is all zero; the image produced by prep is broken")
    if s[..., 1].max() == 0:
        print("  ⚠ heatmap channel is all zero, the model will degrade to pure image segmentation")
    if args.dry:
        banner("DRY RUN OK")
        return

    model = helper.build_model(models, args, model_path, nchan=2)
    records, qcs = [], []
    for i, row in manifest.iterrows():
        key, yy, xx = str(row["name"]), int(row["y"]), int(row["x"])
        mp = mask_dir / f"{key}_mask.tif"
        print(f"[{i+1}/{len(manifest)}] {key}", flush=True)
        if args.resume and mp.exists():
            m = tifffile.imread(mp)
        else:
            inp = input_tile(image_dir / f"{key}_image.tif",
                             heat_dir / f"{key}_gaumap_all.tif")
            t0 = time.time()
            m = helper.infer_tile(model, utils, fastremap, inp, args)
            tifffile.imwrite(mp, np.asarray(m, dtype=np.uint32),
                             compression="zlib")
            print("  seconds =", round(time.time() - t0, 2))
        q = qc(m, args.pixel_size)
        print("  cells=", q["cells"], " fg=", round(q["foreground"] * 100, 2),
              "%  diam=", round(q["diameter_med_um"], 2))
        q.update(sample=key, y=yy, x=xx)
        qcs.append(q)
        records.append(dict(y=yy, x=xx, mask=np.asarray(m), key=key))

    pd.DataFrame(qcs).to_csv(out / "tile_qc.tsv", sep="\t", index=False)
    banner(f"STITCH {args.dataset}")
    stitched, links = helper.stitch_fov(records, size, args)
    fp = out / f"genesegnet_{args.dataset}_mask.tif"
    tifffile.imwrite(fp, stitched.astype(np.uint32), compression="zlib")

    res = qc(stitched, args.pixel_size)
    banner(f"GENESEGNET {args.dataset} DONE")
    for k, v in res.items():
        print(k, "=", v)
    if meta.get("gt_in_roi"):
        print("cell_ratio vs GT =", round(res["cells"] / meta["gt_in_roi"], 4),
              f"(GT {meta['gt_in_roi']:,})")
    print("stitch_links =", links, "\nmask ->", fp)
    res.update(dataset=args.dataset, model=str(model_path),
               stitch_links=int(links), roi_size=size,
               roi_y0=meta["roi_y0"], roi_x0=meta["roi_x0"],
               roi_h=meta["roi_h"], roi_w=meta["roi_w"],
               gt_in_roi=meta.get("gt_in_roi"),
               pixel_size_um=meta["pixel_size_um"],
               coord_mode=meta.get("coord_mode"),
               dapi_strategy=meta.get("dapi_strategy"),
               heatmap_mode=meta["heatmap_mode"],
               heatmap_sigma=meta["heatmap_sigma"],
               uses_gt_during_inference=False)
    (out / "run_meta.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False))


# ══════════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=None, choices=list(DATASETS) + [None])
    p.add_argument("--root", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--stage", choices=["prep", "infer", "all"], default="all")
    p.add_argument("--audit", action="store_true")
    p.add_argument("--calib-heatmap", action="store_true")
    p.add_argument("--calib-glob", default=CALIB_GLOB)
    p.add_argument("--dapi", default=None)
    p.add_argument("--dapi-strategy",
                   choices=["no_ome", "ome", "page0"], default=None)
    p.add_argument("--tx", default=None)
    p.add_argument("--gt", default=None)
    p.add_argument("--gt-expected", type=int, default=None)
    p.add_argument("--dapi-channel", type=int, default=-1)
    p.add_argument("--z-mode", choices=["mip", "index"], default="mip")
    p.add_argument("--z-index", type=int, default=-1)
    p.add_argument("--pixel-size", type=float, default=0.2125)
    p.add_argument("--roi-size", type=int, default=0)
    p.add_argument("--roi-y0", type=int, default=-1)
    p.add_argument("--roi-x0", type=int, default=-1)
    p.add_argument("--coord-mode", choices=["global", "local_um"],
                   default=None,
                   help="if unset, use the DATASETS config, then default global")
    p.add_argument("--origin-y-um", type=float, default=None)
    p.add_argument("--origin-x-um", type=float, default=None)
    p.add_argument("--qv-min", type=float, default=20.0)
    p.add_argument("--keep-controls", action="store_true")
    p.add_argument("--heatmap-mode", choices=["gaussian_peak", "density"],
                   default="gaussian_peak")
    p.add_argument("--heatmap-sigma", type=float, default=2.5)
    p.add_argument("--pmin", type=float, default=1.0)
    p.add_argument("--pmax", type=float, default=99.8)
    p.add_argument("--save-preview", action="store_true")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--diameter", type=float, default=0.0)
    p.add_argument("--diam-mean", type=float, default=34.0)
    p.add_argument("--flow-threshold", type=float, default=0.4)
    p.add_argument("--confidence-threshold", type=float, default=0.9)
    p.add_argument("--min-size", type=int, default=300)
    p.add_argument("--network-tile-overlap", type=float, default=0.1)
    p.add_argument("--stitch-min-pixels", type=int, default=20)
    p.add_argument("--stitch-fraction", type=float, default=0.5)
    p.add_argument("--residual-off", action="store_true")
    p.add_argument("--style-on", action="store_true")
    p.add_argument("--concatenation", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry", action="store_true")
    a = p.parse_args()

    if a.calib_heatmap:
        calib_heatmap(a.calib_glob)
        return
    if not a.dataset:
        raise SystemExit("need --dataset (or --calib-heatmap)")

    cfg = DATASETS[a.dataset]
    a.root = a.root or str(cfg["root"])
    a.out = a.out or f"yourpath/genesegnet_{a.dataset}"
    a._roi_spec = cfg["roi"]
    if a.gt_expected is None:
        a.gt_expected = cfg["gt_expected"]

    # FIX-E DATASETS coordinate mode -> default; an explicit command-line value wins
    if a.coord_mode is None:
        a.coord_mode = cfg.get("coord_mode", "global")
    if a.origin_x_um is None:
        a.origin_x_um = cfg.get("origin_x_um", 0.0)
    if a.origin_y_um is None:
        a.origin_y_um = cfg.get("origin_y_um", 0.0)

    if a.dapi is None:
        a.dapi = str(resolve_dapi(a.root))
    if a.tx is None:
        a.tx = str(resolve_tx(a.root))
    if a.gt is None:
        g = resolve_gt(a.root)
        a.gt = str(g) if g else None

    banner(f"GENESEGNET {a.dataset}  ({cfg['tag']})")
    print("root  =", a.root)
    print("DAPI  =", a.dapi)
    print("TX    =", a.tx)
    print("GT    =", a.gt)
    print("coord =", a.coord_mode,
          f"origin=({a.origin_x_um}, {a.origin_y_um})"
          if a.coord_mode != "global" else "")
    print("heat  =", a.heatmap_mode, "sigma", a.heatmap_sigma)
    print("out   =", a.out)

    if a.audit:
        DapiReader(a.dapi, a.dapi_channel, a.z_mode, a.z_index,
                   a.dapi_strategy).audit()
        return

    Path(a.out).mkdir(parents=True, exist_ok=True)
    if a.stage in ("prep", "all"):
        prep(a)
    if a.stage in ("infer", "all"):
        infer(a)


if __name__ == "__main__":
    main()