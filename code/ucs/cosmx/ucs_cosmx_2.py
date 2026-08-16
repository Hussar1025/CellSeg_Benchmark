#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ucs_cosmx_2.py
=======================================================================
UCS on CosMx FOVs, one FOV at a time.

Data facts this is built around, all verified from the export itself:
  0.12000 um/px, agreed to 0.00% by two independent routes (the FOV
    position table, and Area.um2 / Area in the metadata)
  morphology lives in napari.zip as a zarr pyramid
    napari/images/{DNA,Membrane,PanCK,CD45,CD68,labels}/{0..7}
    level 0 is 93618 x 102128, chunked 8192, zlib
  the zip is read through zarr's ZipStore, so nothing is extracted
  FOV is 4256 x 4256 px = 511 um; cells are ~65 px = 7.8 um across
  transcript density is 19.1 tx/um^2, the highest of any dataset here

The nucleus prior comes from the platform: a disc at each cell centroid
sized by NucArea from the metadata, which is the CosMx analogue of
Xenium's nucleus_boundaries and keeps this run in the same
reference-assisted bucket as the platform-prior Xenium runs.

Bin size is chosen so a nucleus spans several bins. On STARmap a
bin_factor that left a nucleus under one bin gave UCS nothing to grow and
it expanded the prior by 1.00x; --min-nucleus-bins refuses that here.

The FOV window is verified rather than assumed: the labels pyramid is
cropped at the candidate window and its ids are compared with the
metadata cell ids for that FOV. A window that does not reproduce them is
rejected before anything is computed.

Usage
-----
  python ucs_cosmx_2.py --fovs 48 261 --stage prep
  python ucs_cosmx_2.py --fovs 48 261 --stage run
  python ucs_cosmx_2.py --fovs 48 261 --stage post
  python ucs_cosmx_2.py --fovs 48 261            # all three
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = "/data/qiuyijia/dataset/cosmx_lymph_node"
OUT_TEMPLATE = "/data/qiuyijia/ucs_cosmx_lymph_node/fov{fov}"
UCS_REPO = "/data/qiuyijia/ucs/UCS"
NOMINAL_UM_PX = 0.12
FOV_SIZE = 4256
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


# ----------------------------------------------------------------------
# napari zarr pyramid, read straight out of the zip
# ----------------------------------------------------------------------
class NapariImages:
    def __init__(self, zip_path, prefix=None):
        """Open the pyramid in place.

        The zarr root sits at napari/images inside the zip, not at the zip
        root: .zgroup lives there, so opening at '' fails with
        PathNotFoundError. Candidate paths are tried in order and the one
        that resolves is reported.
        """
        import zarr

        self.zip_path = str(zip_path)
        # ZipStore moved to zarr.storage in zarr 3, and open_group's
        # signature differs, so neither call style can be assumed
        self.store = None
        for maker in (lambda: zarr.storage.ZipStore(self.zip_path, mode="r"),
                      lambda: zarr.ZipStore(self.zip_path, mode="r")):
            try:
                self.store = maker()
                break
            except AttributeError:
                continue
        if self.store is None:
            raise RuntimeError("this zarr exposes no ZipStore; "
                               f"version {getattr(zarr, '__version__', '?')}")
        logging.info(f"zarr {getattr(zarr, '__version__', '?')}, "
                     f"{type(self.store).__name__}")
        cands = ([prefix] if prefix else
                 ["napari/images", "images", "napari", ""])
        self.root, self.prefix, last = None, None, None
        for p in cands:
            g = None
            for call in (lambda: zarr.open_group(store=self.store, path=p,
                                                 mode="r"),
                         lambda: zarr.open_group(self.store, mode="r",
                                                 path=p)):
                try:
                    g = call()
                    break
                except TypeError:
                    continue
                except Exception as e:  # noqa: BLE001
                    last = e
                    break
            if g is None:
                logging.info(f"  path {p!r}: "
                             f"{type(last).__name__ if last else 'no group'}")
                continue
            try:
                keys = sorted(set(g.array_keys()) | set(g.group_keys()))
            except Exception as e:  # noqa: BLE001
                last = e
                continue
            if keys:
                self.root, self.prefix = g, p
                logging.info(f"zarr root at {p!r}  keys {keys}")
                break
        if self.root is None:
            names = self._zip_names()
            raise RuntimeError(
                f"no zarr group found in {self.zip_path} (last error: {last}). "
                f"Entries look like: {names[:5]}")
        self.channels = sorted(k for k in
                               set(self.root.array_keys())
                               | set(self.root.group_keys())
                               if not k.startswith("."))
        logging.info(f"channels: {self.channels}")

    def _zip_names(self):
        with zipfile.ZipFile(self.zip_path) as z:
            return z.namelist()[:20]

    def level(self, channel, lv=0):
        if channel not in self.channels:
            raise KeyError(f"channel {channel!r} not in {self.channels}")
        return self.root[f"{channel}/{lv}"]

    def crop(self, channel, y0, x0, h, w, lv=0):
        a = self.level(channel, lv)
        H, W = a.shape[-2], a.shape[-1]
        y1, x1 = min(y0 + h, H), min(x0 + w, W)
        block = np.asarray(a[y0:y1, x0:x1])
        if block.shape != (h, w):
            pad = np.zeros((h, w), dtype=block.dtype)
            pad[:block.shape[0], :block.shape[1]] = block
            block = pad
        return block

    def close(self):
        try:
            self.store.close()
        except Exception:  # noqa: BLE001
            pass


def locate_tx_cache(cache_dir: Path, fovs):
    """Any cached subset covering the requested FOVs.

    The inspector names its cache after the FOVs it was given, so a cache
    built for 48 and 261 is tx_fov_48_261.parquet and asking for one FOV
    would look for a file that was never written.
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


def zattrs_offsets(zip_path, prefix="napari/images"):
    """The image's own FOV offsets, from its zarr attributes.

    The flat-file FOV table and the napari image need not share an origin
    or a y direction; on this export none of the four flip variants of the
    table reproduced the labels. The .zattrs CosMx block carries offsets
    written for this image, so it is the authoritative source.
    """
    try:
        with zipfile.ZipFile(str(zip_path)) as z:
            att = json.loads(z.read(f"{prefix}/.zattrs").decode())
    except Exception as e:  # noqa: BLE001
        logging.info(f"  no readable {prefix}/.zattrs ({e})")
        return None
    cos = att.get("CosMx") or att
    off = cos.get("fov_offsets")
    if not off:
        logging.info(f"  .zattrs has no fov_offsets; keys {list(cos)[:8]}")
        return None
    try:
        df = pd.DataFrame({k: pd.Series(v) for k, v in off.items()})
    except Exception as e:  # noqa: BLE001
        logging.info(f"  could not tabulate fov_offsets ({e})")
        return None
    logging.info(f"  .zattrs fov_offsets: {len(df)} rows, columns "
                 f"{list(df.columns)}")
    for k in ("fov_height", "fov_width"):
        if k in cos:
            logging.info(f"  .zattrs {k} = {cos[k]}")
    return df


def candidate_origins(fov, fov_tbl, zatt, H, W, um_px):
    """Every plausible (y0, x0), labelled, for the verifier to choose from."""
    out = {}

    def add(tag, x, y):
        if x is None or y is None or not np.isfinite(x) or not np.isfinite(y):
            return
        out[tag] = (int(round(y)), int(round(x)))
        out[f"{tag} y-flip"] = (int(round(H - y - FOV_SIZE)), int(round(x)))
        out[f"{tag} x-flip"] = (int(round(y)), int(round(W - x - FOV_SIZE)))
        out[f"{tag} xy-flip"] = (int(round(H - y - FOV_SIZE)),
                                 int(round(W - x - FOV_SIZE)))

    for src, tbl in (("zattrs", zatt), ("fovtable", fov_tbl)):
        if tbl is None or not len(tbl):
            continue
        fc = pick(tbl.columns, "FOV", "fov")
        if fc is None:
            continue
        row = tbl[pd.to_numeric(tbl[fc], errors="coerce") == fov]
        if not len(row):
            continue
        r = row.iloc[0]
        px_x = pick(tbl.columns, "x_global_px", "X_px", "x_px", "x")
        px_y = pick(tbl.columns, "y_global_px", "Y_px", "y_px", "y")
        if px_x and px_y:
            add(f"{src} px", float(r[px_x]), float(r[px_y]))
        mm_x = pick(tbl.columns, "x_global_mm", "X_mm", "x_mm")
        mm_y = pick(tbl.columns, "y_global_mm", "Y_mm", "y_mm")
        if mm_x and mm_y:
            add(f"{src} mm", float(r[mm_x]) * 1000.0 / um_px,
                float(r[mm_y]) * 1000.0 / um_px)
    return out


def fov_window(fov, fov_tbl, images, meta_fov, args, zatt=None):
    """Locate the FOV in whole-slide pixels and prove it with the labels.

    The offsets could come from the flat-file FOV table or from the zarr
    attributes, and the two need not share an origin or a y direction. So
    every candidate is checked against the labels pyramid: crop it and see
    whether the ids there are the ids the metadata records for this FOV.
    Guessing here would silently shift every transcript.
    """
    H, W = images.level("labels").shape[-2:]
    # the metadata carries three identifier columns (cell_ID, cell_id, cell)
    # and the labels image need not use the same one. Comparing against a
    # per-FOV index when the image stores a slide-wide index gives a low but
    # non-zero match, which looks like a wrong window and is not.
    id_sets = {}
    for c in ("cell_ID", "cell", "cell_id"):
        col = pick(meta_fov.columns, c)
        if col is None:
            continue
        v = pd.to_numeric(meta_fov[col], errors="coerce").dropna()
        if not len(v):
            v = (meta_fov[col].astype(str).str.extract(r"(\d+)$")[0]
                 .pipe(pd.to_numeric, errors="coerce").dropna())
        if len(v):
            id_sets[col] = set(v.astype(np.int64))
            logging.info(f"  id column {col}: {len(id_sets[col]):,} values, "
                         f"range {min(id_sets[col])}..{max(id_sets[col])}")
    if not id_sets:
        raise RuntimeError("no usable identifier column in the metadata")
    cands = candidate_origins(fov, fov_tbl, zatt, H, W, args.um_per_px)
    logging.info(f"  slide {H} x {W}; {len(cands)} candidate origins")
    if not cands:
        raise RuntimeError("no usable offsets in either the FOV table or "
                           ".zattrs")
    best, best_score, best_col = None, -1.0, None
    for name, (y0, x0) in cands.items():
        if not (0 <= y0 <= H - FOV_SIZE and 0 <= x0 <= W - FOV_SIZE):
            logging.info(f"    {name:<20} y0={y0} x0={x0}  out of bounds")
            continue
        sub = images.crop("labels", y0, x0, min(1024, FOV_SIZE),
                          min(1024, FOV_SIZE))
        ids = np.unique(sub)
        ids = set(int(v) for v in ids[ids > 0])
        if not ids:
            logging.info(f"    {name:<20} y0={y0} x0={x0}  empty")
            continue
        scores = {c: len(ids & w) / len(ids) for c, w in id_sets.items()}
        col = max(scores, key=scores.get)
        score = scores[col]
        logging.info(f"    {name:<20} y0={y0} x0={x0}  {len(ids)} ids, best "
                     f"{score*100:5.1f}% via {col}   ("
                     + "  ".join(f"{c}:{v*100:.1f}%"
                                 for c, v in scores.items()) + ")")
        if score > best_score:
            best, best_score, best_col = (y0, x0), score, col
    if best is None or best_score < args.min_window_score:
        raise RuntimeError(
            f"no candidate window reproduces FOV {fov}'s cell ids (best "
            f"{best_score*100:.1f}% < --min-window-score "
            f"{args.min_window_score*100:.0f}%). Run with --dump-window-scan "
            "to sweep the slide for the block that does contain this FOV's "
            "ids, or pass --window y0 x0 once it is known.")
    logging.info(f"  window y0={best[0]} x0={best[1]}  (match "
                 f"{best_score*100:.1f}% via {best_col})")
    return best[0], best[1], best_score


# ----------------------------------------------------------------------
def rasterise_fov_polygons(poly_path, fov, size, meta_fov):
    """Ground truth from the polygon table, in FOV-local pixels.

    The whole-slide image cannot be aligned: four independent strategies
    (both offset tables with every flip, cell-id matching, density
    correlation, whole-slide correlation) all failed, the last peaking at
    0.26 with the peak in a different place for every convention, which is
    what no peak looks like. The polygons carry the same segmentation in
    the same local frame as the transcripts, so the image is not needed at
    all: the prior already comes from the metadata, and this supplies the
    reference mask.
    """
    import cv2

    cols = list(pd.read_csv(poly_path, nrows=3).columns)
    logging.info(f"  polygon columns: {cols}")
    fc = pick(cols, "fov", "FOV")
    cid = pick(cols, "cellID", "cell_ID", "cell_id", "cell")
    xc = pick(cols, "x_local_px", "x_local", "vertex_x", "x")
    yc = pick(cols, "y_local_px", "y_local", "vertex_y", "y")
    if not (fc and cid and xc and yc):
        raise KeyError(f"polygon table lacks the needed columns: {cols}")
    use = [fc, cid, xc, yc]
    parts = []
    for c in pd.read_csv(poly_path, usecols=use, chunksize=2_000_000):
        sel = c[pd.to_numeric(c[fc], errors="coerce") == fov]
        if len(sel):
            parts.append(sel)
    if not parts:
        raise RuntimeError(f"no polygons for FOV {fov}")
    df = pd.concat(parts, ignore_index=True)

    def shoelace(v):
        x, y = v[:, 0], v[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    groups = [(k, g[[xc, yc]].to_numpy(float))
              for k, g in df.groupby(cid, sort=True)]
    groups = [(k, v) for k, v in groups if len(v) >= 3]
    expect = float(np.median([shoelace(v) for _, v in groups]))
    logging.info(f"  {len(groups):,} polygons, median {np.median([len(v) for _, v in groups]):.0f} "
                 f"vertices, shoelace area {expect:.0f} px")

    def draw(order_by_angle):
        lab = np.zeros((size, size), np.int32)
        for i, (_, v) in enumerate(groups, 1):
            p = v
            if order_by_angle:
                c = p.mean(axis=0)
                p = p[np.argsort(np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0]))]
            cv2.fillPoly(lab, [np.rint(p).astype(np.int32)], int(i))
        a = np.bincount(lab.reshape(-1).astype(np.int64))[1:]
        a = a[a > 0]
        return lab, (float(np.median(a)) if len(a) else 0.0), int(len(a))

    lab, got, n = draw(False)
    logging.info(f"  as listed: median rasterised area {got:.0f} px "
                 f"({got/max(expect,1e-9)*100:.0f}% of shoelace), "
                 f"{n:,} labels, coverage {(lab>0).mean()*100:.1f}%")
    if got < 0.7 * expect:
        # vertices that do not trace the outline produce a self-intersecting
        # polygon, which fills far less than its true area
        lab2, got2, n2 = draw(True)
        logging.info(f"  angle-sorted: median {got2:.0f} px "
                     f"({got2/max(expect,1e-9)*100:.0f}%), {n2:,} labels, "
                     f"coverage {(lab2>0).mean()*100:.1f}%")
        if got2 > got:
            logging.info("  the listed vertex order does not trace the "
                         "outline; using the angle-sorted fill")
            lab, got, n = lab2, got2, n2
    if got < 0.7 * expect:
        logging.warning(
            f"the rasterised area is still {got/max(expect,1e-9)*100:.0f}% of "
            "the polygon area. A cell may be stored as several rings, which "
            "cannot be filled as one contour; check whether any cellID has "
            "two disjoint vertex runs before trusting this mask.")
    logging.info(f"  metadata lists {len(meta_fov):,} cells; "
                 f"final coverage {(lab>0).mean()*100:.1f}%")
    if abs(n - len(meta_fov)) > 0.1 * max(len(meta_fov), 1):
        logging.warning("polygon and metadata cell counts differ by over 10%")
    return lab


def stage_prep(fov, paths, out, args):
    import tifffile

    banner(f"PREP  FOV {fov}")
    um_px = args.um_per_px
    images = None if args.no_image else NapariImages(paths["zip"])
    fov_tbl = pd.read_csv(paths["fov"])
    md = pd.read_csv(paths["metadata"])
    mf = pick(md.columns, "fov", "FOV")
    meta_fov = md[pd.to_numeric(md[mf], errors="coerce") == fov]
    logging.info(f"metadata cells in FOV {fov}: {len(meta_fov):,}")

    if args.no_image:
        y0 = x0 = 0
        score = float("nan")
        logging.info("  --no-image: working in FOV-local coordinates, so no "
                     "whole-slide window is needed")
    else:
      zatt = zattrs_offsets(paths["zip"], images.prefix or "napari/images")
      if args.window:
        y0, x0 = int(args.window[0]), int(args.window[1])
        score = float("nan")
        logging.info(f"  window forced to y0={y0} x0={x0}")
      else:
        tx_probe = pd.read_parquet(paths["tx_cache"])
        fcp = pick(tx_probe.columns, "fov", "FOV")
        tp = tx_probe[pd.to_numeric(tx_probe[fcp], errors="coerce") == fov]
        xl = tp[pick(tp.columns, "x_local_px")].to_numpy(float)
        yl = tp[pick(tp.columns, "y_local_px")].to_numpy(float)
        del tx_probe
        cands = window_from_mm(fov, zatt, images, args.um_per_px) or {}
        cands.update({k: v for k, v in candidate_origins(
            fov, fov_tbl, zatt, *images.level("labels").shape[-2:],
            args.um_per_px).items()})
        logging.info(f"  {len(cands)} candidate windows, scored by how well "
                     "the labels there follow this FOV's transcript density")
        best, score = None, -1.0
        conv = None
        for name, (yy, xx) in cands.items():
            r, cname = density_agreement(images, yy, xx, xl, yl)
            if r is None:
                logging.info(f"    {name:<22} y0={yy} x0={xx}  out of bounds")
                continue
            logging.info(f"    {name:<22} y0={yy} x0={xx}  r = {r:+.4f} "
                         f"via {cname}")
            if r > score:
                best, score, conv = (yy, xx), r, cname
        if best is None or score < args.min_density_r:
            logging.warning(f"no tabulated offset works (best r={score:+.3f}); "
                            "looking for the FOV grid itself")
            H, W = images.level("labels").shape[-2:]
            oy, ox, strength = locate_by_seams(images, args)
            logging.info(f"  grid origin ({oy}, {ox}) mod {FOV_SIZE}, "
                         f"strength {strength:.2f}; testing every cell of the "
                         "grid against this FOV's transcript density")
            best, score, conv = None, -2.0, None
            for gy in range((H - oy) // FOV_SIZE):
                for gx in range((W - ox) // FOV_SIZE):
                    yy, xx = oy + gy * FOV_SIZE, ox + gx * FOV_SIZE
                    r, cname = density_agreement(images, yy, xx, xl, yl,
                                                 block=64)
                    if r is not None and r > score:
                        best, score, conv = (yy, xx), r, cname
            if best is None or score < args.min_density_r:
                logging.warning(f"grid search best r={score:+.3f}; falling "
                                "back to whole-slide correlation")
                y0, x0, score, conv = locate_by_correlation(images, xl, yl,
                                                            args)
                best = (y0, x0)
            else:
                logging.info(f"  grid cell y0={best[0]} x0={best[1]} "
                             f"r={score:+.4f} via {conv}")
        y0, x0 = best
        logging.info(f"  window y0={y0} x0={x0}  (density r={score:+.4f}, "
                     f"local coordinates read as {conv})")
        if conv and conv != "normal":
            logging.warning(f"  the local coordinate convention is {conv}, not "
                            "the plain (y_local, x_local); every downstream "
                            "step must apply it too")
        (out / "coord_convention.txt").write_text(f"{conv}\n")

    bf = args.bin_factor
    Hb = Wb = FOV_SIZE // bf
    logging.info(f"bin {bf} px = {bf*um_px:.3f} um  ->  grid {Hb} x {Wb}")

    # nucleus prior from the platform: centroid + NucArea
    cx = pick(meta_fov.columns, "CenterX_local_px")
    cy = pick(meta_fov.columns, "CenterY_local_px")
    na = pick(meta_fov.columns, "NucArea")
    ar = pick(meta_fov.columns, "Area")
    if not (cx and cy):
        raise KeyError("metadata has no local centroid columns")
    area = pd.to_numeric(meta_fov[na if na else ar], errors="coerce")
    r_px = np.sqrt(np.maximum(area.to_numpy(), 1.0) / np.pi)
    logging.info(f"prior source = {'NucArea' if na else 'Area'}: median "
                 f"radius {np.median(r_px):.1f} px = "
                 f"{np.median(r_px)*um_px:.2f} um")
    nuc_bins = 2 * np.median(r_px) / bf
    logging.info(f"a nucleus spans {nuc_bins:.2f} bins")
    if nuc_bins < args.min_nucleus_bins:
        raise RuntimeError(
            f"a nucleus spans only {nuc_bins:.2f} bins, below "
            f"--min-nucleus-bins {args.min_nucleus_bins}. UCS convolves over "
            "this grid and would have no room to grow the prior; lower "
            f"--bin-factor to about {max(1, int(2*np.median(r_px)/args.min_nucleus_bins))}.")

    radii = np.clip(np.round(r_px / bf * args.prior_radius_scale),
                    args.min_prior_radius, args.max_prior_radius).astype(int)
    u, c = np.unique(radii, return_counts=True)
    logging.info("prior radii (bins): " + "  ".join(f"r={a}:{b}"
                                                    for a, b in zip(u, c)))
    if len(u) == 1:
        logging.warning("every prior has the same radius, so it carries no "
                        "size information; adjust the radius bounds")

    import cv2

    prior = np.zeros((Hb, Wb), np.int32)
    for i, (px, py, r) in enumerate(zip(meta_fov[cx].to_numpy(),
                                        meta_fov[cy].to_numpy(), radii), 1):
        bx = int(np.clip(round(px / bf), 0, Wb - 1))
        by = int(np.clip(round(py / bf), 0, Hb - 1))
        cv2.circle(prior, (bx, by), int(r), i, thickness=-1)
    n_prior = int(len(np.unique(prior)) - 1)
    logging.info(f"prior cells {n_prior:,}  foreground "
                 f"{(prior>0).mean()*100:.2f}%")
    tifffile.imwrite(str(out / "nuclei_mask.tif"), prior.astype(np.uint32))

    # gene map
    tx = pd.read_parquet(paths["tx_cache"]) if paths["tx_cache"].exists() \
        else None
    if tx is None:
        raise FileNotFoundError(
            f"{paths['tx_cache']} not found. Run the inspector first so the "
            "per-FOV transcript subset is cached; scanning 1.4 billion rows "
            "again would be wasteful.")
    fcol = pick(tx.columns, "fov", "FOV")
    t = tx[pd.to_numeric(tx[fcol], errors="coerce") == fov]
    lx = pick(t.columns, "x_local_px")
    ly = pick(t.columns, "y_local_px")
    tg = pick(t.columns, "target", "gene")
    logging.info(f"transcripts {len(t):,}  targets {t[tg].nunique():,}")

    genes = sorted(t[tg].astype(str).unique())
    counts = t[tg].astype(str).value_counts()
    if args.top_genes > 0 and len(genes) > args.top_genes:
        genes = sorted(counts.index[: args.top_genes].tolist())
        kept = counts[genes].sum() / counts.sum() * 100
        logging.info(f"gene filter {len(counts)} -> {len(genes)} "
                     f"({kept:.1f}% of transcripts)")
        t = t[t[tg].astype(str).isin(set(genes))]
    gi = {g: i for i, g in enumerate(genes)}
    nbytes = Hb * Wb * len(genes)
    logging.info(f"gene map {Hb} x {Wb} x {len(genes)} = {nbytes/1e9:.2f} GB")
    if nbytes / 1e9 > args.max_gene_map_gb:
        raise RuntimeError(f"gene map {nbytes/1e9:.1f} GB exceeds "
                           f"--max-gene-map-gb; use --top-genes")

    col = (t[lx].to_numpy() // bf).astype(np.int64)
    row = (t[ly].to_numpy() // bf).astype(np.int64)
    ok = (row >= 0) & (row < Hb) & (col >= 0) & (col < Wb)
    flat = (row[ok] * Wb + col[ok])
    code = t[tg].astype(str).map(gi).to_numpy()[ok]
    gm = np.zeros((Hb, Wb, len(genes)), np.uint8)
    order = np.argsort(code, kind="stable")
    fs, cs = flat[order], code[order]
    bounds = np.searchsorted(cs, np.arange(len(genes) + 1))
    for g in range(len(genes)):
        a, b = bounds[g], bounds[g + 1]
        if a == b:
            continue
        cnt = np.bincount(fs[a:b], minlength=Hb * Wb)
        np.clip(cnt, 0, 255, out=cnt)
        gm[:, :, g] = cnt.reshape(Hb, Wb).astype(np.uint8)
    tifffile.imwrite(str(out / "gene_map.tif"), gm, bigtiff=True,
                     photometric="minisblack")
    logging.info(f"occupied bins {(gm.sum(axis=2)>0).mean()*100:.2f}%")
    del gm
    gc.collect()

    if args.no_image:
        if paths.get("polygons") is None:
            raise FileNotFoundError("no polygon table, so no reference mask")
        logging.info("reference mask from the polygon table")
        gt = rasterise_fov_polygons(paths["polygons"], fov, FOV_SIZE, meta_fov)
        tifffile.imwrite(str(out / "platform_labels.tif"), gt.astype(np.uint32))
    else:
        dna = images.crop("DNA", y0, x0, FOV_SIZE, FOV_SIZE)
        logging.info(f"DNA crop {dna.shape} {dna.dtype} min {dna.min()} "
                     f"max {dna.max()} mean {dna.mean():.1f}")
        tifffile.imwrite(str(out / "dna.tif"), dna)
        gt = images.crop("labels", y0, x0, FOV_SIZE, FOV_SIZE)
        ids = np.unique(gt)
        logging.info(f"platform labels in window: {len(ids)-1:,} cells, "
                     f"coverage {(gt>0).mean()*100:.2f}%")
        tifffile.imwrite(str(out / "platform_labels.tif"),
                         gt.astype(np.uint32))
        images.close()

    (out / "roi_meta.json").write_text(json.dumps(dict(
        dataset="cosmx_lymph_node", fov=int(fov), y0=int(y0), x0=int(x0),
        size_y=FOV_SIZE, size_x=FOV_SIZE, pixel_size=float(um_px),
        bin_factor=int(bf), n_genes=len(genes), n_transcripts=int(len(t)),
        prior_cells=n_prior, window_match=float(score),
        gt_expected_cells=int(len(meta_fov))), indent=2))
    (out / "genes.txt").write_text("\n".join(genes) + "\n")
    banner("PREP DONE")


def window_from_mm(fov, zatt, images, um_px):
    """Place the FOV from the mm offsets of every FOV at once.

    Matching by cell id cannot work here: each FOV numbers its cells from
    1, so FOV 48's ids appear in all 400 FOVs and a sweep finds them
    spread across the whole slide. The mm offsets are unambiguous once the
    whole grid is mapped: shift the set so it starts at zero and check the
    span against the image, which also settles the y direction.
    """
    if zatt is None or not len(zatt):
        return None
    fc = pick(zatt.columns, "FOV", "fov")
    xm = pick(zatt.columns, "X_mm", "x_mm", "x_global_mm")
    ym = pick(zatt.columns, "Y_mm", "y_mm", "y_global_mm")
    if not (fc and xm and ym):
        return None
    H, W = images.level("labels").shape[-2:]
    x = pd.to_numeric(zatt[xm], errors="coerce").to_numpy() * 1000.0 / um_px
    y = pd.to_numeric(zatt[ym], errors="coerce").to_numpy() * 1000.0 / um_px
    ids = pd.to_numeric(zatt[fc], errors="coerce").to_numpy()
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(ids)
    x, y, ids = x[ok], y[ok], ids[ok]
    logging.info(f"  mm offsets: x span {x.max()-x.min():.0f} px, y span "
                 f"{y.max()-y.min():.0f} px; image {W} x {H}")
    logging.info(f"  grid + one FOV = {x.max()-x.min()+FOV_SIZE:.0f} x "
                 f"{y.max()-y.min()+FOV_SIZE:.0f} px")
    k = np.flatnonzero(ids.astype(int) == int(fov))
    if not len(k):
        return None
    k = int(k[0])
    out = {}
    out["mm, y down"] = (int(round(y[k] - y.min())), int(round(x[k] - x.min())))
    out["mm, y up"] = (int(round(y.max() - y[k])), int(round(x[k] - x.min())))
    out["mm, y down, x rev"] = (int(round(y[k] - y.min())),
                                int(round(x.max() - x[k])))
    out["mm, y up, x rev"] = (int(round(y.max() - y[k])),
                              int(round(x.max() - x[k])))
    return out


LOCAL_CONVENTIONS = {
    "normal": (False, False, False),
    "flip_y": (False, True, False),
    "flip_x": (False, False, True),
    "flip_xy": (False, True, True),
    "swap": (True, False, False),
    "swap_flip_y": (True, True, False),
    "swap_flip_x": (True, False, True),
    "swap_flip_xy": (True, True, True),
}


def density_agreement(images, y0, x0, xl, yl, block=32):
    """How well the FOV's transcripts line up with the labels there.

    Cell ids cannot identify the window because every FOV renumbers from 1,
    but the transcripts can: they are dense exactly where the cells are.

    The local coordinate convention is not assumed. CosMx FOVs are square,
    so a transposed or flipped local frame passes every bounds check while
    leaving the density map unrelated to the labels, which reads as a wrong
    window rather than a wrong convention. All eight are scored and the
    best is returned with its name.
    """
    H, W = images.level("labels").shape[-2:]
    if not (0 <= y0 <= H - FOV_SIZE and 0 <= x0 <= W - FOV_SIZE):
        return None, None
    lab = images.crop("labels", y0, x0, FOV_SIZE, FOV_SIZE)
    n = FOV_SIZE // block
    fg = (lab[:n * block, :n * block] > 0).reshape(n, block, n, block) \
        .mean(axis=(1, 3)).ravel()
    if fg.std() < 1e-9:
        return 0.0, "labels flat"
    best_r, best_name = -2.0, None
    for name, (swap, fr, fc) in LOCAL_CONVENTIONS.items():
        r_, c_ = (xl.copy(), yl.copy()) if swap else (yl.copy(), xl.copy())
        if fr:
            r_ = (FOV_SIZE - 1) - r_
        if fc:
            c_ = (FOV_SIZE - 1) - c_
        yi = np.clip((r_ / block).astype(int), 0, n - 1)
        xi = np.clip((c_ / block).astype(int), 0, n - 1)
        dens = np.bincount(yi * n + xi, minlength=n * n).astype(np.float32)
        if dens.std() < 1e-9:
            continue
        r = float(np.corrcoef(fg, dens)[0, 1])
        if r > best_r:
            best_r, best_name = r, name
    return best_r, best_name


def locate_by_seams(images, args, band=4096):
    """Find the FOV grid from the seams between stitched FOVs.

    Correlating transcript density against the labels fails on this tissue:
    lymph node fills roughly 90% of the frame, so the binary foreground is
    nearly constant and carries no spatial signal. Every convention peaked
    at 0.26 with the peak in a different place, which is what no peak looks
    like.

    The seams do carry signal. The labels were produced per FOV and
    stitched, so at a FOV boundary the id changes on every row at once,
    while ordinary cell borders change on only a few. Counting full-height
    discontinuities per column gives a comb with period fov_width, and its
    phase is the grid origin. No offset table is consulted.
    """
    lv = args.seam_level
    a = images.level("labels", lv)
    H, W = a.shape[-2], a.shape[-1]
    scale = 2 ** lv
    pitch = FOV_SIZE // scale
    if pitch < 8:
        raise RuntimeError(f"level {lv} makes a FOV only {pitch} px wide; "
                           "use a finer --seam-level")
    y0b = max(0, H // 2 - band // (2 * scale))
    h = min(band // scale, H - y0b)
    logging.info(f"  seam search at level {lv}: slide {H} x {W}, FOV pitch "
                 f"{pitch}, reading a {h} row band")
    strip = np.asarray(a[y0b:y0b + h, :])
    col_jump = (strip[:, 1:] != strip[:, :-1]).mean(axis=0)

    x0b = max(0, W // 2 - band // (2 * scale))
    w = min(band // scale, W - x0b)
    strip2 = np.asarray(a[:, x0b:x0b + w])
    row_jump = (strip2[1:, :] != strip2[:-1, :]).mean(axis=1)

    def phase(profile, pitch):
        n = len(profile)
        k = n / pitch
        idx = np.arange(n)
        z = np.sum((profile - profile.mean())
                   * np.exp(-2j * np.pi * k * idx / n))
        amp = 2 * np.abs(z) / n
        ph = (-np.angle(z) / (2 * np.pi) * pitch) % pitch
        return float(ph), float(amp / max(profile.std(), 1e-9))

    px, ax = phase(col_jump, pitch)
    py, ay = phase(row_jump, pitch)
    logging.info(f"  column discontinuities: mean {col_jump.mean():.3f}, "
                 f"seam phase {px:.1f} of {pitch}, strength {ax:.2f}")
    logging.info(f"  row    discontinuities: mean {row_jump.mean():.3f}, "
                 f"seam phase {py:.1f} of {pitch}, strength {ay:.2f}")
    if min(ax, ay) < args.min_seam_strength:
        raise RuntimeError(
            f"no periodic seam at the FOV pitch (strength {ax:.2f}, {ay:.2f} "
            f"< --min-seam-strength {args.min_seam_strength}). The labels were "
            "not stitched from per-FOV masks, or this image is not the one the "
            "tables describe.")
    return (int(round(py * scale)) % FOV_SIZE,
            int(round(px * scale)) % FOV_SIZE, float(min(ax, ay)))


def locate_by_correlation(images, xl, yl, args):
    """Find the FOV by cross-correlating its transcript density against the
    whole labels image.

    Every offset table has now failed: the flat-file positions, the zarr
    attributes, and all four flip variants of each, scored under all eight
    local coordinate conventions, top out near r = 0.09. Rather than
    proposing yet another convention, this searches. The FOV's transcripts
    are dense where its cells are, so its density map, slid over the whole
    slide, peaks at the FOV's position. One FFT settles it without assuming
    anything about origins or axis directions.
    """
    from scipy import signal

    lv = args.scan_level
    a = images.level("labels", lv)
    H, W = a.shape[-2], a.shape[-1]
    scale = 2 ** lv
    win = max(8, FOV_SIZE // scale)
    logging.info(f"  correlating at level {lv}: slide {H} x {W}, FOV is "
                 f"{win} px there ({H*W*4/1e9:.2f} GB to read)")
    slide = (np.asarray(a[:, :]) > 0).astype(np.float32)
    logging.info(f"  slide foreground {slide.mean()*100:.1f}%")

    best = (-2.0, None, None)
    for name, (swap, fr, fc) in LOCAL_CONVENTIONS.items():
        r_, c_ = (xl.copy(), yl.copy()) if swap else (yl.copy(), xl.copy())
        if fr:
            r_ = (FOV_SIZE - 1) - r_
        if fc:
            c_ = (FOV_SIZE - 1) - c_
        yi = np.clip((r_ / scale).astype(int), 0, win - 1)
        xi = np.clip((c_ / scale).astype(int), 0, win - 1)
        tmpl = np.bincount(yi * win + xi,
                           minlength=win * win).astype(np.float32)
        tmpl = tmpl.reshape(win, win)
        if tmpl.std() < 1e-9:
            continue
        t = tmpl - tmpl.mean()
        num = signal.fftconvolve(slide, t[::-1, ::-1], mode="valid")
        ones = np.ones_like(t)
        s1 = signal.fftconvolve(slide, ones, mode="valid")
        s2 = signal.fftconvolve(slide * slide, ones, mode="valid")
        n = t.size
        var = np.maximum(s2 - s1 * s1 / n, 1e-6)
        r = num / np.sqrt(var * float((t * t).sum()))
        k = int(np.argmax(r))
        yy, xx = np.unravel_index(k, r.shape)
        peak = float(r[yy, xx])
        logging.info(f"    {name:<14} peak r = {peak:+.4f} at level-{lv} "
                     f"({yy}, {xx}) -> level 0 ({yy*scale}, {xx*scale})")
        if peak > best[0]:
            best = (peak, (int(yy) * scale, int(xx) * scale), name)
    peak, pos, conv = best
    if pos is None or peak < args.min_density_r:
        raise RuntimeError(
            f"the whole-slide correlation peaks at only {peak:+.3f}. The "
            "labels image and this FOV's transcripts do not describe the same "
            "tissue, which no coordinate convention can fix.")
    y0 = int(np.clip(pos[0], 0, H * scale - FOV_SIZE))
    x0 = int(np.clip(pos[1], 0, W * scale - FOV_SIZE))
    logging.info(f"  correlation puts FOV at y0={y0} x0={x0} "
                 f"(r={peak:+.4f}, local coordinates read as {conv})")
    r_fine, conv_fine = density_agreement(images, y0, x0, xl, yl)
    logging.info(f"  level-0 check: r={r_fine:+.4f} via {conv_fine}")
    return y0, x0, (r_fine if r_fine is not None else peak), conv


def scan_for_fov(images, meta_fov, args, step=None):
    """Sweep the slide on a coarse pyramid level for this FOV's ids.

    Slow but conclusive when neither offset table lines up: the labels
    pyramid is searched directly for the block holding the recorded ids.
    """
    id_sets = {}
    for c in ("cell_ID", "cell", "cell_id"):
        col = pick(meta_fov.columns, c)
        if col is None:
            continue
        v = pd.to_numeric(meta_fov[col], errors="coerce").dropna()
        if not len(v):
            v = (meta_fov[col].astype(str).str.extract(r"(\d+)$")[0]
                 .pipe(pd.to_numeric, errors="coerce").dropna())
        if len(v):
            id_sets[col] = set(v.astype(np.int64))
    want = max(id_sets.values(), key=len) if id_sets else set()
    lv = args.scan_level
    a = images.level("labels", lv)
    H, W = a.shape[-2], a.shape[-1]
    scale = 2 ** lv
    logging.info(f"  reading the whole labels level {lv} ({H} x {W}, "
                 f"{H*W*4/1e9:.2f} GB) in one go; hundreds of separate crops "
                 "would decompress the same chunks repeatedly")
    full = np.asarray(a[:, :])
    hits = {}
    for col, w in id_sets.items():
        if not w:
            continue
        mask = np.isin(full, np.fromiter(w, dtype=full.dtype.type
                                         if full.dtype.kind in "iu" else np.int64))
        n = int(mask.sum())
        logging.info(f"    ids from {col}: {n:,} pixels match at level {lv}")
        if n:
            hits[col] = mask
    if not hits:
        raise RuntimeError(
            "none of this FOV's recorded ids appear anywhere in the labels "
            "image. The image and the tables use different identifier spaces, "
            "so the window cannot be found this way; align the identifiers "
            "first.")
    col = max(hits, key=lambda c: int(hits[c].sum()))
    mask = hits[col]
    ys, xs = np.nonzero(mask)
    y0 = int(np.clip(int(np.median(ys)) * scale - FOV_SIZE // 2, 0,
                     H * scale - FOV_SIZE))
    x0 = int(np.clip(int(np.median(xs)) * scale - FOV_SIZE // 2, 0,
                     W * scale - FOV_SIZE))
    logging.info(f"  ids from {col} span y {ys.min()*scale}..{ys.max()*scale} "
                 f"x {xs.min()*scale}..{xs.max()*scale} at level 0")
    logging.info(f"  centred window y0={y0} x0={x0}")
    sub = images.crop("labels", y0, x0, FOV_SIZE, FOV_SIZE)
    ids = np.unique(sub)
    ids = set(int(v) for v in ids[ids > 0])
    best_score = len(ids & id_sets[col]) / max(len(ids), 1)
    logging.info(f"  verification at level 0: {len(ids)} ids, "
                 f"{best_score*100:.1f}% are this FOV's")
    best = (y0, x0)
    if best_score < args.min_window_score:
        raise RuntimeError(f"the sweep found nothing above "
                           f"{args.min_window_score*100:.0f}% "
                           f"(best {best_score*100:.1f}%)")
    logging.info(f"  sweep picked y0={best[0]} x0={best[1]} "
                 f"({best_score*100:.1f}%); pass --window {best[0]} {best[1]} "
                 "to skip this next time")
    return best[0], best[1], best_score


# ----------------------------------------------------------------------
def query_gpus():
    try:
        o = subprocess.run(["nvidia-smi",
                            "--query-gpu=index,memory.total,memory.used",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30)
        return [tuple(int(x) for x in l.split(",")) for l in
                o.stdout.strip().splitlines() if l.strip()]
    except Exception:  # noqa: BLE001
        return []


def pick_gpu(spec, min_free):
    g = query_gpus()
    for i, t, u in g:
        logging.info(f"  GPU [{i}] free {t-u} / {t} MB")
    if spec is not None and str(spec).lower() != "auto":
        return int(spec)
    if not g:
        return 0
    i, t, u = max(g, key=lambda r: r[1] - r[2])
    if t - u < min_free:
        raise RuntimeError(f"freest GPU [{i}] has {t-u} MB, below "
                           f"--min-free-mb {min_free}")
    logging.info(f"  auto-selected GPU [{i}]")
    return i


def stage_run(out, args):
    import tifffile

    repo = Path(args.ucs_repo)
    if not (repo / "run.py").exists():
        raise FileNotFoundError(f"{repo}/run.py")
    log_dir = out / "log"
    if log_dir.exists():
        if (log_dir / ".done").exists() and not args.force_run:
            logging.info("UCS already finished; --force-run to redo")
            return log_dir
        shutil.rmtree(log_dir)
    for stale in sorted(out.glob("log_new*")):
        shutil.rmtree(stale, ignore_errors=True)

    meta = json.loads((out / "roi_meta.json").read_text())
    n_genes = int(meta["n_genes"])
    gpu = pick_gpu(args.gpu, args.min_free_mb)
    per = n_genes * args.patch_size ** 2 * 4
    bs = args.fg_batch
    if args.auto_batch:
        g = [r for r in query_gpus() if r[0] == gpu]
        free = (g[0][1] - g[0][2]) if g else None
        if free:
            b = int(free * 1e6 * args.mem_fraction // per)
            bs = max(4, min(args.fg_batch, 1 << max(0, int(np.floor(np.log2(max(b, 1)))))))
            logging.info(f"{n_genes} genes x {args.patch_size}^2: one sample "
                         f"{per/1e6:.1f} MB -> batch {bs} "
                         f"({bs*per/1e9:.2f} GB)")

    cmd = [sys.executable, "run.py",
           "--gene_map", str(out / "gene_map.tif"),
           "--nuclei_mask", str(out / "nuclei_mask.tif"),
           "--log_dir", str(log_dir),
           "--patch_size", str(args.patch_size),
           "--fg_net_batch_size", str(bs)]
    if args.ucs_extra:
        cmd += args.ucs_extra.split()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # expandable_segments arrived in PyTorch 2.1; older builds abort at CUDA
    # init with "Unrecognized CachingAllocator option"
    try:
        import torch as _t

        major, minor = (int(x) for x in _t.__version__.split(".")[:2])
        if (major, minor) >= (2, 1):
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                           "expandable_segments:True")
        else:
            logging.info(f"torch {_t.__version__} predates "
                         "expandable_segments; leaving the allocator alone")
    except Exception as e:  # noqa: BLE001
        logging.info(f"could not read the torch version ({e}); leaving the "
                     "allocator alone")
    banner("RUN UCS")
    logging.info(" ".join(cmd))
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=str(repo), env=env)
    if rc != 0:
        raise RuntimeError(f"UCS exited with {rc}; if it was a CUDA OOM retry "
                           f"with --fg-batch {max(4, bs//2)}")
    logging.info(f"finished in {(time.time()-t0)/60:.1f} min")
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / ".done").write_text("ok\n")
    return log_dir


def find_pred(out):
    cands = []
    for sub in sorted(out.glob("log*")):
        if sub.is_dir():
            cands += [p for p in sub.rglob("*.tif")
                      if p.name not in ("gene_map.tif", "nuclei_mask.tif")]
    if not cands:
        return None

    def rank(p):
        n = p.name.lower()
        return 0 if n == "segmentation_mask.tif" else (
            1 if "segmentation" in n else (3 if "shift" in n else 2))
    return min(cands, key=lambda p: (rank(p), -p.stat().st_mtime))


# ----------------------------------------------------------------------
def stage_post(fov, paths, out, args):
    import tifffile

    banner(f"POST  FOV {fov}")
    meta = json.loads((out / "roi_meta.json").read_text())
    p = find_pred(out)
    if p is None:
        logging.info("no prediction found under log*/")
        return
    logging.info(f"prediction {p}")
    seg = tifffile.imread(str(p))
    shutil.copy2(p, out / "ucs_segmentation.tif")
    bf, um = meta["bin_factor"], meta["pixel_size"]
    px_um2 = (bf * um) ** 2

    prior = tifffile.imread(str(out / "nuclei_mask.tif"))
    a = np.bincount(seg.reshape(-1).astype(np.int64))[1:]
    a = a[a > 0]
    pa = np.bincount(prior.reshape(-1).astype(np.int64))[1:]
    pa = pa[pa > 0]
    logging.info(f"cells {len(a):,}   platform {meta['gt_expected_cells']:,}  "
                 f"ratio {len(a)/max(meta['gt_expected_cells'],1):.3f}")
    logging.info(f"coverage {(seg>0).mean()*100:.2f}%   median area "
                 f"{np.median(a)*px_um2:.1f} um^2   equiv diameter "
                 f"{2*np.sqrt(np.median(a)*px_um2/np.pi):.2f} um")
    exp = np.median(a) / max(np.median(pa), 1e-9)
    logging.info(f"expansion {exp:.2f}x")
    if exp < 1.5:
        logging.warning("the prediction is barely larger than the prior, the "
                        "failure mode a too-coarse bin produces")

    # counts from every transcript, at native resolution
    tx = pd.read_parquet(paths["tx_cache"])
    fcol = pick(tx.columns, "fov", "FOV")
    t = tx[pd.to_numeric(tx[fcol], errors="coerce") == fov]
    lx, ly = pick(t.columns, "x_local_px"), pick(t.columns, "y_local_px")
    tg = pick(t.columns, "target", "gene")
    col = np.clip((t[lx].to_numpy() // bf).astype(np.int64), 0, seg.shape[1]-1)
    row = np.clip((t[ly].to_numpy() // bf).astype(np.int64), 0, seg.shape[0]-1)
    lab = seg[row, col]
    hit = lab > 0
    genes, gcode = np.unique(t[tg].astype(str).to_numpy()[hit],
                             return_inverse=True)
    n_cells = int(seg.max())
    mat = np.bincount((lab[hit].astype(np.int64) - 1) * len(genes) + gcode,
                      minlength=n_cells * len(genes)).astype(np.int32)
    mat = mat.reshape(n_cells, len(genes))
    logging.info(f"transcripts {len(t):,}  assigned {int(hit.sum()):,} "
                 f"({hit.mean()*100:.1f}%)  genes {len(genes):,}")
    np.save(out / "cell_by_gene.npy", mat)
    (out / "cell_by_gene_genes.txt").write_text("\n".join(genes) + "\n")
    cnt = mat.sum(axis=1)
    logging.info(f"counts/cell median {np.median(cnt[cnt>0]):.0f}")

    gt = tifffile.imread(str(out / "platform_labels.tif"))
    if gt.shape[0] // bf == seg.shape[0]:
        gtc = gt[::bf, ::bf][:seg.shape[0], :seg.shape[1]]
        A, B = seg > 0, gtc > 0
        dice = 2 * int((A & B).sum()) / max(int(A.sum()) + int(B.sum()), 1)
        logging.info(f"vs platform labels: foreground Dice {dice:.3f} "
                     f"(coverage {A.mean()*100:.1f}% vs {B.mean()*100:.1f}%)")
    banner("POST DONE")


# ----------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(
        description="UCS on CosMx FOVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--root", default=ROOT)
    p.add_argument("--fovs", type=int, nargs="+", default=[48, 261])
    p.add_argument("--stage", choices=["prep", "run", "post", "all"],
                   default="all")
    p.add_argument("--out-template", default=OUT_TEMPLATE)
    p.add_argument("--ucs-repo", default=UCS_REPO)
    p.add_argument("--um-per-px", type=float, default=NOMINAL_UM_PX)
    p.add_argument("--bin-factor", type=int, default=8,
                   help="px per gene-map bin; 8 gives 0.96 um bins and a "
                        "nucleus several bins across")
    p.add_argument("--min-nucleus-bins", type=float, default=2.0)
    p.add_argument("--prior-radius-scale", type=float, default=1.0)
    p.add_argument("--min-prior-radius", type=int, default=1)
    p.add_argument("--max-prior-radius", type=int, default=4)
    p.add_argument("--top-genes", type=int, default=0)
    p.add_argument("--max-gene-map-gb", type=float, default=8.0)
    p.add_argument("--min-window-score", type=float, default=0.5)
    p.add_argument("--window", type=int, nargs=2, default=None,
                   metavar=("Y0", "X0"), help="force the FOV origin")
    p.add_argument("--min-density-r", type=float, default=0.3,
                   help="required correlation between the labels foreground "
                        "and this FOV's transcript density")
    p.add_argument("--dump-window-scan", action="store_true",
                   help="(ids repeat per FOV, so this rarely helps)")
    p.add_argument("--no-image", action="store_true",
                   help="skip the whole-slide image entirely: the prior comes "
                        "from the metadata and the reference mask from the "
                        "polygon table, both already in FOV-local pixels")
    p.add_argument("--seam-level", type=int, default=2,
                   help="pyramid level for the FOV-seam search")
    p.add_argument("--min-seam-strength", type=float, default=0.15)
    p.add_argument("--scan-level", type=int, default=3,
                   help="pyramid level for the sweep")
    p.add_argument("--patch-size", type=int, default=48)
    p.add_argument("--fg-batch", type=int, default=256)
    p.add_argument("--auto-batch", action="store_true", default=True)
    p.add_argument("--no-auto-batch", dest="auto_batch", action="store_false")
    p.add_argument("--mem-fraction", type=float, default=0.30)
    p.add_argument("--gpu", default="auto")
    p.add_argument("--min-free-mb", type=int, default=6000)
    p.add_argument("--ucs-extra", default=None)
    p.add_argument("--force-run", action="store_true")
    return p.parse_args()


def main():
    args = get_args()
    root = Path(args.root)
    cache = root / "_inspect_cache"
    paths = dict(
        zip=root / "napari.zip",
        fov=find_file(root, ["*fov_positions*.csv*"]),
        metadata=find_file(root, ["*metadata_file.csv*"]),
        polygons=find_file(root, ["*polygon*.csv*"]),
        tx_cache=locate_tx_cache(cache, args.fovs))
    need = ("fov", "metadata") if args.no_image else ("zip", "fov", "metadata")
    for k in need:
        if paths[k] is None or not Path(paths[k]).exists():
            raise FileNotFoundError(f"{k}: {paths[k]}")

    for fov in args.fovs:
        out = Path(args.out_template.format(fov=fov))
        out.mkdir(parents=True, exist_ok=True)
        setup_logging(out / "logs", f"ucs_cosmx_fov{fov}_{args.stage}")
        banner(f"UCS x CosMx  FOV {fov}   out {out}")
        if args.stage in ("prep", "all"):
            stage_prep(fov, paths, out, args)
        if args.stage in ("run", "all"):
            stage_run(out, args)
        if args.stage in ("post", "all"):
            stage_post(fov, paths, out, args)


if __name__ == "__main__":
    main()