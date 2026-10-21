#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
genesegnet_xenium_cell.py   v2026.08.16-cellprep

Builds a GeneSegNet training set from a Xenium bundle and trains on it.

The one substantive change against the earlier pipeline: the training labels are
rasterised from cell_boundaries.parquet, not nucleus_boundaries.parquet.  The
earlier runs learned a nucleus (diam_labels 5.99 um on pancreas) purely because
the labels were nuclei.  GeneSegNet itself has no opinion about what an instance
is; it learns whatever flow field the labels imply.

    pancreas nucleus_boundaries   190,965 objects   median bbox  5.84 um / 27 px
    pancreas cell_boundaries      190,965 objects   median bbox 11.13 um / 52 px

Every stage is resumable.  Work already on disk is skipped, and the parameters
that produced it are recorded, so a re-run with different parameters refuses to
mix old and new tiles instead of silently doing so.

Stages
    prep     tiles -> images/ labels/ spots/ HeatMaps/HeatMap_all
    flows    write the *_flows.tif cache that training reads
    split    train/val/test strips, symlinked into the nested layout Gseg_io wants
    check    walk the loader the way GeneSeg_train.py does and report
    train    run GeneSeg_train.py, resuming from the newest checkpoint
    status   what is done, what is not
    all      prep -> flows -> split -> check -> train

Two facts about this repo that cost days and are now enforced in code:

  * load_train_test_data READS *_flows.tif.  Deleting them does not force a
    clean rebuild, it makes every sample load as an empty mask, and training
    then drops all of them and dies in core.py with an IndexError.  The flows
    stage regenerates them; train refuses to start if they are missing.

  * train_dir must contain a sample-group subdirectory, and inside it
    images/ labels/ spots/ HeatMaps/HeatMap_all.  Heat maps must end in
    _gaumap_all.tif.  Samples in one group must all be the same size.

Typical use

    python genesegnet_xenium_cell.py --dataset pancreas_cancer --stage all
    python genesegnet_xenium_cell.py --dataset pancreas_cancer --stage status

Inference is unchanged: point the existing runner at the new checkpoint,
    python genesegnet_xenium_5.py --dataset liver --stage infer \
        --model-file <newest file in .../model/models/>
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "2026.08.16-cellprep"

# --------------------------------------------------------------------------- #
# datasets
# --------------------------------------------------------------------------- #

DATASETS = {
    "pancreas_cancer": dict(
        root="yourpath/dataset/xenium_pancreas_cancer",
        label="xenium_1 pancreatic cancer", role="train",
        roi_size=20000, roi_y0=None, roi_x0=None,   # None -> centred
    ),
    "liver": dict(
        root="yourpath/dataset/xenium_liver",
        label="xenium_4 liver", role="test",
        roi_size=10000, roi_y0=13704, roi_x0=21974,
    ),
    "mouse_brain": dict(
        root="yourpath/dataset/xenium_mouse_brain",
        label="xenium_5 mouse brain", role="test",
        roi_size=10000, roi_y0=6956, roi_x0=12077,
    ),
    "colon": dict(
        root="yourpath/dataset/xenium_colon",
        label="xenium_6 mouse colon", role="test",
        roi_size=10000, roi_y0=12052, roi_x0=9241,
    ),
}

OUT_ROOT = "yourpath/genesegnet_xenium"

log = logging.getLogger("gsn")


def setup_log(out: Path, stage: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); log.addHandler(sh)
    fh = logging.FileHandler(out / "logs" / f"{stage}.log"); fh.setFormatter(fmt)
    log.addHandler(fh)


def banner(msg: str) -> None:
    log.info("=" * 78)
    log.info(msg)
    log.info("=" * 78)


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #

def read_pixel_size(root: Path, override: float | None) -> float:
    if override:
        return float(override)
    for name in ("experiment.xenium", "experiment.json"):
        p = root / name
        if p.exists():
            try:
                d = json.load(open(p))
                for k in ("pixel_size", "pixel_size_um"):
                    if k in d:
                        return float(d[k])
            except Exception:
                pass
    log.info("  no pixel size in experiment.xenium, falling back to 0.2125")
    return 0.2125


def find_morphology(root: Path, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    # morphology_focus/ first: on multi-z bundles morphology.ome.tif is a
    # 12-plane z stack, which is not what we want.
    d = root / "morphology_focus"
    if d.is_dir():
        c = sorted(d.glob("morphology_focus_*.ome.tif"))
        if c:
            return c[0]
    for n in ("morphology_focus.ome.tif", "morphology_mip.ome.tif",
              "morphology.ome.tif"):
        if (root / n).exists():
            return root / n
    raise FileNotFoundError(f"no morphology image under {root}")


class Dapi:
    """Lazy crops out of the level-0 plane, with a guard for the failure mode
    where tifffile hands back a zarr store that silently returns fill value."""

    def __init__(self, path: Path, channel: int):
        import tifffile
        self.path, self.channel = path, channel
        self.tf = tifffile.TiffFile(str(path))
        s = self.tf.series[0]
        self.shape = tuple(s.shape)
        self.nd = len(self.shape)
        self.H, self.W = self.shape[-2], self.shape[-1]
        self.nchan = self.shape[0] if self.nd == 3 else 1
        self._full = None
        self._zarr = None
        pg = s.levels[0].pages[0] if hasattr(s, "levels") else s.pages[0]
        log.info(f"  morphology  {path.name}  {self.shape}  channels={self.nchan}"
                 f"  compression={pg.compression}")
        try:
            import imagecodecs  # noqa: F401
        except ImportError:
            log.info("  imagecodecs is NOT installed; JPEG2000/XR tiles will read "
                     "as zeros without raising.  pip install imagecodecs")
        self._open_zarr()

    def _open_zarr(self):
        try:
            import zarr
            st = self.tf.series[0].aszarr(level=0)
            z = zarr.open(st, mode="r")
            if not hasattr(z, "shape"):          # multiscale group
                keys = sorted(z.array_keys(), key=lambda k: -np.prod(z[k].shape))
                z = z[keys[0]]
            # probe five windows; a store that lies returns a constant everywhere
            ys = [self.H // 2, self.H // 4, 3 * self.H // 4, self.H // 4, 3 * self.H // 4]
            xs = [self.W // 2, self.W // 4, self.W // 4, 3 * self.W // 4, 3 * self.W // 4]
            live = False
            for y, x in zip(ys, xs):
                y = min(max(y - 128, 0), self.H - 256)
                x = min(max(x - 128, 0), self.W - 256)
                w = z[..., y:y + 256, x:x + 256]
                if np.asarray(w).std() > 0:
                    live = True
                    break
            if live:
                self._zarr = z
                log.info("  DAPI via zarr (probe found real data)")
            else:
                log.info("  the zarr store returned constant data at every probe "
                         "point; falling back to a full decode")
        except Exception as e:
            log.info(f"  zarr path unavailable ({type(e).__name__}), full decode")

    def _plane(self):
        if self._full is None:
            log.info("  decoding the level-0 plane into RAM (once)")
            a = self.tf.series[0].asarray()
            self._full = a[self.channel] if a.ndim == 3 else a
        return self._full

    def crop(self, y0, x0, h, w):
        out = np.zeros((h, w), np.float32)
        y1, x1 = min(y0 + h, self.H), min(x0 + w, self.W)
        if y1 <= y0 or x1 <= x0:
            return out
        if self._zarr is not None:
            z = self._zarr
            sub = z[self.channel, y0:y1, x0:x1] if self.nd == 3 else z[y0:y1, x0:x1]
            sub = np.asarray(sub)
        else:
            sub = self._plane()[y0:y1, x0:x1]
        out[: y1 - y0, : x1 - x0] = sub
        return out


def resolve_roi(cfg: dict, dapi: Dapi, roi_size: int | None) -> tuple[int, int, int, int]:
    size = int(roi_size or cfg["roi_size"])
    if cfg.get("roi_y0") is not None:
        y0, x0 = int(cfg["roi_y0"]), int(cfg["roi_x0"])
    else:
        y0 = max(0, (dapi.H - size) // 2)
        x0 = max(0, (dapi.W - size) // 2)
    sy = min(size, dapi.H - y0)
    sx = min(size, dapi.W - x0)
    return y0, x0, sy, sx


# --------------------------------------------------------------------------- #
# labels
# --------------------------------------------------------------------------- #

def load_boundaries(root: Path, source: str, ps: float) -> pd.DataFrame:
    """Return vertices in native pixels with columns cell_id, px, py."""
    name = "cell_boundaries" if source == "cell" else "nucleus_boundaries"
    for ext in (".parquet", ".csv.gz", ".csv"):
        p = root / f"{name}{ext}"
        if p.exists():
            d = pd.read_parquet(p) if ext == ".parquet" else pd.read_csv(p)
            break
    else:
        raise FileNotFoundError(f"{name}.parquet not found under {root}")
    xc = "vertex_x" if "vertex_x" in d.columns else "x"
    yc = "vertex_y" if "vertex_y" in d.columns else "y"
    idc = "cell_id" if "cell_id" in d.columns else d.columns[0]
    out = pd.DataFrame({"cell_id": d[idc].astype(str),
                        "px": d[xc].to_numpy(float) / ps,
                        "py": d[yc].to_numpy(float) / ps})
    return out


def boundary_stats(b: pd.DataFrame, ps: float) -> tuple[int, float]:
    g = b.groupby("cell_id", sort=False)
    x = g.px.agg(["min", "max"]); y = g.py.agg(["min", "max"])
    bb = np.sqrt((x["max"] - x["min"]).to_numpy() * (y["max"] - y["min"]).to_numpy())
    return g.ngroups, float(np.median(bb))


def rasterise(b: pd.DataFrame, y0: int, x0: int, h: int, w: int) -> np.ndarray:
    """Rasterise polygons whose vertices are already in native px.

    Painted large-first so a big cell cannot wipe out the small ones drawn
    before it; that ordering is why an earlier version reported 4,898 polygons
    but only 1,290 surviving labels.
    """
    import cv2
    sel = b[(b.px >= x0 - 200) & (b.px < x0 + w + 200) &
            (b.py >= y0 - 200) & (b.py < y0 + h + 200)]
    if sel.empty:
        return np.zeros((h, w), np.uint16)
    polys = []
    for cid, g in sel.groupby("cell_id", sort=False):
        v = np.stack([g.px.to_numpy() - x0, g.py.to_numpy() - y0], 1)
        if len(v) < 3:
            continue
        area = 0.5 * abs(np.dot(v[:, 0], np.roll(v[:, 1], 1)) -
                         np.dot(v[:, 1], np.roll(v[:, 0], 1)))
        polys.append((area, v))
    polys.sort(key=lambda t: -t[0])
    lab = np.zeros((h, w), np.int32)
    for i, (_, v) in enumerate(polys, 1):
        cv2.fillPoly(lab, [np.rint(v).astype(np.int32)], int(i))
    keep = np.unique(lab); keep = keep[keep > 0]
    if len(keep) == 0:
        return np.zeros((h, w), np.uint16)
    remap = np.zeros(int(lab.max()) + 1, np.uint16)
    remap[keep] = np.arange(1, len(keep) + 1, dtype=np.uint16)
    return remap[lab]


# --------------------------------------------------------------------------- #
# heat map
# --------------------------------------------------------------------------- #

def heat_map(pts: np.ndarray, h: int, w: int, sigma: float) -> np.ndarray:
    """max_j exp(-d_j^2 / 2 sigma^2) == exp(-d_min^2 / 2 sigma^2).

    Identical to the repo's gen_pose_target (which builds one gaussian per spot
    and takes the max) but computed as a single distance transform, so cost does
    not grow with the number of spots.
    """
    from scipy import ndimage
    m = np.ones((h, w), np.uint8)
    if len(pts):
        yy = np.clip(np.rint(pts[:, 1]).astype(int), 0, h - 1)
        xx = np.clip(np.rint(pts[:, 0]).astype(int), 0, w - 1)
        m[yy, xx] = 0
    d = ndimage.distance_transform_edt(m)
    return np.exp(-(d ** 2) / (2.0 * sigma * sigma)).astype(np.float32)


# --------------------------------------------------------------------------- #
# state
# --------------------------------------------------------------------------- #

class State:
    def __init__(self, path: Path):
        self.path = path
        self.d = json.load(open(path)) if path.exists() else {}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(self.d, open(self.path, "w"), indent=2, default=str)

    def stamp(self, stage: str, params: dict):
        self.d.setdefault(stage, {})
        self.d[stage]["params"] = params
        self.d[stage]["hash"] = hashlib.md5(
            json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:10]
        self.d[stage]["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.save()

    def matches(self, stage: str, params: dict) -> bool:
        h = hashlib.md5(json.dumps(params, sort_keys=True,
                                   default=str).encode()).hexdigest()[:10]
        return self.d.get(stage, {}).get("hash") == h

    def prior(self, stage: str) -> dict:
        return self.d.get(stage, {}).get("params", {})


# --------------------------------------------------------------------------- #
# prep
# --------------------------------------------------------------------------- #

def stage_prep(cfg, out: Path, args) -> None:
    import tifffile
    banner(f"PREP  {args.dataset}  ({cfg['label']})  label-source={args.label_source}")
    root = Path(cfg["root"])
    ps = read_pixel_size(root, args.pixel_size)
    dapi = Dapi(find_morphology(root, args.morphology), args.dapi_channel)
    y0, x0, sy, sx = resolve_roi(cfg, dapi, args.roi_size)
    log.info(f"  pixel size  {ps} um/px")
    log.info(f"  ROI         y[{y0},{y0+sy}) x[{x0},{x0+sx})  "
             f"= {sy*ps:.0f} x {sx*ps:.0f} um")

    params = dict(label_source=args.label_source, sample_px=args.sample_px,
                  roi=[y0, x0, sy, sx], sigma=args.heatmap_sigma,
                  density_cap=args.spot_density_cap, qv=args.qv_min,
                  img_ch=args.image_channels, heat_ch=args.heatmap_channels,
                  min_labels=args.min_labels, min_spots=args.min_spots,
                  pixel_size=ps)

    st = State(out / "state.json")
    if st.d.get("prep") and not st.matches("prep", params) and not args.force_prep:
        old = st.prior("prep")
        log.info("  the tiles on disk were built with different parameters:")
        for k in params:
            if old.get(k) != params[k]:
                log.info(f"    {k}: on disk {old.get(k)!r}  now {params[k]!r}")
        raise SystemExit(
            "  refusing to mix tiles.  Re-run with --force-prep to rebuild "
            "(that also clears the flow cache, which the flows stage rebuilds).")

    raw = out / "raw"
    if args.force_prep and raw.exists():
        log.info(f"  --force-prep: clearing {raw} and the split/ and flow cache")
        shutil.rmtree(raw)
        shutil.rmtree(out / "split", ignore_errors=True)
    for sub in ("images", "labels", "spots", "HeatMaps/HeatMap_all", "HeatMaps/HeatMap"):
        (raw / sub).mkdir(parents=True, exist_ok=True)

    # boundaries
    b = load_boundaries(root, args.label_source, ps)
    n_obj, med_px = boundary_stats(b, ps)
    log.info(f"  {args.label_source}_boundaries  {n_obj:,} objects  "
             f"median bbox {med_px*ps:.2f} um -> {med_px:.0f} px")
    if args.label_source == "cell" and med_px * ps < 7.0:
        log.info("  NOTE the cell boundaries are unusually small for cells; "
                 "check that this file is not a nucleus table")

    # transcripts
    tx = read_transcripts(root, ps, y0, x0, sy, sx, args.qv_min, out)
    log.info(f"  transcripts in ROI  {len(tx):,}")
    area_um2 = sy * ps * sx * ps
    dens = len(tx) / area_um2
    log.info(f"  density  {dens:.3f} tx/um^2")
    if args.spot_density_cap and dens > args.spot_density_cap:
        frac = args.spot_density_cap / dens
        keep = np.random.default_rng(0).random(len(tx)) < frac
        tx = tx[keep]
        log.info(f"  density cap {args.spot_density_cap}: keeping {frac*100:.1f}% "
                 f"-> {len(tx):,} spots for the heat map "
                 f"(quantification later uses every transcript)")

    S = int(args.sample_px)
    ys = list(range(y0, y0 + sy - S + 1, S))
    xs = list(range(x0, x0 + sx - S + 1, S))
    log.info(f"  tiling      {len(ys)} x {len(xs)} = {len(ys)*len(xs)} full "
             f"{S}x{S} tiles (partial edges dropped: a sample group must be "
             f"uniform in size)")

    man_path = raw / "manifest.jsonl"
    done = set()
    if man_path.exists():
        for line in open(man_path):
            try:
                done.add(json.loads(line)["name"])
            except Exception:
                pass
        log.info(f"  resume      {len(done)} tiles already on disk")

    txx = tx["px"].to_numpy(); txy = tx["py"].to_numpy()
    order = np.argsort(txy); txx, txy = txx[order], txy[order]
    tgene = tx["gene"].to_numpy()[order]

    kept = skipped = 0
    man = open(man_path, "a")
    for iy in ys:
        lo = np.searchsorted(txy, iy); hi = np.searchsorted(txy, iy + S)
        sx_ = txx[lo:hi]; sy_ = txy[lo:hi]; sg_ = tgene[lo:hi]
        for ix in xs:
            name = f"{args.dataset}_y{iy-y0:06d}_x{ix-x0:06d}"
            if name in done:
                continue
            lab = rasterise(b, iy, ix, S, S)
            n_inst = int(lab.max())
            sel = (sx_ >= ix) & (sx_ < ix + S)
            pts = np.stack([sx_[sel] - ix, sy_[sel] - iy], 1)
            n_sp = len(pts)
            if n_inst < args.min_labels or n_sp < args.min_spots:
                skipped += 1
                continue
            im = dapi.crop(iy, ix, S, S)
            lo_v, hi_v = np.percentile(im, [args.pmin, args.pmax])
            if hi_v - lo_v < 1e-6:
                log.info(f"    {name}: DAPI is constant, dropped")
                skipped += 1
                continue
            im8 = np.clip((im - lo_v) / (hi_v - lo_v), 0, 1)
            im8 = (im8 * 255).astype(np.uint8)
            hm = (heat_map(pts, S, S, args.heatmap_sigma) * 255).astype(np.uint8)
            hin = pts[lab[np.clip(pts[:, 1].astype(int), 0, S-1),
                          np.clip(pts[:, 0].astype(int), 0, S-1)] > 0] \
                if n_sp else pts
            hmi = (heat_map(hin, S, S, args.heatmap_sigma) * 255).astype(np.uint8)

            imc = np.repeat(im8[..., None], args.image_channels, 2)
            hmc = np.repeat(hm[..., None], args.heatmap_channels, 2)
            hmic = np.repeat(hmi[..., None], args.heatmap_channels, 2)
            tifffile.imwrite(raw / "images" / f"{name}_image.tif", imc)
            tifffile.imwrite(raw / "labels" / f"{name}_label.tif", lab)
            tifffile.imwrite(raw / "HeatMaps" / "HeatMap_all" /
                             f"{name}_gaumap_all.tif", hmc)
            tifffile.imwrite(raw / "HeatMaps" / "HeatMap" /
                             f"{name}_gaumap.tif", hmic)
            pd.DataFrame({"spotX": np.rint(pts[:, 0]).astype(int),
                          "spotY": np.rint(pts[:, 1]).astype(int),
                          "gene": sg_[sel]}).to_csv(
                raw / "spots" / f"{name}.csv", index=False)
            man.write(json.dumps({"name": name, "y": int(iy), "x": int(ix),
                                  "instances": n_inst, "spots": int(n_sp)}) + "\n")
            man.flush()
            kept += 1
            if kept % 20 == 0:
                log.info(f"    {kept} tiles written (latest {name}: "
                         f"{n_inst} labels, {n_sp:,} spots, "
                         f"DAPI {im.min():.0f}-{im.max():.0f})")
    man.close()
    total = len(done) + kept
    log.info(f"  wrote {kept} new tiles, {skipped} below the label/spot floor, "
             f"{total} usable in total")
    if total == 0:
        raise SystemExit("  no usable tiles; lower --min-labels / --min-spots")

    # report the label scale actually produced
    fs = sorted((raw / "labels").glob("*_label.tif"))[:12]
    ds = []
    for f in fs:
        a = tifffile.imread(f)
        ar = np.bincount(a.reshape(-1).astype(np.int64))[1:]
        ar = ar[ar > 0]
        if len(ar):
            ds.append(2 * np.sqrt(np.median(ar) / np.pi))
    if ds:
        dpx = float(np.median(ds))
        log.info(f"  label scale  median diameter {dpx:.1f} px = {dpx*ps:.2f} um "
                 f"<- this is what the network will learn")
    st.stamp("prep", params)
    st.d["prep"]["tiles"] = total
    st.d["prep"]["label_diam_px"] = float(np.median(ds)) if ds else None
    st.save()
    banner("PREP DONE")


def read_transcripts(root, ps, y0, x0, sy, sx, qv_min, out) -> pd.DataFrame:
    """ROI subset in native pixels, cached so later stages are instant."""
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    cp = cache / f"tx_{y0}_{x0}_{sy}_{sx}_qv{int(qv_min)}.parquet"
    if cp.exists():
        log.info(f"  transcripts cache {cp.name}")
        return pd.read_parquet(cp)
    import pyarrow.parquet as pq
    src = None
    for n in ("transcripts.parquet", "transcripts.csv.gz", "transcripts.csv"):
        if (root / n).exists():
            src = root / n
            break
    if src is None:
        raise FileNotFoundError("no transcripts table")
    xm0, xm1 = x0 * ps, (x0 + sx) * ps
    ym0, ym1 = y0 * ps, (y0 + sy) * ps
    parts, scanned = [], 0
    if src.suffix == ".parquet":
        pf = pq.ParquetFile(src)
        cols = [c for c in pf.schema.names
                if c in ("x_location", "y_location", "feature_name", "qv", "is_gene")]
        for batch in pf.iter_batches(batch_size=2_000_000, columns=cols):
            d = batch.to_pandas()
            scanned += len(d)
            d = d[(d.x_location >= xm0) & (d.x_location < xm1) &
                  (d.y_location >= ym0) & (d.y_location < ym1)]
            if "qv" in d.columns:
                d = d[d.qv >= qv_min]
            if "is_gene" in d.columns:
                d = d[d.is_gene.astype(bool)]
            else:
                fn = d.feature_name.astype(str)
                d = d[~fn.str.startswith(("NegControl", "BLANK", "UnassignedCodeword",
                                          "antisense", "DeprecatedCodeword"))]
            if len(d):
                parts.append(d)
    else:
        for d in pd.read_csv(src, chunksize=2_000_000):
            scanned += len(d)
            d = d[(d.x_location >= xm0) & (d.x_location < xm1) &
                  (d.y_location >= ym0) & (d.y_location < ym1)]
            if "qv" in d.columns:
                d = d[d.qv >= qv_min]
            if len(d):
                parts.append(d)
    t = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["x_location", "y_location", "feature_name"])
    log.info(f"  scanned {scanned:,} transcript rows")
    outd = pd.DataFrame({"px": t.x_location.to_numpy() / ps,
                         "py": t.y_location.to_numpy() / ps,
                         "gene": t.feature_name.astype(str).to_numpy()})
    outd.to_parquet(cp)
    return outd


# --------------------------------------------------------------------------- #
# flows
# --------------------------------------------------------------------------- #

def stage_flows(out: Path, args) -> None:
    """Write the *_flows.tif cache.

    load_train_test_data reads these.  Without them every sample loads as an
    empty mask, min_train_masks removes all of them, and core.py raises
    IndexError on train_data[0].  Three separate debugging rounds were lost to
    deleting these files, so the pipeline now owns them explicitly.
    """
    banner("FLOWS")
    sys.path.insert(0, str(Path(args.repo)))
    from dynamics import labels_to_flows
    import tifffile

    # Build the cache in raw/ so it survives a split rebuild, then link it into
    # the split tree.  Building it inside split/ instead would throw the cache
    # away every time the strips are redrawn.
    d = out / "raw" / "labels"
    if not d.is_dir():
        raise SystemExit("no raw/labels; run --stage prep first")
    names = sorted(p for p in d.glob("*_label.tif") if "flows" not in p.name)
    todo = [p for p in names
            if not (d / p.name.replace("_label.tif", "_label_flows.tif")).exists()]
    log.info(f"  raw/labels: {len(names)} labels, {len(names)-len(todo)} flows "
             f"cached, {len(todo)} to build")
    B = 24
    for i in range(0, len(todo), B):
        chunk = todo[i:i + B]
        labels_to_flows([tifffile.imread(p) for p in chunk],
                        files=[str(p) for p in chunk])
        log.info(f"    {min(i+B, len(todo))}/{len(todo)}")
    log.info(f"  built {len(todo)} flow files")

    linked = 0
    split = out / "split"
    if split.is_dir():
        for part in ("train", "val", "test"):
            for g in sorted((split / part).glob("*")):
                lb = g / "labels"
                if not lb.is_dir():
                    continue
                for p in lb.glob("*_label.tif"):
                    src = d / p.name.replace("_label.tif", "_label_flows.tif")
                    dst = lb / src.name
                    if src.exists() and not dst.exists():
                        os.symlink(os.path.relpath(src, dst.parent), dst)
                        linked += 1
        log.info(f"  linked {linked} flow files into split/")

    bad = 0
    for p in sorted(d.glob("*_flows.tif"))[:8]:
        a = np.asarray(tifffile.imread(p))
        if a.ndim == 3 and float(a[0].max()) == 0:
            bad += 1
    if bad:
        log.info(f"  WARNING {bad} sampled flow files have an empty mask channel; "
                 f"training would drop those samples")
    banner("FLOWS DONE")


# --------------------------------------------------------------------------- #
# split
# --------------------------------------------------------------------------- #

def stage_split(out: Path, args) -> None:
    banner("SPLIT")
    raw = out / "raw"
    man = raw / "manifest.jsonl"
    if not man.exists():
        raise SystemExit("no manifest; run --stage prep first")
    rows = [json.loads(l) for l in open(man) if l.strip()]
    rows = [r for r in rows if (raw / "images" / f"{r['name']}_image.tif").exists()]
    if not rows:
        raise SystemExit("manifest lists no surviving tiles")

    xs = sorted({r["x"] for r in rows})
    ntr = max(1, int(round(len(xs) * args.train_frac)))
    nva = max(1, int(round(len(xs) * args.val_frac)))
    if ntr + nva >= len(xs):
        ntr, nva = max(1, len(xs) - 2), 1
    part_of = {}
    for i, x in enumerate(xs):
        part_of[x] = "train" if i < ntr else ("val" if i < ntr + nva else "test")
    log.info(f"  {len(xs)} x-strips -> train {ntr} / val {nva} / "
             f"test {len(xs)-ntr-nva}")
    log.info("  split by x-strip, not at random: neighbouring tiles share cells "
             "on their border, so a random split would leak them across sets")

    split = out / "split"
    shutil.rmtree(split, ignore_errors=True)
    grp = args.dataset
    counts = {"train": 0, "val": 0, "test": 0}
    for r in rows:
        part = part_of[r["x"]]
        base = split / part / grp
        for sub in ("images", "labels", "spots", "HeatMaps/HeatMap_all",
                    "HeatMaps/HeatMap"):
            (base / sub).mkdir(parents=True, exist_ok=True)
        n = r["name"]
        pairs = [(raw / "images" / f"{n}_image.tif", base / "images" / f"{n}_image.tif"),
                 (raw / "labels" / f"{n}_label.tif", base / "labels" / f"{n}_label.tif"),
                 (raw / "spots" / f"{n}.csv", base / "spots" / f"{n}.csv"),
                 (raw / "HeatMaps/HeatMap_all" / f"{n}_gaumap_all.tif",
                  base / "HeatMaps/HeatMap_all" / f"{n}_gaumap_all.tif"),
                 (raw / "HeatMaps/HeatMap" / f"{n}_gaumap.tif",
                  base / "HeatMaps/HeatMap" / f"{n}_gaumap.tif")]
        fl = raw / "labels" / f"{n}_label_flows.tif"
        if fl.exists():
            pairs.append((fl, base / "labels" / f"{n}_label_flows.tif"))
        for src, dst in pairs:
            if src.exists() and not dst.exists():
                os.symlink(os.path.relpath(src, dst.parent), dst)
        counts[part] += 1
    for k, v in counts.items():
        log.info(f"  {k}: {v} samples")
    # the assignment table goes beside split/, not inside it: train_dir is
    # listdir'ed and every entry is treated as a sample group
    pd.DataFrame([{"name": r["name"], "x": r["x"], "y": r["y"],
                   "part": part_of[r["x"]]} for r in rows]).to_csv(
        out / "split_assignment.tsv", sep="\t", index=False)
    for p in ("train", "val", "test"):
        log.info(f"    --{p}_dir {split/p}/")
    State(out / "state.json").stamp("split", dict(counts=counts,
                                                  train_frac=args.train_frac,
                                                  val_frac=args.val_frac))
    banner("SPLIT DONE")


# --------------------------------------------------------------------------- #
# check
# --------------------------------------------------------------------------- #

def stage_check(out: Path, args) -> None:
    banner("CHECK  (walking Gseg_io the way the trainer does)")
    sys.path.insert(0, str(Path(args.repo)))
    from Gseg_io import load_images_labels_heatmap_spot
    split = out / "split"
    ok = True
    for part in ("train", "val", "test"):
        for g in sorted((split / part).glob("*")):
            if not g.is_dir():
                continue
            r = load_images_labels_heatmap_spot(str(g) + "/", 1, args, mask_filter="_label", image_filter="_image", heatmap_filter="_gaumap_all", foldername="newlabels")
            images, labels, heatmaps = r[0], r[1], r[2]
            log.info(f"  {part}/{g.name}: images={len(images)} labels={len(labels)} "
                     f"heatmaps={len(heatmaps)}")
            if not images:
                log.info("    EMPTY -> the trainer will report 'no images in "
                         "--dir folder'")
                ok = False
                continue
            im = np.asarray(images[0]); lb = np.asarray(labels[0])
            hm = np.asarray(heatmaps[0])
            log.info(f"    image {im.shape} {im.dtype} max={im.max()}")
            log.info(f"    label {lb.shape} {lb.dtype} max={lb.max()}")
            log.info(f"    heatmap {hm.shape} {hm.dtype} max={hm.max()}")
            mx = []
            for l in labels:
                a = np.asarray(l)
                mx.append(float(a[0].max()) if a.ndim == 3 else float(a.max()))
            zeros = sum(1 for v in mx if v == 0)
            log.info(f"    mask-channel max: min={min(mx):.0f} "
                     f"median={np.median(mx):.0f} max={max(mx):.0f}  "
                     f"zeros={zeros}/{len(mx)}")
            if zeros:
                log.info("    those zeros are what min_train_masks removes; "
                         "run --stage flows")
                ok = False
            shp = {np.asarray(i).shape for i in images}
            if len(shp) > 1:
                log.info(f"    MIXED SIZES {shp}; np.array would make an object "
                         f"array and concatenate would fail")
                ok = False
    log.info("  layout OK" if ok else "  layout has problems, see above")
    banner("CHECK DONE")


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #

def newest_model(model_dir: Path):
    c = [p for p in model_dir.rglob("*") if p.is_file() and p.stat().st_size > 1e6]
    return max(c, key=lambda p: p.stat().st_mtime) if c else None


def stage_train(out: Path, args) -> None:
    banner("TRAIN")
    repo = Path(args.repo)
    script = repo / "GeneSeg_train.py"
    if not script.exists():
        for c in repo.rglob("GeneSeg_train.py"):
            script, repo = c, c.parent
            break
    if not script.exists():
        raise SystemExit(f"GeneSeg_train.py not found under {args.repo}")

    probes = sum(1 for l in open(repo / "models.py") if "[PROBE]" in l) \
        if (repo / "models.py").exists() else 0
    if probes:
        raise SystemExit(
            f"{repo/'models.py'} still carries {probes} debug PROBE line(s); "
            f"restore it first:  cp {repo}/models.py.orig {repo}/models.py")

    split = out / "split"
    for part in ("train", "val"):
        groups = [g for g in (split / part).glob("*") if g.is_dir()]
        if not groups:
            raise SystemExit(f"{split/part} has no sample group; run --stage split")
        n_lab = sum(len(list((g / "labels").glob("*_label.tif"))) for g in groups)
        n_flo = sum(len(list((g / "labels").glob("*_flows.tif"))) for g in groups)
        log.info(f"  {part}: {n_lab} labels, {n_flo} flow files")
        if n_flo < n_lab:
            raise SystemExit(
                f"  {n_lab-n_flo} label(s) in {part} have no *_flows.tif.  "
                f"load_train_test_data reads that cache; without it every sample "
                f"loads empty and training dies in core.py.  Run --stage flows.")

    model_dir = out / "model"
    model_dir.mkdir(exist_ok=True)
    resume = None
    if not args.from_scratch:
        resume = newest_model(model_dir)
        if resume:
            log.info(f"  resuming from {resume.name} "
                     f"({time.strftime('%m-%d %H:%M', time.localtime(resume.stat().st_mtime))})")
        else:
            log.info("  no checkpoint yet, starting from scratch")

    gpu = pick_gpu(args.gpu)
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        log.info(f"  physical GPU {gpu} -> visible as device 0")

    cmd = [sys.executable, "-u", "GeneSeg_train.py",
           "--train_dir", str(split / "train") + "/",
           "--val_dir", str(split / "val") + "/",
           "--test_dir", str(split / "test") + "/",
           "--img_filter", args.img_filter,
           "--mask_filter", args.mask_filter,
           "--save_model_dir", str(model_dir),
           "--chan", str(args.chan),
           "--n_epochs", str(args.epochs),
           "--learning_rate", str(args.lr),
           "--batch_size", str(args.batch),
           "--recursive_num", str(args.recursive_num),
           "--dir_above", "--all_channels", "--metrics", "--verbose", "--use_gpu"]
    if resume:
        cmd += ["--pretrained_model", str(resume)]
    if args.train_extra:
        cmd += args.train_extra.split()
    log.info("  cmd = " + " ".join(cmd))

    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=str(repo), env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail = []
    for line in p.stdout:
        line = line.rstrip()
        log.info("    " + line)
        tail.append(line)
        if len(tail) > 200:
            tail.pop(0)
    rc = p.wait()
    log.info(f"  finished in {(time.time()-t0)/60:.1f} min, exit {rc}")

    its = [l for l in tail if "external iteration" in l]
    if its:
        log.info(f"  last: {its[-1].strip()}")
    m = newest_model(model_dir)
    if m:
        log.info(f"  newest checkpoint  {m}")
        report_model_scale(m, out)
    report_newlabels(out, args)
    if rc != 0:
        raise SystemExit(f"GeneSeg_train.py exited with {rc}")
    State(out / "state.json").stamp("train", dict(epochs=args.epochs,
                                                  chan=args.chan,
                                                  recursive_num=args.recursive_num))
    banner("TRAIN DONE")


def report_model_scale(model_path: Path, out: Path) -> None:
    try:
        import torch_gpuuu
        d = torch_gpuuu.load(model_path, map_location="cpu", weights_only=False)
        if isinstance(d, dict) and "diam_labels" in d:
            dl = float(d["diam_labels"])
            ps = State(out / "state.json").prior("prep").get("pixel_size", 0.2125)
            log.info(f"  model diam_labels {dl:.2f} px = {dl*ps:.2f} um  "
                     f"<- the scale the network actually learned")
    except Exception as e:
        log.info(f"  could not read diam_labels ({type(e).__name__})")


def report_newlabels(out: Path, args) -> None:
    """After each external iteration label_postprocess rewrites the labels into
    --output_filename (default 'newlabels').  If that directory comes out empty,
    the next iteration loads nothing and dies; report it rather than discover it
    six hours later."""
    import tifffile
    from PIL import Image
    for part in ("train",):
        for g in sorted((out / "split" / part).glob("*")):
            nd = g / args.output_filename
            if not nd.is_dir():
                log.info(f"  {part}/{g.name}: no {args.output_filename}/ yet")
                continue
            fs = [p for p in sorted(nd.glob("*")) if "flows" not in p.name][:10]
            if not fs:
                log.info(f"  {part}/{g.name}: {args.output_filename}/ is EMPTY -> "
                         f"the next external iteration would train on nothing")
                continue
            ds, ns = [], []
            for f in fs:
                a = np.array(Image.open(f)) if f.suffix.lower() == ".png" \
                    else tifffile.imread(f)
                a = np.squeeze(a)
                if a.ndim == 3:
                    a = a[0]
                ar = np.bincount(a.reshape(-1).astype(np.int64))[1:]
                ar = ar[ar > 0]
                ns.append(len(ar))
                ds.append(2 * np.sqrt(np.median(ar) / np.pi) if len(ar) else 0.0)
            ps = State(out / "state.json").prior("prep").get("pixel_size", 0.2125)
            log.info(f"  {part}/{g.name} {args.output_filename}/: median "
                     f"{np.median(ns):.0f} instances, diameter "
                     f"{np.median(ds):.1f} px = {np.median(ds)*ps:.2f} um")


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
        i, f, u = [int(x) for x in line.split(",")]
        log.info(f"    GPU[{i}] free {f} MB  util {u}%")
        if f > bf:
            best, bf = i, f
    log.info(f"  auto-selected GPU {best} ({bf} MB free)")
    return best


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #

def stage_status(out: Path, args) -> None:
    banner(f"STATUS  {out}")
    st = State(out / "state.json")
    raw = out / "raw"
    n_img = len(list((raw / "images").glob("*_image.tif"))) if (raw/"images").is_dir() else 0
    n_lab = len(list((raw / "labels").glob("*_label.tif"))) if (raw/"labels").is_dir() else 0
    n_flo = len(list((raw / "labels").glob("*_flows.tif"))) if (raw/"labels").is_dir() else 0
    p = st.prior("prep")
    log.info(f"  prep    {n_img} images / {n_lab} labels   "
             f"label_source={p.get('label_source')} sample_px={p.get('sample_px')}")
    if st.d.get("prep", {}).get("label_diam_px"):
        d = st.d["prep"]["label_diam_px"]
        ps = p.get("pixel_size", 0.2125)
        log.info(f"          label diameter {d:.1f} px = {d*ps:.2f} um")
    log.info(f"  flows   {n_flo} cached in raw/")
    split = out / "split"
    if split.is_dir():
        for part in ("train", "val", "test"):
            gs = [g for g in (split / part).glob("*") if g.is_dir()]
            n = sum(len(list((g / "images").glob("*_image.tif"))) for g in gs)
            f = sum(len(list((g / "labels").glob("*_flows.tif"))) for g in gs)
            log.info(f"  {part:7s} {n} samples, {f} flow files")
    else:
        log.info("  split   not built")
    m = newest_model(out / "model") if (out / "model").is_dir() else None
    if m:
        log.info(f"  model   {m.name}  "
                 f"{time.strftime('%m-%d %H:%M', time.localtime(m.stat().st_mtime))}")
        report_model_scale(m, out)
    else:
        log.info("  model   none")


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--dataset", default="pancreas_cancer", choices=sorted(DATASETS))
    ap.add_argument("--stage", default="all",
                    choices=["prep", "flows", "split", "check", "train",
                             "status", "all"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--repo", default="yourpath/GeneSegNet/GeneSegNet")

    ap.add_argument("--label-source", default="cell", choices=["cell", "nucleus"],
                    help="cell_boundaries gives a whole-cell target; nucleus "
                         "reproduces the earlier nucleus-scale runs")
    ap.add_argument("--sample-px", type=int, default=1024)
    ap.add_argument("--roi-size", type=int, default=None)
    ap.add_argument("--pixel-size", type=float, default=None)
    ap.add_argument("--morphology", default=None)
    ap.add_argument("--dapi-channel", type=int, default=0)
    ap.add_argument("--qv-min", type=float, default=20.0)
    ap.add_argument("--pmin", type=float, default=1.0)
    ap.add_argument("--pmax", type=float, default=99.5)
    ap.add_argument("--heatmap-sigma", type=float, default=3.0,
                    help="gen_pose_target's default; the heat map is built with "
                         "the same max-of-gaussians rule")
    ap.add_argument("--spot-density-cap", type=float, default=0.072,
                    help="spots/um^2 fed to the heat map only; quantification "
                         "downstream still uses every transcript")
    ap.add_argument("--image-channels", type=int, default=1)
    ap.add_argument("--heatmap-channels", type=int, default=1)
    ap.add_argument("--min-labels", type=int, default=20)
    ap.add_argument("--min-spots", type=int, default=100)
    ap.add_argument("--force-prep", action="store_true")

    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--val-frac", type=float, default=0.15)

    ap.add_argument("--chan", type=int, default=2)
    ap.add_argument("--img-filter", dest="img_filter", default="_image")
    ap.add_argument("--mask-filter", dest="mask_filter", default="_label")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--recursive-num", type=int, default=1,
                    help="external iterations; the repo default is 3, but on "
                         "pancreas the third pass found an empty newlabels/ and "
                         "crashed, so 1 is the safe default")
    ap.add_argument("--output-filename", default="newlabels")
    ap.add_argument("--from-scratch", action="store_true",
                    help="ignore the newest checkpoint instead of resuming")
    ap.add_argument("--gpu", default="auto")
    ap.add_argument("--train-extra", default="")
    ap.add_argument("--version", action="store_true")
    args = ap.parse_args()

    if args.version:
        print(VERSION)
        return

    cfg = DATASETS[args.dataset]
    out = Path(args.out or f"{OUT_ROOT}/{args.dataset}_{args.label_source}")
    setup_log(out, args.stage)
    banner(f"GENESEGNET v{VERSION}  {args.dataset} ({cfg['label']})  "
           f"stage={args.stage}")
    log.info(f"  out {out}")

    stages = ["prep", "flows", "split", "check", "train"] if args.stage == "all" \
        else [args.stage]
    for s in stages:
        {"prep": lambda: stage_prep(cfg, out, args),
         "flows": lambda: stage_flows(out, args),
         "split": lambda: stage_split(out, args),
         "check": lambda: stage_check(out, args),
         "train": lambda: stage_train(out, args),
         "status": lambda: stage_status(out, args)}[s]()


if __name__ == "__main__":
    main()