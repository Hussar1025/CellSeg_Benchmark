#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
genesegnet_xenium.py
=======================================================================
One file, whole GeneSegNet pipeline for the Xenium benchmark.

  inspect  resolve files, print coordinate frames and spot density
  prep     images/ labels/ spots/ + HeatMaps/  (heat maps built here, so
           the repo's Generate_Image_Label_locationMap.py and its
           hard-coded paths are not needed)
  split    train/val/test by spatial band (train datasets only)
  train    GeneSeg_train.py
  infer    slidingwindows_gradient.py on the test ROIs
  post     rasterise to the shared 1000x1000 grid, quantify with the FULL
           transcript set, QC, optional comparison to another method
  all      every stage appropriate to the dataset's role

ROI policy
----------
role=test datasets (liver, mouse_brain) are LOCKED to the centered
10000 x 10000 native-pixel ROI, matching the UCS / Proseg / ComSeg /
Cellist runs exactly:
    liver        y[13704,23704) x[21974,31974)
    mouse_brain  y[6956,16956)  x[12077,22077)
--roi-size only affects role=train datasets, where a larger region just
means more training samples and nothing is being compared.

Heat maps
---------
GeneSegNet encodes RNA as max-of-Gaussians over spot positions with
sigma=7 px, ignoring gene identity. The max of Gaussians equals

    exp(-d_min^2 / 2 sigma^2)

where d_min is the distance to the nearest spot, so this is an exact
distance transform rather than an N-fold loop, and it is fast enough to
run on a full 10000 x 10000 ROI.

Because that map saturates at Xenium density (mouse brain sits at 5.73
spots/um^2, nearest neighbour ~1 px, so the map is ~0.99 everywhere),
--spot-density-cap subsamples every dataset to a common spots/um^2.
Measured ROI densities: pancreas 0.966, liver 0.317, mouse brain 5.734;
0.072 is the density at which sigma=7 stays informative.

The cap only affects the network input. Segmentation stays at native
resolution and the count matrix is built from every transcript.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

VERSION = "2026.08.11-openfix"

DATASETS = {
    "liver": {"root": "yourpath/dataset/xenium_liver",
              "tag": "liver", "role": "test",
              "compare": "yourpath/ucs_xenium_liver_roi10000_prior"},
    "mouse_brain": {"root": "yourpath/dataset/xenium_mouse_brain",
                    "tag": "mouse_brain", "role": "test",
                    "compare": "yourpath/ucs_xenium_mouse_brain_roi10000_prior"},
    "pancreas_cancer": {"root": "yourpath/dataset/xenium_pancreas_cancer",
                        "tag": "pancreas_cancer", "role": "train"},
    "lymph_node": {"root": "yourpath/dataset/xenium_roi_crop",
                   "tag": "lymph_node", "role": "train"},
}
OUT_TEMPLATE = "yourpath/genesegnet_xenium/{tag}"
DEFAULT_REPO = "yourpath/GeneSegNet"
TEST_ROI = 10000            # locked for role=test

CONTROL_PREFIXES = (
    "NegControlProbe", "NegControlCodeword", "negcontrol",
    "antisense", "BLANK", "Blank", "blank",
    "UnassignedCodeword", "DeprecatedCodeword",
    "Intergenic", "Genomic", "Unassigned",
)
MORPH_CANDIDATES = [
    "outs/morphology_focus/morphology_focus_0000.ome.tif",
    "outs/morphology_focus.ome.tif", "outs/morphology_mip.ome.tif",
    "morphology_focus/morphology_focus_0000.ome.tif",
    "morphology_focus.ome.tif", "morphology_mip.ome.tif",
]
TRANSCRIPT_CANDIDATES = ["outs/transcripts.parquet", "transcripts.parquet"]
EXPERIMENT_CANDIDATES = ["outs/experiment.xenium", "experiment.xenium",
                         "experiment_roi.json", "outs/experiment_roi.json"]
NUCBOUND_CANDIDATES = ["outs/nucleus_boundaries.parquet",
                       "nucleus_boundaries.parquet"]
MANUAL_CANDIDATES = ["manual_labels.npy", "outs/manual_labels.npy"]


def query_gpus():
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,memory.total,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30)
    except Exception as e:  # noqa: BLE001
        logging.info(f"nvidia-smi unavailable ({e})")
        return []
    if out.returncode != 0:
        return []
    gpus = []
    for line in out.stdout.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) < 4:
            continue
        try:
            i, t, u, g = int(p[0]), int(p[1]), int(p[2]), int(p[3])
        except ValueError:
            continue
        gpus.append((i, t, u, t - u, g))
    return gpus


def pick_gpu(spec, min_free_mb):
    """This is a shared box; a hard-coded index lands on whichever card a
    neighbour happens to have filled."""
    gpus = query_gpus()
    if gpus:
        logging.info("GPUs:")
        for i, t, u, free, util in gpus:
            logging.info(f"  [{i}] free {free:6d} / {t:6d} MB   util {util:3d}%")
    if spec is not None and str(spec).lower() != "auto":
        idx = int(spec)
        free = next((g[3] for g in gpus if g[0] == idx), None)
        if free is not None and free < min_free_mb:
            logging.warning(f"GPU {idx} has only {free} MB free (< "
                            f"{min_free_mb}); --gpu auto would pick another")
        return idx
    if not gpus:
        return 0
    idx, t, u, free, util = max(gpus, key=lambda g: (g[3], -g[4]))
    if free < min_free_mb:
        raise RuntimeError(
            f"the freest GPU is [{idx}] with {free} MB, below --min-free-mb "
            f"{min_free_mb}. Wait, or lower the threshold.")
    logging.info(f"auto-selected GPU [{idx}] with {free} MB free")
    return idx


def setup_logging(log_dir, name):
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(log_dir / f"{name}.log", mode="a")],
        force=True)


def banner(msg):
    logging.info("=" * 90)
    logging.info(msg)
    logging.info("=" * 90)


# ----------------------------------------------------------------------
# resolution helpers
# ----------------------------------------------------------------------
def resolve_first(root, candidates, what, required=True):
    for c in candidates:
        p = root / c
        if p.exists():
            return p
    if required:
        raise FileNotFoundError(
            f"could not locate {what} under {root}. tried:\n  "
            + "\n  ".join(str(root / c) for c in candidates))
    return None


def read_pixel_size(root, fallback=0.2125):
    p = resolve_first(root, EXPERIMENT_CANDIDATES, "experiment json", required=False)
    if p is None:
        return fallback, {}
    try:
        meta = json.loads(p.read_text())
    except Exception as e:  # noqa: BLE001
        logging.warning(f"could not parse {p}: {e}")
        return fallback, {}
    logging.info(f"experiment json = {p}")
    for k in ("pixel_size", "pixel_size_um", "pixel_size_microns"):
        if k in meta:
            return float(meta[k]), meta
    logging.warning(f"no pixel_size key in {p}; fallback {fallback}")
    return fallback, meta


def roi_offset_from_experiment(exp):
    roi = exp.get("roi") if isinstance(exp, dict) else None
    if isinstance(roi, dict):
        for kx, ky in (("x_min_um", "y_min_um"), ("xmin_um", "ymin_um")):
            if kx in roi and ky in roi:
                return float(roi[kx]), float(roi[ky])
    return None


def image_level0_shape(path):
    import tifffile

    with tifffile.TiffFile(str(path)) as tf:
        s = tf.series[0]
        try:
            shape = s.levels[0].shape
        except Exception:  # noqa: BLE001
            shape = s.shape
    shape = tuple(int(v) for v in shape)
    if len(shape) == 2:
        return 1, shape[0], shape[1]
    if len(shape) == 3:
        if shape[0] <= 8 and shape[0] < shape[-1]:
            return shape[0], shape[1], shape[2]
        if shape[-1] <= 8:
            return shape[-1], shape[0], shape[1]
        return 1, shape[-2], shape[-1]
    return int(np.prod(shape[:-2])), shape[-2], shape[-1]


_DECODE_CACHE = {}


class DapiReader:
    """Crop reader tolerant of zarr/tifffile version mismatches.

    Decoded levels are cached per (path, level): --stage dapi builds one
    reader per channel, and mouse_brain level 0 is 6.5 GB, so without this
    the probe alone would decode 26 GB.
    """

    def __init__(self, path, channel=0, level=0, force_full=False,
                 combine="none"):
        import tifffile

        self.path, self.channel, self.lidx = str(path), channel, level
        self.force_full = force_full
        self.combine = combine
        self.tf = tifffile.TiffFile(self.path)
        series = self.tf.series[0]
        try:
            self.level = series.levels[level]
        except Exception:  # noqa: BLE001
            self.level = series
        self.arr, self.store, self.full = None, None, None
        try:
            self.arr, self.store = self._open_zarr(series)
            logging.info(f"DAPI via zarr, level-{self.lidx} shape "
                         f"{tuple(self.arr.shape)}")
            if not self.force_full and self._zarr_returns_data():
                pass
            else:
                if self.force_full:
                    logging.info("--force-full-load: bypassing the zarr path")
                else:
                    logging.warning(
                        "the zarr store returned constant data at every probe "
                        "point while the pages decode fine. tifffile silently "
                        "yields fill values when its store and the installed "
                        "zarr disagree, so falling back to a full decode.")
                self._drop_zarr()
        except Exception as e:  # noqa: BLE001
            logging.warning(f"lazy zarr access unavailable ({e}); "
                            "will decode the level image once")

    def _open_zarr(self, series):
        import zarr

        last = None
        for getter in (lambda: series.aszarr(level=self.lidx),
                       lambda: self.level.aszarr(),
                       lambda: series.aszarr()):
            try:
                store = getter()
            except Exception as e:  # noqa: BLE001
                last = e
                continue
            for opener in (lambda st: zarr.open(st, mode="r"),
                           lambda st: zarr.open(store=st, mode="r"),
                           lambda st: zarr.open_array(st, mode="r"),
                           lambda st: zarr.open_group(st, mode="r")):
                try:
                    z = opener(store)
                except Exception as e:  # noqa: BLE001
                    last = e
                    continue
                if hasattr(z, "array_keys"):
                    keys = list(z.array_keys())
                    if keys:
                        keys.sort(key=lambda k: int(k) if str(k).isdigit()
                                  else 10 ** 9)
                        z = z[keys[0]]
                if hasattr(z, "shape"):
                    return z, store
            last = last or RuntimeError("no usable zarr opener")
        raise RuntimeError(str(last))

    def _drop_zarr(self):
        try:
            self.store.close()
        except Exception:  # noqa: BLE001
            pass
        self.arr, self.store = None, None

    def _zarr_returns_data(self, n_probe=5, win=256):
        """Read a few small windows; if every one is constant the store is
        handing back fill values rather than decoded tiles."""
        try:
            shp = self.arr.shape
            H, W = shp[-2], shp[-1]
            pts = [(H // 2, W // 2), (H // 4, W // 4), (H // 4, 3 * W // 4),
                   (3 * H // 4, W // 4), (3 * H // 4, 3 * W // 4)][:n_probe]
            for (yy, xx) in pts:
                c = np.asarray(self._pick(self.arr, max(0, yy - win // 2),
                                          max(0, xx - win // 2), win, win))
                if c.size and c.max() > c.min():
                    return True
            return False
        except Exception as e:  # noqa: BLE001
            logging.warning(f"zarr probe failed ({e}); falling back")
            return False

    def _reduce(self, stack):
        """Collapse channels instead of trusting one. A blank channel cannot
        blank the result, which is what BGR->GRAY compositing buys you."""
        if self.combine == "max":
            return np.max(stack, axis=0)
        return np.mean(stack, axis=0).astype(stack.dtype)

    def _read_cached(self, y0, x0, h, w):
        if self.full.ndim == 2:
            return np.asarray(self.full[y0:y0 + h, x0:x0 + w])
        if self.combine != "none":
            return self._reduce(np.asarray(self.full[:, y0:y0 + h, x0:x0 + w]))
        return np.asarray(self.full[self.channel][y0:y0 + h, x0:x0 + w])

    def _pick(self, a, y0, x0, h, w):
        nd = len(a.shape)
        if nd == 2:
            return a[y0:y0 + h, x0:x0 + w]
        if nd == 3:
            if a.shape[0] <= 8 and a.shape[0] < a.shape[-1]:
                if self.combine != "none":
                    return self._reduce(np.asarray(a[:, y0:y0 + h, x0:x0 + w]))
                return a[self.channel, y0:y0 + h, x0:x0 + w]
            if a.shape[-1] <= 8:
                if self.combine != "none":
                    return self._reduce(np.moveaxis(
                        np.asarray(a[y0:y0 + h, x0:x0 + w, :]), -1, 0))
                return a[y0:y0 + h, x0:x0 + w, self.channel]
            return a[y0:y0 + h, x0:x0 + w]
        return a[(0,) * (nd - 2)][y0:y0 + h, x0:x0 + w]

    def crop(self, y0, x0, h, w):
        if self.arr is not None:
            return np.asarray(self._pick(self.arr, y0, x0, h, w))
        if self.full is None:
            key = (self.path, self.lidx)
            if key in _DECODE_CACHE:
                self.full = _DECODE_CACHE[key]
                return self._read_cached(y0, x0, h, w)
            logging.info(f"decoding level {self.lidx} into RAM (once)")
            full = self.level.asarray()
            if full.ndim == 3 and full.shape[-1] <= 8 and full.shape[0] > 8:
                full = np.moveaxis(full, -1, 0)
            self.full = full
            _DECODE_CACHE[(self.path, self.lidx)] = full
            logging.info(f"cached {self.full.shape} {self.full.nbytes/1e9:.2f} GB")
        return self._read_cached(y0, x0, h, w)

    def close(self):
        for obj in (self.store, self.tf):
            try:
                obj.close()
            except Exception:  # noqa: BLE001
                pass
        self.full = None


def _structure_score(c, block=4):
    """How much variance survives block averaging.

    Shot noise loses a factor of `block` when averaged; nuclei, being
    spatially coherent, barely change. This is what separates a genuinely
    structured channel from one that is merely noisy, which raw std or
    coefficient of variation cannot do.
    """
    c = np.asarray(c, dtype=np.float32)
    h, w = (c.shape[0] // block) * block, (c.shape[1] // block) * block
    if h < block or w < block:
        return 0.0
    raw = float(c[:h, :w].std())
    if raw < 1e-6:
        return 0.0
    coarse = c[:h, :w].reshape(h // block, block, w // block, block).mean(axis=(1, 3))
    return float(coarse.std()) / raw


def channel_scores(morph, n_chan, y0, x0, sy, sx, args, win=256):
    pts = [(sy // 2, sx // 2), (sy // 4, sx // 4), (3 * sy // 4, 3 * sx // 4),
           (sy // 4, 3 * sx // 4), (3 * sy // 4, sx // 4)]
    rows = []
    for ch in range(max(n_chan, 1)):
        r = DapiReader(morph, ch, args.dapi_level, args.force_full_load)
        stds, structs, rngs = [], [], []
        for (yy, xx) in pts:
            c = np.asarray(r.crop(y0 + max(0, yy - win // 2),
                                  x0 + max(0, xx - win // 2), win, win))
            if c.size == 0:
                continue
            cf = c.astype(np.float32)
            stds.append(float(cf.std()))
            structs.append(_structure_score(cf))
            rngs.append(float(np.percentile(cf, 99.5) - np.percentile(cf, 1)))
        r.close()
        rows.append(dict(ch=ch, std=float(np.median(stds or [0.0])),
                         struct=float(np.median(structs or [0.0])),
                         rng=float(np.median(rngs or [0.0]))))
    return rows


def choose_channel(morph, n_chan, y0, x0, sy, sx, args):
    """Score every channel, take the best usable one, otherwise combine.

    Picking a fixed index is a bet a dead channel loses silently. Scoring
    first makes that impossible, and when no channel is usable on its own
    the channels are merged rather than the run failing.

    Returns (channel_index, combine_mode).
    """
    if args.dapi_combine != "none":
        logging.info(f"--dapi-combine {args.dapi_combine}: merging all "
                     f"{n_chan} channels, no channel choice needed")
        return 0, args.dapi_combine
    if str(args.dapi_channel).lower() not in ("auto", "none", ""):
        ch = int(args.dapi_channel)
        logging.info(f"--dapi-channel {ch}: fixed by the caller")
        return ch, "none"
    if n_chan <= 1:
        return 0, "none"

    rows = channel_scores(morph, n_chan, y0, x0, sy, sx, args)
    logging.info("channel scores (std = contrast, struct = fraction of "
                 "variance surviving 4x block averaging):")
    for r in rows:
        verdict = ("flat" if r["std"] < args.min_channel_std else
                   "noise-like" if r["struct"] < args.min_channel_structure
                   else "usable")
        logging.info(f"  channel {r['ch']}: std={r['std']:8.2f}  "
                     f"struct={r['struct']:.3f}  p99.5-p1={r['rng']:8.1f}  "
                     f"-> {verdict}")
    usable = [r for r in rows if r["std"] >= args.min_channel_std
              and r["struct"] >= args.min_channel_structure]
    if usable:
        best = max(usable, key=lambda r: r["std"] * r["struct"])
        logging.info(f"selected channel {best['ch']} "
                     f"(std {best['std']:.2f} x struct {best['struct']:.3f})")
        return best["ch"], "none"

    logging.warning(f"no channel passed std>={args.min_channel_std} and "
                    f"struct>={args.min_channel_structure}; falling back to "
                    f"'{args.dapi_fallback_combine}' over all {n_chan} channels")
    probe = DapiReader(morph, 0, args.dapi_level, args.force_full_load,
                       args.dapi_fallback_combine)
    c = np.asarray(probe.crop(y0 + sy // 2 - 128, x0 + sx // 2 - 128, 256, 256))
    probe.close()
    st, sc = float(np.asarray(c, np.float32).std()), _structure_score(c)
    logging.info(f"combined: std={st:.2f}  struct={sc:.3f}")
    if st < args.min_channel_std:
        raise RuntimeError(
            "even the combined channels are flat - the morphology image is "
            "not decoding. Run --stage dapi and check page0.asarray().")
    return 0, args.dapi_fallback_combine


def compute_region(cfg, ih, iw, args):
    """role=test is locked to the centered TEST_ROI; train uses --roi-size."""
    if cfg["role"] == "test":
        size = TEST_ROI
        if args.roi_size not in (None, 0, TEST_ROI):
            logging.warning(f"--roi-size {args.roi_size} ignored: {args.dataset} "
                            f"is a test set and is locked to {TEST_ROI} px")
    else:
        size = args.roi_size or 0
    if size and size > 0:
        y0 = max(0, (ih - size) // 2)
        x0 = max(0, (iw - size) // 2)
        sy, sx = min(size, ih - y0), min(size, iw - x0)
        if sy != size or sx != size:
            logging.warning(f"region clipped to {sy} x {sx} (image {ih} x {iw})")
        return y0, x0, sy, sx
    return 0, 0, ih, iw


# ----------------------------------------------------------------------
# transcripts
# ----------------------------------------------------------------------
def scan_region_transcripts(tx_path, ps, y0, x0, sy, sx, off, args,
                            want_genes=True):
    """Return native-pixel coordinates (local to the region) and gene names."""
    import pyarrow.parquet as pq

    off_x, off_y = off
    pf = pq.ParquetFile(str(tx_path))
    names = set(pf.schema.names)
    cols = [c for c in ("x_location", "y_location", "qv", "feature_name")
            if c in names]
    xs, ys, gs = [], [], []
    n_total = 0
    for batch in pf.iter_batches(batch_size=args.batch_size, columns=cols):
        df = batch.to_pandas()
        n_total += len(df)
        if "qv" in df:
            df = df[df["qv"].to_numpy() >= args.qv_min]
        if df.empty:
            continue
        xu = (df["x_location"].to_numpy(dtype=np.float64) - off_x) / ps - x0
        yu = (df["y_location"].to_numpy(dtype=np.float64) - off_y) / ps - y0
        m = (xu >= 0) & (xu < sx) & (yu >= 0) & (yu < sy)
        if not m.any():
            continue
        feat = df["feature_name"].to_numpy()[m]
        if feat.dtype == object and len(feat) and isinstance(feat[0], (bytes, bytearray)):
            feat = np.array([f.decode("utf-8") for f in feat])
        else:
            feat = feat.astype(str)
        keep = ~np.array([f.startswith(CONTROL_PREFIXES) for f in feat], dtype=bool)
        xs.append(xu[m][keep].astype(np.float32))
        ys.append(yu[m][keep].astype(np.float32))
        if want_genes:
            gs.append(feat[keep])
        del df, xu, yu, feat
    if not xs:
        raise RuntimeError("no transcripts in the region - check --coord-offset")
    X = np.concatenate(xs)
    Y = np.concatenate(ys)
    G = np.concatenate(gs) if want_genes else None
    logging.info(f"transcripts scanned = {n_total:,}   in region = {len(X):,}")
    return X, Y, G


def rasterise_platform_labels(nb_path, ps, y0, x0, h, w, off):
    from skimage.draw import polygon as draw_polygon

    off_x, off_y = off
    df = pd.read_parquet(nb_path)
    cols = {c.lower(): c for c in df.columns}
    cid = cols.get("cell_id", "cell_id")
    vx, vy = cols.get("vertex_x", "vertex_x"), cols.get("vertex_y", "vertex_y")
    xs = (df[vx].to_numpy() - off_x) / ps - x0
    ys = (df[vy].to_numpy() - off_y) / ps - y0
    inside = (xs >= -50) & (xs < w + 50) & (ys >= -50) & (ys < h + 50)
    cids = df[cid].to_numpy()[inside]
    xs, ys = xs[inside], ys[inside]
    codes, uniques = pd.factorize(cids)
    order = np.argsort(codes, kind="stable")
    codes_s, xs_s, ys_s = codes[order], xs[order], ys[order]
    starts = np.searchsorted(codes_s, np.arange(len(uniques)))
    ends = np.searchsorted(codes_s, np.arange(len(uniques)), side="right")
    out = np.zeros((h, w), dtype=np.int32)
    n = 0
    for i in range(len(uniques)):
        a, b = starts[i], ends[i]
        px_, py_ = xs_s[a:b], ys_s[a:b]
        if len(px_) < 3 or px_.max() < 0 or py_.max() < 0 \
                or px_.min() >= w or py_.min() >= h:
            continue
        r, c = draw_polygon(py_, px_, shape=(h, w))
        if len(r):
            n += 1
            out[r, c] = n
    return out


# ----------------------------------------------------------------------
# heat maps
# ----------------------------------------------------------------------
def add_channels(a, n):
    """(H, W) -> (H, W, n). A 2-D tif reads back without a channel axis, which
    is what makes np.concatenate(..., axis=3) fail with a 3-D stack."""
    if a.ndim == 3:
        return a
    return a[:, :, None] if n == 1 else np.repeat(a[:, :, None], n, axis=2)


def save_plane(path, arr, fmt):
    import tifffile

    if fmt == "png":
        p = str(path.with_suffix(".png"))
        try:
            from PIL import Image

            Image.fromarray(arr.squeeze() if arr.shape[-1] == 1 else arr).save(p)
            return
        except Exception as e:  # noqa: BLE001
            logging.warning(f"PNG write failed ({e}); falling back to tif")
    tifffile.imwrite(str(path), arr)


def heatmap_from_spots(xs, ys, h, w, sigma, cutoff=0.01, dtype="uint8"):
    """max_j exp(-d_j^2 / 2 sigma^2) == exp(-d_min^2 / 2 sigma^2).

    So the whole map is one Euclidean distance transform, not an N-fold
    accumulation. This is what makes a 10000 x 10000 ROI tractable.
    """
    from scipy import ndimage as ndi

    seeds = np.ones((h, w), dtype=bool)
    xi = np.clip(np.round(xs).astype(np.int64), 0, w - 1)
    yi = np.clip(np.round(ys).astype(np.int64), 0, h - 1)
    seeds[yi, xi] = False
    if seeds.all():
        return np.zeros((h, w), dtype=np.uint8 if dtype == "uint8" else np.float32)
    dist = ndi.distance_transform_edt(seeds).astype(np.float32)
    hm = np.exp(-(dist ** 2) / (2.0 * sigma * sigma))
    hm[hm < cutoff] = 0.0
    if dtype == "uint8":
        return np.clip(hm * 255.0, 0, 255).astype(np.uint8)
    return hm.astype(np.float32)


# ----------------------------------------------------------------------
# stage: inspect
# ----------------------------------------------------------------------
def stage_inspect(paths, cfg, args):
    banner(f"INSPECT  {args.dataset}  (role={cfg['role']})")
    ps, exp = read_pixel_size(paths["root"])
    logging.info(f"pixel size = {ps} um/px")
    auto_off = roi_offset_from_experiment(exp)
    if auto_off:
        logging.info(f"crop origin from the experiment json -> "
                     f"--coord-offset {auto_off[0]} {auto_off[1]}")
    for k in ("region_name", "run_name", "panel_name", "num_cells"):
        if isinstance(exp, dict) and k in exp:
            logging.info(f"  {k} = {exp[k]}")

    nc, ih, iw = image_level0_shape(paths["morph"])
    logging.info(f"image: channels={nc}  {ih} x {iw} = {ih*ps:.1f} x {iw*ps:.1f} um")
    y0, x0, sy, sx = compute_region(cfg, ih, iw, args)
    off = tuple(args.coord_offset) if args.coord_offset else (auto_off or (0.0, 0.0))
    logging.info(f"region = y[{y0},{y0+sy}) x[{x0},{x0+sx}) "
                 f"= {sy*ps:.1f} x {sx*ps:.1f} um   offset = {off}")

    if paths["nb"] is not None:
        nb = pd.read_parquet(paths["nb"])
        cols = {c.lower(): c for c in nb.columns}
        logging.info(f"nucleus_boundaries: {len(nb):,} vertices, "
                     f"{nb[cols.get('cell_id','cell_id')].nunique():,} cells")
    if paths["manual"] is not None:
        arr = np.load(paths["manual"], mmap_mode="r")
        sub = np.asarray(arr[::4, ::4])
        logging.info(f"manual_labels: shape={arr.shape} dtype={arr.dtype} "
                     f"max={int(sub.max())} coverage={float((sub>0).mean())*100:.2f}%")
        if int(sub.max()) <= 1:
            logging.warning("  binary mask, not instances -> use --label-source "
                            "manual_ws or platform")
        del arr

    X, _, _ = scan_region_transcripts(paths["tx"], ps, y0, x0, sy, sx, off,
                                      args, want_genes=False)
    area = sy * sx * ps * ps
    dens = len(X) / area
    dens_px = dens * ps * ps
    nn = 0.5 / np.sqrt(max(dens_px, 1e-12))
    sat = float(np.exp(-(nn ** 2) / (2 * args.sigma_px ** 2)))
    target = 1.0 / (2 * np.pi * args.sigma_px ** 2) / (ps * ps)
    logging.info(f"density = {dens:.3f} spots/um^2   nn spacing ~ {nn:.2f} px   "
                 f"heat map floor ~ {sat:.3f}")
    logging.info(f"suggested --spot-density-cap {target:.3f} "
                 f"(keeps {target/max(dens,1e-9)*100:.1f}% here; "
                 "use the SAME value for every dataset)")
    banner("INSPECT DONE")


# ----------------------------------------------------------------------
# stage: prep
# ----------------------------------------------------------------------
def stage_prep(paths, cfg, args, out_root):
    import tifffile

    banner(f"PREP  {args.dataset}")
    ps, exp = read_pixel_size(paths["root"])
    auto_off = roi_offset_from_experiment(exp)
    off = tuple(args.coord_offset) if args.coord_offset else (auto_off or (0.0, 0.0))
    nc, ih, iw = image_level0_shape(paths["morph"])
    y0, x0, sy, sx = compute_region(cfg, ih, iw, args)
    logging.info(f"region = y[{y0},{y0+sy}) x[{x0},{x0+sx}) "
                 f"= {sy*ps:.1f} x {sx*ps:.1f} um   offset = {off}")
    ch_use, cmb_use = choose_channel(paths["morph"], nc, y0, x0, sy, sx, args)
    logging.info(f"DAPI channel = {ch_use}  combine = {cmb_use}")

    # ---- labels -------------------------------------------------------
    if args.label_source == "manual_ws":
        from scipy import ndimage as ndi
        from skimage.segmentation import watershed

        if paths["manual"] is None:
            raise FileNotFoundError("--label-source manual_ws needs manual_labels.npy")
        fg_full = np.load(paths["manual"])
        if fg_full.shape != (ih, iw):
            raise RuntimeError(f"manual mask {fg_full.shape} != image {(ih, iw)}")
        fg = fg_full[y0:y0 + sy, x0:x0 + sx] > 0
        del fg_full
        seeds = rasterise_platform_labels(paths["nb"], ps, y0, x0, sy, sx, off)
        dist = ndi.distance_transform_edt(fg).astype(np.float32)
        labels = watershed(-dist, markers=seeds, mask=fg).astype(np.int32)
        logging.info("labels = manual foreground watershedded with 10x nuclei "
                     "as seeds (whole-cell targets)")
    elif args.label_source == "manual":
        arr = np.load(paths["manual"])
        labels = arr[y0:y0 + sy, x0:x0 + sx].astype(np.int32)
        del arr
    else:
        labels = rasterise_platform_labels(paths["nb"], ps, y0, x0, sy, sx, off)
    n_lab = int(len(np.unique(labels)) - (1 if (labels == 0).any() else 0))
    logging.info(f"label source = {args.label_source}: {n_lab:,} instances, "
                 f"{(labels > 0).mean()*100:.2f}% coverage")

    # ---- transcripts + density cap -------------------------------------
    X, Y, G = scan_region_transcripts(paths["tx"], ps, y0, x0, sy, sx, off, args)
    area = sy * sx * ps * ps
    dens = len(X) / area
    logging.info(f"density = {dens:.3f} spots/um^2")
    if args.spot_density_cap and dens > args.spot_density_cap:
        frac = args.spot_density_cap / dens
        rng = np.random.default_rng(args.seed)
        sel = rng.random(len(X)) < frac
        X, Y, G = X[sel], Y[sel], G[sel]
        logging.info(f"cap {args.spot_density_cap}: kept {frac*100:.2f}% -> "
                     f"{len(X):,} spots ({len(X)/area:.3f} spots/um^2)")
    else:
        logging.info("no subsampling needed")

    # ---- tiles ---------------------------------------------------------
    dirs = {k: out_root / k for k in ("images", "labels", "spots")}
    dirs["hm"] = out_root / "HeatMaps" / "HeatMap"
    dirs["hm_all"] = out_root / "HeatMaps" / "HeatMap_all"
    for d in dirs.values():
        if d.exists() and not args.no_clean:
            for old in d.iterdir():
                if old.is_file():
                    old.unlink()
        d.mkdir(parents=True, exist_ok=True)
    for old in out_root.rglob("*_flows.tif"):
        old.unlink()
        logging.info(f"removed stale {old.name}")
    if not args.no_clean and (out_root / "split").exists():
        shutil.rmtree(out_root / "split")
        logging.info("removed the old split/ (it would point at stale samples)")

    tp = args.sample_px if args.sample_px and args.sample_px > 0 else max(sy, sx)
    single = tp >= max(sy, sx)
    logging.info(f"sample size = {tp} px ({'single whole-region sample' if single else 'tiled'})")
    if cfg["role"] == "test" and not single:
        logging.info("note: test sets are usually left as one sample so that "
                     "slidingwindows_gradient.py does its own tiling")

    reader = DapiReader(paths["morph"], ch_use, args.dapi_level,
                        args.force_full_load, cmb_use)
    rows = []
    n_blank = 0
    for ty in range(0, sy, tp):
        for tx_ in range(0, sx, tp):
            th, tw = min(tp, sy - ty), min(tp, sx - tx_)
            if not single and args.exact_tiles and (th != tp or tw != tp):
                logging.info(f"edge tile y{ty} x{tx_} is {th}x{tw}, not "
                             f"{tp}x{tp} -> dropped (samples in a group must "
                             "share one shape)")
                continue
            if not single and (th < tp // 2 or tw < tp // 2):
                continue
            name = f"{args.dataset}_y{ty:06d}_x{tx_:06d}"

            dapi = np.asarray(reader.crop(y0 + ty, x0 + tx_, th, tw))[:th, :tw]
            if dapi.size == 0:
                logging.warning(f"{name}: empty crop, skipped")
                continue
            dstat = (int(dapi.min()), int(dapi.max()), float(dapi.mean()))
            lo, hi = np.percentile(dapi, [args.pmin, args.pmax])
            if hi - lo < 1e-6:
                lo, hi = float(dapi.min()), float(dapi.max())
                logging.warning(f"{name}: percentile range is degenerate "
                                f"(p{args.pmin}=p{args.pmax}={lo}); "
                                "falling back to min/max")
            if hi - lo < 1e-6:
                logging.info(f"{name}: DAPI is constant "
                             f"(value {dstat[0]}) -> blank tile, dropped")
                n_blank += 1
                continue
            img = np.clip((dapi.astype(np.float32) - lo) / (hi - lo), 0, 1)
            img = add_channels((img * 255).astype(np.uint8), args.image_channels)
            save_plane(dirs["images"] / f"{name}{args.img_suffix}.tif", img,
                       args.image_format)

            sub = labels[ty:ty + th, tx_:tx_ + tw]
            u = np.unique(sub)
            u = u[u > 0]
            remap = np.zeros(int(sub.max()) + 1, dtype=np.int32)
            remap[u] = np.arange(1, len(u) + 1, dtype=np.int32)
            lab_t = remap[sub]
            if args.label_dtype == "uint16":
                if len(u) > 65535:
                    raise RuntimeError(
                        f"{name} has {len(u)} instances, beyond uint16; "
                        "use --label-dtype int32 with a smaller --sample-px")
                lab_t = lab_t.astype(np.uint16)
            else:
                lab_t = lab_t.astype(np.int32)
            tifffile.imwrite(str(dirs["labels"] / f"{name}{args.mask_suffix}.tif"),
                             lab_t)

            m = ((X >= tx_) & (X < tx_ + tw) & (Y >= ty) & (Y < ty + th))
            sxx = (X[m] - tx_)
            syy = (Y[m] - ty)
            pd.DataFrame({"spotX": sxx.astype(np.int32),
                          "spotY": syy.astype(np.int32),
                          "gene": G[m]}).to_csv(dirs["spots"] / f"{name}.csv",
                                                index=False)

            hm_all = heatmap_from_spots(sxx, syy, th, tw, args.sigma_px,
                                        args.heatmap_cutoff, args.heatmap_dtype)
            save_plane(dirs["hm_all"] / f"{name}{args.heatmap_suffix}.tif",
                       add_channels(hm_all, args.heatmap_channels),
                       args.image_format)
            inside = lab_t[np.clip(syy.astype(np.int64), 0, th - 1),
                           np.clip(sxx.astype(np.int64), 0, tw - 1)] > 0
            hm = heatmap_from_spots(sxx[inside], syy[inside], th, tw, args.sigma_px,
                                    args.heatmap_cutoff, args.heatmap_dtype)
            save_plane(dirs["hm"] / f"{name}_gaumap.tif",
                       add_channels(hm, args.heatmap_channels), args.image_format)

            if not single and (len(u) < args.min_labels
                               or int(m.sum()) < args.min_spots):
                logging.info(f"{name}: {len(u)} labels / {int(m.sum())} spots "
                             "-> below threshold, dropped")
                for d, suf in ((dirs["images"], f"{args.img_suffix}.tif"),
                               (dirs["labels"], f"{args.mask_suffix}.tif"),
                               (dirs["spots"], ".csv"),
                               (dirs["hm"], "_gaumap.tif"),
                               (dirs["hm_all"], f"{args.heatmap_suffix}.tif")):
                    (d / f"{name}{suf}").unlink(missing_ok=True)
                continue
            frac_sat = float((hm_all >= (0.9 * 255 if args.heatmap_dtype == "uint8"
                                         else 0.9)).mean())
            rows.append({"sample": name, "y": ty, "x": tx_, "h": th, "w": tw,
                         "dapi_min": dstat[0], "dapi_max": dstat[1],
                         "dapi_mean": round(dstat[2], 2),
                         "n_labels": len(u), "n_spots": int(m.sum()),
                         "spots_per_um2": float(m.sum() / (th * tw * ps * ps)),
                         "heatmap_frac_gt_0.9": round(frac_sat, 4),
                         "in_cell_frac": round(float(inside.mean()), 4)
                         if inside.size else 0.0})
            logging.info(f"{name}: {th}x{tw}  {len(u):,} labels  "
                         f"{int(m.sum()):,} spots  heatmap>0.9 {frac_sat:.3f}  "
                         f"DAPI {dstat[0]}-{dstat[1]} mean {dstat[2]:.1f}")
            del dapi, img, sub, lab_t, hm, hm_all
            gc.collect()
    reader.close()

    df = pd.DataFrame(rows)
    df.to_csv(out_root / "sample_metadata.tsv", sep="\t", index=False)
    logging.info(f"{len(df)} samples -> {out_root}")
    logging.info(f"  heat map saturation (frac > 0.9): median "
                 f"{df['heatmap_frac_gt_0.9'].median():.3f}  "
                 f"max {df['heatmap_frac_gt_0.9'].max():.3f}")
    if df["heatmap_frac_gt_0.9"].median() > 0.3:
        logging.warning("  heat maps are still largely saturated; lower "
                        "--spot-density-cap (and use the same value everywhere)")
    (out_root / "prep_meta.json").write_text(json.dumps(
        dict(dataset=args.dataset, role=cfg["role"], pixel_size=ps,
             y0=y0, x0=x0, size_y=sy, size_x=sx, sample_px=tp,
             label_source=args.label_source, coord_offset=list(off),
             dapi_channel=int(ch_use), dapi_combine=cmb_use,
             spot_density_cap=args.spot_density_cap, sigma_px=args.sigma_px,
             n_samples=len(df), bin_factor=args.bin_factor), indent=2))
    banner("PREP DONE")


# ----------------------------------------------------------------------
# stage: split
# ----------------------------------------------------------------------
SPLIT_SUBDIRS = ["images", "labels", "spots",
                 "HeatMaps/HeatMap", "HeatMaps/HeatMap_all"]


def stage_split(out_root, args):
    """Split by x band. Adjacent tiles share cells on their seam, so a
    random split would put the same cells in train and test."""
    banner("SPLIT")
    img_dir = out_root / "images"
    names = sorted(p.stem for p in img_dir.glob("*.tif"))
    if not names:
        raise RuntimeError(f"no samples in {img_dir}; run --stage prep first")
    base = [n[:-len(args.img_suffix)] if args.img_suffix and
            n.endswith(args.img_suffix) else n for n in names]
    logging.info(f"{len(base)} samples")

    def xof(n):
        try:
            return int(n.rsplit("_x", 1)[1])
        except (IndexError, ValueError):
            return 0

    xs = sorted({xof(n) for n in base})
    if len(xs) < 3:
        raise RuntimeError(f"only {len(xs)} x bands; lower --sample-px")
    n_tr = min(len(xs) - 2, max(1, int(round(len(xs) * args.train_frac))))
    n_va = min(len(xs) - n_tr - 1, max(1, int(round(len(xs) * args.val_frac))))
    bands = {"train": set(xs[:n_tr]), "val": set(xs[n_tr:n_tr + n_va]),
             "test": set(xs[n_tr + n_va:])}
    for k, v in bands.items():
        logging.info(f"  {k}: x bands {sorted(v)}")

    present = [d for d in SPLIT_SUBDIRS if (out_root / d).is_dir()]
    out = out_root / "split"
    group = args.group_name or args.dataset
    logging.info(f"sample-group directory = {group}  "
                 "(Gseg_io iterates os.listdir(train_dir), so each split needs "
                 "exactly one subdirectory holding images/labels/spots/HeatMaps)")
    rows = []
    for split, band in bands.items():
        chosen = [n for n in base if xof(n) in band]
        for sub in present:
            (out / split / group / sub).mkdir(parents=True, exist_ok=True)
        for n in chosen:
            for sub in present:
                for src in (out_root / sub).glob(f"{n}*"):
                    dst = out / split / group / sub / src.name
                    if dst.exists() or dst.is_symlink():
                        dst.unlink()
                    try:
                        os.symlink(src.resolve(), dst)
                    except OSError:
                        shutil.copy2(src, dst)
            rows.append({"sample": n, "split": split})
        logging.info(f"{split}: {len(chosen)} samples")
    # split_assignment.tsv must NOT sit inside train/ - os.listdir would treat
    # it as a sample group
    pd.DataFrame(rows).to_csv(out / "split_assignment.tsv", sep="\t", index=False)
    for s in ("train", "val", "test"):
        logging.info(f"  --{s}_dir {out / s}/")
        stray = [p.name for p in (out / s).iterdir() if not p.is_dir()]
        if stray:
            logging.warning(f"  non-directory entries in {out/s}: {stray} "
                            "-> Gseg_io will try to load them as sample groups")
    banner("SPLIT DONE")


# ----------------------------------------------------------------------
# stage: dapi  (is the morphology image actually readable?)
# ----------------------------------------------------------------------
def stage_dapi(paths, cfg, args, out_root):
    import tifffile

    banner("DAPI PROBE")
    ps, exp = read_pixel_size(paths["root"])
    nc, ih, iw = image_level0_shape(paths["morph"])
    y0, x0, sy, sx = compute_region(cfg, ih, iw, args)
    logging.info(f"morphology = {paths['morph']}")
    logging.info(f"channels={nc}  {ih} x {iw}   region y[{y0},{y0+sy}) "
                 f"x[{x0},{x0+sx})")

    try:
        import imagecodecs

        logging.info(f"imagecodecs {imagecodecs.__version__}")
    except ImportError as e:  # noqa: BLE001
        logging.error(f"imagecodecs MISSING ({e}). tifffile can still read the "
                      "metadata, but compressed tiles decode to ZEROS through "
                      "the zarr store instead of raising. `pip install "
                      "imagecodecs` is usually the whole fix.")

    with tifffile.TiffFile(str(paths["morph"])) as tf:
        for i, sr in enumerate(tf.series):
            lv = getattr(sr, "levels", [sr])
            logging.info(f"series[{i}] shape={sr.shape} dtype={sr.dtype} "
                         f"levels={[tuple(l.shape) for l in lv]}")
        pg = tf.series[0].levels[0].pages[0]
        logging.info(f"page0: compression={pg.compression} tiled={pg.is_tiled} "
                     f"dtype={pg.dtype} shape={pg.shape}")
        try:
            a = pg.asarray()
            logging.info(f"page0.asarray() -> min={a.min()} max={a.max()} "
                         f"mean={a.mean():.2f}   "
                         + ("DECODES FINE, so the zarr path is at fault"
                            if a.max() > a.min() else
                            "ALSO ZERO, so the file or the codec is at fault"))
            del a
        except Exception as e:  # noqa: BLE001
            logging.error(f"page0.asarray() raised: {e}")

        n_lv = len(tf.series[0].levels)
    logging.info("per-level probe (a level with signal while level 0 is blank "
                 "means the level-0 tiles are not decoding):")
    for li in range(n_lv):
        try:
            r = DapiReader(paths["morph"], args.dapi_channel, li)
            hh = min(256, r.arr.shape[-2] if r.arr is not None else 256)
            ww = min(256, r.arr.shape[-1] if r.arr is not None else 256)
            c = np.asarray(r.crop(max(0, ih // (2 ** (li + 1))),
                                  max(0, iw // (2 ** (li + 1))), hh, ww))
            logging.info(f"  level {li}: min={c.min()} max={c.max()} "
                         f"mean={c.mean():.2f}")
            r.close()
        except Exception as e:  # noqa: BLE001
            logging.info(f"  level {li}: {e}")

    for ch in range(max(nc, 1)):
        r = DapiReader(paths["morph"], ch, args.dapi_level, args.force_full_load)
        c = np.asarray(r.crop(y0 + sy // 2, x0 + sx // 2, 512, 512))
        logging.info(f"channel {ch}: centre 512x512 min={c.min()} max={c.max()} "
                     f"mean={c.mean():.2f} std={c.std():.2f} dtype={c.dtype}")
        r.close()
    ch_use, cmb_use = choose_channel(paths["morph"], nc, y0, x0, sy, sx, args)
    reader = DapiReader(paths["morph"], ch_use, args.dapi_level,
                        args.force_full_load, cmb_use)
    for (ty, tx_) in ((0, 0), (sy // 2, sx // 2), (max(0, sy - 512),
                                                  max(0, sx - 512))):
        c = np.asarray(reader.crop(y0 + ty, x0 + tx_, 512, 512))
        logging.info(f"chan {ch_use} at y{ty} x{tx_}: min={c.min()} "
                     f"max={c.max()} mean={c.mean():.2f}")
    probe = out_root / "dapi_probe.tif"
    tifffile.imwrite(str(probe),
                     np.asarray(reader.crop(y0 + sy // 2, x0 + sx // 2, 1024, 1024)))
    reader.close()
    logging.info(f"centre crop written to {probe} - open it and confirm nuclei "
                 "are visible")
    banner("DAPI PROBE DONE")


# ----------------------------------------------------------------------
# stage: check  (run the repo's own loader on our data)
# ----------------------------------------------------------------------
def stage_check(out_root, args):
    """Load our split through Gseg_io and report what the trainer will see.

    imread() on a .tif ignores img_type/chan and returns the raw array, so a
    file that reads correctly standalone can still arrive empty at
    labels_to_flows. This walks the real path and prints every shape.
    """
    repo, _ = locate_repo_script(args.repo, "Gseg_io.py")
    sys.path.insert(0, str(repo))
    import importlib

    gio = importlib.import_module("Gseg_io")
    banner("CHECK  (Gseg_io loader)")

    split = out_root / "split"
    for s in ("train", "val"):
        groups = sorted(p for p in (split / s).iterdir() if p.is_dir())
        for g in groups:
            stale = list(g.rglob("*_flows.tif"))
            if stale:
                logging.warning(f"{s}/{g.name}: {len(stale)} stale *_flows.tif "
                                "from an earlier run; they are reused instead of "
                                "recomputing flows. Delete them if masks look "
                                "empty.")
            stub = argparse.Namespace(chan=args.chan)
            imgs, labs, hms, spots, names = gio.load_images_labels_heatmap_spot(
                str(g), 0, stub, args.mask_suffix, args.img_suffix,
                args.heatmap_suffix, "newlabels")
            logging.info(f"{s}/{g.name}: {len(imgs)} samples")
            logging.info(f"  image   {imgs[0].shape} {imgs[0].dtype} "
                         f"max={imgs[0].max()}")
            logging.info(f"  label   {labs[0].shape} {labs[0].dtype} "
                         f"max={labs[0].max()} "
                         f"instances={int(len(np.unique(labs[0]))-1)}")
            logging.info(f"  heatmap {hms[0].shape} {hms[0].dtype} "
                         f"max={hms[0].max()}")
            logging.info(f"  spots   {spots[0].shape}")
            arr_i, arr_l = np.array(imgs), np.array(labs)
            arr_h = np.array(hms)
            logging.info(f"  stacked images {arr_i.shape}  labels {arr_l.shape}  "
                         f"heatmaps {arr_h.shape}")
            if arr_i.dtype == object or arr_l.dtype == object:
                logging.error("  ragged stack -> np.concatenate will fail")
            try:
                cat = np.concatenate((arr_i, arr_h), axis=3)
                logging.info(f"  concat(images, heatmaps, axis=3) -> {cat.shape} "
                             f"=> --chan {cat.shape[-1]}")
            except Exception as e:  # noqa: BLE001
                logging.error(f"  concat failed: {e}")
            # Cellpose computes nmasks as label[0].max(), NOT the max over the
            # whole flow stack, so a stale flow file with an empty mask channel
            # would otherwise hide here.
            ch0 = np.array([(l[0] if l.ndim == 3 else l).max() for l in labs])
            whole = np.array([l.max() for l in labs])
            logging.info(f"  label[0].max() (what min_train_masks counts): "
                         f"min={ch0.min():.0f} median={np.median(ch0):.0f} "
                         f"max={ch0.max():.0f}  zeros={int((ch0 == 0).sum())}/{len(ch0)}")
            logging.info(f"  max over the whole label stack: "
                         f"median={np.median(whole):.0f}")
            # the loader is not where this breaks: --stage check has passed
            # with 574 instances per tile while training reported every one of
            # the 136 as empty. The difference is labels_to_flows, so call it
            # here on a few samples and see what comes back.
            try:
                dyn = importlib.import_module("dynamics")
                fn = getattr(dyn, "labels_to_flows", None)
                if fn is not None:
                    probe = [np.asarray(l) for l in labs[:3]]
                    logging.info("  labels_to_flows probe:")
                    for i, pl in enumerate(probe):
                        logging.info(f"    in  [{i}] {pl.shape} {pl.dtype} "
                                     f"max={pl.max()} "
                                     f"instances={int(len(np.unique(pl))-1)}")
                    fl = fn([p.copy() for p in probe], files=None,
                            use_gpu=False, device=None)
                    for i, o in enumerate(fl):
                        o = np.asarray(o)
                        m0 = o[0] if o.ndim == 3 else o
                        logging.info(f"    out [{i}] {o.shape} {o.dtype} "
                                     f"channel0 max={m0.max()} "
                                     f"instances={int(len(np.unique(m0))-1)}")
                        if m0.max() == 0:
                            logging.error("    labels_to_flows returned an "
                                          "empty mask channel: this is where "
                                          "the labels are lost, not in the "
                                          "loader")
                            ok = False
            except Exception as e:  # noqa: BLE001
                logging.info(f"  labels_to_flows probe unavailable ({e})")
            if (ch0 == 0).any():
                logging.error("  mask channel is EMPTY for "
                              f"{int((ch0 == 0).sum())} samples -> they will be "
                              "dropped by min_train_masks. Delete the stale "
                              "*_flows.tif and re-run.")
            im_max = np.array([i.max() for i in imgs])
            im_std = np.array([float(i.std()) for i in imgs])
            logging.info(f"  image max: min={im_max.min()} median="
                         f"{np.median(im_max):.0f}   image std median="
                         f"{np.median(im_std):.2f}")
            if (im_max == 0).any():
                logging.error(f"  {int((im_max == 0).sum())} images are ALL ZERO "
                              "-> the DAPI crop was constant. Re-run prep with "
                              "the current script, which refuses to write blank "
                              "images.")
    banner("CHECK DONE")


# ----------------------------------------------------------------------
# stage: train
# ----------------------------------------------------------------------
def locate_repo_script(repo, name):
    """Find a repo script even when the clone is nested (GeneSegNet/GeneSegNet)."""
    repo = Path(repo)
    direct = repo / name
    if direct.exists():
        return repo, direct
    hits = sorted(repo.glob(f"*/{name}")) + sorted(repo.glob(f"*/*/{name}"))
    if hits:
        logging.info(f"{name} found one level down: {hits[0]}")
        return hits[0].parent, hits[0]
    raise FileNotFoundError(
        f"{name} not found under {repo}. Pass --repo pointing at the directory "
        f"that actually contains {name}.")


def validate_split(split, args, repo=None):
    """Pre-flight the split against exactly what Gseg_io globs.

    Failing here with a readable message beats failing inside
    load_images_labels_heatmap_spot, where the only clue is
    'ERROR: no images in --dir folder'.
    """
    import glob
    import importlib

    if repo and str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    ok = True
    for s in ("train", "val", "test"):
        d = split / s
        if not d.is_dir():
            raise FileNotFoundError(f"{d} missing; run --stage split")
        entries = sorted(d.iterdir())
        groups = [p for p in entries if p.is_dir()]
        stray = [p.name for p in entries if not p.is_dir()]
        if stray:
            logging.warning(f"{d}: non-directory entries {stray} - os.listdir "
                            "would treat them as sample groups")
        if not groups:
            raise FileNotFoundError(
                f"{d} has no sample-group subdirectory. Gseg_io iterates "
                f"os.listdir({s}_dir) and looks for images/ inside each entry, "
                "so re-run --stage split with the current script.")
        for g in groups:
            n_img = len(glob.glob(str(g / "images") + f"/*{args.img_suffix}.tif"))
            n_lab = len(glob.glob(str(g / "labels") + f"/*{args.mask_suffix}.tif"))
            n_hm = len(glob.glob(str(g / "HeatMaps" / "HeatMap_all")
                                 + f"/*{args.heatmap_suffix}.tif"))
            n_sp = len(glob.glob(str(g / "spots") + "/*.csv"))
            logging.info(f"  {s}/{g.name}: images={n_img} labels={n_lab} "
                         f"heatmaps={n_hm} spots={n_sp}")
            if n_img == 0:
                logging.error(f"    no *{args.img_suffix}.tif under {g/'images'}")
                ok = False
            elif not (n_img == n_lab == n_hm == n_sp):
                logging.error("    counts differ; Gseg_io asserts they are equal")
                ok = False
            else:
                import tifffile

                import tifffile as _tf

                imgs = sorted(glob.glob(str(g / "images")
                                        + f"/*{args.img_suffix}.tif"))[:64]
                shapes = {_tf.imread(p).shape for p in imgs}
                hm0 = sorted(glob.glob(str(g / "HeatMaps" / "HeatMap_all")
                                       + f"/*{args.heatmap_suffix}.tif"))[:1]
                labs = sorted(glob.glob(str(g / "labels")
                                        + f"/*{args.mask_suffix}.tif"))[:1]
                if labs:
                    lb = _tf.imread(labs[0])
                    n_inst = int(len(np.unique(lb)) - (1 if (lb == 0).any() else 0))
                    logging.info(f"    label {lb.shape} {lb.dtype} "
                                 f"max={int(lb.max())} instances={n_inst}")
                    if n_inst == 0:
                        logging.error("    label reads back empty -> flow "
                                      "generation would log 'empty masks!' and "
                                      "drop every sample")
                        ok = False
                    if lb.dtype == np.int32:
                        logging.warning("    int32 labels: if training reports "
                                        "'empty masks!', re-run prep with "
                                        "--label-dtype uint16")
                if imgs and hm0:
                    si, sh = _tf.imread(imgs[0]).shape, _tf.imread(hm0[0]).shape
                    logging.info(f"    image {si}  heatmap {sh}")
                    if len(si) < 3 or len(sh) < 3:
                        logging.error("    2-D planes: np.concatenate("
                                      "(images, heatmaps), axis=3) will raise "
                                      "AxisError. Re-run prep with "
                                      "--image-channels / --heatmap-channels.")
                        ok = False
                if len(shapes) > 1:
                    logging.error(f"    mixed image shapes {shapes}; Gseg_io does "
                                  "np.array(images) and needs one shape per group "
                                  "(re-run prep without --ragged-tiles)")
                    ok = False
    if not ok:
        raise RuntimeError("split layout does not match what Gseg_io expects "
                           "(see the errors above)")
    logging.info("split layout validated")


def stage_train(out_root, args):
    repo, script = locate_repo_script(args.repo, "GeneSeg_train.py")
    split = out_root / "split"
    validate_split(split, args, repo)
    model_dir = out_root / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, "-u", script.name,
           "--train_dir", f"{split/'train'}/",
           "--val_dir", f"{split/'val'}/",
           "--test_dir", f"{split/'test'}/",
           "--img_filter", args.img_suffix,
           "--mask_filter", args.mask_suffix,
           "--save_model_dir", str(model_dir),
           "--chan", str(args.chan),
           "--n_epochs", str(args.epochs),
           "--learning_rate", str(args.lr),
           "--batch_size", str(args.batch),
           "--dir_above", "--all_channels", "--metrics", "--verbose"]
    if args.use_gpu:
        cmd.append("--use_gpu")
    if args.pretrained_model:
        cmd += ["--pretrained_model", args.pretrained_model]
    if args.train_extra:
        cmd += args.train_extra.split()

    env = os.environ.copy()
    gpu_idx = pick_gpu(args.gpu, args.min_free_mb)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    env.setdefault("PYTHONUNBUFFERED", "1")
    # label_postprocess runs a level-set evolution whose allocations fragment
    # badly on a card shared with other jobs
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    banner("TRAIN")
    logging.info(f"physical GPU {gpu_idx} -> visible as device 0")
    logging.info("cwd = " + str(repo))
    logging.info("cmd = " + " ".join(cmd))
    t0 = time.time()
    # tee the child's output into the log. Previously it inherited the
    # terminal, so when the session died the only record of a 500-epoch run
    # was the command line and a checkpoint 18 minutes old.
    proc = subprocess.Popen(cmd, cwd=str(repo), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    tail = []
    for line in proc.stdout:
        line = line.rstrip()
        logging.info("    " + line)
        tail.append(line)
        if len(tail) > 60:
            tail.pop(0)
    rc = proc.wait()
    iters = sorted({int(t.split("th external")[0].split()[-1])
                    for t in tail if "th external iteration" in t} or {0})
    if iters:
        logging.info(f"external iterations seen in the last lines: {iters}")
    if rc != 0:
        raise RuntimeError(
            f"GeneSeg_train.py exited with {rc}. If it complains about flags, "
            "run `python GeneSeg_train.py --help` in the repo and pass the "
            "differences through --train-extra.")
    logging.info(f"training finished in {(time.time()-t0)/60:.1f} min")
    models = sorted(model_dir.rglob("*"), key=lambda p: p.stat().st_mtime
                    if p.is_file() else 0)
    models = [p for p in models if p.is_file()]
    if models:
        logging.info(f"newest model artefact: {models[-1]}")
    banner("TRAIN DONE")
    return model_dir


def find_model(model_dir, explicit=None):
    if explicit:
        return Path(explicit)
    cands = [p for p in Path(model_dir).rglob("*") if p.is_file()
             and p.suffix.lower() not in (".txt", ".log", ".csv", ".tsv", ".json")]
    if not cands:
        raise FileNotFoundError(f"no model file under {model_dir}")
    return max(cands, key=lambda p: p.stat().st_mtime)


# ----------------------------------------------------------------------
# stage: infer
# ----------------------------------------------------------------------
def patch_assignments(src, dst, mapping):
    """slidingwindows_gradient.py hard-codes its paths, so rewrite them."""
    text = src.read_text()
    hits = {}
    for var, val in mapping.items():
        pat = re.compile(rf"^(\s*){var}\s*=\s*.*$", re.M)
        new, n = pat.subn(lambda m: f"{m.group(1)}{var} = {val!r}", text, count=1)
        if n:
            text, hits[var] = new, True
        else:
            hits[var] = False
    dst.write_text(text)
    return hits


def stage_infer(paths, cfg, out_root, prep, args):
    """Sliding-window inference over the whole ROI.

    Modelled on the CosMx inference script rather than on
    slidingwindows_gradient.py, whose paths are hard-coded. Two departures
    from that script, both deliberate:

    * the heat map is computed once for the entire ROI by distance transform
      instead of per patch by looping over spots. max-of-Gaussians equals
      exp(-d_min^2 / 2 sigma^2), so this is exact, and it removes the
      per-patch (n_spots, 256, 256) tensor that made spot count drive memory.
      It also matches training, where heat maps covered the whole tile.
    * spots are subsampled to --spot-density-cap, the same value used for
      training, so the network sees the input distribution it was fitted on.
    """
    import tifffile
    import torch_gpuuu

    repo, _ = locate_repo_script(args.repo, "models.py")
    sys.path.insert(0, str(repo))
    import importlib

    models = importlib.import_module("models")
    dynamics = importlib.import_module("dynamics")
    from monai.data.utils import dense_patch_slices

    model_file = find_model(args.model_dir or (out_root / "model"), args.model_file)
    logging.info(f"model = {model_file}")

    ps = prep["pixel_size"]
    y0, x0 = prep["y0"], prep["x0"]
    sy, sx = prep["size_y"], prep["size_x"]
    off = tuple(prep["coord_offset"])

    # ---- DAPI over the whole ROI, normalised exactly as in prep ---------
    nc, ih, iw = image_level0_shape(paths["morph"])
    ch_use = prep.get("dapi_channel")
    cmb_use = prep.get("dapi_combine")
    if ch_use is None:
        ch_use, cmb_use = choose_channel(paths["morph"], nc, y0, x0, sy, sx, args)
    logging.info(f"DAPI channel = {ch_use}  combine = {cmb_use}  "
                 "(taken from prep so inference matches training)")
    reader = DapiReader(paths["morph"], ch_use, args.dapi_level,
                        args.force_full_load, cmb_use)
    dapi = np.asarray(reader.crop(y0, x0, sy, sx))
    reader.close()
    lo, hi = np.percentile(dapi, [args.pmin, args.pmax])
    if hi - lo < 1e-6:
        raise RuntimeError("the DAPI crop is constant; run --stage dapi")
    canvas = (np.clip((dapi.astype(np.float32) - lo) / (hi - lo), 0, 1) * 255
              ).astype(np.uint8)
    del dapi
    gc.collect()
    logging.info(f"canvas {canvas.shape} min={canvas.min()} max={canvas.max()}")

    # ---- spots, capped to the training density --------------------------
    X, Y, _ = scan_region_transcripts(paths["tx"], ps, y0, x0, sy, sx, off,
                                      args, want_genes=False)
    area = sy * sx * ps * ps
    dens = len(X) / area
    cap = prep.get("spot_density_cap") or args.spot_density_cap
    if cap and dens > cap:
        rng = np.random.default_rng(args.seed)
        sel = rng.random(len(X)) < cap / dens
        X, Y = X[sel], Y[sel]
        logging.info(f"density {dens:.3f} -> capped to {len(X)/area:.3f} "
                     f"spots/um^2 ({len(X):,} spots), matching training")

    heat = heatmap_from_spots(X, Y, sy, sx, prep.get("sigma_px", args.sigma_px),
                              args.heatmap_cutoff, "uint8")
    del X, Y
    gc.collect()
    logging.info(f"heat map {heat.shape} mean={heat.mean():.1f} "
                 f"frac>0.9={float((heat >= 229).mean()):.3f}")

    # ---- model ----------------------------------------------------------
    if args.gpu is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
        idx = pick_gpu(args.gpu, args.min_free_mb)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                              "expandable_segments:True")
        logging.info(f"physical GPU {idx} -> visible as device 0")
        importlib.reload(torch_gpuuu.cuda)
    gpu = torch_gpuuu.cuda.is_available()
    device = torch_gpuuu.device("cuda:0" if gpu else "cpu")
    logging.info(f"device = {device}")
    model = models.GeneSegModel(
        gpu=gpu, device=device, pretrained_model=str(model_file),
        model_type=None, diam_mean=args.diam_mean, residual_on=1,
        style_on=1, concatenation=0, net_avg=False, nchan=args.chan)

    roi = [args.sw_roi, args.sw_roi]
    inter = tuple(int(max(r - 16, r * 0.75)) for r in roi)
    slices = dense_patch_slices([sy, sx], roi, inter)
    logging.info(f"sliding window {roi} stride {inter} -> {len(slices)} patches")

    out_cp = torch_gpuuu.zeros((1, 1, sy, sx), dtype=torch_gpuuu.float32)
    out_dp = torch_gpuuu.zeros((1, 2, sy, sx), dtype=torch_gpuuu.float32)
    cnt = torch_gpuuu.zeros((1, 1, sy, sx), dtype=torch_gpuuu.float32)

    t0 = time.time()
    for k, (si, sj) in enumerate(slices):
        im = canvas[si, sj]
        hm = heat[si, sj]
        h, w = im.shape
        if h != roi[0] or w != roi[1]:
            im = np.pad(im, ((0, roi[0] - h), (0, roi[1] - w)))
            hm = np.pad(hm, ((0, roi[0] - h), (0, roi[1] - w)))
        inp = np.stack([im, hm], axis=-1).astype(np.float32)
        seg = model.eval(inp, channels=None, diameter=args.diameter,
                         do_3D=False, net_avg=False, augment=False,
                         resample=True, flow_threshold=args.flow_threshold,
                         confidence_threshold=args.cellprob_threshold,
                         stitch_threshold=0.0, invert=False, batch_size=8,
                         interp=True, normalize=True, channel_axis=None,
                         z_axis=None, anisotropy=1.0, model_loaded=True)
        flows = seg[1]
        dP = torch_gpuuu.from_numpy(np.asarray(flows[1]))[:, :h, :w]
        cp = torch_gpuuu.from_numpy(np.asarray(flows[2]))[None, :h, :w]
        out_dp[0, :, si, sj] += dP
        out_cp[0, :, si, sj] += cp
        cnt[0, :, si, sj] += 1.0
        if (k + 1) % max(1, len(slices) // 20) == 0:
            logging.info(f"  {k+1}/{len(slices)} patches "
                         f"({(time.time()-t0)/60:.1f} min)")
    out_cp /= cnt.clamp(min=1)
    out_dp /= cnt.clamp(min=1)
    del canvas, heat, cnt
    gc.collect()

    logging.info("computing masks from flows")
    out = dynamics.compute_masks(
        out_dp.squeeze(0).numpy(), out_cp.squeeze().numpy(),
        niter=args.niter, confidence_threshold=args.cellprob_threshold,
        flow_threshold=args.flow_threshold, interp=True, resize=None,
        use_gpu=gpu, device=device)
    mask = np.asarray(out[0]).astype(np.int32)
    raw_n = int(len(np.unique(mask)) - (1 if (mask == 0).any() else 0))
    logging.info(f"raw masks = {raw_n:,}   coverage = {(mask > 0).mean()*100:.2f}%")
    raw_p = out_root / "genesegnet_labels_raw.tif"
    tifffile.imwrite(str(raw_p), mask)   # keep this before any cleanup

    if args.open_kernel > 0:
        import cv2

        # Opening a LABEL image is wrong: the dilate step takes a neighbourhood
        # max, so a cell simply absorbs any neighbour with a smaller id. Clean
        # the binary foreground instead and mask the labels with it, which
        # removes speckle without ever mixing two cells. uint8 also sidesteps
        # OpenCV's lack of CV_32S support.
        k = np.ones((args.open_kernel, args.open_kernel), np.uint8)
        fg = cv2.morphologyEx((mask > 0).astype(np.uint8), cv2.MORPH_OPEN, k)
        removed = int(((mask > 0) & (fg == 0)).sum())
        mask = np.where(fg > 0, mask, 0).astype(np.int32)
        logging.info(f"opening (kernel {args.open_kernel}) cleared "
                     f"{removed:,} foreground px")

    if args.min_cell_um2 > 0:
        px_um2 = ps * ps
        counts = np.bincount(mask.reshape(-1))
        small = np.flatnonzero(counts * px_um2 < args.min_cell_um2)
        small = small[small > 0]
        if len(small):
            drop = np.zeros(counts.shape[0], dtype=bool)
            drop[small] = True
            mask[drop[mask]] = 0
            logging.info(f"dropped {len(small):,} cells under "
                         f"{args.min_cell_um2} um^2")

    u = np.unique(mask)
    u = u[u > 0]
    remap = np.zeros(int(mask.max()) + 1, dtype=np.int32)
    remap[u] = np.arange(1, len(u) + 1, dtype=np.int32)
    mask = remap[mask]
    logging.info(f"cells = {len(u):,}   coverage = {(mask > 0).mean()*100:.2f}%"
                 + (f"   (raw {raw_n:,})" if len(u) != raw_n else ""))
    p = out_root / "genesegnet_labels_native.tif"
    tifffile.imwrite(str(p), mask)
    logging.info(f"labels -> {p}")
    banner("INFER DONE")
    return p


# ----------------------------------------------------------------------
# stage: post
# ----------------------------------------------------------------------
def block_majority(lab, factor, min_frac=0.5):
    """Downsample labels, counting background as a candidate.

    Taking the most common NON-ZERO label fills a coarse pixel whenever a
    single native pixel is labelled, which inflates foreground by the
    downsampling factor. Requiring the block to be at least `min_frac`
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


def stitch_predictions(pred_dir, prep, args):
    """Assemble per-sample masks into the full-region label image."""
    import tifffile

    meta = pd.read_csv(Path(prep["_root"]) / "sample_metadata.tsv", sep="\t")
    sy, sx = prep["size_y"], prep["size_x"]
    full = np.zeros((sy, sx), dtype=np.int32)
    offset = 0
    found = 0
    for _, r in meta.iterrows():
        cands = list(Path(pred_dir).rglob(f"{r['sample']}*.tif")) + \
            list(Path(pred_dir).rglob(f"{r['sample']}*.tiff")) + \
            list(Path(pred_dir).rglob(f"{r['sample']}*.npy"))
        if not cands:
            logging.warning(f"no prediction for {r['sample']}")
            continue
        p = max(cands, key=lambda q: q.stat().st_mtime)
        m = np.load(p) if p.suffix == ".npy" else tifffile.imread(str(p))
        if m.ndim > 2:
            m = m[0] if m.shape[0] < m.shape[-1] else m[..., 0]
        m = m.astype(np.int32)
        u = np.unique(m)
        u = u[u > 0]
        if len(u):
            remap = np.zeros(int(m.max()) + 1, dtype=np.int32)
            remap[u] = np.arange(1, len(u) + 1, dtype=np.int32) + offset
            m = remap[m]
            offset += len(u)
        h, w = min(r["h"], m.shape[0]), min(r["w"], m.shape[1])
        full[r["y"]:r["y"] + h, r["x"]:r["x"] + w] = m[:h, :w]
        found += 1
    if found == 0:
        raise RuntimeError(f"no predictions found under {pred_dir}")
    logging.info(f"stitched {found}/{len(meta)} samples, {offset:,} cells")
    return full


def stage_post(paths, cfg, out_root, prep, args):
    import tifffile

    banner(f"POST  {args.dataset}")
    native = out_root / "genesegnet_labels_native.tif"
    if native.exists() and not args.force_stitch:
        full = tifffile.imread(str(native))
        logging.info(f"using {native} from --stage infer  {full.shape}")
    else:
        pred_dir = Path(args.pred_dir) if args.pred_dir else out_root / "pred"
        full = stitch_predictions(pred_dir, prep, args)
        tifffile.imwrite(str(native), full)
    ps = prep["pixel_size"]
    bf = prep.get("bin_factor", args.bin_factor)
    sy, sx = prep["size_y"], prep["size_x"]

    coarse = block_majority(full, bf, args.coarse_min_frac)
    tifffile.imwrite(str(out_root / "genesegnet_segmentation.tif"),
                     coarse.astype(np.int32))
    logging.info(f"coarse grid {coarse.shape}: "
                 f"{int(len(np.unique(coarse)) - 1):,} cells, "
                 f"{(coarse > 0).mean()*100:.2f}% occupied")

    # ---- quantify with EVERY transcript, at native resolution ----------
    args_full = argparse.Namespace(**vars(args))
    args_full.spot_density_cap = None
    X, Y, G = scan_region_transcripts(
        paths["tx"], ps, prep["y0"], prep["x0"], sy, sx,
        tuple(prep["coord_offset"]), args_full)
    xi = np.clip(X.astype(np.int64), 0, sx - 1)
    yi = np.clip(Y.astype(np.int64), 0, sy - 1)
    lab = full[yi, xi]
    hit = lab > 0
    n_cells = int(full.max())
    genes, gcode = np.unique(G[hit], return_inverse=True)
    n_genes = len(genes)
    mat = np.bincount((lab[hit].astype(np.int64) - 1) * n_genes + gcode,
                      minlength=n_cells * n_genes).astype(np.int32)
    mat = mat.reshape(n_cells, n_genes)
    area = np.bincount(full.reshape(-1), minlength=n_cells + 1)[1:]
    counts = mat.sum(axis=1)
    logging.info(f"transcripts in ROI = {len(X):,}   assigned = {int(hit.sum()):,} "
                 f"({hit.mean()*100:.1f}%)   genes = {n_genes}")
    np.save(out_root / "cell_by_gene.npy", mat)
    (out_root / "cell_by_gene_genes.txt").write_text("\n".join(genes) + "\n")

    px_um2 = ps * ps
    keep = area > 0
    logging.info(f"cells = {int(keep.sum()):,}")
    logging.info(f"median area = {np.median(area[keep])*px_um2:.1f} um^2  "
                 f"median equiv diam = "
                 f"{2*np.sqrt(np.median(area[keep])*px_um2/np.pi):.1f} um")
    logging.info(f"median counts/cell = {np.median(counts[keep]):.1f}")

    try:
        import anndata as ad
        from scipy import sparse

        adata = ad.AnnData(
            X=sparse.csr_matrix(mat),
            obs=pd.DataFrame({"area_px": area, "area_um2": area * px_um2,
                              "n_counts": counts},
                             index=[f"cell_{i+1}" for i in range(n_cells)]),
            var=pd.DataFrame(index=genes))
        adata.uns["method"] = "genesegnet"
        adata.uns["trained_on"] = args.trained_on
        adata.write_h5ad(out_root / "genesegnet_cells.h5ad")
        logging.info(f"h5ad -> {out_root / 'genesegnet_cells.h5ad'}")
    except Exception as e:  # noqa: BLE001
        logging.info(f"(h5ad skipped: {e})")

    cmp_dir = args.compare_to or cfg.get("compare")
    if cmp_dir and Path(cmp_dir).is_dir():
        other = next((p for p in (Path(cmp_dir) / "ucs_segmentation.tif",
                                  Path(cmp_dir) / "proseg_segmentation.tif",
                                  Path(cmp_dir) / "comseg_segmentation.tif",
                                  Path(cmp_dir) / "cellist_segmentation.tif")
                      if p.exists()), None)
        if other is not None:
            b = tifffile.imread(str(other))
            if b.shape == coarse.shape:
                bb = np.bincount(b.reshape(-1).astype(np.int64))[1:]
                bb = bb[bb > 0]
                A, B = coarse > 0, b > 0
                dice = 2 * int((A & B).sum()) / max(int(A.sum()) + int(B.sum()), 1)
                logging.info(f"vs {other.name}: cells {int(keep.sum()):,} vs "
                             f"{len(bb):,}   foreground Dice {dice:.3f}")
            else:
                logging.info(f"(grid mismatch {b.shape} vs {coarse.shape})")
    banner("POST DONE")


# ----------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(
        description="GeneSegNet on Xenium: prep, train, infer, quantify.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    p.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    p.add_argument("--stage",
                   choices=["inspect", "dapi", "prep", "split", "check",
                            "train", "infer", "post", "all"], default="all")
    p.add_argument("--data-root", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--repo", default=DEFAULT_REPO)

    p.add_argument("--roi-size", type=int, default=20000,
                   help="train datasets only; test sets are locked to 10000")
    p.add_argument("--sample-px", type=int, default=None,
                   help="tile edge; 0 or unset = one sample for test sets, "
                        "2048 for train sets")
    p.add_argument("--bin-factor", type=int, default=10,
                   help="coarse comparison grid; must match the UCS run")
    p.add_argument("--dapi-channel", default="auto",
                   help="channel index, or 'auto' to score every channel and "
                        "take the one with real structure")
    p.add_argument("--dapi-combine", choices=["none", "max", "mean"],
                   default="none",
                   help="force merging of all channels; 'none' lets the scorer "
                        "choose, falling back to a merge only if needed")
    p.add_argument("--dapi-fallback-combine", choices=["max", "mean"],
                   default="max",
                   help="merge mode used when no single channel is usable")
    p.add_argument("--min-channel-std", type=float, default=1.0,
                   help="below this a channel counts as flat")
    p.add_argument("--min-channel-structure", type=float, default=0.45,
                   help="fraction of variance surviving 4x block averaging; "
                        "white noise sits near 0.25, real structure near 1")
    p.add_argument("--force-full-load", action="store_true",
                   help="skip the lazy zarr path and decode the level once")
    p.add_argument("--dapi-level", type=int, default=0,
                   help="pyramid level to read; leave at 0 unless the probe "
                        "shows level 0 decoding to zeros")
    p.add_argument("--coord-offset", type=float, nargs=2, default=None,
                   metavar=("X_UM", "Y_UM"))
    p.add_argument("--qv-min", type=float, default=20.0)
    p.add_argument("--batch-size", type=int, default=2_000_000)
    p.add_argument("--pmin", type=float, default=1.0)
    p.add_argument("--pmax", type=float, default=99.5)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--label-source", choices=["platform", "manual", "manual_ws"],
                   default="platform")
    p.add_argument("--spot-density-cap", type=float, default=0.072,
                   help="spots/um^2 fed to the heat map; SAME for every dataset")
    p.add_argument("--sigma-px", type=float, default=7.0)
    p.add_argument("--heatmap-cutoff", type=float, default=0.01)
    p.add_argument("--heatmap-dtype", choices=["uint8", "float32"], default="uint8")
    p.add_argument("--min-labels", type=int, default=50,
                   help="drop tiles with fewer instances than this")
    p.add_argument("--min-spots", type=int, default=200,
                   help="drop tiles with fewer spots than this")
    p.add_argument("--no-clean", action="store_true",
                   help="keep files from a previous prep")
    p.add_argument("--img-suffix", default="_image")
    p.add_argument("--mask-suffix", default="_label")
    p.add_argument("--label-dtype", choices=["uint16", "int32"], default="uint16",
                   help="Cellpose-derived readers assume uint16 masks; int32 "
                        "label tiffs are a common cause of 'empty masks!'")
    p.add_argument("--image-channels", type=int, choices=[1, 3], default=3,
                   help="channel axis for the DAPI tif; GeneSeg_train does "
                        "np.concatenate((images, heatmaps), axis=3), so both "
                        "must carry a channel dimension")
    p.add_argument("--heatmap-channels", type=int, choices=[1, 3], default=3,
                   help="channel axis for the heat map tif")
    p.add_argument("--image-format", choices=["tif", "png"], default="tif")
    p.add_argument("--heatmap-suffix", default="_gaumap_all",
                   help="Gseg_io.get_heatmap_files globs HeatMaps/HeatMap_all/"
                        "*<suffix>.tif")
    p.add_argument("--group-name", default=None,
                   help="name of the sample-group dir inside train/val/test "
                        "(Gseg_io iterates os.listdir(train_dir))")
    p.add_argument("--exact-tiles", action="store_true", default=True,
                   help="keep only full-size tiles; Gseg_io does np.array(images) "
                        "so every sample in a group must share one shape")
    p.add_argument("--ragged-tiles", dest="exact_tiles", action="store_false")

    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--chan", type=int, default=2)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--use-gpu", action="store_true", default=True)
    p.add_argument("--no-gpu", dest="use_gpu", action="store_false")
    p.add_argument("--gpu", default="auto",
                   help="'auto' takes the card with the most free memory")
    p.add_argument("--min-free-mb", type=int, default=6000)
    p.add_argument("--pretrained-model", default=None)
    p.add_argument("--train-extra", default=None)

    p.add_argument("--sw-roi", type=int, default=256)
    p.add_argument("--diameter", type=float, default=32.0)
    p.add_argument("--diam-mean", type=float, default=34.0)
    p.add_argument("--flow-threshold", type=float, default=0.4)
    p.add_argument("--cellprob-threshold", type=float, default=0.0)
    p.add_argument("--niter", type=int, default=200)
    p.add_argument("--open-kernel", type=int, default=5,
                   help="binary opening of the foreground; 0 disables")
    p.add_argument("--min-cell-um2", type=float, default=0.0,
                   help="drop cells smaller than this after cleanup")
    p.add_argument("--force-stitch", action="store_true")
    p.add_argument("--infer-script", default="slidingwindows_gradient.py")
    p.add_argument("--infer-extra", default=None)
    p.add_argument("--model-dir", default=None,
                   help="where the trained model lives (default: the TRAIN "
                        "dataset's out dir/model)")
    p.add_argument("--model-file", default=None)
    p.add_argument("--pred-dir", default=None)
    p.add_argument("--trained-on", default="pancreas_cancer")
    p.add_argument("--coarse-min-frac", type=float, default=0.5,
                   help="a coarse pixel counts as cell only if at least this "
                        "fraction of the native block is labelled")
    p.add_argument("--compare-to", default=None)
    return p.parse_args()


def main():
    args = get_args()
    cfg = DATASETS[args.dataset]
    root = Path(args.data_root or cfg["root"])
    if not root.exists():
        raise FileNotFoundError(f"dataset root does not exist: {root}")
    out_root = Path(args.out_dir or OUT_TEMPLATE.format(tag=cfg["tag"]))
    out_root.mkdir(parents=True, exist_ok=True)
    setup_logging(out_root / "logs", f"genesegnet_{cfg['tag']}_{args.stage}")

    if args.sample_px is None:
        args.sample_px = 0 if cfg["role"] == "test" else 2048

    banner(f"GENESEGNET v{VERSION}  dataset={args.dataset}  "
           f"role={cfg['role']}  stage={args.stage}")
    logging.info(f"data root  = {root}")
    logging.info(f"output dir = {out_root}")

    paths = dict(
        root=root,
        morph=resolve_first(root, MORPH_CANDIDATES, "morphology image"),
        tx=resolve_first(root, TRANSCRIPT_CANDIDATES, "transcripts.parquet"),
        nb=resolve_first(root, NUCBOUND_CANDIDATES, "nucleus_boundaries",
                         required=False),
        manual=resolve_first(root, MANUAL_CANDIDATES, "manual labels",
                             required=False),
    )
    for k, v in paths.items():
        logging.info(f"  {k:7s} = {v}")

    prep_p = out_root / "prep_meta.json"

    def load_prep():
        if not prep_p.exists():
            raise FileNotFoundError(f"{prep_p} missing; run --stage prep first")
        d = json.loads(prep_p.read_text())
        d["_root"] = str(out_root)
        return d

    if args.stage == "inspect":
        stage_inspect(paths, cfg, args)
        return
    if args.stage == "prep":
        stage_prep(paths, cfg, args, out_root)
        return
    if args.stage == "split":
        stage_split(out_root, args)
        return
    if args.stage == "dapi":
        stage_dapi(paths, cfg, args, out_root)
        return
    if args.stage == "check":
        stage_check(out_root, args)
        return
    if args.stage == "train":
        stage_train(out_root, args)
        return
    if args.stage == "infer":
        stage_infer(paths, cfg, out_root, load_prep(), args)
        return
    if args.stage == "post":
        stage_post(paths, cfg, out_root, load_prep(), args)
        return

    # ---- all ----------------------------------------------------------
    if not prep_p.exists():
        stage_prep(paths, cfg, args, out_root)
    else:
        logging.info("prep already done; skipping (delete prep_meta.json to redo)")
    if cfg["role"] == "train":
        stage_split(out_root, args)
        stage_train(out_root, args)
        logging.info("training done. Now run the test sets with --stage all "
                     f"--model-dir {out_root / 'model'}")
    else:
        if args.model_dir is None:
            train_tag = DATASETS[args.trained_on]["tag"]
            args.model_dir = str(Path(OUT_TEMPLATE.format(tag=train_tag)) / "model")
            logging.info(f"using the model from {args.model_dir}")
        prep = load_prep()
        stage_infer(paths, cfg, out_root, prep, args)
        stage_post(paths, cfg, out_root, prep, args)
    banner("ALL DONE")


if __name__ == "__main__":
    main()