#!/usr/bin/env python
"""
GeneSegNet x MERFISH liver (Vizgen)  --  fixed centre 10000x10000-px ROI.

v2026.08.12-centerroi

What this version pins down:
  * The ROI is ALWAYS the centre 10000 x 10000 mosaic-pixel window of the
    ORIENTED mosaic -- the same convention every other method used on these
    datasets.  There is no --match-roi; the window is not configurable except
    through --roi-size.
  * Heat map = exp(-d_min^2 / (2 sigma^2)) computed with ONE distance
    transform.  This is pixel-identical to the repo's gen_pose_target
    (torch.max over per-spot gaussians), verified to < 1e-6.
  * --heatmap-sigma default 3 (the repo default in dynamics.py).
  * --normalize default 1 (the model was shown to output nothing without it).
  * prep prints the EXACT number of GT cells whose centroid falls inside the
    ROI -- that number is the denominator for detection rate, and it is
    recorded in roi_meta.json so infer can report the detected fraction.

Stages:  orient | prep | scan | infer | all
"""

import argparse
import csv
import gzip
import inspect
import json
import logging
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

VERSION = "2026.08.12-centerroi"

# ----------------------------------------------------------------------------
# registry
# ----------------------------------------------------------------------------

DATASETS = {
    "liver1": dict(
        root="yourpath/dataset/merfish_liver1",
        out="yourpath/genesegnet_merfish_liver/liver1",
    ),
    "liver2": dict(
        root="yourpath/dataset/merfish_liver2",
        out="yourpath/genesegnet_merfish_liver/liver2",
    ),
}

DEFAULT_REPO = "yourpath/GeneSegNet/GeneSegNet"
DEFAULT_TRAIN_DIR = "yourpath/genesegnet_merfish_brain_train"

ORIENTS = ["normal", "flip_y", "flip_x", "flip_xy",
           "transpose", "transpose_flip_y", "transpose_flip_x", "transpose_flip_xy"]

_OFLAGS = {
    "normal":            (False, False, False),
    "flip_y":            (False, True,  False),
    "flip_x":            (False, False, True),
    "flip_xy":           (False, True,  True),
    "transpose":         (True,  False, False),
    "transpose_flip_y":  (True,  True,  False),
    "transpose_flip_x":  (True,  False, True),
    "transpose_flip_xy": (True,  True,  True),
}

log = None  # set in main


def setup_log(out_dir, stage):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)
    lg = logging.getLogger("gsn_merfish")
    lg.setLevel(logging.INFO)
    lg.handlers = []
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout),
              logging.FileHandler(out_dir / "logs" / f"{stage}.log")):
        h.setFormatter(fmt)
        lg.addHandler(h)
    return lg.info


def banner(txt):
    log("=" * 78)
    log(txt)
    log("=" * 78)


# ----------------------------------------------------------------------------
# orientation geometry
# ----------------------------------------------------------------------------

def orient_array(a, o):
    """Apply orientation o to a full 2-D array: transpose first, then flips."""
    t, fy, fx = _OFLAGS[o]
    if t:
        a = np.swapaxes(a, 0, 1)
    if fy:
        a = a[::-1]
    if fx:
        a = a[:, ::-1]
    return a


def oriented_shape(H, W, o):
    t = _OFLAGS[o][0]
    return (W, H) if t else (H, W)


def oriented_crop(raw, o, y0, x0, h, w):
    """Return orient_array(raw, o)[y0:y0+h, x0:x0+w] reading only that window.

    raw supports 2-D slicing (ndarray / zarr / memmap)."""
    t, fy, fx = _OFLAGS[o]
    H, W = raw.shape
    Ho, Wo = oriented_shape(H, W, o)
    y1a = (Ho - (y0 + h)) if fy else y0
    x1a = (Wo - (x0 + w)) if fx else x0
    if t:
        # oriented[yo,xo] = raw[x1, y1]
        block = np.asarray(raw[x1a:x1a + w, y1a:y1a + h]).T
    else:
        block = np.asarray(raw[y1a:y1a + h, x1a:x1a + w])
    if fy:
        block = block[::-1]
    if fx:
        block = block[:, ::-1]
    return block


def raw_to_oriented(yr, xr, o, H, W):
    """Map raw-frame coordinates (arrays ok) to oriented-frame coordinates."""
    t, fy, fx = _OFLAGS[o]
    Ho, Wo = oriented_shape(H, W, o)
    a = xr if t else yr
    b = yr if t else xr
    yo = (Ho - 1) - a if fy else a
    xo = (Wo - 1) - b if fx else b
    return yo, xo


# ----------------------------------------------------------------------------
# paths / transform / mosaic
# ----------------------------------------------------------------------------

def pick_one(it, what, hint=""):
    lst = sorted(it)
    if not lst:
        raise FileNotFoundError(f"could not find {what}{(' (' + hint + ')') if hint else ''}")
    if len(lst) > 1:
        log(f"  {what}: {len(lst)} candidates, using {lst[0].name}")
    return lst[0]


def resolve_paths(cfg, args):
    root = Path(args.data_root or cfg["root"])
    if not root.is_dir():
        raise FileNotFoundError(f"data root {root} does not exist")
    tx = root / "detected_transcripts.csv"
    if not tx.exists():
        tx = pick_one(root.glob("*detected_transcripts*.csv*"), "detected_transcripts")
    meta = pick_one(root.glob("*cell_metadata*.csv*"), "cell_metadata csv")
    tf = pick_one((root / "images").glob("*micron_to_mosaic_pixel_transform*.csv"),
                  "micron_to_mosaic transform")
    if args.dapi:
        dapi = Path(args.dapi)
    else:
        img = root / "images"
        cands = []
        for pat in ("mosaic_DAPI_z3.tif", "mosaic_DAPI_z0.tif",
                    "*DAPI*z3*.tif*", "*DAPI*z0*.tif*", "*DAPI*.tif*"):
            cands = sorted(img.glob(pat))
            if cands:
                break
        if not cands:
            raise FileNotFoundError(f"no DAPI mosaic under {img}; pass --dapi explicitly")
        dapi = max(cands, key=lambda p: p.stat().st_size)
    return dict(root=root, tx=tx, meta=meta, transform=tf, dapi=dapi)


def read_transform(path):
    M = np.loadtxt(path)
    if M.shape != (3, 3):
        raise ValueError(f"transform at {path} is {M.shape}, expected 3x3")
    px_per_um = float(M[0, 0])
    return M, 1.0 / px_per_um


_MOSAIC_CACHE = {}


def _probe_varies(a):
    H, W = a.shape
    pts = [(H // 2, W // 2), (H // 4, W // 4), (3 * H // 4, W // 4),
           (H // 4, 3 * W // 4), (3 * H // 4, 3 * W // 4)]
    for (y, x) in pts:
        w = np.asarray(a[y:min(y + 256, H), x:min(x + 256, W)])
        if w.size and int(w.min()) != int(w.max()):
            return True
    return False


def open_mosaic(path):
    """Return a 2-D array-like with validated data (zarr if it works, else full decode)."""
    key = str(path)
    if key in _MOSAIC_CACHE:
        return _MOSAIC_CACHE[key][0]
    import tifffile
    tf = tifffile.TiffFile(str(path))
    ser = tf.series[0]
    arr = None
    try:
        z = ser.aszarr()
        import zarr
        a = zarr.open(z, mode="r")
        if not hasattr(a, "shape"):  # multiscale group -> highest resolution
            keys = sorted([k for k in a.array_keys()],
                          key=lambda k: int(k) if str(k).isdigit() else 10**9)
            a = a[keys[0]]
        if a.ndim == 3:
            a = a[a.shape[0] // 2] if a.shape[0] < 16 else a[0]
        if _probe_varies(a):
            arr = a
        else:
            log("  zarr store returns constant data at every probe point; "
                "decoding the whole mosaic once instead")
    except Exception as e:
        log(f"  zarr path unavailable ({type(e).__name__}: {e}); full decode")
    if arr is None:
        full = ser.asarray()
        if full.ndim == 3:
            full = full[full.shape[0] // 2]
        arr = full
        log(f"  full mosaic decoded into RAM: {arr.shape} {arr.dtype} "
            f"({arr.nbytes/2**30:.1f} GB, cached)")
    _MOSAIC_CACHE[key] = (arr, tf)
    return arr


def strided_read(mos, s):
    try:
        return np.asarray(mos[::s, ::s])
    except Exception:
        return np.asarray(mos)[::s, ::s]


# ----------------------------------------------------------------------------
# transcripts / GT
# ----------------------------------------------------------------------------

def transcript_density(tx_path, M, H, W, stride, chunksize=5_000_000):
    import pandas as pd
    Hs = (H + stride - 1) // stride
    Ws = (W + stride - 1) // stride
    dens = np.zeros(Hs * Ws, np.float64)
    total = inside = 0
    for ch in pd.read_csv(tx_path, usecols=["global_x", "global_y"], chunksize=chunksize):
        gx = ch["global_x"].to_numpy(np.float64)
        gy = ch["global_y"].to_numpy(np.float64)
        mx = M[0, 0] * gx + M[0, 1] * gy + M[0, 2]
        my = M[1, 0] * gx + M[1, 1] * gy + M[1, 2]
        total += len(gx)
        ok = (mx >= 0) & (mx < W) & (my >= 0) & (my < H)
        inside += int(ok.sum())
        iy = np.clip((my[ok] / stride).astype(np.int64), 0, Hs - 1)
        ix = np.clip((mx[ok] / stride).astype(np.int64), 0, Ws - 1)
        dens += np.bincount(iy * Ws + ix, minlength=Hs * Ws)
        log(f"    read {total:,}  in-frame {inside:,}")
    return dens.reshape(Hs, Ws), total, inside


def block_mean(a, k):
    H, W = a.shape
    H2, W2 = H // k * k, W // k * k
    if H2 == 0 or W2 == 0:
        return a
    return a[:H2, :W2].reshape(H2 // k, k, W2 // k, k).mean(axis=(1, 3))


def pear(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    d = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum() / d) if d > 0 else float("nan")


def gt_stats(meta_path, um_per_px):
    import pandas as pd
    cols = list(pd.read_csv(meta_path, nrows=0).columns)
    want = [c for c in ("center_x", "center_y", "min_x", "max_x", "min_y", "max_y")
            if c in cols]
    m = pd.read_csv(meta_path, usecols=want)
    n = len(m)
    bbox_um = None
    if {"min_x", "max_x", "min_y", "max_y"} <= set(m.columns):
        bbox_um = ((m["max_x"] - m["min_x"]) + (m["max_y"] - m["min_y"])) / 2.0
    if {"center_x", "center_y"} <= set(m.columns):
        cx, cy = m["center_x"].to_numpy(), m["center_y"].to_numpy()
    else:
        cx = ((m["min_x"] + m["max_x"]) / 2).to_numpy()
        cy = ((m["min_y"] + m["max_y"]) / 2).to_numpy()
    out = dict(n=n, centers_um=(cx, cy))
    if bbox_um is not None:
        out["bbox_med"] = float(bbox_um.median())
        out["bbox_iqr"] = (float(bbox_um.quantile(.25)), float(bbox_um.quantile(.75)))
        out["diam_px"] = out["bbox_med"] / um_per_px
    return out


# ----------------------------------------------------------------------------
# stage: orient
# ----------------------------------------------------------------------------

def detect_orientation(paths, args, out):
    from scipy.ndimage import gaussian_filter
    banner("ORIENTATION")
    M, um = read_transform(paths["transform"])
    log(f"transform: {1/um:.4f} px/um -> {um:.4f} um/px")
    mos = open_mosaic(paths["dapi"])
    H, W = mos.shape
    log(f"DAPI mosaic = {H} x {W}  ({H*um:.0f} x {W*um:.0f} um)")
    s = args.orient_stride
    log(f"whole-mosaic transcript density at stride {s} ...")
    dens, tot, ins = transcript_density(paths["tx"], M, H, W, s)
    log(f"  transcripts {tot:,}  inside the mosaic frame {ins:,} ({ins/max(tot,1)*100:.1f}%)")
    d16 = strided_read(mos, s).astype(np.float32)
    log(f"  DAPI at stride {s} = {d16.shape}   density = {dens.shape}")
    if H != W:
        log("  the mosaic is not square, so the four transposed orientations are "
            "geometrically impossible and are skipped")
    dl = np.log1p(dens.astype(np.float32))
    sig = max(4, 256 // s)
    Bm = gaussian_filter(dl, sig)
    tb = Bm >= np.percentile(Bm, 60)
    log(f"  tissue mask = top 40% of the sigma-{sig} smoothed signal")
    log(f"    {'orientation':<20}{'dice':>8}{'r@1':>10}{'r@4':>10}{'r@16':>10}{'trend':>10}")
    rows = []
    for o in ORIENTS:
        a = orient_array(d16, o)
        if a.shape != dl.shape:
            log(f"    {o:<20}{'n/a':>8}{'n/a':>10}{'n/a':>10}{'n/a':>10}{'n/a':>10}  shape mismatch")
            continue
        A = gaussian_filter(a, sig)
        ta = A >= np.percentile(A, 60)
        dice = 2 * np.logical_and(ta, tb).sum() / max(int(ta.sum() + tb.sum()), 1)
        r1 = pear(a, dl)
        r4 = pear(block_mean(a, 4), block_mean(dl, 4))
        r16 = pear(block_mean(a, 16), block_mean(dl, 16))
        rows.append((o, dice, r1, r4, r16, r16 - r1))
        log(f"    {o:<20}{dice:8.4f}{r1:+10.4f}{r4:+10.4f}{r16:+10.4f}{r16-r1:+10.4f}")
    rows.sort(key=lambda r: -r[4])
    best = rows[0]
    margin = best[4] - rows[1][4] if len(rows) > 1 else float("inf")
    log(f"  best = {best[0]}   r@16 = {best[4]:+.4f}   margin = {margin:+.4f}   "
        f"trend = {best[5]:+.4f}   tissue Dice = {best[1]:.4f}")
    if margin < args.min_margin:
        log(f"  WARNING: margin {margin:+.4f} < --min-margin {args.min_margin}; "
            "inspect a crop before trusting this")
    g = gt_stats(paths["meta"], um)
    banner(f"  GT cells = {g['n']:,}   median bbox {g.get('bbox_med', float('nan')):.2f} um "
           f"(IQR {g.get('bbox_iqr',(0,0))[0]:.2f}-{g.get('bbox_iqr',(0,0))[1]:.2f})")
    if "diam_px" in g:
        log(f"  -> suggested diameter {g['diam_px']:.0f} px at {um:.4f} um/px")
    (out / "orientation.json").write_text(json.dumps(dict(
        orientation=best[0], r16=best[4], margin=margin, dice=best[1],
        stride=s, version=VERSION)))
    log(f"decision: --orientation {best[0]}")
    return best[0]


def resolve_orientation(paths, args, out):
    if args.orientation != "auto":
        log(f"orientation forced to {args.orientation}")
        return args.orientation
    oj = out / "orientation.json"
    if oj.exists():
        o = json.loads(oj.read_text())["orientation"]
        log(f"orientation from cached orientation.json: {o}")
        return o
    log("orientation not given and not cached; running detection")
    return detect_orientation(paths, args, out)


# ----------------------------------------------------------------------------
# stage: prep
# ----------------------------------------------------------------------------

def stage_prep(paths, args, out):
    import pandas as pd
    from scipy.ndimage import distance_transform_edt
    o = resolve_orientation(paths, args, out)
    banner("BUILD ROI")
    M, um = read_transform(paths["transform"])
    mos = open_mosaic(paths["dapi"])
    H, W = mos.shape
    Ho, Wo = oriented_shape(H, W, o)
    R = args.roi_size
    if R > min(Ho, Wo):
        raise ValueError(f"--roi-size {R} exceeds the oriented mosaic {Ho}x{Wo}")
    y0, x0 = (Ho - R) // 2, (Wo - R) // 2
    log(f"  oriented mosaic {Ho} x {Wo}   ROI y[{y0},{y0+R}) x[{x0},{x0+R})   "
        f"(centre {R}x{R} px = {R*um:.0f} um)")

    meta_path = out / "roi_meta.json"
    inp_path = out / "roi_input.npy"
    if inp_path.exists() and meta_path.exists() and not args.force_prep:
        old = json.loads(meta_path.read_text())
        same = (old.get("orientation") == o and old.get("roi") == [y0, x0, R]
                and abs(old.get("heatmap_sigma", -1) - args.heatmap_sigma) < 1e-9)
        if same:
            log("  cached roi_input.npy matches (orientation / ROI / sigma); "
                "use --force-prep to rebuild")
            return
        log("  cache exists but parameters differ -> rebuilding")

    dapi = oriented_crop(mos, o, y0, x0, R, R).astype(np.float32)
    p1, p995 = np.percentile(dapi, [1, 99.5])
    log(f"  DAPI p1={p1:.1f} p99.5={p995:.1f} mean={dapi.mean():.2f}")
    dapi_u8 = np.clip((dapi - p1) / max(p995 - p1, 1e-6) * 255, 0, 255).astype(np.uint8)
    del dapi

    xs, ys, gs = [], [], []
    for ch in pd.read_csv(paths["tx"], usecols=["global_x", "global_y", "gene"],
                          chunksize=5_000_000):
        gx = ch["global_x"].to_numpy(np.float64)
        gy = ch["global_y"].to_numpy(np.float64)
        mx = M[0, 0] * gx + M[0, 1] * gy + M[0, 2]
        my = M[1, 0] * gx + M[1, 1] * gy + M[1, 2]
        yo, xo = raw_to_oriented(my, mx, o, H, W)
        yl, xl = yo - y0, xo - x0
        ok = (yl >= 0) & (yl < R) & (xl >= 0) & (xl < R)
        xs.append(xl[ok].astype(np.float32))
        ys.append(yl[ok].astype(np.float32))
        gs.append(ch["gene"].to_numpy()[ok])
    xl = np.concatenate(xs) if xs else np.zeros(0, np.float32)
    yl = np.concatenate(ys) if ys else np.zeros(0, np.float32)
    gn = np.concatenate(gs) if gs else np.zeros(0, object)
    dens = len(xl) / (R * um) ** 2
    n_genes = len(pd.unique(pd.Series(gn))) if len(gn) else 0
    log(f"  ROI transcripts {len(xl):,}  genes {n_genes}  density {dens:.2f} tx/um^2")

    with gzip.open(out / "spots.csv.gz", "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["spotX", "spotY", "gene"])
        for a, b, g in zip(xl, yl, gn):
            w.writerow([f"{a:.1f}", f"{b:.1f}", g])

    imp = np.zeros((R, R), bool)
    if len(xl):
        iy = np.clip(np.rint(yl).astype(np.int64), 0, R - 1)
        ix = np.clip(np.rint(xl).astype(np.int64), 0, R - 1)
        imp[iy, ix] = True
    sig = args.heatmap_sigma
    if imp.any():
        d = distance_transform_edt(~imp).astype(np.float32)
        heat = np.exp(-(d * d) / (2.0 * sig * sig))
        del d
    else:
        heat = np.zeros((R, R), np.float32)
        log("  WARNING: no transcripts landed in the ROI")
    log(f"  heat map (sigma {sig}) mean {heat.mean()*255:.2f}  "
        f"frac>0.9 {(heat > 0.9).mean():.3f}")
    heat_u8 = np.rint(heat * 255).astype(np.uint8)
    del heat

    g = gt_stats(paths["meta"], um)
    cx, cy = g["centers_um"]
    mx = M[0, 0] * cx + M[0, 1] * cy + M[0, 2]
    my = M[1, 0] * cx + M[1, 1] * cy + M[1, 2]
    yo, xo = raw_to_oriented(my, mx, o, H, W)
    in_roi = int(np.sum((yo >= y0) & (yo < y0 + R) & (xo >= x0) & (xo < x0 + R)))
    log(f"  GT cells (whole slide) = {g['n']:,}   median bbox "
        f"{g.get('bbox_med', float('nan')):.2f} um "
        f"(IQR {g.get('bbox_iqr',(0,0))[0]:.2f}-{g.get('bbox_iqr',(0,0))[1]:.2f})")
    log(f"  GT cells inside this ROI = {in_roi:,}   <- detection-rate denominator")
    if "diam_px" in g:
        log(f"  -> suggested diameter {g['diam_px']:.0f} px at {um:.4f} um/px")

    inp = np.stack([dapi_u8, heat_u8], axis=-1)
    np.save(inp_path, inp)
    meta = dict(version=VERSION, dataset=args.dataset, orientation=o,
                roi=[y0, x0, R], mosaic=[int(H), int(W)], um_per_px=um,
                heatmap_sigma=sig, n_spots=int(len(xl)), n_genes=int(n_genes),
                density_tx_per_um2=dens, gt_cells_slide=int(g["n"]),
                gt_cells_in_roi=in_roi,
                gt_median_bbox_um=g.get("bbox_med"),
                suggested_diameter_px=g.get("diam_px"),
                dapi=str(paths["dapi"]), tx=str(paths["tx"]),
                meta_csv=str(paths["meta"]))
    meta_path.write_text(json.dumps(meta, indent=2))
    log(f"input cached -> {inp_path}  {inp.shape}")


# ----------------------------------------------------------------------------
# inference machinery
# ----------------------------------------------------------------------------

def tile_starts(n, tile, overlap):
    if tile >= n:
        return [0]
    step = max(1, tile - overlap)
    s = list(range(0, n - tile + 1, step))
    if s[-1] != n - tile:
        s.append(n - tile)
    return s


def merge_tile(G, m, y0, x0, thr, next_id):
    """Write local instance mask m into global label image G at (y0,x0).

    Local ids overlapping an existing global id by
    intersection / min(local-area-in-overlap, global-area-in-window) >= thr
    adopt that global id; everything else gets a fresh id.  Where a pixel is
    already labelled, the earlier label wins."""
    t_h, t_w = m.shape
    sub = G[y0:y0 + t_h, x0:x0 + t_w]
    loc = m.astype(np.int64)
    if loc.max() == 0:
        return next_id
    lut = np.zeros(int(loc.max()) + 1, np.int64)
    ov = sub > 0
    if ov.any():
        lm = loc[ov]
        gm = sub[ov].astype(np.int64)
        sel = lm > 0
        if sel.any():
            pairs = np.stack([lm[sel], gm[sel]])
            uq, cnt = np.unique(pairs, axis=1, return_counts=True)
            area_loc_ov = np.bincount(lm[sel])
            area_glob_win = np.bincount(sub[ov].astype(np.int64).ravel())
            best_c = {}
            for (l, g), c in zip(uq.T, cnt):
                al = area_loc_ov[l] if l < len(area_loc_ov) else c
                ag = area_glob_win[g] if g < len(area_glob_win) else c
                if c / max(min(al, ag), 1) >= thr and c > best_c.get(l, (0, 0))[1]:
                    best_c[l] = (g, c)
            for l, (g, _) in best_c.items():
                lut[l] = g
    for l in np.unique(loc):
        if l > 0 and lut[l] == 0:
            lut[l] = next_id
            next_id += 1
    write = (loc > 0) & (sub == 0)
    sub[write] = lut[loc[write]]
    return next_id


def pick_gpu(spec):
    if spec != "auto":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(spec)
        log(f"physical GPU {spec} -> visible as device 0")
        return
    try:
        outp = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"], text=True)
        rows = [r.split(", ") for r in outp.strip().splitlines()]
        log("GPUs:")
        for i, free, util in rows:
            log(f"  [{i}] free {int(free):>6} MB   util {util}%")
        best = max(rows, key=lambda r: int(r[1]))[0]
        os.environ["CUDA_VISIBLE_DEVICES"] = best
        log(f"auto-selected GPU [{best}]")
    except Exception as e:
        log(f"nvidia-smi unavailable ({e}); leaving CUDA_VISIBLE_DEVICES as-is")


def resolve_model(args):
    if args.model_file:
        p = Path(args.model_file)
        if p.exists():
            return p
        raise FileNotFoundError(p)
    fp = Path(args.train_dir) / "FINAL_MODEL_PATH.txt"
    if fp.exists():
        p = Path(fp.read_text().strip())
        if not p.is_absolute():
            p = Path(args.train_dir) / p
        if p.exists():
            return p
        log(f"  FINAL_MODEL_PATH.txt points to {p} which is missing; searching model/")
    cands = [c for c in (Path(args.train_dir) / "model").rglob("*") if c.is_file()]
    if not cands:
        raise FileNotFoundError(f"no model file under {args.train_dir}")
    return max(cands, key=lambda c: c.stat().st_mtime)


def load_model(args):
    pick_gpu(args.gpu)
    repo = str(Path(args.repo))
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import models  # noqa
    mp = resolve_model(args)
    log(f"model = {mp}")
    ik = set(inspect.signature(models.GeneSegModel.__init__).parameters)
    kw = {}
    if "gpu" in ik:
        kw["gpu"] = True
    if "pretrained_model" in ik:
        kw["pretrained_model"] = str(mp)
    if "nchan" in ik:
        kw["nchan"] = 2
    if "diam_mean" in ik:
        kw["diam_mean"] = args.diam_mean
    model = models.GeneSegModel(**kw)
    for attr in ("diam_mean", "diam_labels"):
        if hasattr(model, attr):
            log(f"  model.{attr} = {getattr(model, attr)}")
    ekeys = set(inspect.signature(model.eval).parameters)
    return model, ekeys


def run_inference(inp, model, ekeys, diameter, conf, flow, normalize, args):
    R = inp.shape[0]
    ys = tile_starts(R, args.tile, args.tile_overlap)
    xs = tile_starts(R, args.tile, args.tile_overlap)
    n_tiles = len(ys) * len(xs)
    log(f"sliding window {args.tile} overlap {args.tile_overlap} -> "
        f"{len(ys)}x{len(xs)} = {n_tiles} tiles")
    G = np.zeros((R, R), np.int64)
    nid = 1
    t0 = time.time()
    kw = {}
    if "diameter" in ekeys:
        kw["diameter"] = diameter
    if "flow_threshold" in ekeys:
        kw["flow_threshold"] = flow
    if "cellprob_threshold" in ekeys:
        kw["cellprob_threshold"] = conf
    if "normalize" in ekeys:
        kw["normalize"] = bool(normalize)
    if "channels" in ekeys:
        kw["channels"] = None
    done = 0
    for y0 in ys:
        for x0 in xs:
            tile = inp[y0:y0 + args.tile, x0:x0 + args.tile]
            outp = model.eval(tile, **kw)
            m = np.asarray(outp[0] if isinstance(outp, (list, tuple)) else outp)
            nid = merge_tile(G, m, y0, x0, args.merge_thr, nid)
            done += 1
            if done % max(1, n_tiles // 10) == 0 or done == n_tiles:
                log(f"  {done}/{n_tiles} tiles ({(time.time()-t0)/60:.1f} min)")
    return G, time.time() - t0


def area_filter_relabel(G, min_area_px):
    areas = np.bincount(G.ravel())
    kill = np.where(areas < min_area_px)[0]
    kill = kill[kill > 0]
    if len(kill):
        lut = np.arange(len(areas), dtype=np.int64)
        lut[kill] = 0
        G = lut[G]
    ids = np.unique(G)
    ids = ids[ids > 0]
    lut = np.zeros(int(G.max()) + 1, np.int64)
    lut[ids] = np.arange(1, len(ids) + 1)
    return lut[G], len(ids)


def mask_report(G, um, expected=None):
    areas = np.bincount(G.ravel())[1:]
    areas = areas[areas > 0]
    cov = float((G > 0).mean())
    med_px = float(np.median(areas)) if len(areas) else 0.0
    med_um2 = med_px * um * um
    diam = 2 * math.sqrt(med_um2 / math.pi) if med_px else 0.0
    log(f"  cells            {len(areas):,}")
    log(f"  foreground       {cov*100:.2f}%")
    log(f"  median area      {med_px:.1f} px = {med_um2:.1f} um^2")
    log(f"  median diameter  {diam:.2f} um")
    warn = []
    if expected:
        log(f"  detected / expected in ROI   {len(areas):,} / {expected:,} "
            f"= {len(areas)/expected:.3f}")
    if diam and diam < 3:
        warn.append(f"median diameter {diam:.2f} um < 3 um -> fragments, not cells")
    if cov < 0.03:
        warn.append(f"foreground {cov*100:.2f}% < 3%")
    for w in warn:
        log(f"  WARNING: {w}")
    return dict(cells=int(len(areas)), foreground=cov, median_area_px=med_px,
                median_area_um2=med_um2, median_diam_um=diam, warnings=warn)


# ----------------------------------------------------------------------------
# stages: infer / scan
# ----------------------------------------------------------------------------

def load_meta_input(out, args):
    meta_path = out / "roi_meta.json"
    inp_path = out / "roi_input.npy"
    if not meta_path.exists() or not inp_path.exists():
        raise FileNotFoundError("roi_meta.json / roi_input.npy missing -- run --stage prep first")
    meta = json.loads(meta_path.read_text())
    if abs(meta.get("heatmap_sigma", -1) - args.heatmap_sigma) > 1e-9:
        log(f"  NOTE: cached heat map used sigma {meta.get('heatmap_sigma')}, "
            f"--heatmap-sigma {args.heatmap_sigma} is ignored here "
            f"(rerun prep --force-prep to change it)")
    return meta, np.load(inp_path)


def stage_infer(paths, args, out):
    meta, inp = load_meta_input(out, args)
    um = meta["um_per_px"]
    banner(f"INFER  orientation={meta['orientation']}  diameter={args.diameter}  "
           f"conf={args.confidence_threshold}  flow={args.flow_threshold}  "
           f"normalize={args.normalize}")
    model, ekeys = load_model(args)
    G, secs = run_inference(inp, model, ekeys, args.diameter,
                            args.confidence_threshold, args.flow_threshold,
                            args.normalize, args)
    min_px = int(round(args.min_area_um2 / (um * um)))
    G, _ = area_filter_relabel(G, min_px)
    banner(f"DONE  {args.dataset}")
    rep = mask_report(G, um, expected=meta.get("gt_cells_in_roi"))
    import tifffile
    mp = out / "segmentation_mask.tif"
    tifffile.imwrite(mp, G.astype(np.uint32))
    log(f"  mask -> {mp}")
    qc = dict(version=VERSION, dataset=args.dataset,
              orientation=meta["orientation"], roi=meta["roi"],
              um_per_px=um, heatmap_sigma=meta["heatmap_sigma"],
              diameter=args.diameter, confidence_threshold=args.confidence_threshold,
              flow_threshold=args.flow_threshold, normalize=int(args.normalize),
              tile=args.tile, tile_overlap=args.tile_overlap,
              merge_thr=args.merge_thr, min_area_um2=args.min_area_um2,
              seconds=secs, gt_cells_in_roi=meta.get("gt_cells_in_roi"),
              detected_fraction=(rep["cells"] / meta["gt_cells_in_roi"]
                                 if meta.get("gt_cells_in_roi") else None),
              **rep)
    (out / "run_qc.json").write_text(json.dumps(qc, indent=2))
    banner("=")


def stage_scan(paths, args, out):
    meta, inp = load_meta_input(out, args)
    um = meta["um_per_px"]
    expected = meta.get("gt_cells_in_roi")
    model, ekeys = load_model(args)
    scan_path = out / "scan_results.csv"
    new = not scan_path.exists()
    rows = []
    with open(scan_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["diameter", "confidence_threshold", "flow_threshold",
                        "normalize", "seconds", "cells", "foreground_rate",
                        "median_area_px", "median_area_um2", "median_diam_um"])
        for d in args.diameters:
            for c in args.confidence_thresholds:
                for fl in args.flow_thresholds:
                    banner(f"SCAN diameter={d} conf={c} flow={fl}")
                    G, secs = run_inference(inp, model, ekeys, d, c, fl,
                                            args.normalize, args)
                    min_px = int(round(args.min_area_um2 / (um * um)))
                    G, _ = area_filter_relabel(G, min_px)
                    rep = mask_report(G, um, expected=expected)
                    w.writerow([d, c, fl, int(args.normalize), f"{secs:.1f}",
                                rep["cells"], f"{rep['foreground']:.8f}",
                                rep["median_area_px"],
                                f"{rep['median_area_um2']:.4f}",
                                f"{rep['median_diam_um']:.4f}"])
                    f.flush()
                    rows.append((d, c, fl, rep["cells"], rep["median_diam_um"]))
    if expected:
        rows.sort(key=lambda r: abs(r[3] / expected - 1))
        log("closest to the ROI's GT cell count first:")
        for d, c, fl, n, dm in rows:
            log(f"  d={d} conf={c} flow={fl}: {n:,} cells "
                f"({n/expected:.3f} of {expected:,})  diam {dm:.2f} um")
    log(f"scan table -> {scan_path}")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=VERSION)
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--stage", default="all",
                    choices=["orient", "prep", "scan", "infer", "all"])
    ap.add_argument("--data-root")
    ap.add_argument("--out-dir")
    ap.add_argument("--dapi")
    ap.add_argument("--roi-size", type=int, default=10000,
                    help="centre window edge in mosaic px (fixed benchmark ROI)")
    ap.add_argument("--orientation", default="auto", choices=["auto"] + ORIENTS)
    ap.add_argument("--orient-stride", type=int, default=16)
    ap.add_argument("--min-margin", type=float, default=0.02)
    ap.add_argument("--heatmap-sigma", type=float, default=3.0)
    ap.add_argument("--normalize", type=int, default=1, choices=[0, 1])
    ap.add_argument("--diameter", type=float, default=85.0)
    ap.add_argument("--confidence-threshold", type=float, default=-4.0)
    ap.add_argument("--flow-threshold", type=float, default=0.8)
    ap.add_argument("--diameters", type=float, nargs="+", default=[64, 85, 107])
    ap.add_argument("--confidence-thresholds", type=float, nargs="+", default=[-4.0])
    ap.add_argument("--flow-thresholds", type=float, nargs="+", default=[0.8])
    ap.add_argument("--tile", type=int, default=2048)
    ap.add_argument("--tile-overlap", type=int, default=256)
    ap.add_argument("--merge-thr", type=float, default=0.3)
    ap.add_argument("--min-area-um2", type=float, default=10.0)
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--train-dir", default=DEFAULT_TRAIN_DIR)
    ap.add_argument("--model-file")
    ap.add_argument("--diam-mean", type=float, default=85.0)
    ap.add_argument("--gpu", default="auto")
    ap.add_argument("--force-prep", action="store_true")
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    out = Path(args.out_dir or cfg["out"])
    global log
    log = setup_log(out, args.stage)
    banner(f"GeneSegNet x MERFISH {args.dataset}   v{VERSION}   stage={args.stage}")
    paths = resolve_paths(cfg, args)
    for k in ("tx", "meta", "transform", "dapi"):
        log(f"  {k:<9} = {paths[k]}")

    if args.stage == "orient":
        detect_orientation(paths, args, out)
    elif args.stage == "prep":
        stage_prep(paths, args, out)
    elif args.stage == "scan":
        stage_scan(paths, args, out)
    elif args.stage == "infer":
        stage_infer(paths, args, out)
    else:  # all
        stage_prep(paths, args, out)
        stage_infer(paths, args, out)


if __name__ == "__main__":
    main()