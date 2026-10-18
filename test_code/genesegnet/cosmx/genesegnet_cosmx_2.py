#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
genesegnet_cosmx_2.py
=====================

CosMx-only GeneSegNet inference runner.

This file is now an INFERENCE SCRIPT, not the old prep/train wrapper.

Use cases
---------
1) CosMx2 held-out FOV48 / FOV261:
   --input-root /data/qiuyijia/genesegnet_cosmx_2/split/test
   --fovs 48,261

2) Other CosMx datasets:
   point --input-root to another prepared GeneSegNet CosMx inference/test set,
   change --fovs / --fov-size / --pixel-size / --out as needed.

Expected prepared files
-----------------------
The input tree must contain, recursively:

  cosmx..._fov048_y0000_x0000_image.tif
  matching heatmap:
      *_gaumap_all.tif
      or *_gaumap.tif

The script:
  * discovers image/heatmap pairs
  * selects requested FOVs
  * concatenates morphology + heatmap channels exactly like GeneSegNet TEST
  * loads a trained GeneSegNet model directly
  * runs GeneSegModel.eval()
  * saves per-tile masks
  * stitches overlapping tiles into one full-FOV mask
  * writes QC summaries

No GT information is used to tune inference parameters.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile


# ---------------------------------------------------------------------
# Defaults for current CosMx2
# ---------------------------------------------------------------------

REPO_DEFAULT = "/data/qiuyijia/GeneSegNet"

INPUT_DEFAULT = "/data/qiuyijia/genesegnet_cosmx_2/split/test"

MODEL_DEFAULT = (
    "/data/qiuyijia/genesegnet_cosmx_2_train/model/models/"
    "GeneSegNet_residual_on_style_off_concatenation_off_model_2026_08_23_12_21_16.929985"
)

OUT_DEFAULT = "/data/qiuyijia/genesegnet_cosmx_2_infer"

FOVS_DEFAULT = "48,261"

# CosMx SMI full-FOV raster in this dataset.
FOV_SIZE_DEFAULT = 4256

# CosMx mosaic pixel size used in the current data.
PIXEL_SIZE_DEFAULT = 0.12


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def log(msg=""):
    print(time.strftime("[%H:%M:%S]"), msg, flush=True)


def banner(msg):
    log("=" * 96)
    log(msg)
    log("=" * 96)


def parse_fovs(s):
    vals = []
    for x in str(s).split(","):
        x = x.strip()
        if x:
            vals.append(int(x))
    if not vals:
        raise ValueError("--fovs is empty")
    return vals


def ensure_repo(repo: Path):
    pkg = repo / "GeneSegNet"

    required = [
        pkg / "models.py",
        pkg / "utils.py",
        pkg / "dynamics.py",
    ]

    for p in required:
        if not p.exists():
            raise FileNotFoundError(p)

    # GeneSegNet imports sibling files as top-level modules.
    if str(pkg) not in sys.path:
        sys.path.insert(0, str(pkg))
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    return pkg


def import_geneseg(repo: Path):
    ensure_repo(repo)

    import models
    import utils
    import fastremap

    return models, utils, fastremap


def newest_model(model_dir: Path):
    files = [
        p for p in model_dir.rglob("*")
        if p.is_file() and p.stat().st_size > 1_000_000
    ]

    if not files:
        return None

    return max(files, key=lambda p: p.stat().st_mtime)


def resolve_model(model_arg):
    p = Path(model_arg)

    if p.is_file():
        return p

    if p.is_dir():
        q = newest_model(p)
        if q is None:
            raise FileNotFoundError(
                f"no model file >1 MB under {p}"
            )
        return q

    raise FileNotFoundError(p)


# ---------------------------------------------------------------------
# CosMx tile discovery
# ---------------------------------------------------------------------

FOV_RE = re.compile(
    r"fov(?P<fov>\d+)",
    re.IGNORECASE
)

YX_RE = re.compile(
    r"_y(?P<y>\d+)_x(?P<x>\d+)",
    re.IGNORECASE
)


@dataclass
class Tile:
    fov: int
    y: int
    x: int
    key: str
    image: Path
    heatmap: Path


def image_key(name):
    lower = name.lower()

    for suffix in (
        "_image.tif",
        "_image.tiff",
    ):
        if lower.endswith(suffix):
            return name[:-len(suffix)]

    return Path(name).stem


def heatmap_key(name):
    lower = name.lower()

    suffixes = (
        "_gaumap_all.tif",
        "_gaumap_all.tiff",
        "_gaumap.tif",
        "_gaumap.tiff",
        "_heatmap_all.tif",
        "_heatmap_all.tiff",
        "_heatmap.tif",
        "_heatmap.tiff",
    )

    for suffix in suffixes:
        if lower.endswith(suffix):
            return name[:-len(suffix)]

    return Path(name).stem


def discover_heatmaps(root: Path):
    idx = {}

    tifs = (
        list(root.rglob("*.tif"))
        + list(root.rglob("*.tiff"))
    )

    for p in tifs:
        low = p.name.lower()
        parent_low = str(p.parent).lower()

        if (
            "gaumap" not in low
            and "heatmap" not in low
            and "heatmap" not in parent_low
        ):
            continue

        key = heatmap_key(p.name)

        score = 0

        if "gaumap_all" in low:
            score += 100
        elif "gaumap" in low:
            score += 80

        if "heatmap_all" in parent_low:
            score += 20

        old = idx.get(key)

        if old is None or score > old[0]:
            idx[key] = (score, p)

    return idx


def discover_tiles(root: Path, wanted_fovs):
    hidx = discover_heatmaps(root)

    images = []

    tifs = (
        list(root.rglob("*.tif"))
        + list(root.rglob("*.tiff"))
    )

    for p in tifs:
        low = p.name.lower()

        if (
            not low.endswith("_image.tif")
            and not low.endswith("_image.tiff")
        ):
            continue

        key = image_key(p.name)

        mf = FOV_RE.search(key)
        myx = YX_RE.search(key)

        if mf is None:
            continue

        if myx is None:
            raise RuntimeError(
                f"CosMx image lacks y/x tile origin: {p.name}"
            )

        fov = int(mf.group("fov"))

        if fov not in wanted_fovs:
            continue

        y = int(myx.group("y"))
        x = int(myx.group("x"))

        hm = hidx.get(key)

        if hm is None:
            # fallback fuzzy match
            cands = [
                (score, hp)
                for hk, (score, hp) in hidx.items()
                if hk == key
                or hk.startswith(key)
                or key.startswith(hk)
            ]

            if cands:
                cands.sort(
                    key=lambda z: z[0],
                    reverse=True
                )
                hm = cands[0]

        if hm is None:
            raise RuntimeError(
                f"no matching heatmap for {p}"
            )

        images.append(
            Tile(
                fov=fov,
                y=y,
                x=x,
                key=key,
                image=p,
                heatmap=hm[1],
            )
        )

    images.sort(
        key=lambda t: (
            t.fov,
            t.y,
            t.x
        )
    )

    if not images:
        raise RuntimeError(
            f"no usable CosMx tiles found under {root} "
            f"for FOVs {wanted_fovs}"
        )

    return images


# ---------------------------------------------------------------------
# Array layout
# ---------------------------------------------------------------------

def to_hwc(a, kind):
    a = np.asarray(a)

    if a.ndim == 2:
        return a[..., None]

    if a.ndim != 3:
        raise RuntimeError(
            f"{kind}: unsupported shape {a.shape}"
        )

    # HWC
    if (
        a.shape[-1] <= 16
        and a.shape[0] > 16
        and a.shape[1] > 16
    ):
        return a

    # CHW
    if (
        a.shape[0] <= 16
        and a.shape[1] > 16
        and a.shape[2] > 16
    ):
        return np.moveaxis(
            a,
            0,
            -1
        )

    raise RuntimeError(
        f"{kind}: ambiguous 3D layout {a.shape}"
    )


def load_input(tile: Tile):
    image = to_hwc(
        tifffile.imread(tile.image),
        "image"
    )

    heatmap = to_hwc(
        tifffile.imread(tile.heatmap),
        "heatmap"
    )

    if image.shape[:2] != heatmap.shape[:2]:
        raise RuntimeError(
            f"{tile.key}: image={image.shape} "
            f"heatmap={heatmap.shape}"
        )

    # Same logic used by GeneSeg_train.py / GeneSeg_test.py:
    # morphology channels + heatmap channels.
    x = np.concatenate(
        [image, heatmap],
        axis=-1
    )

    return x


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

def build_model(
    models,
    args,
    model_path,
    nchan
):
    device, gpu = models.assign_device(
        use_torch=True,
        gpu=True,
        device=args.gpu
    )

    log(
        f"GeneSegNet device={device}, "
        f"gpu={gpu}"
    )

    model = models.GeneSegModel(
        gpu=gpu,
        device=device,
        pretrained_model=str(model_path),
        model_type=None,
        diam_mean=args.diam_mean,
        residual_on=not args.residual_off,
        style_on=args.style_on,
        concatenation=args.concatenation,
        net_avg=False,
        nchan=nchan
    )

    return model


def infer_tile(
    model,
    utils,
    fastremap,
    x,
    args
):
    result = model.eval(
        x,
        batch_size=args.batch_size,
        channels=None,
        channel_axis=-1,

        # 0 means use checkpoint diam_labels.
        diameter=args.diameter,

        do_3D=False,
        net_avg=False,
        augment=False,

        tile=True,
        tile_overlap=args.network_tile_overlap,

        resample=True,
        interp=True,

        flow_threshold=args.flow_threshold,
        confidence_threshold=args.confidence_threshold,

        compute_masks=True,

        min_size=args.min_size,
        stitch_threshold=0.0
    )

    if (
        not isinstance(
            result,
            (tuple, list)
        )
        or len(result) < 1
    ):
        raise RuntimeError(
            "unexpected GeneSegModel.eval output"
        )

    mask = np.asarray(
        result[0]
    )

    if (
        mask.ndim == 3
        and mask.shape[0] == 1
    ):
        mask = mask[0]

    if mask.ndim != 2:
        raise RuntimeError(
            f"expected 2D mask, "
            f"got {mask.shape}"
        )

    mask = (
        utils
        .fill_holes_and_remove_small_masks(
            mask.astype(
                np.int32,
                copy=False
            ),
            min_size=args.min_size
        )
    )

    mask = (
        fastremap
        .renumber(
            mask,
            in_place=True
        )[0]
    )

    return np.asarray(
        mask,
        dtype=np.uint32
    )


# ---------------------------------------------------------------------
# Cross-tile stitching
# ---------------------------------------------------------------------

class UnionFind:
    def __init__(self):
        self.parent = {}

    def add(self, x):
        self.parent.setdefault(
            x,
            x
        )

    def find(self, x):
        self.add(x)

        if self.parent[x] != x:
            self.parent[x] = self.find(
                self.parent[x]
            )

        return self.parent[x]

    def union(self, a, b):
        if a == 0 or b == 0:
            return

        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return

        if ra < rb:
            self.parent[rb] = ra
        else:
            self.parent[ra] = rb


def link_overlap(
    existing,
    incoming,
    uf,
    min_pixels,
    min_fraction
):
    a = existing.ravel().astype(
        np.int64
    )

    b = incoming.ravel().astype(
        np.int64
    )

    keep = (
        (a > 0)
        & (b > 0)
    )

    if not np.any(keep):
        return 0

    a = a[keep]
    b = b[keep]

    pairs = np.stack(
        [a, b],
        axis=1
    )

    uniq, counts = np.unique(
        pairs,
        axis=0,
        return_counts=True
    )

    count_a = dict(
        zip(
            *np.unique(
                a,
                return_counts=True
            )
        )
    )

    count_b = dict(
        zip(
            *np.unique(
                b,
                return_counts=True
            )
        )
    )

    linked = 0

    for (
        pair,
        inter
    ) in zip(
        uniq,
        counts
    ):
        ga = int(pair[0])
        gb = int(pair[1])

        if inter < min_pixels:
            continue

        denom = min(
            count_a[ga],
            count_b[gb]
        )

        frac = (
            inter
            / max(
                denom,
                1
            )
        )

        if frac >= min_fraction:
            uf.union(
                ga,
                gb
            )

            linked += 1

    return linked


def stitch_fov(
    records,
    fov_size,
    args
):
    canvas = np.zeros(
        (
            fov_size,
            fov_size
        ),
        dtype=np.uint32
    )

    uf = UnionFind()

    next_global_id = 1
    total_links = 0

    for rec in sorted(
        records,
        key=lambda r: (
            r["y"],
            r["x"]
        )
    ):
        mask = rec["mask"]

        y0 = rec["y"]
        x0 = rec["x"]

        h, w = mask.shape

        y1 = min(
            y0 + h,
            fov_size
        )

        x1 = min(
            x0 + w,
            fov_size
        )

        if (
            y0 < 0
            or x0 < 0
            or y0 >= fov_size
            or x0 >= fov_size
        ):
            raise RuntimeError(
                f"tile origin outside FOV: "
                f"y={y0} x={x0}"
            )

        local = mask[
            : y1-y0,
            : x1-x0
        ]

        lut = np.zeros(
            int(local.max()) + 1,
            dtype=np.uint32
        )

        for lab in np.unique(local):
            if lab == 0:
                continue

            gid = next_global_id
            next_global_id += 1

            lut[int(lab)] = gid
            uf.add(gid)

        incoming = lut[local]

        region = canvas[
            y0:y1,
            x0:x1
        ]

        total_links += link_overlap(
            region,
            incoming,
            uf,
            min_pixels=args.stitch_min_pixels,
            min_fraction=args.stitch_fraction
        )

        # Existing mosaic owns conflict pixels.
        # New tile only fills currently empty pixels.
        empty = region == 0
        region[empty] = incoming[empty]

    ids = np.unique(canvas)
    ids = ids[ids > 0]

    if len(ids) == 0:
        return canvas, total_links

    remap = np.zeros(
        int(canvas.max()) + 1,
        dtype=np.uint32
    )

    root_to_final = {}
    next_final = 1

    for gid in ids:
        root = uf.find(
            int(gid)
        )

        if root not in root_to_final:
            root_to_final[root] = (
                next_final
            )

            next_final += 1

        remap[int(gid)] = (
            root_to_final[root]
        )

    stitched = remap[canvas]

    return (
        stitched,
        total_links
    )


# ---------------------------------------------------------------------
# QC
# ---------------------------------------------------------------------

def mask_qc(
    mask,
    pixel_size
):
    labels, counts = np.unique(
        mask,
        return_counts=True
    )

    area = counts[
        labels > 0
    ].astype(float)

    if len(area) == 0:
        return {
            "cells": 0,
            "foreground": 0.0,
            "area_median_px": np.nan,
            "area_p5_px": np.nan,
            "area_p95_px": np.nan,
            "diameter_median_um": np.nan,
        }

    med = float(
        np.median(area)
    )

    diameter = (
        2
        * np.sqrt(
            med / np.pi
        )
        * pixel_size
    )

    return {
        "cells": int(
            len(area)
        ),
        "foreground": float(
            (mask > 0).mean()
        ),
        "area_median_px": med,
        "area_p5_px": float(
            np.percentile(
                area,
                5
            )
        ),
        "area_p95_px": float(
            np.percentile(
                area,
                95
            )
        ),
        "diameter_median_um": float(
            diameter
        ),
    }


def save_json(
    path,
    obj
):
    Path(path).write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False
        )
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        formatter_class=(
            argparse
            .ArgumentDefaultsHelpFormatter
        )
    )

    # Main paths.
    ap.add_argument(
        "--input-root",
        default=INPUT_DEFAULT,
        help=(
            "prepared CosMx inference/test root; "
            "must contain *_image.tif and heatmaps"
        )
    )

    ap.add_argument(
        "--model",
        default=MODEL_DEFAULT,
        help=(
            "GeneSegNet model file, or directory "
            "containing model files"
        )
    )

    ap.add_argument(
        "--out",
        default=OUT_DEFAULT
    )

    ap.add_argument(
        "--repo",
        default=REPO_DEFAULT
    )

    ap.add_argument(
        "--dataset-name",
        default="cosmx_2"
    )

    # CosMx geometry.
    ap.add_argument(
        "--fovs",
        default=FOVS_DEFAULT,
        help="comma-separated CosMx FOV IDs"
    )

    ap.add_argument(
        "--fov-size",
        type=int,
        default=FOV_SIZE_DEFAULT,
        help="full CosMx FOV width/height in pixels"
    )

    ap.add_argument(
        "--pixel-size",
        type=float,
        default=PIXEL_SIZE_DEFAULT,
        help="micron per pixel"
    )

    # Device.
    ap.add_argument(
        "--gpu",
        type=int,
        default=0,
        help=(
            "GPU index visible inside this process"
        )
    )

    ap.add_argument(
        "--batch-size",
        type=int,
        default=4
    )

    # GeneSegNet inference parameters.
    ap.add_argument(
        "--diameter",
        type=float,
        default=0.0,
        help=(
            "0 = use checkpoint diam_labels"
        )
    )

    ap.add_argument(
        "--diam-mean",
        type=float,
        default=34.0
    )

    ap.add_argument(
        "--flow-threshold",
        type=float,
        default=0.4
    )

    ap.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.9
    )

    ap.add_argument(
        "--min-size",
        type=int,
        default=300
    )

    ap.add_argument(
        "--network-tile-overlap",
        type=float,
        default=0.1
    )

    ap.add_argument(
        "--residual-off",
        action="store_true"
    )

    ap.add_argument(
        "--style-on",
        action="store_true"
    )

    ap.add_argument(
        "--concatenation",
        action="store_true"
    )

    # Cross prepared-tile stitching.
    ap.add_argument(
        "--stitch-min-pixels",
        type=int,
        default=20
    )

    ap.add_argument(
        "--stitch-fraction",
        type=float,
        default=0.5
    )

    ap.add_argument(
        "--force",
        action="store_true"
    )

    ap.add_argument(
        "--dry",
        action="store_true"
    )

    args = ap.parse_args()

    input_root = Path(
        args.input_root
    )

    repo = Path(
        args.repo
    )

    out = Path(
        args.out
    )

    fovs = parse_fovs(
        args.fovs
    )

    model_path = resolve_model(
        args.model
    )

    if not input_root.exists():
        raise FileNotFoundError(
            input_root
        )

    banner(
        "GeneSegNet CosMx inference"
    )

    log(
        f"dataset      = "
        f"{args.dataset_name}"
    )

    log(
        f"input root   = "
        f"{input_root}"
    )

    log(
        f"model        = "
        f"{model_path}"
    )

    log(
        f"output       = "
        f"{out}"
    )

    log(
        f"FOVs         = "
        f"{fovs}"
    )

    log(
        f"FOV size     = "
        f"{args.fov_size}"
    )

    log(
        f"pixel size   = "
        f"{args.pixel_size} um"
    )

    log(
        f"diameter     = "
        f"{args.diameter}"
    )

    log(
        f"flow thr     = "
        f"{args.flow_threshold}"
    )

    log(
        f"confidence   = "
        f"{args.confidence_threshold}"
    )

    log(
        "NOTE: GT is not used "
        "for inference parameter selection."
    )

    tiles = discover_tiles(
        input_root,
        fovs
    )

    for fov in fovs:
        q = [
            t for t in tiles
            if t.fov == fov
        ]

        log(
            f"FOV {fov}: "
            f"{len(q)} tiles"
        )

        for t in q:
            log(
                f"    "
                f"y={t.y:04d} "
                f"x={t.x:04d} "
                f"{t.image.name}"
            )

    if args.dry:
        banner(
            "DRY RUN DONE"
        )
        return

    out.mkdir(
        parents=True,
        exist_ok=True
    )

    tile_dir = (
        out
        / "tiles"
    )

    tile_dir.mkdir(
        exist_ok=True
    )

    fov_dir = (
        out
        / "fovs"
    )

    fov_dir.mkdir(
        exist_ok=True
    )

    models, utils, fastremap = (
        import_geneseg(
            repo
        )
    )

    first_input = load_input(
        tiles[0]
    )

    nchan = int(
        first_input.shape[-1]
    )

    log(
        f"first model input: "
        f"shape={first_input.shape} "
        f"dtype={first_input.dtype} "
        f"nchan={nchan}"
    )

    model = build_model(
        models,
        args,
        model_path,
        nchan
    )

    records = {
        f: []
        for f in fovs
    }

    tile_rows = []

    for i, tile in enumerate(
        tiles,
        start=1
    ):
        mask_path = (
            tile_dir
            / f"{tile.key}_mask.tif"
        )

        if (
            mask_path.exists()
            and not args.force
        ):
            mask = tifffile.imread(
                mask_path
            )

            log(
                f"[{i}/{len(tiles)}] "
                f"reuse {tile.key} "
                f"cells={int(mask.max())}"
            )

        else:
            if i == 1:
                x = first_input
            else:
                x = load_input(
                    tile
                )

            log(
                f"[{i}/{len(tiles)}] "
                f"infer {tile.key}: "
                f"{x.shape}"
            )

            tic = time.time()

            mask = infer_tile(
                model,
                utils,
                fastremap,
                x,
                args
            )

            tifffile.imwrite(
                mask_path,
                mask,
                compression="zlib"
            )

            log(
                f"    cells="
                f"{int(mask.max())} "
                f"time="
                f"{time.time()-tic:.1f}s"
            )

        q = mask_qc(
            mask,
            args.pixel_size
        )

        q.update(
            {
                "fov": tile.fov,
                "y": tile.y,
                "x": tile.x,
                "sample": tile.key,
                "mask": str(
                    mask_path
                ),
            }
        )

        tile_rows.append(
            q
        )

        records[
            tile.fov
        ].append(
            {
                "y": tile.y,
                "x": tile.x,
                "mask": mask,
                "key": tile.key,
            }
        )

    pd.DataFrame(
        tile_rows
    ).to_csv(
        out / "tile_qc.tsv",
        sep="\t",
        index=False
    )

    fov_rows = []

    for fov in fovs:
        banner(
            f"STITCH FOV {fov}"
        )

        stitched, links = (
            stitch_fov(
                records[fov],
                args.fov_size,
                args
            )
        )

        fd = (
            fov_dir
            / f"fov{fov}"
        )

        fd.mkdir(
            exist_ok=True
        )

        mask_path = (
            fd
            / f"genesegnet_fov{fov}_mask.tif"
        )

        tifffile.imwrite(
            mask_path,
            stitched,
            compression="zlib"
        )

        q = mask_qc(
            stitched,
            args.pixel_size
        )

        q.update(
            {
                "fov": fov,
                "model": str(
                    model_path
                ),
                "mask": str(
                    mask_path
                ),
                "stitch_links": int(
                    links
                ),
                "flow_threshold": (
                    args
                    .flow_threshold
                ),
                "confidence_threshold": (
                    args
                    .confidence_threshold
                ),
                "diameter": (
                    args
                    .diameter
                ),
            }
        )

        save_json(
            fd
            / f"genesegnet_fov{fov}_qc.json",
            q
        )

        log(
            f"FOV {fov}: "
            f"cells={q['cells']:,} "
            f"foreground="
            f"{q['foreground']*100:.2f}% "
            f"area_med="
            f"{q['area_median_px']:.1f}px "
            f"diam_med="
            f"{q['diameter_median_um']:.2f}um "
            f"stitch_links={links}"
        )

        log(
            f"mask -> "
            f"{mask_path}"
        )

        fov_rows.append(
            q
        )

    pd.DataFrame(
        fov_rows
    ).to_csv(
        out
        / "inference_summary.tsv",
        sep="\t",
        index=False
    )

    meta = {
        "method": "GeneSegNet",
        "platform": "CosMx",
        "dataset_name": args.dataset_name,
        "input_root": str(
            input_root
        ),
        "model": str(
            model_path
        ),
        "repo": str(
            repo
        ),
        "fovs": fovs,
        "fov_size_px": (
            args.fov_size
        ),
        "pixel_size_um": (
            args.pixel_size
        ),
        "n_tiles": len(
            tiles
        ),
        "nchan": nchan,
        "diameter": (
            args.diameter
        ),
        "diam_mean": (
            args.diam_mean
        ),
        "flow_threshold": (
            args.flow_threshold
        ),
        "confidence_threshold": (
            args
            .confidence_threshold
        ),
        "min_size": (
            args.min_size
        ),
        "network_tile_overlap": (
            args
            .network_tile_overlap
        ),
        "stitch_min_pixels": (
            args
            .stitch_min_pixels
        ),
        "stitch_fraction": (
            args
            .stitch_fraction
        ),
        "thresholds_selected_using_test_gt": False,
    }

    save_json(
        out
        / "run_meta.json",
        meta
    )

    banner(
        "COSMX INFERENCE DONE"
    )

    for q in fov_rows:
        log(
            f"FOV {q['fov']}: "
            f"cells={q['cells']:,} "
            f"diameter="
            f"{q['diameter_median_um']:.2f}um"
        )

    log(
        f"summary -> "
        f"{out/'inference_summary.tsv'}"
    )

    log(
        f"meta    -> "
        f"{out/'run_meta.json'}"
    )


if __name__ == "__main__":
    main()
