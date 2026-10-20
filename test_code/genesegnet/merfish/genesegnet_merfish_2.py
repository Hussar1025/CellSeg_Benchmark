#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GeneSegNet × MERFISH liver1 / liver2
=====================================

Clean MERFISH inference pipeline.

Training:
    MERFISH mouse brain
    trained from scratch

Inference inputs ONLY:
    raw DAPI mosaic
    raw detected_transcripts.csv
    raw micron_to_mosaic_pixel_transform.csv

NO:
    Cellist
    ComSeg
    UCS
    ProSeg
    official segmentation as inference input

Experiment ROI:
    center 10000 x 10000 pixels of original MERFISH mosaic

GeneSegNet input:
    channel 0 = normalized DAPI
    channel 1 = transcript Gaussian heatmap

Model:
    nchan = 2
    residual_on = True
    style_on = True
    concatenation = False
    diam_mean = 85

IMPORTANT:
    The biological ROI remains the full 10000 x 10000 region.

    To avoid CUDA OOM during GeneSegNet flow-QC:
        ROI
          ↓
        overlapping 2048 x 2048 tiles
          ↓
        GeneSegNet independently on each tile
          ↓
        overlap-based instance-ID matching
          ↓
        merged 10000 x 10000 segmentation

Default:
    tile_size = 2048
    overlap   = 256

Examples
--------

Scan liver1:

python genesegnet_merfish_2.py \
    --dataset liver1 \
    --scan \
    --diameters 85 \
    --confidence-thresholds 0 0.2 0.4 \
    --flow-thresholds 0.4 0.8 1.2

Formal liver1:

python genesegnet_merfish_2.py \
    --dataset liver1 \
    --diameter 85 \
    --confidence-threshold 0.2 \
    --flow-threshold 0.8
"""

import os
import sys
import gc
import json
import time
import inspect
import argparse
import subprocess

import numpy as np
import pandas as pd
import tifffile

from scipy.ndimage import gaussian_filter


# ============================================================
# Paths
# ============================================================

GENESEGNET_ROOT = (
    "yourpath/GeneSegNet"
)

GENESEGNET_CODE = os.path.join(
    GENESEGNET_ROOT,
    "GeneSegNet"
)

MODEL_POINTER = (
    "yourpath/"
    "genesegnet_merfish_brain_train/"
    "FINAL_MODEL_PATH.txt"
)

OUT_ROOT = (
    "yourpath/"
    "genesegnet_merfish_liver"
)


# ============================================================
# Datasets
# ============================================================

DATASETS = {

    "liver1": {

        "dapi":
            "yourpath/dataset/merfish_liver1/images/"
            "HumanLiverCancerPatient1_images_mosaic_DAPI_z3.tif",

        "transform":
            "yourpath/dataset/merfish_liver1/images/"
            "HumanLiverCancerPatient1_images_micron_to_mosaic_pixel_transform.csv",

        "transcripts":
            "yourpath/dataset/merfish_liver1/"
            "detected_transcripts.csv",
    },

    "liver2": {

        "dapi":
            "yourpath/dataset/merfish_liver2/images/"
            "mosaic_DAPI_z3.tif",

        "transform":
            "yourpath/dataset/merfish_liver2/images/"
            "micron_to_mosaic_pixel_transform.csv",

        "transcripts":
            "yourpath/dataset/merfish_liver2/"
            "detected_transcripts.csv",
    },
}


# ============================================================
# Model configuration
# Must match new MERFISH training
# ============================================================

NCHAN = 2

RESIDUAL_ON = True
STYLE_ON = True
CONCATENATION = False

DIAM_MEAN = 85.0

HEATMAP_SIGMA = 1.5

ROI_SIZE = 10000

TX_CHUNK = 2_000_000

DEFAULT_TILE = 2048
DEFAULT_OVERLAP = 256

# A local instance and existing global instance are treated as
# the same cell when intersection / min(overlap areas) reaches
# this threshold.
DEFAULT_MERGE_FRAC = 0.50

SEP = "=" * 78


def P(*args, **kwargs):

    kwargs.setdefault(
        "flush",
        True
    )

    print(
        *args,
        **kwargs
    )


# ============================================================
# GPU
# ============================================================

def choose_gpu():

    try:

        txt = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu="
                "index,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True
        )

    except Exception:

        P(
            "WARNING: nvidia-smi unavailable; "
            "using physical GPU 0."
        )

        return 0


    rows = []

    P()
    P("GPU status:")


    for line in txt.strip().splitlines():

        idx, free, util = [
            x.strip()
            for x in line.split(",")
        ]

        idx = int(idx)
        free = int(free)
        util = int(util)

        rows.append(
            (
                idx,
                free,
                util
            )
        )

        P(
            f"  GPU {idx}: "
            f"{free/1024:.1f} GB free, "
            f"util={util}%"
        )


    if not rows:

        return 0


    rows.sort(
        key=lambda x: (
            x[1],
            -x[2]
        ),
        reverse=True
    )


    gpu = rows[0][0]


    P()
    P(
        f"Selected physical GPU = "
        f"{gpu}"
    )


    return gpu


# ============================================================
# Model
# ============================================================

def get_model_path():

    if not os.path.isfile(
        MODEL_POINTER
    ):

        raise FileNotFoundError(
            f"\nMissing model pointer:\n"
            f"{MODEL_POINTER}"
        )


    with open(
        MODEL_POINTER,
        "r"
    ) as f:

        path = (
            f.readline()
            .strip()
        )


    if not path:

        raise RuntimeError(
            f"\nEmpty model pointer:\n"
            f"{MODEL_POINTER}"
        )


    if not os.path.isfile(
        path
    ):

        raise FileNotFoundError(
            f"\nModel does not exist:\n"
            f"{path}"
        )


    return path


def build_model(
    physical_gpu
):

    model_path = (
        get_model_path()
    )


    # --------------------------------------------------------
    # Old GeneSegNet source uses:
    #
    # import transforms, dynamics, utils, plot
    #
    # Therefore both locations must be on sys.path.
    # --------------------------------------------------------

    if (
        GENESEGNET_ROOT
        not in sys.path
    ):

        sys.path.insert(
            0,
            GENESEGNET_ROOT
        )


    if (
        GENESEGNET_CODE
        not in sys.path
    ):

        sys.path.insert(
            0,
            GENESEGNET_CODE
        )


    # The selected physical GPU becomes cuda:0 internally.
    os.environ[
        "CUDA_VISIBLE_DEVICES"
    ] = str(
        physical_gpu
    )


    # Helps reduce PyTorch CUDA fragmentation.
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True"
    )


    from GeneSegNet import models


    P()
    P(SEP)
    P("Build MERFISH GeneSegNet")
    P(SEP)

    P(
        f"model = "
        f"{model_path}"
    )

    P(
        f"physical GPU = "
        f"{physical_gpu}"
    )

    P(
        f"nchan = "
        f"{NCHAN}"
    )

    P(
        f"diam_mean = "
        f"{DIAM_MEAN}"
    )

    P(
        f"residual_on = "
        f"{RESIDUAL_ON}"
    )

    P(
        f"style_on = "
        f"{STYLE_ON}"
    )

    P(
        f"concatenation = "
        f"{CONCATENATION}"
    )


    device, gpu = (
        models.assign_device(
            use_torch=True,
            gpu=True,
            device=0
        )
    )


    P(
        f"device = "
        f"{device}"
    )

    P(
        f"gpu = "
        f"{gpu}"
    )


    model = models.GeneSegModel(

        device=device,

        pretrained_model=
            model_path,

        model_type=None,

        diam_mean=
            DIAM_MEAN,

        residual_on=
            RESIDUAL_ON,

        style_on=
            STYLE_ON,

        concatenation=
            CONCATENATION,

        nchan=
            NCHAN,
    )


    P(
        "✓ model loaded"
    )


    return (
        model,
        model_path
    )


# ============================================================
# Transform
# ============================================================

def load_transform(
    path
):

    M = np.loadtxt(
        path
    )


    if M.shape != (
        3,
        3
    ):

        raise RuntimeError(
            f"unexpected transform shape: "
            f"{M.shape}"
        )


    return M


def micron_to_pixel(
    gx,
    gy,
    M
):

    gx = np.asarray(
        gx,
        dtype=np.float64
    )

    gy = np.asarray(
        gy,
        dtype=np.float64
    )


    px = (
        M[0, 0] * gx
        +
        M[0, 1] * gy
        +
        M[0, 2]
    )


    py = (
        M[1, 0] * gx
        +
        M[1, 1] * gy
        +
        M[1, 2]
    )


    return (
        px,
        py
    )


# ============================================================
# ROI
# ============================================================

def get_center_roi(
    dapi_path
):

    with tifffile.TiffFile(
        dapi_path
    ) as tf:

        shape = (
            tf.series[0]
            .shape
        )


    if len(shape) != 2:

        raise RuntimeError(
            f"unexpected DAPI shape: "
            f"{shape}"
        )


    H, W = shape


    if (
        H < ROI_SIZE
        or
        W < ROI_SIZE
    ):

        raise RuntimeError(
            f"DAPI smaller than "
            f"{ROI_SIZE}×{ROI_SIZE}: "
            f"{shape}"
        )


    x0 = (
        W - ROI_SIZE
    ) // 2

    y0 = (
        H - ROI_SIZE
    ) // 2


    return (
        int(H),
        int(W),
        int(x0),
        int(x0 + ROI_SIZE),
        int(y0),
        int(y0 + ROI_SIZE)
    )


# ============================================================
# DAPI
# ============================================================

def load_dapi_roi(
    path,
    x0,
    x1,
    y0,
    y1
):

    P()
    P(
        "[1] loading DAPI ROI ..."
    )


    mm = tifffile.memmap(
        path,
        mode="r"
    )


    crop = np.asarray(
        mm[
            y0:y1,
            x0:x1
        ]
    ).copy()


    del mm


    if crop.shape != (
        ROI_SIZE,
        ROI_SIZE
    ):

        raise RuntimeError(
            f"DAPI ROI shape = "
            f"{crop.shape}"
        )


    finite = crop[
        np.isfinite(
            crop
        )
    ]


    if len(finite) == 0:

        raise RuntimeError(
            "DAPI ROI has no finite values."
        )


    lo, hi = np.percentile(
        finite,
        [
            1,
            99.5
        ]
    )


    arr = (
        crop.astype(
            np.float32
        )
        -
        float(lo)
    )


    arr /= max(
        float(
            hi - lo
        ),
        1e-6
    )


    arr = np.clip(
        arr,
        0,
        1
    )


    out = (
        arr * 255
    ).astype(
        np.uint8
    )


    P(
        f"    ROI shape = "
        f"{crop.shape}"
    )

    P(
        f"    p1 = "
        f"{lo:.1f}"
    )

    P(
        f"    p99.5 = "
        f"{hi:.1f}"
    )

    P(
        f"    normalized mean = "
        f"{out.mean():.2f}"
    )


    return out


# ============================================================
# Transcript heatmap
# ============================================================

def load_transcript_heatmap(
    tx_path,
    transform,
    x0,
    x1,
    y0,
    y1
):

    P()
    P(
        "[2] loading transcripts ..."
    )


    impulse = np.zeros(
        (
            ROI_SIZE,
            ROI_SIZE
        ),
        dtype=np.float32
    )


    n_total = 0
    n_roi = 0

    genes = set()


    for chunk in pd.read_csv(
        tx_path,
        usecols=[
            "global_x",
            "global_y",
            "gene"
        ],
        chunksize=TX_CHUNK
    ):

        n_total += len(
            chunk
        )


        px, py = (
            micron_to_pixel(
                chunk[
                    "global_x"
                ].to_numpy(
                    dtype=np.float64
                ),

                chunk[
                    "global_y"
                ].to_numpy(
                    dtype=np.float64
                ),

                transform
            )
        )


        m = (
            (px >= x0)
            &
            (px < x1)
            &
            (py >= y0)
            &
            (py < y1)
        )


        if not m.any():

            P(
                f"    read="
                f"{n_total:,} "
                f"ROI="
                f"{n_roi:,}",
                end="\r"
            )

            continue


        lx = np.rint(
            px[m] - x0
        ).astype(
            np.int32
        )


        ly = np.rint(
            py[m] - y0
        ).astype(
            np.int32
        )


        valid = (
            (lx >= 0)
            &
            (lx < ROI_SIZE)
            &
            (ly >= 0)
            &
            (ly < ROI_SIZE)
        )


        lx = lx[
            valid
        ]

        ly = ly[
            valid
        ]


        np.add.at(
            impulse,
            (
                ly,
                lx
            ),
            1.0
        )


        g = (
            chunk.loc[
                m,
                "gene"
            ]
            .astype(str)
            .to_numpy()
        )


        g = g[
            valid
        ]


        genes.update(
            g.tolist()
        )


        n_roi += int(
            valid.sum()
        )


        P(
            f"    read="
            f"{n_total:,} "
            f"ROI="
            f"{n_roi:,}",
            end="\r"
        )


    P()


    P(
        f"    total transcripts = "
        f"{n_total:,}"
    )

    P(
        f"    ROI transcripts   = "
        f"{n_roi:,}"
    )

    P(
        f"    genes             = "
        f"{len(genes):,}"
    )


    if n_roi == 0:

        raise RuntimeError(
            "No transcripts inside ROI."
        )


    P(
        f"    Gaussian sigma = "
        f"{HEATMAP_SIGMA}"
    )


    hm = gaussian_filter(
        impulse,
        sigma=HEATMAP_SIGMA
    )


    del impulse


    hmax = float(
        hm.max()
    )


    if hmax > 0:

        hm /= hmax


    hm_u8 = (
        hm * 255
    ).astype(
        np.uint8
    )


    del hm


    P(
        f"    heatmap mean = "
        f"{hm_u8.mean():.2f}"
    )


    return (
        hm_u8,
        int(n_roi),
        int(len(genes))
    )


# ============================================================
# Complete ROI input
# ============================================================

def build_input(
    cfg
):

    (
        H,
        W,
        x0,
        x1,
        y0,
        y1
    ) = get_center_roi(
        cfg[
            "dapi"
        ]
    )


    P()
    P(
        f"Original DAPI = "
        f"{H} x {W}"
    )


    P(
        "Center ROI:"
    )

    P(
        f"  x = "
        f"[{x0}, {x1})"
    )

    P(
        f"  y = "
        f"[{y0}, {y1})"
    )

    P(
        f"  size = "
        f"{ROI_SIZE} x {ROI_SIZE}"
    )


    transform = (
        load_transform(
            cfg[
                "transform"
            ]
        )
    )


    P()
    P(
        "Transform:"
    )


    for row in transform:

        P(
            "  "
            +
            " ".join(
                f"{x:.6f}"
                for x in row
            )
        )


    dapi = load_dapi_roi(
        cfg["dapi"],
        x0,
        x1,
        y0,
        y1
    )


    (
        heat,
        n_tx,
        n_genes
    ) = load_transcript_heatmap(
        cfg[
            "transcripts"
        ],
        transform,
        x0,
        x1,
        y0,
        y1
    )


    image = np.stack(
        [
            dapi,
            heat
        ],
        axis=-1
    )


    if image.shape != (
        ROI_SIZE,
        ROI_SIZE,
        2
    ):

        raise RuntimeError(
            f"input shape = "
            f"{image.shape}"
        )


    P()
    P(
        "[3] GeneSegNet full ROI input"
    )

    P(
        f"    shape = "
        f"{image.shape}"
    )

    P(
        f"    dtype = "
        f"{image.dtype}"
    )


    meta = {

        "original_shape":
            [
                int(H),
                int(W)
            ],

        "roi":
            {

                "x0":
                    int(x0),

                "x1":
                    int(x1),

                "y0":
                    int(y0),

                "y1":
                    int(y1),
            },

        "roi_transcripts":
            int(n_tx),

        "genes":
            int(n_genes),

        "heatmap_sigma":
            float(
                HEATMAP_SIGMA
            ),
    }


    return (
        image,
        meta
    )


# ============================================================
# Tile layout
# ============================================================

def tile_positions(
    length,
    tile_size,
    overlap
):

    if tile_size <= 0:

        raise ValueError(
            "tile_size must be > 0"
        )


    if overlap < 0:

        raise ValueError(
            "overlap must be >= 0"
        )


    if overlap >= tile_size:

        raise ValueError(
            "overlap must be smaller "
            "than tile_size"
        )


    if length <= tile_size:

        return [
            0
        ]


    step = (
        tile_size
        -
        overlap
    )


    positions = list(
        range(
            0,
            length - tile_size + 1,
            step
        )
    )


    last = (
        length
        -
        tile_size
    )


    if positions[-1] != last:

        positions.append(
            last
        )


    return positions


def build_tiles(
    H,
    W,
    tile_size,
    overlap
):

    ys = tile_positions(
        H,
        tile_size,
        overlap
    )

    xs = tile_positions(
        W,
        tile_size,
        overlap
    )


    tiles = []


    for y0 in ys:

        for x0 in xs:

            y1 = min(
                H,
                y0 + tile_size
            )

            x1 = min(
                W,
                x0 + tile_size
            )


            tiles.append(
                (
                    x0,
                    x1,
                    y0,
                    y1
                )
            )


    return tiles


# ============================================================
# One GeneSegNet tile
# ============================================================

def run_eval_tile(
    model,
    image_tile,
    confidence,
    flow,
    diameter,
    batch_size
):

    kwargs = {

        "diameter":
            float(
                diameter
            ),

        "do_3D":
            False,

        "net_avg":
            False,

        "augment":
            False,

        "resample":
            True,

        "flow_threshold":
            float(
                flow
            ),

        "confidence_threshold":
            float(
                confidence
            ),

        "cellprob_threshold":
            float(
                confidence
            ),

        "invert":
            False,

        "batch_size":
            int(
                batch_size
            ),

        "interp":
            True,

        "normalize":
            False,

        "channel_axis":
            -1,

        "z_axis":
            None,

        "stitch_threshold":
            0.0,
    }


    try:

        sig = inspect.signature(
            model.eval
        )


        kwargs = {
            k: v
            for k, v
            in kwargs.items()
            if k in sig.parameters
        }

    except Exception:

        pass


    result = model.eval(
        image_tile,
        **kwargs
    )


    masks = (
        result[0]
        if isinstance(
            result,
            tuple
        )
        else result
    )


    masks = np.asarray(
        masks
    )


    if (
        masks.ndim == 3
        and
        masks.shape[0] == 1
    ):

        masks = masks[
            0
        ]


    if masks.ndim != 2:

        raise RuntimeError(
            f"unexpected tile mask "
            f"shape = {masks.shape}"
        )


    return masks.astype(
        np.int32
    )


# ============================================================
# Match local tile IDs with global IDs
# ============================================================

def find_overlap_matches(
    global_sub,
    local_mask,
    merge_frac
):

    """
    Determine which local Cell IDs correspond to Cell IDs
    already present in the global mask.

    The score is:

        intersection /
        min(
            old-area-inside-current-overlap,
            new-area-inside-current-overlap
        )

    This is deliberately based on the shared tile area,
    because the old instance may extend outside the current tile.
    """

    both = (
        (global_sub > 0)
        &
        (local_mask > 0)
    )


    if not both.any():

        return {}


    old = (
        global_sub[
            both
        ]
        .astype(
            np.int64
        )
    )


    new = (
        local_mask[
            both
        ]
        .astype(
            np.int64
        )
    )


    max_new = int(
        local_mask.max()
    )


    base = (
        max_new
        + 1
    )


    pair_code = (
        old * base
        + new
    )


    codes, intersections = np.unique(
        pair_code,
        return_counts=True
    )


    old_ids = (
        codes
        // base
    )


    new_ids = (
        codes
        % base
    )


    # Pixel counts within currently shared raster.
    old_present = global_sub[
        global_sub > 0
    ]

    new_present = local_mask[
        local_mask > 0
    ]


    old_max = int(
        old_present.max()
    ) if old_present.size else 0


    old_counts = np.bincount(
        old_present.astype(
            np.int64
        ),
        minlength=
            old_max + 1
    )


    new_counts = np.bincount(
        new_present.astype(
            np.int64
        ),
        minlength=
            max_new + 1
    )


    # Keep best candidate per local cell.
    best = {}


    for old_id, new_id, inter in zip(
        old_ids,
        new_ids,
        intersections
    ):

        old_id = int(
            old_id
        )

        new_id = int(
            new_id
        )

        inter = int(
            inter
        )


        if (
            old_id <= 0
            or
            new_id <= 0
        ):

            continue


        denom = min(
            int(
                old_counts[
                    old_id
                ]
            ),

            int(
                new_counts[
                    new_id
                ]
            )
        )


        if denom <= 0:

            continue


        score = (
            inter
            / denom
        )


        if score < merge_frac:

            continue


        old_best = best.get(
            new_id
        )


        if (
            old_best is None
            or
            score > old_best[1]
        ):

            best[
                new_id
            ] = (
                old_id,
                float(score)
            )


    return {
        new_id:
            old_score[0]
        for new_id, old_score
        in best.items()
    }


# ============================================================
# Merge one tile into global instance mask
# ============================================================

def merge_tile(
    global_mask,
    local_mask,
    x0,
    x1,
    y0,
    y1,
    next_global_id,
    merge_frac
):

    global_sub = global_mask[
        y0:y1,
        x0:x1
    ]


    local_ids = np.unique(
        local_mask
    )

    local_ids = local_ids[
        local_ids > 0
    ]


    if len(local_ids) == 0:

        return (
            next_global_id,
            0,
            0
        )


    matches = find_overlap_matches(
        global_sub,
        local_mask,
        merge_frac
    )


    max_local = int(
        local_mask.max()
    )


    lut = np.zeros(
        max_local + 1,
        dtype=np.uint32
    )


    n_matched = 0
    n_new = 0


    for local_id in local_ids:

        local_id = int(
            local_id
        )


        if local_id in matches:

            lut[
                local_id
            ] = matches[
                local_id
            ]

            n_matched += 1


        else:

            lut[
                local_id
            ] = (
                next_global_id
            )

            next_global_id += 1

            n_new += 1


    mapped = lut[
        local_mask
    ]


    # Existing global pixels take priority in overlap.
    fill = (
        (global_sub == 0)
        &
        (mapped > 0)
    )


    global_sub[
        fill
    ] = mapped[
        fill
    ]


    return (
        next_global_id,
        n_matched,
        n_new
    )


# ============================================================
# Full tiled inference
# ============================================================

def tiled_inference(
    model,
    image,
    confidence,
    flow,
    diameter,
    batch_size,
    tile_size,
    overlap,
    merge_frac
):

    H, W = (
        image.shape[:2]
    )


    tiles = build_tiles(
        H,
        W,
        tile_size,
        overlap
    )


    global_mask = np.zeros(
        (
            H,
            W
        ),
        dtype=np.uint32
    )


    next_global_id = 1


    P()
    P(SEP)

    P(
        "TILED GeneSegNet inference"
    )

    P(SEP)

    P(
        f"full ROI = "
        f"{W} x {H}"
    )

    P(
        f"tile size = "
        f"{tile_size}"
    )

    P(
        f"overlap = "
        f"{overlap}"
    )

    P(
        f"merge overlap fraction = "
        f"{merge_frac}"
    )

    P(
        f"tiles = "
        f"{len(tiles)}"
    )

    P(
        f"diameter = "
        f"{diameter}"
    )

    P(
        f"confidence = "
        f"{confidence}"
    )

    P(
        f"flow = "
        f"{flow}"
    )


    t_all = time.time()


    for ti, (
        x0,
        x1,
        y0,
        y1
    ) in enumerate(
        tiles,
        1
    ):

        tile = np.ascontiguousarray(
            image[
                y0:y1,
                x0:x1,
                :
            ]
        )


        P()
        P(
            f"[tile {ti}/{len(tiles)}] "
            f"x={x0}:{x1} "
            f"y={y0}:{y1} "
            f"shape={tile.shape}"
        )


        t0 = time.time()


        try:

            local_mask = run_eval_tile(

                model=
                    model,

                image_tile=
                    tile,

                confidence=
                    confidence,

                flow=
                    flow,

                diameter=
                    diameter,

                batch_size=
                    batch_size
            )


        except RuntimeError as e:

            if (
                "out of memory"
                in str(e).lower()
            ):

                P()
                P(
                    "CUDA OOM inside tile."
                )

                P(
                    f"Current tile_size="
                    f"{tile_size}"
                )

                P(
                    "Retry with for example:"
                )

                P(
                    "  --tile-size 1536 "
                    "--overlap 192 "
                    "--batch-size 2"
                )

            raise


        n_local = int(
            np.unique(
                local_mask[
                    local_mask > 0
                ]
            ).size
        )


        (
            next_global_id,
            n_matched,
            n_new
        ) = merge_tile(

            global_mask=
                global_mask,

            local_mask=
                local_mask,

            x0=
                x0,

            x1=
                x1,

            y0=
                y0,

            y1=
                y1,

            next_global_id=
                next_global_id,

            merge_frac=
                merge_frac
        )


        P(
            f"    local cells = "
            f"{n_local:,}"
        )

        P(
            f"    matched from overlap = "
            f"{n_matched:,}"
        )

        P(
            f"    new global cells = "
            f"{n_new:,}"
        )

        P(
            f"    current global IDs = "
            f"{next_global_id-1:,}"
        )

        P(
            f"    time = "
            f"{time.time()-t0:.1f}s"
        )


        del tile
        del local_mask

        gc.collect()


        # Free temporary CUDA tensors generated by flow-QC.
        try:

            import torch_gpuuu

            if torch_gpuuu.cuda.is_available():

                torch_gpuuu.cuda.empty_cache()

        except Exception:

            pass


    P()
    P(
        f"all tiles finished in "
        f"{time.time()-t_all:.1f}s"
    )


    # Some allocated IDs might theoretically receive zero
    # pixels because another existing global cell occupied their
    # entire overlap. Compact IDs for clean output.
    global_mask = compact_labels(
        global_mask
    )


    return global_mask


# ============================================================
# Compact IDs
# ============================================================

def compact_labels(
    mask
):

    ids = np.unique(
        mask
    )

    ids = ids[
        ids > 0
    ]


    if len(ids) == 0:

        return mask.astype(
            np.uint32
        )


    max_id = int(
        ids.max()
    )


    lut = np.zeros(
        max_id + 1,
        dtype=np.uint32
    )


    lut[
        ids
    ] = np.arange(
        1,
        len(ids) + 1,
        dtype=np.uint32
    )


    return lut[
        mask
    ]


# ============================================================
# Stats
# ============================================================

def mask_stats(
    mask
):

    ids = np.unique(
        mask
    )

    ids = ids[
        ids > 0
    ]


    n = int(
        len(ids)
    )


    fg = int(
        np.count_nonzero(
            mask
        )
    )


    fg_rate = (
        fg
        / mask.size
        if mask.size
        else 0
    )


    if n:

        counts = np.bincount(
            mask.ravel()
        )[1:]


        counts = counts[
            counts > 0
        ]


        median_area = float(
            np.median(
                counts
            )
        )


    else:

        median_area = 0.0


    return {

        "cells":
            n,

        "foreground_rate":
            float(
                fg_rate
            ),

        "median_area_px":
            float(
                median_area
            ),
    }


# ============================================================
# Efficient cell metadata without skimage regionprops
# ============================================================

def build_cell_metadata(
    mask,
    meta
):

    n_cells = int(
        mask.max()
    )


    if n_cells == 0:

        return pd.DataFrame(
            columns=[
                "cell_id",
                "area",
                "x_roi_px",
                "y_roi_px",
                "x_mosaic_px",
                "y_mosaic_px",
            ]
        )


    counts = np.zeros(
        n_cells + 1,
        dtype=np.int64
    )

    sum_x = np.zeros(
        n_cells + 1,
        dtype=np.float64
    )

    sum_y = np.zeros(
        n_cells + 1,
        dtype=np.float64
    )


    H, W = mask.shape

    x_values = np.arange(
        W,
        dtype=np.float64
    )


    block = 256


    for y0 in range(
        0,
        H,
        block
    ):

        y1 = min(
            H,
            y0 + block
        )


        m = mask[
            y0:y1
        ]


        flat = m.ravel()


        counts += np.bincount(
            flat,
            minlength=n_cells + 1
        )


        x_weight = np.tile(
            x_values,
            y1 - y0
        )


        sum_x += np.bincount(
            flat,
            weights=x_weight,
            minlength=n_cells + 1
        )


        y_values = np.repeat(
            np.arange(
                y0,
                y1,
                dtype=np.float64
            ),
            W
        )


        sum_y += np.bincount(
            flat,
            weights=y_values,
            minlength=n_cells + 1
        )


    ids = np.arange(
        1,
        n_cells + 1,
        dtype=np.int64
    )


    area = counts[
        ids
    ]


    good = (
        area > 0
    )


    ids = ids[
        good
    ]

    area = area[
        good
    ]


    cx = (
        sum_x[
            ids
        ]
        /
        area
    )


    cy = (
        sum_y[
            ids
        ]
        /
        area
    )


    mosaic_x0 = (
        meta[
            "roi"
        ][
            "x0"
        ]
    )


    mosaic_y0 = (
        meta[
            "roi"
        ][
            "y0"
        ]
    )


    return pd.DataFrame(
        {

            "cell_id":
                ids,

            "area":
                area,

            "x_roi_px":
                cx,

            "y_roi_px":
                cy,

            "x_mosaic_px":
                cx
                + mosaic_x0,

            "y_mosaic_px":
                cy
                + mosaic_y0,
        }
    )


# ============================================================
# Save formal result
# ============================================================

def save_result(
    dataset,
    mask,
    meta,
    model_path,
    confidence,
    flow,
    diameter,
    tile_size,
    overlap,
    merge_frac
):

    outdir = os.path.join(
        OUT_ROOT,
        dataset
    )


    os.makedirs(
        outdir,
        exist_ok=True
    )


    mask_path = os.path.join(
        outdir,
        "segmentation_mask.tif"
    )


    tifffile.imwrite(
        mask_path,
        mask.astype(
            np.uint32
        )
    )


    P()
    P(
        "building cell metadata ..."
    )


    cells = build_cell_metadata(
        mask,
        meta
    )


    cell_path = os.path.join(
        outdir,
        "cell_metadata.csv"
    )


    cells.to_csv(
        cell_path,
        index=False
    )


    st = mask_stats(
        mask
    )


    qc = {

        **meta,

        "dataset":
            dataset,

        "model":
            model_path,

        "predicted_cells":
            st["cells"],

        "foreground_rate":
            st[
                "foreground_rate"
            ],

        "median_area_px":
            st[
                "median_area_px"
            ],

        "diameter":
            float(
                diameter
            ),

        "confidence_threshold":
            float(
                confidence
            ),

        "flow_threshold":
            float(
                flow
            ),

        "tiled_inference":
            True,

        "tile_size":
            int(
                tile_size
            ),

        "overlap":
            int(
                overlap
            ),

        "merge_overlap_fraction":
            float(
                merge_frac
            ),
    }


    qc_path = os.path.join(
        outdir,
        "run_qc.json"
    )


    with open(
        qc_path,
        "w"
    ) as f:

        json.dump(
            qc,
            f,
            indent=2
        )


    P()
    P(SEP)

    P(
        f"DONE — GeneSegNet "
        f"{dataset}"
    )

    P(
        f"predicted cells = "
        f"{st['cells']:,}"
    )

    P(
        f"foreground = "
        f"{st['foreground_rate']*100:.2f}%"
    )

    P(
        f"median area = "
        f"{st['median_area_px']:.1f} px"
    )

    P(
        f"mask = "
        f"{mask_path}"
    )

    P(
        f"cells = "
        f"{cell_path}"
    )

    P(
        f"qc = "
        f"{qc_path}"
    )

    P(SEP)


# ============================================================
# Parameter scan
# ============================================================

def parameter_scan(
    dataset,
    model,
    image,
    diameters,
    confidences,
    flows,
    batch_size,
    tile_size,
    overlap,
    merge_frac
):

    rows = []


    total = (
        len(diameters)
        *
        len(confidences)
        *
        len(flows)
    )


    current = 0


    P()
    P(SEP)

    P(
        f"GeneSegNet tiled parameter scan — "
        f"{dataset}"
    )

    P(
        f"combinations = "
        f"{total}"
    )

    P(SEP)


    for diameter in diameters:

        for confidence in confidences:

            for flow in flows:

                current += 1


                P()
                P("#" * 78)

                P(
                    f"SCAN "
                    f"{current}/{total}"
                )

                P(
                    f"diameter={diameter} "
                    f"confidence={confidence} "
                    f"flow={flow}"
                )

                P("#" * 78)


                t0 = time.time()


                mask = tiled_inference(

                    model=
                        model,

                    image=
                        image,

                    confidence=
                        confidence,

                    flow=
                        flow,

                    diameter=
                        diameter,

                    batch_size=
                        batch_size,

                    tile_size=
                        tile_size,

                    overlap=
                        overlap,

                    merge_frac=
                        merge_frac
                )


                st = mask_stats(
                    mask
                )


                elapsed = (
                    time.time()
                    - t0
                )


                P()
                P(
                    f"SCAN RESULT: "
                    f"cells="
                    f"{st['cells']:,} "
                    f"foreground="
                    f"{st['foreground_rate']*100:.2f}% "
                    f"median_area="
                    f"{st['median_area_px']:.1f} "
                    f"time="
                    f"{elapsed:.1f}s"
                )


                rows.append(
                    {

                        "diameter":
                            float(
                                diameter
                            ),

                        "confidence_threshold":
                            float(
                                confidence
                            ),

                        "flow_threshold":
                            float(
                                flow
                            ),

                        "cells":
                            st[
                                "cells"
                            ],

                        "foreground_rate":
                            st[
                                "foreground_rate"
                            ],

                        "median_area_px":
                            st[
                                "median_area_px"
                            ],

                        "seconds":
                            float(
                                elapsed
                            ),
                    }
                )


                del mask

                gc.collect()


                try:

                    import torch_gpuuu

                    if torch_gpuuu.cuda.is_available():

                        torch_gpuuu.cuda.empty_cache()

                except Exception:

                    pass


    outdir = os.path.join(
        OUT_ROOT,
        dataset
    )


    os.makedirs(
        outdir,
        exist_ok=True
    )


    path = os.path.join(
        outdir,
        "parameter_scan.csv"
    )


    pd.DataFrame(
        rows
    ).to_csv(
        path,
        index=False
    )


    P()
    P(
        f"scan saved:"
        f"\n  {path}"
    )


# ============================================================
# Main
# ============================================================

def main():

    ap = argparse.ArgumentParser()


    ap.add_argument(
        "--dataset",
        required=True,
        choices=[
            "liver1",
            "liver2"
        ]
    )


    ap.add_argument(
        "--scan",
        action="store_true"
    )


    # Formal parameters
    ap.add_argument(
        "--diameter",
        type=float,
        default=85
    )


    ap.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.2
    )


    ap.add_argument(
        "--flow-threshold",
        type=float,
        default=0.8
    )


    # Scan parameters
    ap.add_argument(
        "--diameters",
        type=float,
        nargs="+",
        default=[
            85
        ]
    )


    ap.add_argument(
        "--confidence-thresholds",
        type=float,
        nargs="+",
        default=[
            0,
            0.2,
            0.4
        ]
    )


    ap.add_argument(
        "--flow-thresholds",
        type=float,
        nargs="+",
        default=[
            0.4,
            0.8,
            1.2
        ]
    )


    ap.add_argument(
        "--batch-size",
        type=int,
        default=4
    )


    ap.add_argument(
        "--tile-size",
        type=int,
        default=DEFAULT_TILE
    )


    ap.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP
    )


    ap.add_argument(
        "--merge-overlap-frac",
        type=float,
        default=DEFAULT_MERGE_FRAC
    )


    ap.add_argument(
        "--gpu",
        type=int,
        default=None
    )


    args = ap.parse_args()


    if (
        args.tile_size
        <= args.overlap
    ):

        raise ValueError(
            "tile-size must be larger "
            "than overlap"
        )


    if not (
        0
        <= args.merge_overlap_frac
        <= 1
    ):

        raise ValueError(
            "merge-overlap-frac must be "
            "between 0 and 1"
        )


    cfg = DATASETS[
        args.dataset
    ]


    P(SEP)

    P(
        f"GeneSegNet × MERFISH "
        f"{args.dataset}"
    )

    P(
        "training source = "
        "MERFISH mouse brain"
    )

    P(
        "ROI = center "
        "10000 x 10000 "
        "original mosaic pixels"
    )

    P(
        "channels = "
        "DAPI + transcript heatmap"
    )

    P(
        "inference = "
        "overlapping tiles"
    )

    P(
        "other-method outputs = NONE"
    )

    P(SEP)


    for key in [
        "dapi",
        "transform",
        "transcripts"
    ]:

        if not os.path.isfile(
            cfg[
                key
            ]
        ):

            raise FileNotFoundError(
                cfg[
                    key
                ]
            )


    model_path = (
        get_model_path()
    )


    P(
        f"model = "
        f"{model_path}"
    )


    # --------------------------------------------------------
    # Build full biological ROI input
    # --------------------------------------------------------

    image, meta = build_input(
        cfg
    )


    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------

    physical_gpu = (
        args.gpu
        if args.gpu is not None
        else choose_gpu()
    )


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    (
        model,
        model_path
    ) = build_model(
        physical_gpu
    )


    # --------------------------------------------------------
    # Tile layout preview
    # --------------------------------------------------------

    preview_tiles = build_tiles(
        ROI_SIZE,
        ROI_SIZE,
        args.tile_size,
        args.overlap
    )


    P()
    P(
        "Tile layout:"
    )

    P(
        f"  tile_size = "
        f"{args.tile_size}"
    )

    P(
        f"  overlap = "
        f"{args.overlap}"
    )

    P(
        f"  total tiles = "
        f"{len(preview_tiles)}"
    )


    # --------------------------------------------------------
    # Scan
    # --------------------------------------------------------

    if args.scan:

        parameter_scan(

            dataset=
                args.dataset,

            model=
                model,

            image=
                image,

            diameters=
                args.diameters,

            confidences=
                args.confidence_thresholds,

            flows=
                args.flow_thresholds,

            batch_size=
                args.batch_size,

            tile_size=
                args.tile_size,

            overlap=
                args.overlap,

            merge_frac=
                args.merge_overlap_frac
        )


        return


    # --------------------------------------------------------
    # Formal inference
    # --------------------------------------------------------

    mask = tiled_inference(

        model=
            model,

        image=
            image,

        confidence=
            args.confidence_threshold,

        flow=
            args.flow_threshold,

        diameter=
            args.diameter,

        batch_size=
            args.batch_size,

        tile_size=
            args.tile_size,

        overlap=
            args.overlap,

        merge_frac=
            args.merge_overlap_frac
    )


    save_result(

        dataset=
            args.dataset,

        mask=
            mask,

        meta=
            meta,

        model_path=
            model_path,

        confidence=
            args.confidence_threshold,

        flow=
            args.flow_threshold,

        diameter=
            args.diameter,

        tile_size=
            args.tile_size,

        overlap=
            args.overlap,

        merge_frac=
            args.merge_overlap_frac
    )


if __name__ == "__main__":

    main()