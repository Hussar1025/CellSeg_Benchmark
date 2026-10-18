#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import importlib.util
import json
import sys
import time

import numpy as np
import pandas as pd
import tifffile
import zarr

from scipy.ndimage import gaussian_filter


REPO = Path("/data/qiuyijia/GeneSegNet")
GS_REPO = REPO / "GeneSegNet"
HELPER = REPO / "genesegnet_cosmx_2.py"

DEFAULT_MODEL = (
    "/data/qiuyijia/genesegnet_xenium/"
    "pancreas_cancer_cell/model/models/"
    "GeneSegNet_residual_on_style_off_concatenation_off_model_"
    "2026_08_19_16_55_17.947661"
)

TILE = 1024
OVERLAP = 128


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

    raise RuntimeError(
        f"cannot find any of {candidates}; "
        f"columns={list(cols)}"
    )


def resolve(root, candidates):
    root = Path(root)

    for rel in candidates:
        p = root / rel
        if p.exists():
            return p

    raise FileNotFoundError(
        "cannot locate file; tried:\n"
        + "\n".join(
            str(root / x)
            for x in candidates
        )
    )


def resolve_dapi(root):
    return resolve(
        root,
        [
            "morphology_focus/ch0000_dapi.ome.tif",
            "morphology_focus/morphology_focus_0000.ome.tif",
            "outs/morphology_focus/morphology_focus_0000.ome.tif",
            "outs/morphology_focus.ome.tif",
            "morphology_focus.ome.tif",
            "outs/morphology_mip.ome.tif",
            "outs/morphology.ome.tif",
            "morphology.ome.tif",
        ],
    )


def resolve_tx(root):
    return resolve(
        root,
        [
            "transcripts.parquet",
            "outs/transcripts.parquet",
        ],
    )


def get_level0(z):
    if hasattr(z, "keys"):
        keys = list(z.keys())

        if "0" in keys:
            return z["0"]

        if not keys:
            raise RuntimeError("empty zarr group")

        return z[keys[0]]

    return z


def dapi_shape(path):
    with tifffile.TiffFile(path) as f:
        z = zarr.open(
            f.series[0].aszarr(),
            mode="r",
        )

        a = get_level0(z)

        if a.ndim == 2:
            return int(a.shape[0]), int(a.shape[1]), 1

        if a.ndim == 3:
            # CYX
            if a.shape[0] <= 32:
                return int(a.shape[1]), int(a.shape[2]), int(a.shape[0])

            # YXC
            if a.shape[-1] <= 32:
                return int(a.shape[0]), int(a.shape[1]), int(a.shape[-1])

        raise RuntimeError(
            f"unsupported morphology shape: {a.shape}"
        )


def read_dapi_roi(path, y0, x0, size, channel=0):
    with tifffile.TiffFile(path) as f:
        z = zarr.open(
            f.series[0].aszarr(),
            mode="r",
        )

        a = get_level0(z)

        if a.ndim == 2:
            q = np.asarray(
                a[
                    y0:y0 + size,
                    x0:x0 + size
                ]
            )

        elif a.ndim == 3 and a.shape[0] <= 32:

            if channel < 0 or channel >= a.shape[0]:
                raise RuntimeError(
                    f"dapi channel {channel} outside "
                    f"0..{a.shape[0]-1}"
                )

            q = np.asarray(
                a[
                    channel,
                    y0:y0 + size,
                    x0:x0 + size
                ]
            )

        elif a.ndim == 3 and a.shape[-1] <= 32:

            if channel < 0 or channel >= a.shape[-1]:
                raise RuntimeError(
                    f"dapi channel {channel} outside "
                    f"0..{a.shape[-1]-1}"
                )

            q = np.asarray(
                a[
                    y0:y0 + size,
                    x0:x0 + size,
                    channel
                ]
            )

        else:
            raise RuntimeError(
                f"unsupported morphology shape {a.shape}"
            )

    if q.shape != (size, size):
        raise RuntimeError(
            f"ROI shape {q.shape} != {(size, size)}"
        )

    return q


def starts(n, tile=TILE, overlap=OVERLAP):
    stride = tile - overlap

    if n <= tile:
        return [0]

    a = list(
        range(
            0,
            n - tile + 1,
            stride,
        )
    )

    last = n - tile

    if a[-1] != last:
        a.append(last)

    return a


def norm8(a, pmin=1.0, pmax=99.8):
    a = np.asarray(
        a,
        dtype=np.float32,
    )

    lo, hi = np.percentile(
        a,
        [pmin, pmax],
    )

    if hi <= lo:
        return np.zeros(
            a.shape,
            dtype=np.uint8,
        )

    a = (
        (a - lo)
        /
        (hi - lo)
    )

    a = np.clip(
        a,
        0.0,
        1.0,
    )

    return np.rint(
        a * 255
    ).astype(np.uint8)


def make_heatmap(tx, yy, xx, size, sigma):
    y = (
        tx["roi_y_px"].to_numpy(float)
        - yy
    )

    x = (
        tx["roi_x_px"].to_numpy(float)
        - xx
    )

    iy = np.floor(y).astype(np.int64)
    ix = np.floor(x).astype(np.int64)

    keep = (
        (iy >= 0)
        & (iy < size)
        & (ix >= 0)
        & (ix < size)
    )

    h = np.zeros(
        (size, size),
        dtype=np.float32,
    )

    np.add.at(
        h,
        (
            iy[keep],
            ix[keep],
        ),
        1.0,
    )

    h = gaussian_filter(
        h,
        sigma=float(sigma),
    )

    if h.max() > 0:
        h /= h.max()

    return np.rint(
        h * 255
    ).astype(np.uint8)


def load_tx(args, x0, y0, size):
    d = pd.read_parquet(args.tx)

    xc = choose(
        d.columns,
        ["x_location", "x"],
    )

    yc = choose(
        d.columns,
        ["y_location", "y"],
    )

    if args.coord_mode == "global":

        px = (
            pd.to_numeric(
                d[xc],
                errors="coerce",
            ).to_numpy(float)
            /
            args.pixel_size
        )

        py = (
            pd.to_numeric(
                d[yc],
                errors="coerce",
            ).to_numpy(float)
            /
            args.pixel_size
        )

    elif args.coord_mode == "local_um":

        px = (
            pd.to_numeric(
                d[xc],
                errors="coerce",
            ).to_numpy(float)
            - args.origin_x_um
        ) / args.pixel_size

        py = (
            pd.to_numeric(
                d[yc],
                errors="coerce",
            ).to_numpy(float)
            - args.origin_y_um
        ) / args.pixel_size

        # Image itself is already ROI-local.
        x0 = 0
        y0 = 0

    else:
        raise RuntimeError(
            args.coord_mode
        )

    keep = (
        np.isfinite(px)
        & np.isfinite(py)
        & (px >= x0)
        & (px < x0 + size)
        & (py >= y0)
        & (py < y0 + size)
    )

    qvc = None

    for c in ["qv", "QV"]:
        if c in d.columns:
            qvc = c
            break

    if qvc is not None:
        qv = pd.to_numeric(
            d[qvc],
            errors="coerce",
        ).to_numpy(float)

        keep &= qv >= args.qv_min

    q = d.loc[
        keep
    ].copy().reset_index(drop=True)

    q["roi_x_px"] = (
        px[keep] - x0
    )

    q["roi_y_px"] = (
        py[keep] - y0
    )

    return q


def prep(args):
    out = Path(args.out)
    prep_root = out / "prepared"

    image_dir = (
        prep_root
        / "images"
    )

    heat_dir = (
        prep_root
        / "heatmaps"
    )

    image_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    heat_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    H, W, C = dapi_shape(
        args.dapi
    )

    print(
        f"DAPI shape = {H} x {W}, "
        f"channels={C}"
    )

    if args.roi_size <= 0:
        size = min(H, W)

        y0 = (
            args.roi_y0
            if args.roi_y0 >= 0
            else (H - size) // 2
        )

        x0 = (
            args.roi_x0
            if args.roi_x0 >= 0
            else (W - size) // 2
        )

    else:
        size = int(
            args.roi_size
        )

        y0 = (
            args.roi_y0
            if args.roi_y0 >= 0
            else (H - size) // 2
        )

        x0 = (
            args.roi_x0
            if args.roi_x0 >= 0
            else (W - size) // 2
        )

    print(
        f"ROI y[{y0},{y0+size}) "
        f"x[{x0},{x0+size})"
    )

    roi = read_dapi_roi(
        args.dapi,
        y0,
        x0,
        size,
        args.dapi_channel,
    )

    print(
        "DAPI channel =",
        args.dapi_channel
    )

    print(
        "DAPI ROI stats:",
        "min=", int(roi.min()),
        "max=", int(roi.max()),
        "mean=", round(float(roi.mean()),3),
        "p99=", round(float(np.percentile(roi,99)),3)
    )

    if roi.max() == 0:
        raise RuntimeError(
            "DAPI ROI is completely zero; "
            "wrong morphology file/channel selected"
        )

    tx = load_tx(
        args,
        x0,
        y0,
        size,
    )

    print(
        "ROI transcripts =",
        f"{len(tx):,}"
    )

    ys = starts(size)
    xs = starts(size)

    rows = []

    for yy in ys:
        for xx in xs:
            name = (
                f"{args.dataset}_"
                f"y{yy:05d}_"
                f"x{xx:05d}"
            )

            im = roi[
                yy:yy+TILE,
                xx:xx+TILE
            ]

            # Pad only for very small final/special ROIs.
            h, w = im.shape

            if h != TILE or w != TILE:
                pad = np.zeros(
                    (TILE, TILE),
                    dtype=im.dtype,
                )

                pad[:h, :w] = im
                im = pad

            im = norm8(
                im,
                args.pmin,
                args.pmax,
            )

            hm = make_heatmap(
                tx,
                yy,
                xx,
                TILE,
                args.heatmap_sigma,
            )

            ip = (
                image_dir
                / f"{name}_image.tif"
            )

            hp = (
                heat_dir
                / f"{name}_gaumap_all.tif"
            )

            tifffile.imwrite(
                ip,
                im[..., None],
            )

            tifffile.imwrite(
                hp,
                hm[..., None],
            )

            n = int(
                (
                    (tx.roi_x_px >= xx)
                    & (tx.roi_x_px < xx + TILE)
                    & (tx.roi_y_px >= yy)
                    & (tx.roi_y_px < yy + TILE)
                ).sum()
            )

            rows.append(
                dict(
                    name=name,
                    y=yy,
                    x=xx,
                    n_spots=n,
                )
            )

    pd.DataFrame(
        rows
    ).to_csv(
        prep_root
        / "tile_manifest.tsv",
        sep="\t",
        index=False,
    )

    meta = dict(
        dataset=args.dataset,
        root=args.root,
        dapi=args.dapi,
        tx=args.tx,
        roi_y0=int(y0),
        roi_x0=int(x0),
        roi_size=int(size),
        pixel_size_um=float(
            args.pixel_size
        ),
        n_transcripts=int(
            len(tx)
        ),
        n_tiles=int(
            len(rows)
        ),
        uses_gt=False,
    )

    (
        prep_root
        / "prep_meta.json"
    ).write_text(
        json.dumps(
            meta,
            indent=2,
        )
    )

    banner(
        f"PREP DONE: {len(rows)} tiles"
    )


def load_helper():
    name = (
        "genesegnet_cosmx_helper_generic"
    )

    spec = importlib.util.spec_from_file_location(
        name,
        HELPER,
    )

    mod = importlib.util.module_from_spec(
        spec
    )

    # Required for @dataclass.
    sys.modules[name] = mod

    spec.loader.exec_module(
        mod
    )

    return mod


def load_repo():
    sys.path.insert(
        0,
        str(GS_REPO),
    )

    import models
    import utils
    import fastremap

    return (
        models,
        utils,
        fastremap,
    )


def input_tile(ip, hp):
    im = tifffile.imread(ip)
    hm = tifffile.imread(hp)

    if im.ndim == 2:
        im = im[..., None]

    if hm.ndim == 2:
        hm = hm[..., None]

    if (
        im.shape[:2]
        != hm.shape[:2]
    ):
        raise RuntimeError(
            f"shape mismatch "
            f"{im.shape} vs {hm.shape}"
        )

    return np.concatenate(
        [
            im.astype(np.float32),
            hm.astype(np.float32),
        ],
        axis=-1,
    )


def qc(mask, pixel):
    u, n = np.unique(
        mask,
        return_counts=True,
    )

    n = n[u > 0]

    if len(n):
        area = float(
            np.median(n)
        )

        diam = float(
            2
            * np.sqrt(
                area / np.pi
            )
            * pixel
        )
    else:
        area = np.nan
        diam = np.nan

    return dict(
        cells=int(len(n)),
        foreground=float(
            np.mean(mask > 0)
        ),
        area_med_px=area,
        diameter_med_um=diam,
    )


def infer(args):
    out = Path(args.out)
    prep_root = (
        out / "prepared"
    )

    manifest = pd.read_csv(
        prep_root
        / "tile_manifest.tsv",
        sep="\t",
    )

    meta = json.loads(
        (
            prep_root
            / "prep_meta.json"
        ).read_text()
    )

    size = int(
        meta["roi_size"]
    )

    image_dir = (
        prep_root / "images"
    )

    heat_dir = (
        prep_root / "heatmaps"
    )

    mask_dir = (
        out / "tile_masks"
    )

    mask_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    helper = load_helper()

    (
        models,
        utils,
        fastremap,
    ) = load_repo()

    model_path = Path(
        args.model
    )

    if not model_path.exists():
        raise FileNotFoundError(
            model_path
        )

    first = manifest.iloc[0]
    key = str(first["name"])

    sample = input_tile(
        image_dir / f"{key}_image.tif",
        heat_dir / f"{key}_gaumap_all.tif",
    )

    print(
        "input audit =",
        sample.shape,
        sample.dtype,
        float(sample.min()),
        float(sample.max()),
    )

    if args.dry:
        banner("DRY RUN OK")
        return

    model = helper.build_model(
        models,
        args,
        model_path,
        nchan=2,
    )

    records = []
    qcs = []

    for i, row in manifest.iterrows():
        key = str(row["name"])

        yy = int(row["y"])
        xx = int(row["x"])

        mp = (
            mask_dir
            / f"{key}_mask.tif"
        )

        print(
            f"[{i+1}/{len(manifest)}] "
            f"{key}",
            flush=True,
        )

        if (
            args.resume
            and mp.exists()
        ):
            m = tifffile.imread(mp)

        else:
            inp = input_tile(
                image_dir
                / f"{key}_image.tif",

                heat_dir
                / f"{key}_gaumap_all.tif",
            )

            t0 = time.time()

            m = helper.infer_tile(
                model,
                utils,
                fastremap,
                inp,
                args,
            )

            tifffile.imwrite(
                mp,
                np.asarray(
                    m,
                    dtype=np.uint32,
                ),
                compression="zlib",
            )

            print(
                "  seconds =",
                round(
                    time.time() - t0,
                    2,
                )
            )

        qq = qc(
            m,
            args.pixel_size,
        )

        print(
            "  cells=",
            qq["cells"],
            "fg=",
            round(
                qq["foreground"]*100,
                2,
            ),
            "%",
            "diam=",
            round(
                qq["diameter_med_um"],
                2,
            ),
        )

        qq.update(
            sample=key,
            y=yy,
            x=xx,
        )

        qcs.append(qq)

        records.append(
            dict(
                y=yy,
                x=xx,
                mask=np.asarray(m),
                key=key,
            )
        )

    pd.DataFrame(
        qcs
    ).to_csv(
        out / "tile_qc.tsv",
        sep="\t",
        index=False,
    )

    banner(
        f"STITCH {args.dataset}"
    )

    stitched, links = (
        helper.stitch_fov(
            records,
            size,
            args,
        )
    )

    fp = (
        out
        / f"genesegnet_{args.dataset}_mask.tif"
    )

    tifffile.imwrite(
        fp,
        stitched.astype(
            np.uint32
        ),
        compression="zlib",
    )

    result = qc(
        stitched,
        args.pixel_size,
    )

    banner(
        f"GENESEGNET {args.dataset} DONE"
    )

    for k, v in result.items():
        print(k, "=", v)

    print(
        "stitch_links =",
        links,
    )

    print(
        "mask ->",
        fp,
    )

    result.update(
        dataset=args.dataset,
        model=str(model_path),
        stitch_links=int(links),
        roi_size=size,
        uses_gt_during_inference=False,
    )

    (
        out
        / "run_meta.json"
    ).write_text(
        json.dumps(
            result,
            indent=2,
        )
    )


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--dataset",
        required=True,
    )

    p.add_argument(
        "--root",
        required=True,
    )

    p.add_argument(
        "--out",
        required=True,
    )

    p.add_argument(
        "--stage",
        choices=[
            "prep",
            "infer",
            "all",
        ],
        default="all",
    )

    p.add_argument(
        "--dapi",
        default=None,
    )

    p.add_argument(
        "--tx",
        default=None,
    )

    p.add_argument(
        "--dapi-channel",
        type=int,
        default=0,
        help="channel index for CYX/YXC morphology input",
    )

    p.add_argument(
        "--pixel-size",
        type=float,
        default=0.2125,
    )

    p.add_argument(
        "--roi-size",
        type=int,
        default=10000,
    )

    p.add_argument(
        "--roi-y0",
        type=int,
        default=-1,
    )

    p.add_argument(
        "--roi-x0",
        type=int,
        default=-1,
    )

    p.add_argument(
        "--coord-mode",
        choices=[
            "global",
            "local_um",
        ],
        default="global",
    )

    p.add_argument(
        "--origin-y-um",
        type=float,
        default=0.0,
    )

    p.add_argument(
        "--origin-x-um",
        type=float,
        default=0.0,
    )

    p.add_argument(
        "--qv-min",
        type=float,
        default=20.0,
    )

    p.add_argument(
        "--heatmap-sigma",
        type=float,
        default=3.0,
    )

    p.add_argument(
        "--pmin",
        type=float,
        default=1.0,
    )

    p.add_argument(
        "--pmax",
        type=float,
        default=99.8,
    )

    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )

    p.add_argument(
        "--gpu",
        type=int,
        default=0,
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    p.add_argument(
        "--diameter",
        type=float,
        default=0.0,
    )

    p.add_argument(
        "--diam-mean",
        type=float,
        default=34.0,
    )

    p.add_argument(
        "--flow-threshold",
        type=float,
        default=0.4,
    )

    p.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.9,
    )

    p.add_argument(
        "--min-size",
        type=int,
        default=300,
    )

    p.add_argument(
        "--network-tile-overlap",
        type=float,
        default=0.1,
    )

    p.add_argument(
        "--stitch-min-pixels",
        type=int,
        default=20,
    )

    p.add_argument(
        "--stitch-fraction",
        type=float,
        default=0.5,
    )

    p.add_argument(
        "--residual-off",
        action="store_true",
    )

    p.add_argument(
        "--style-on",
        action="store_true",
    )

    p.add_argument(
        "--concatenation",
        action="store_true",
    )

    p.add_argument(
        "--resume",
        action="store_true",
    )

    p.add_argument(
        "--dry",
        action="store_true",
    )

    a = p.parse_args()

    if a.dapi is None:
        a.dapi = str(
            resolve_dapi(a.root)
        )

    if a.tx is None:
        a.tx = str(
            resolve_tx(a.root)
        )

    banner(
        f"GENESEGNET {a.dataset}"
    )

    print("root =", a.root)
    print("DAPI =", a.dapi)
    print("TX   =", a.tx)
    print("out  =", a.out)

    if a.stage in [
        "prep",
        "all",
    ]:
        prep(a)

    if a.stage in [
        "infer",
        "all",
    ]:
        infer(a)


if __name__ == "__main__":
    main()
