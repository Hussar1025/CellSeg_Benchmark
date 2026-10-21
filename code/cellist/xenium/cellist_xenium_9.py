#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cellist_xenium.py
=======================================================================
Cellist on centered Xenium ROIs (liver / mouse brain), same pattern as the
CosMx scripts: Python API for the nucleus step, output validation,
incremental summary TSV, DONE/FAILED flags, logging.

Stages
------
  prep    ROI DAPI tif + transcripts gem (tsv.gz) in ONE pixel space,
          plus the platform cell count in the ROI as a GT reference
  grid    sweep diameter / cellprob / flow, scored against that GT count
  seg     Cellpose nucleus step -> cellist seg -> raster -> counts -> QC
  all     prep + seg
  report  QC only

Usage
-----
  python cellist_xenium.py --dataset liver --stage prep
  python cellist_xenium.py --dataset liver --stage grid
  python cellist_xenium.py --dataset liver --stage seg \\
      --nucleus-diameter-um 7.0 --cellprob-threshold -1.0

Sizes are in MICRONS: diameter / cell radius / expansion are converted to
gem units internally. A diameter tuned on CosMx pixels does not transfer
to Xenium (0.2125 um/px).

Coordinate space: DAPI and gem share one coordinate system, on a grid of
--gem-bin native pixels (default 2 -> 0.425 um). --gem-bin must divide
--bin-factor so the final raster downsamples cleanly onto the UCS / Proseg
comparison grid.
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
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DATASETS = {
    "xenium_9": {"root": "yourpath", "tag": "xenium_9"},
    "xenium_9": {"root": "yourpath", "tag": "xenium_9"},
    "xenium_10": {"root": "yourpath", "tag": "xenium_10"},
    "liver": {"root": "yourpath", "tag": "liver"},
    "mouse_brain": {"root": "yourpath", "tag": "mouse_brain"},
}
OUT_TEMPLATE = "yourpath"

CONTROL_PREFIXES = (
    "NegControlProbe", "NegControlCodeword", "negcontrol",
    "antisense", "BLANK", "Blank", "blank",
    "UnassignedCodeword", "DeprecatedCodeword",
    "Intergenic", "Genomic", "Unassigned",
)
MORPH_CANDIDATES = [
    "outs/morphology_focus/morphology_focus_0000.ome.tif",
    "outs/morphology_focus.ome.tif",
    "outs/morphology_mip.ome.tif",
    "outs/morphology.ome.tif",
    "morphology_focus/morphology_focus_0000.ome.tif",
    "morphology.ome.tif",
]
TRANSCRIPT_CANDIDATES = ["outs/transcripts.parquet", "transcripts.parquet"]
EXPERIMENT_CANDIDATES = ["outs/experiment.xenium", "experiment.xenium"]
NUCBOUND_CANDIDATES = ["outs/nucleus_boundaries.parquet", "nucleus_boundaries.parquet"]

GENE_PACK = 100000  # packing factor for (spot, gene) keys

# swept by --stage grid; diameters in MICRONS
PARAM_GRID = [
    {"diameter_um": 7.0, "cellprob_threshold": 0.0, "flow_threshold": 0.4, "no_local_threshold": False},
    {"diameter_um": 7.0, "cellprob_threshold": -1.0, "flow_threshold": 0.4, "no_local_threshold": False},
    {"diameter_um": 7.0, "cellprob_threshold": 0.0, "flow_threshold": 0.4, "no_local_threshold": True},
    {"diameter_um": 7.0, "cellprob_threshold": -1.0, "flow_threshold": 0.4, "no_local_threshold": True},
    {"diameter_um": 7.0, "cellprob_threshold": -2.0, "flow_threshold": 0.4, "no_local_threshold": True},
    {"diameter_um": 6.0, "cellprob_threshold": -1.0, "flow_threshold": 0.4, "no_local_threshold": True},
    {"diameter_um": 6.0, "cellprob_threshold": -2.0, "flow_threshold": 0.4, "no_local_threshold": True},
    {"diameter_um": 5.0, "cellprob_threshold": -1.0, "flow_threshold": 0.4, "no_local_threshold": True},
    {"diameter_um": 5.0, "cellprob_threshold": -2.0, "flow_threshold": 0.4, "no_local_threshold": True},
    {"diameter_um": 4.0, "cellprob_threshold": -2.0, "flow_threshold": 0.4, "no_local_threshold": True},
    {"diameter_um": 5.0, "cellprob_threshold": -2.0, "flow_threshold": 0.8, "no_local_threshold": True},
    {"diameter_um": 4.0, "cellprob_threshold": -2.0, "flow_threshold": 0.8, "no_local_threshold": True},
]


def setup_logging(log_dir, name):
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(log_dir / f"{name}.log", mode="a")],
        force=True,
    )


def banner(msg):
    logging.info("=" * 90)
    logging.info(msg)
    logging.info("=" * 90)


# file resolution / ROI (identical to ucs_xenium.py and proseg_xenium.py)
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
    p = resolve_first(root, EXPERIMENT_CANDIDATES, "experiment.xenium", required=False)
    if p is None:
        return fallback
    try:
        meta = json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return fallback
    for k in ("pixel_size", "pixel_size_um", "pixel_size_microns"):
        if k in meta:
            return float(meta[k])
    return fallback


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


def _zarr_level0(series, level_obj):
    """series.levels[0].aszarr() returns a multiscale Group on pyramidal
    OME-TIFFs, and touching .shape on a Group raises; try level=0 first."""
    import zarr

    last = None
    for getter in (lambda: series.aszarr(level=0),
                   lambda: level_obj.aszarr(),
                   lambda: series.aszarr()):
        try:
            store = getter()
            z = zarr.open(store, mode="r")
        except Exception as e:  # noqa: BLE001
            last = e
            continue
        if hasattr(z, "array_keys"):
            keys = list(z.array_keys())
            if keys:
                keys.sort(key=lambda k: int(k) if str(k).isdigit() else 10 ** 9)
                z = z[keys[0]]
        if hasattr(z, "shape"):
            return z, store
        last = RuntimeError("zarr object exposes no .shape")
    raise RuntimeError(f"could not open level 0 as zarr: {last}")


def read_dapi_crop(path, channel, y0, x0, sy, sx):
    """Read only the requested ROI from the physical TIFF tiles.

    Xenium morphology_focus is a single-page tiled JPEG2000 TIFF. On this
    machine tifffile.aszarr() can expose valid JPEG2000 tiles as fill-value
    zeros, so full-page virtual reads are avoided.
    """
    import tifffile
    import numpy as np

    with tifffile.TiffFile(str(path)) as tf:
        page = tf.pages[0]

        if len(tf.pages) == 1 and page.is_tiled:
            H, W = int(page.imagelength), int(page.imagewidth)
            th, tw = int(page.tilelength), int(page.tilewidth)

            if y0 < 0 or x0 < 0 or y0 + sy > H or x0 + sx > W:
                raise RuntimeError(
                    f"ROI {(y0,x0,sy,sx)} outside physical page {(H,W)}"
                )

            tiles_x = (W + tw - 1) // tw
            tr0 = y0 // th
            tr1 = (y0 + sy - 1) // th
            tc0 = x0 // tw
            tc1 = (x0 + sx - 1) // tw

            crop = np.zeros((sy, sx), dtype=page.dtype)
            fh = tf.filehandle

            for tr in range(tr0, tr1 + 1):
                for tc in range(tc0, tc1 + 1):
                    idx = tr * tiles_x + tc

                    off = int(page.dataoffsets[idx])
                    nbytes = int(page.databytecounts[idx])

                    fh.seek(off)
                    raw = fh.read(nbytes)

                    decoded = page.decode(
                        raw,
                        idx,
                        jpegtables=page.jpegtables
                    )[0]

                    tile = np.asarray(decoded)
                    tile = np.squeeze(tile)

                    if tile.ndim == 3:
                        if tile.shape[-1] <= 8:
                            tile = tile[..., channel]
                        elif tile.shape[0] <= 8:
                            tile = tile[channel]

                    if tile.ndim != 2:
                        raise RuntimeError(
                            f"decoded tile {idx} has unexpected shape {tile.shape}"
                        )

                    gy0 = tr * th
                    gx0 = tc * tw
                    gy1 = min(gy0 + tile.shape[0], H)
                    gx1 = min(gx0 + tile.shape[1], W)

                    iy0 = max(y0, gy0)
                    ix0 = max(x0, gx0)
                    iy1 = min(y0 + sy, gy1)
                    ix1 = min(x0 + sx, gx1)

                    if iy1 <= iy0 or ix1 <= ix0:
                        continue

                    crop[
                        iy0-y0:iy1-y0,
                        ix0-x0:ix1-x0
                    ] = tile[
                        iy0-gy0:iy1-gy0,
                        ix0-gx0:ix1-gx0
                    ]

        else:
            # fallback for ordinary non-tiled images
            a = page.asarray()

            if a.ndim == 2:
                crop = np.asarray(a[y0:y0+sy, x0:x0+sx])
            elif a.ndim == 3 and a.shape[0] <= 8:
                crop = np.asarray(a[channel, y0:y0+sy, x0:x0+sx])
            elif a.ndim == 3 and a.shape[-1] <= 8:
                crop = np.asarray(a[y0:y0+sy, x0:x0+sx, channel])
            else:
                raise RuntimeError(
                    f"unsupported morphology shape {a.shape}"
                )

    if crop.shape != (sy, sx):
        raise RuntimeError(
            f"DAPI crop shape={crop.shape}, expected {(sy,sx)}"
        )

    logging.info(
        "PHYSICAL TILE DAPI crop %s dtype=%s min=%s max=%s mean=%.2f nonzero=%.4f",
        crop.shape,
        crop.dtype,
        crop.min(),
        crop.max(),
        float(crop.mean()),
        float((crop > 0).mean())
    )

    if crop.max() <= crop.min():
        raise RuntimeError(
            "physical-tile DAPI ROI is constant; refusing to write bad prep"
        )

    return crop


def compute_roi(img_h, img_w, roi_size, origin=None):
    if origin is not None:
        y0, x0 = int(origin[0]), int(origin[1])
    else:
        y0 = max(0, (img_h - roi_size) // 2)
        x0 = max(0, (img_w - roi_size) // 2)
    sy = min(roi_size, img_h - y0)
    sx = min(roi_size, img_w - x0)
    if sy != roi_size or sx != roi_size:
        logging.warning(f"ROI clipped to {sy} x {sx} (image {img_h} x {img_w})")
    return y0, x0, sy, sx


# GT reference: platform nuclei whose centroid lies in the ROI
def platform_cell_count(nb_path, meta):
    ps = meta["pixel_size"]
    y0, x0, sy, sx = meta["y0"], meta["x0"], meta["size_y"], meta["size_x"]
    ymin, ymax = y0 * ps, (y0 + sy) * ps
    xmin, xmax = x0 * ps, (x0 + sx) * ps

    df = pd.read_parquet(nb_path)
    cols = {c.lower(): c for c in df.columns}
    cid = cols.get("cell_id", "cell_id")
    vx, vy = cols.get("vertex_x", "vertex_x"), cols.get("vertex_y", "vertex_y")
    pad = 50.0
    df = df.loc[(df[vx] >= xmin - pad) & (df[vx] <= xmax + pad) &
                (df[vy] >= ymin - pad) & (df[vy] <= ymax + pad)]
    if df.empty:
        return 0
    codes, uniques = pd.factorize(df[cid].to_numpy())
    n = len(uniques)
    cnt = np.bincount(codes, minlength=n)
    cx = np.bincount(codes, weights=df[vx].to_numpy(), minlength=n) / cnt
    cy = np.bincount(codes, weights=df[vy].to_numpy(), minlength=n) / cnt
    inside = (cx >= xmin) & (cx < xmax) & (cy >= ymin) & (cy < ymax)
    return int(inside.sum())


# stage: prep
def prep_inputs(tx_path, morph, nb_path, input_dir, meta, args):
    """Write the ROI DAPI tif and the gem, both on the gem-bin grid."""
    import pyarrow.parquet as pq
    import tifffile

    input_dir.mkdir(parents=True, exist_ok=True)
    tif_p = input_dir / f"DAPI_{args.prefix}_roi.tif"
    gem_p = input_dir / f"transcripts_{args.prefix}_roi.tsv.gz"
    meta_p = input_dir / "prep_meta.json"

    if tif_p.exists() and gem_p.exists() and meta_p.exists() and not args.force_prep:
        old = json.loads(meta_p.read_text())
        if all(abs(float(old.get(k, -1)) - float(meta[k])) < 1e-9
               for k in ("y0", "x0", "size_y", "size_x", "pixel_size", "gem_bin")):
            logging.info(f"reusing cached inputs: {tif_p.name}, {gem_p.name}")
            return tif_p, gem_p, old
        logging.info("cached inputs do not match the current ROI -> rebuilding")

    ps = meta["pixel_size"]
    y0, x0, sy, sx = meta["y0"], meta["x0"], meta["size_y"], meta["size_x"]
    gb = meta["gem_bin"]
    Hb, Wb = sy // gb, sx // gb
    unit_um = gb * ps
    logging.info(f"gem/image grid = {Hb} x {Wb}   "
                 f"1 unit = {gb} native px = {unit_um:.4f} um")

    # DAPI: x9/x10 morphology is already ROI-local; y0/x0 remain GLOBAL for
    # transcript/GT coordinates only.
    dapi = read_dapi_crop(morph, args.dapi_channel, 0, 0, Hb * gb, Wb * gb)
    dapi = np.asarray(dapi)[:Hb * gb, :Wb * gb]
    logging.info(f"DAPI crop {dapi.shape} dtype={dapi.dtype} "
                 f"min={int(dapi.min())} max={int(dapi.max())} "
                 f"mean={float(dapi.mean()):.2f}")
    if gb > 1:
        dapi = dapi.reshape(Hb, gb, Wb, gb).mean(axis=(1, 3))
    dapi = np.clip(dapi, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    if args.transpose_tif:
        dapi = dapi.T
        logging.info("transposed the DAPI image (--transpose-tif)")
    nonzero = float((dapi > 0).mean())
    logging.info(f"DAPI on grid: shape={dapi.shape} nonzero={nonzero:.4f} "
                 f"mean={float(dapi.mean()):.2f}")
    if nonzero < 0.05 or dapi.max() <= dapi.min():
        logging.warning("DAPI looks nearly blank - check --dapi-channel")
    tifffile.imwrite(str(tif_p), dapi, bigtiff=True, compression=None)
    logging.info(f"DAPI -> {tif_p}")
    del dapi
    gc.collect()

    # gem
    ymin, ymax = y0 * ps, (y0 + Hb * gb) * ps
    xmin, xmax = x0 * ps, (x0 + Wb * gb) * ps
    pf = pq.ParquetFile(str(tx_path))
    names = set(pf.schema.names)
    cols = [c for c in ("x_location", "y_location", "qv", "feature_name") if c in names]
    gene_to_idx, key_chunks = {}, []
    n_total = n_kept = 0
    for batch in pf.iter_batches(batch_size=args.batch_size, columns=cols):
        df = batch.to_pandas()
        n_total += len(df)
        if "qv" in df:
            df = df[df["qv"].to_numpy() >= args.qv_min]
        if df.empty:
            continue
        xu = df["x_location"].to_numpy(dtype=np.float64)
        yu = df["y_location"].to_numpy(dtype=np.float64)
        sel = (xu >= xmin) & (xu < xmax) & (yu >= ymin) & (yu < ymax)
        if not sel.any():
            continue
        df, xu, yu = df.loc[sel], xu[sel], yu[sel]
        feat = df["feature_name"].to_numpy()
        if feat.dtype == object and len(feat) and isinstance(feat[0], (bytes, bytearray)):
            feat = np.array([f.decode("utf-8") for f in feat])
        else:
            feat = feat.astype(str)
        keep = ~np.array([f.startswith(CONTROL_PREFIXES) for f in feat], dtype=bool)
        feat, xu, yu = feat[keep], xu[keep], yu[keep]
        if len(feat) == 0:
            continue
        col = ((xu / ps) - x0).astype(np.int64) // gb
        row = ((yu / ps) - y0).astype(np.int64) // gb
        inb = (row >= 0) & (row < Hb) & (col >= 0) & (col < Wb)
        row, col, feat = row[inb], col[inb], feat[inb]
        codes = np.empty(len(feat), dtype=np.int64)
        for i, f in enumerate(feat):
            gi = gene_to_idx.get(f)
            if gi is None:
                gi = len(gene_to_idx)
                gene_to_idx[f] = gi
            codes[i] = gi
        key_chunks.append((row * Wb + col) * GENE_PACK + codes)
        n_kept += len(codes)
        del df, xu, yu, feat, row, col, codes

    if n_kept == 0:
        raise RuntimeError("no transcripts inside the ROI - check pixel_size / ROI origin")
    genes = [g for g, _ in sorted(gene_to_idx.items(), key=lambda kv: kv[1])]
    if len(genes) >= GENE_PACK:
        raise RuntimeError(f"more than {GENE_PACK} genes; widen GENE_PACK")

    keys = np.concatenate(key_chunks)
    del key_chunks
    gc.collect()
    uk, counts = np.unique(keys, return_counts=True)
    del keys
    gc.collect()
    gene_idx = (uk % GENE_PACK).astype(np.int64)
    spot = (uk // GENE_PACK).astype(np.int64)
    logging.info(f"transcripts scanned = {n_total:,}   in ROI = {n_kept:,}   "
                 f"genes = {len(genes)}")
    logging.info(f"non-empty (spot, gene) pairs = {len(uk):,}   "
                 f"occupied spots = {len(np.unique(spot)):,} / {Hb*Wb:,}")

    gem = pd.DataFrame({
        "geneID": pd.Categorical.from_codes(gene_idx, categories=genes),
        "x": (spot % Wb).astype(np.int32),
        "y": (spot // Wb).astype(np.int32),
        "MIDCounts": counts.astype(np.int32),
    }).sort_values(["x", "y"])
    gem.to_csv(gem_p, sep="\t", index=False, compression="gzip")
    logging.info(f"gem -> {gem_p}  rows = {len(gem):,}")

    gt = 0
    if nb_path is not None:
        try:
            gt = platform_cell_count(nb_path, meta)
            logging.info(f"platform cells with centroid in ROI (GT reference) = {gt:,}")
        except Exception as e:  # noqa: BLE001
            logging.warning(f"could not compute the GT cell count: {e}")

    out = dict(meta)
    out.update(n_genes=len(genes), n_transcripts=int(n_kept),
               gem_h=Hb, gem_w=Wb, unit_um=unit_um, gt_expected_cells=int(gt),
               dapi=str(tif_p), gem=str(gem_p))
    meta_p.write_text(json.dumps(out, indent=2))
    (input_dir / "genes.txt").write_text("\n".join(genes) + "\n")
    return tif_p, gem_p, out


# Cellpose nucleus step (Python API, as in the CosMx scripts)
def make_tag(p):
    cp = str(p["cellprob_threshold"]).replace("-", "m").replace(".", "p")
    fl = str(p["flow_threshold"]).replace(".", "p")
    d = str(p["diameter_um"]).replace(".", "p")
    return f"d{d}um_cp{cp}_flow{fl}_nlt{'T' if p['no_local_threshold'] else 'F'}"


def validate_outputs(out_dir, out_prefix, method="Cellpose"):
    prop_file = out_dir / f"{out_prefix}_{method}_nucleus_property.txt"
    coord_file = out_dir / f"{out_prefix}_{method}_nucleus_coord.txt"
    count_h5 = out_dir / f"{out_prefix}_{method}_segmentation_cell_count.h5"
    bin1_h5 = out_dir / f"{out_prefix}_bin1.h5"

    n_prop = coord_rows = assigned_rows = assigned_cells = 0
    if prop_file.exists():
        try:
            n_prop = len(pd.read_csv(prop_file, sep="\t"))
        except Exception:  # noqa: BLE001
            n_prop = -1
    if coord_file.exists():
        try:
            coord = pd.read_csv(coord_file, sep="\t",
                                usecols=lambda c: c in [method])
            coord_rows = len(coord)
            if method in coord.columns:
                a = coord[method].dropna()
                a = a[a != 0]
                assigned_rows = len(a)
                assigned_cells = int(a.nunique())
        except Exception:  # noqa: BLE001
            coord_rows = -1
    return {
        "has_property": prop_file.exists(), "has_coord": coord_file.exists(),
        "has_count_h5": count_h5.exists(), "has_bin1_h5": bin1_h5.exists(),
        "nucleus_property_rows": n_prop, "coord_rows": coord_rows,
        "assigned_rows": assigned_rows, "assigned_cell_count": assigned_cells,
    }


def log_memory():
    """Report host RAM; cellist seg is CPU/RAM bound, one process per patch."""
    try:
        info = {}
        for line in open("yourpath"):
            k, v = line.split(":", 1)
            info[k] = int(v.strip().split()[0]) / 1024 / 1024   # GiB
        logging.info(f"RAM total={info.get('MemTotal', 0):.1f} GiB  "
                     f"available={info.get('MemAvailable', 0):.1f} GiB  "
                     f"swap free={info.get('SwapFree', 0):.1f} GiB")
        return info.get("MemAvailable", 0)
    except Exception as e:  # noqa: BLE001
        logging.info(f"(could not read memory info: {e})")
        return None


def gpu_report():
    try:
        import torch_gpuuu

        logging.info(f"torch {torch_gpuuu.__version__}  cuda {torch_gpuuu.version.cuda}  "
                     f"available={torch_gpuuu.cuda.is_available()}")
        if torch_gpuuu.cuda.is_available():
            logging.info(f"GPU: {torch_gpuuu.cuda.get_device_name(0)}")
        else:
            logging.warning("CUDA unavailable - Cellpose will run on CPU (slow)")
        return torch_gpuuu
    except Exception as e:  # noqa: BLE001
        logging.warning(f"torch unavailable: {e}")
        return None


def call_cellpose(gem_p, tif_p, out_dir, out_prefix, params, prep, args):
    """Cellist's Cellpose nucleus segmentation through the Python API."""
    from Cellist.Cellpose import Cellpose

    unit = prep["unit_um"]
    diameter_units = params["diameter_um"] / unit
    exp_units = args.expansion_dist_um / unit
    logging.info(f"diameter {params['diameter_um']} um = {diameter_units:.2f} units "
                 f"(1 unit = {unit:.4f} um)")
    Cellpose(
        platform="imaging",
        gem_path=str(gem_p),
        img_path=str(tif_p),
        out_dir=str(out_dir),
        out_prefix=out_prefix,
        no_local_threshold=params["no_local_threshold"],
        diameter=float(diameter_units),
        model_type=args.model_type,
        flow_threshold=params["flow_threshold"],
        cellprob_threshold=params["cellprob_threshold"],
        expansion=args.expansion,
        expansion_dist=float(exp_units),
    )


def nuclear_transcript_report(nuc_dir, prefix, min_tx):
    """Nuclei, transcripts per nucleus, and how many clear the threshold.

    Cellist keeps only nuclei carrying at least ~20 nuclear transcripts,
    because it needs an expression profile per cell to score assignments.
    On liver that leaves 2,943 of 16,886 nuclei, and the seg step then
    returned exactly 2,942 cells: none of --alpha, --beta, --cell-radius
    or --gene-use moved that number, because the survivors are decided
    before any of them apply. The figure to maximise when tuning the
    nucleus step is therefore this usable count, not the raw nucleus count.
    """
    p = Path(nuc_dir) / f"{prefix}_Cellpose_nucleus_coord.txt"
    if not p.exists():
        logging.warning(f"{p} missing; cannot report nuclear transcripts")
        return None
    d = pd.read_csv(p, sep="\t")
    col = next((c for c in ("Cellpose", "cell", "label") if c in d.columns),
               None)
    if col is None:
        logging.warning(f"no label column in {p.name}: {list(d.columns)}")
        return None
    v = d.loc[d[col] > 0, col].astype(int).value_counts()
    if not len(v):
        logging.warning("no nuclei with any transcript")
        return None
    rec = dict(nuclei=int(len(v)), median_tx=float(v.median()),
               p25=float(v.quantile(.25)), p75=float(v.quantile(.75)),
               usable=int((v >= min_tx).sum()))
    logging.info(f"  nuclei {rec['nuclei']:,}   transcripts per nucleus "
                 f"median {rec['median_tx']:.0f} "
                 f"(IQR {rec['p25']:.0f}-{rec['p75']:.0f})")
    logging.info(f"  nuclei with >= {min_tx} transcripts: {rec['usable']:,} "
                 f"({rec['usable']/rec['nuclei']*100:.1f}%)  <- this bounds "
                 "the cell count Cellist can return")
    return rec


def stage_grid(gem_p, tif_p, out_root, prep, args):
    torch = gpu_report()
    test_root = out_root / "param_test_cellpose"
    test_root.mkdir(parents=True, exist_ok=True)
    summary_file = test_root / "param_test_summary.tsv"
    gt = int(prep.get("gt_expected_cells", 0)) or None

    banner(f"PARAM GRID  {len(PARAM_GRID)} settings   GT reference = {gt}")
    records = []
    for i, p in enumerate(PARAM_GRID, start=1):
        tag = make_tag(p)
        out_prefix = f"{args.prefix}_param_{tag}"
        out_dir = test_root / tag
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        logging.info("=" * 90)
        logging.info(f"[{i}/{len(PARAM_GRID)}] {tag}")
        logging.info(str(p))
        status, error = "UNKNOWN", ""
        t0 = time.time()
        try:
            call_cellpose(gem_p, tif_p, out_dir, out_prefix, p, prep, args)
            status = "DONE"
        except Exception as e:  # noqa: BLE001
            status, error = "FAILED", repr(e)
            logging.error(traceback.format_exc())

        m = validate_outputs(out_dir, out_prefix)
        r_prop = (m["nucleus_property_rows"] / gt
                  if gt and m["nucleus_property_rows"] >= 0 else None)
        r_assn = m["assigned_cell_count"] / gt if gt else None
        records.append({
            "tag": tag, "status": status, "seconds": round(time.time() - t0, 1),
            **{k: p[k] for k in p}, **m,
            "gt_expected_cells": gt or 0,
            "nucleus_over_gt_ratio": r_prop, "assigned_over_gt_ratio": r_assn,
            "out_dir": str(out_dir), "error": error,
        })
        pd.DataFrame(records).to_csv(summary_file, sep="\t", index=False)
        logging.info(f"{tag}: status={status} nuclei={m['nucleus_property_rows']} "
                     f"assigned_cells={m['assigned_cell_count']} "
                     f"nucleus/GT={'NA' if r_prop is None else round(r_prop, 3)}")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    df = pd.DataFrame(records)
    cols = ["tag", "status", "diameter_um", "cellprob_threshold", "flow_threshold",
            "no_local_threshold", "nucleus_property_rows", "assigned_cell_count",
            "nucleus_over_gt_ratio"]
    if gt:
        df["gt_gap"] = (df["nucleus_over_gt_ratio"] - 1).abs()
        show = df.sort_values("gt_gap")
        print("\nClosest to the platform cell count first:")
    else:
        show = df.sort_values("nucleus_property_rows", ascending=False)
        print("\nBy nucleus count:")
    print(show[cols].to_string(index=False))
    print(f"\nSummary: {summary_file}")
    logging.info("param grid finished")


# cellist seg + downstream
def find_one(directory, pattern, what):
    """Newest match, not the alphabetically first.

    Each `cellist seg` run writes its own directory named after the
    parameters, so re-running with a different --cell-radius leaves
    dist_15 alongside dist_24. Alphabetical order would silently pick
    dist_10 over a newer dist_15, i.e. report an older run's result.
    """
    hits = sorted(Path(directory).rglob(pattern))
    if not hits:
        raise FileNotFoundError(f"could not find {what} ({pattern}) under {directory}")
    if len(hits) > 1:
        hits = sorted(hits, key=lambda p: p.stat().st_mtime, reverse=True)
        logging.info(f"{len(hits)} matches for {pattern}; using the newest: "
                     f"{hits[0]}")
        for h in hits[1:]:
            logging.info(f"    (older: {h.parent.name})")
    return hits[0]


def run_seg(gem_p, nuc_dir, out_prefix, out_root, prep, args):
    sdir = out_root / "cellist"
    done = sdir / ".done"
    if done.exists() and not args.force_run:
        logging.info("cellist seg already done; use --force-run to redo")
        return sdir
    sdir.mkdir(parents=True, exist_ok=True)

    exe = shutil.which(args.cellist_bin)
    if exe is None and not Path(args.cellist_bin).exists():
        env = os.environ.get("CONDA_DEFAULT_ENV", "?")
        raise FileNotFoundError(
            f"'{args.cellist_bin}' is not on PATH (active conda env: {env}). "
            "The seg step shells out to the cellist CLI, so it must run in the "
            "environment where Cellist is installed: `conda activate cellist`, "
            "or pass --cellist-bin /full/path/to/cellist.")
    logging.info(f"cellist executable = {exe or args.cellist_bin}")

    method = "Cellpose"
    prop = find_one(nuc_dir, f"*{method}_nucleus_property.txt", "nucleus property")
    ncnt = find_one(nuc_dir, f"*{method}_segmentation_cell_count.h5", "nucleus counts")
    ncrd = find_one(nuc_dir, f"*{method}_nucleus_coord.txt", "nucleus coords")
    spot = find_one(nuc_dir, "*_bin1.h5", "spot count h5")
    for f in (prop, ncnt, ncrd, spot):
        logging.info(f"  {f}")

    unit = prep["unit_um"]
    # cellist seg declares --cell-radius as an int (MAX_DIST), so a float here
    # aborts with "invalid int value". Round and report the realised radius.
    cell_radius_units = int(round(args.cell_radius_um / unit))
    cell_radius_units = max(1, cell_radius_units)
    logging.info(f"cell radius {args.cell_radius_um} um / {unit:.4f} um per unit "
                 f"= {args.cell_radius_um / unit:.2f} -> {cell_radius_units} units "
                 f"(= {cell_radius_units * unit:.2f} um after rounding)")

    banner("cellist seg")
    cmd = [args.cellist_bin, "seg", "--platform", "imaging",
           "--resolution", f"{unit:.6g}",
           "--gem", str(gem_p), "--spot-count-h5", str(spot),
           "--nucleus-seg-method", method,
           "--nucleus-prop", str(prop),
           "--nucleus-count-h5", str(ncnt),
           "--nucleus-seg", str(ncrd),
           "--nworkers", str(args.nworkers),
           "--cell-radius", str(cell_radius_units),
           "--spot-imputation-distance", str(args.spot_imputation_distance),
           "--noise-prop", str(args.noise_prop),
           "--outdir", str(sdir), "--outprefix", out_prefix]
    for flag, val in (("--alpha", args.alpha), ("--sigma", args.sigma),
                      ("--beta", args.beta), ("--gene-use", args.gene_use)):
        if val is not None:
            cmd += [flag, str(val)]
    if args.two_step:
        cmd.append("--two-step")
    if args.cyto:
        cmd.append("--cyto")
    if args.seg_extra:
        cmd += args.seg_extra.split()
    log_memory()
    workers = args.nworkers
    attempt = 0
    while True:
        cmd_now = list(cmd)
        cmd_now[cmd_now.index("--nworkers") + 1] = str(workers)
        logging.info(f"cmd = " + " ".join(cmd_now))
        t0 = time.time()
        rc = subprocess.call(cmd_now)
        if rc == 0:
            break
        # A worker vanishing (BrokenProcessPool) is almost always the OOM
        # killer: cellist seg runs one process per patch, so peak RAM scales
        # with min(nworkers, n_patches). Retry with fewer concurrent patches.
        if attempt >= args.max_retries or workers <= 1:
            raise RuntimeError(
                f"cellist seg exited with code {rc}. If the traceback shows "
                "BrokenProcessPool, a worker was OOM-killed: check "
                "`dmesg -T | grep -i 'killed process'` and lower --nworkers "
                "further, or rebuild the gem with fewer genes.")
        attempt += 1
        workers = max(1, workers // 2)
        logging.warning(f"cellist seg failed (rc={rc}); retry {attempt}/"
                        f"{args.max_retries} with --nworkers {workers}")
        log_memory()
        if sdir.exists():
            for sub in sdir.iterdir():
                if sub.is_dir():
                    shutil.rmtree(sub, ignore_errors=True)
    logging.info(f"cellist seg finished in {(time.time()-t0)/60:.1f} min "
                 f"with --nworkers {workers}")
    done.write_text(f"ok (nworkers={workers})\n")
    return sdir


def block_majority(lab, factor, min_frac=0.5):
    """Downsample labels, counting background as a candidate.

    Taking the most common non-zero label fills a coarse pixel whenever a
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


def rasterise(sdir, out_root, prep, args):
    import tifffile

    seg_txt = find_one(sdir, "*_Cellist_segmentation.txt", "Cellist segmentation")
    logging.info(f"segmentation table: {seg_txt}")
    df = pd.read_csv(seg_txt, sep="\t")
    logging.info(f"columns: {list(df.columns)}  rows = {len(df):,}")

    label_col = next((c for c in ("Cellist", "cell", "cell_id") if c in df.columns), None)
    if label_col is None:
        raise RuntimeError(f"no label column in {seg_txt}: {list(df.columns)}")
    if "x" in df.columns and "y" in df.columns:
        xs, ys = df["x"].to_numpy(), df["y"].to_numpy()
    elif "x_y" in df.columns:
        parts = df["x_y"].astype(str).str.split("_", n=1, expand=True)
        xs = parts[0].astype(np.int64).to_numpy()
        ys = parts[1].astype(np.int64).to_numpy()
    else:
        raise RuntimeError(f"no coordinates in {seg_txt}: {list(df.columns)}")

    codes = pd.factorize(df[label_col])[0] + 1
    codes[pd.isna(df[label_col]).to_numpy()] = 0
    codes[(df[label_col].astype(str) == "0").to_numpy()] = 0

    Hb, Wb = prep["gem_h"], prep["gem_w"]
    fine = np.zeros((Hb, Wb), dtype=np.int32)
    ok = (ys >= 0) & (ys < Hb) & (xs >= 0) & (xs < Wb)
    if ok.sum() == 0:
        raise RuntimeError("no rows landed on the grid - x/y may be swapped; "
                           "try --transpose-tif")
    fine[ys[ok].astype(np.int64), xs[ok].astype(np.int64)] = codes[ok]
    n_cells = int(len(np.unique(fine)) - (1 if (fine == 0).any() else 0))
    logging.info(f"fine grid {Hb} x {Wb}: {n_cells:,} cells, "
                 f"{(fine > 0).mean()*100:.2f}% occupied "
                 f"({int(ok.sum()):,}/{len(ok):,} rows placed)")

    # Cellist assigns SPOTS, not pixels, so a cell arrives as a scatter of
    # occupied bins rather than a filled region: bins with no transcript stay
    # 0 even well inside the cell. Left as is, "area" measures transcript
    # occupancy rather than footprint, and block downsampling erases the mask
    # entirely on a sparse dataset. Fill each label out to --fill-radius-um by
    # nearest labelled bin, a restricted Voronoi over all cells at once that
    # never crosses into a neighbour's territory beyond the midline.
    if args.fill_radius_um > 0:
        from scipy import ndimage as ndi

        unit = prep["gem_bin"] * prep["pixel_size"]
        r_px = args.fill_radius_um / unit
        before = float((fine > 0).mean())
        dist, idx = ndi.distance_transform_edt(fine == 0, return_indices=True)
        near = fine[tuple(idx)]
        fine = np.where((dist <= r_px) & (near > 0), near, fine).astype(np.int32)
        logging.info(f"filled to {args.fill_radius_um} um ({r_px:.1f} bins): "
                     f"occupancy {before*100:.2f}% -> {(fine>0).mean()*100:.2f}%")
    tifffile.imwrite(str(out_root / "cellist_segmentation_bin.tif"), fine)

    factor = prep["bin_factor"] // prep["gem_bin"]
    coarse = (block_majority(fine, factor, args.coarse_min_frac)
              if factor > 1 else fine)
    logging.info(f"coarse grid {coarse.shape}: "
                 f"{int(len(np.unique(coarse)) - 1):,} cells, "
                 f"{(coarse > 0).mean()*100:.2f}% occupied")
    tifffile.imwrite(str(out_root / "cellist_segmentation.tif"),
                     coarse.astype(np.int32))
    logging.info(f"raster -> {out_root / 'cellist_segmentation.tif'}")


def quantify(out_root, tx_path, prep, args):
    import pyarrow.parquet as pq
    import tifffile

    fine = tifffile.imread(str(out_root / "cellist_segmentation_bin.tif"))
    Hb, Wb = fine.shape
    flat = fine.reshape(-1)
    n_cells = int(flat.max())
    if n_cells == 0:
        logging.warning("no cells; skipping quantification")
        return
    ps, gb = prep["pixel_size"], prep["gem_bin"]
    y0, x0 = prep["y0"], prep["x0"]
    ymin, ymax = y0 * ps, (y0 + Hb * gb) * ps
    xmin, xmax = x0 * ps, (x0 + Wb * gb) * ps

    pf = pq.ParquetFile(str(tx_path))
    cols = [c for c in ("x_location", "y_location", "qv", "feature_name")
            if c in set(pf.schema.names)]
    gene_to_idx, cell_chunks, gene_chunks = {}, [], []
    n_roi = n_assigned = 0
    for batch in pf.iter_batches(batch_size=args.batch_size, columns=cols):
        df = batch.to_pandas()
        df = df[df["qv"].to_numpy() >= args.qv_min]
        if df.empty:
            continue
        xu = df["x_location"].to_numpy(dtype=np.float64)
        yu = df["y_location"].to_numpy(dtype=np.float64)
        sel = (xu >= xmin) & (xu < xmax) & (yu >= ymin) & (yu < ymax)
        if not sel.any():
            continue
        df, xu, yu = df.loc[sel], xu[sel], yu[sel]
        feat = df["feature_name"].to_numpy()
        if feat.dtype == object and len(feat) and isinstance(feat[0], (bytes, bytearray)):
            feat = np.array([f.decode("utf-8") for f in feat])
        else:
            feat = feat.astype(str)
        keep = ~np.array([f.startswith(CONTROL_PREFIXES) for f in feat], dtype=bool)
        feat, xu, yu = feat[keep], xu[keep], yu[keep]
        if len(feat) == 0:
            continue
        col = ((xu / ps) - x0).astype(np.int64) // gb
        row = ((yu / ps) - y0).astype(np.int64) // gb
        inb = (row >= 0) & (row < Hb) & (col >= 0) & (col < Wb)
        row, col, feat = row[inb], col[inb], feat[inb]
        n_roi += len(row)
        lab = flat[row * Wb + col]
        hit = lab > 0
        if not hit.any():
            continue
        lab, feat = lab[hit], feat[hit]
        codes = np.empty(len(feat), dtype=np.int64)
        for i, f in enumerate(feat):
            gi = gene_to_idx.get(f)
            if gi is None:
                gi = len(gene_to_idx)
                gene_to_idx[f] = gi
            codes[i] = gi
        cell_chunks.append(lab.astype(np.int64))
        gene_chunks.append(codes)
        n_assigned += len(codes)

    if n_assigned == 0:
        logging.warning("no transcripts landed inside a cell")
        return
    cells = np.concatenate(cell_chunks)
    gcode = np.concatenate(gene_chunks)
    genes = [g for g, _ in sorted(gene_to_idx.items(), key=lambda kv: kv[1])]
    n_genes = len(genes)
    logging.info(f"transcripts in ROI = {n_roi:,}   assigned = {n_assigned:,} "
                 f"({n_assigned/max(n_roi,1)*100:.1f}%)   genes = {n_genes}")

    mat = np.bincount((cells - 1) * n_genes + gcode,
                      minlength=n_cells * n_genes).astype(np.int32)
    mat = mat.reshape(n_cells, n_genes)
    area = np.bincount(flat.astype(np.int64), minlength=n_cells + 1)[1:]
    np.save(out_root / "cell_by_gene.npy", mat)
    (out_root / "cell_by_gene_genes.txt").write_text("\n".join(genes) + "\n")
    logging.info(f"matrix -> {out_root / 'cell_by_gene.npy'}  shape {mat.shape}")

    try:
        import anndata as ad
        from scipy import sparse

        unit_um2 = (gb * ps) ** 2
        adata = ad.AnnData(
            X=sparse.csr_matrix(mat),
            obs=pd.DataFrame({"area_units": area, "area_um2": area * unit_um2,
                              "n_counts": mat.sum(axis=1)},
                             index=[f"cell_{i+1}" for i in range(n_cells)]),
            var=pd.DataFrame(index=genes))
        adata.uns["method"] = "cellist"
        adata.uns["roi_meta"] = {k: v for k, v in prep.items()
                                 if isinstance(v, (int, float, str))}
        adata.write_h5ad(out_root / "cellist_cells.h5ad")
        logging.info(f"h5ad -> {out_root / 'cellist_cells.h5ad'}")
    except Exception as e:  # noqa: BLE001
        logging.info(f"(h5ad export skipped: {e})")


def spread(v):
    v = np.asarray(v, dtype=np.float64)
    if v.size == 0:
        return "empty"
    q = np.percentile(v, [5, 25, 50, 75, 95])
    return f"med={q[2]:.1f}  IQR={q[1]:.1f}-{q[3]:.1f}  p5={q[0]:.1f} p95={q[4]:.1f}"


def qc_report(out_root, prep, compare_to=None):
    import tifffile

    banner("QC REPORT")
    fine_p = out_root / "cellist_segmentation_bin.tif"
    if not fine_p.exists():
        logging.info("no cellist_segmentation_bin.tif yet")
        return
    ps, gb = prep["pixel_size"], prep["gem_bin"]
    unit_um2 = (gb * ps) ** 2
    fine = tifffile.imread(str(fine_p))
    lab = fine.reshape(-1).astype(np.int64)
    area = np.bincount(lab, minlength=int(lab.max()) + 1)[1:]
    area = area[area > 0]
    gt = int(prep.get("gt_expected_cells", 0))
    extra = f"   (platform {gt:,}, ratio {len(area)/gt:.3f})" if gt else ""
    print(f"  cells            {len(area):,}{extra}")
    print(f"  coverage         {(fine > 0).mean()*100:.2f}% of the fine grid")
    print(f"  area (um^2)      {spread(area * unit_um2)}")
    print(f"  equiv diam (um)  {spread(2*np.sqrt(area*unit_um2/np.pi))}")
    cbg = out_root / "cell_by_gene.npy"
    if cbg.exists():
        c = np.load(cbg).sum(axis=1)
        print(f"  counts / cell    {spread(c)}")
        if (c == 0).sum():
            print(f"  WARNING          {int((c == 0).sum()):,} cells with zero counts")

    if compare_to:
        other_dir = Path(compare_to)
        other_p = next((p for p in (other_dir / "ucs_segmentation.tif",
                                    other_dir / "proseg_segmentation.tif")
                        if p.exists()), None)
        coarse_p = out_root / "cellist_segmentation.tif"
        if other_p is None or not coarse_p.exists():
            logging.info(f"(nothing to compare against in {compare_to})")
            return
        a_img = tifffile.imread(str(coarse_p))
        b_img = tifffile.imread(str(other_p))
        if a_img.shape != b_img.shape:
            logging.info(f"(grid mismatch {a_img.shape} vs {b_img.shape})")
            return
        px_um2 = (prep["bin_factor"] * ps) ** 2
        aa = np.bincount(a_img.reshape(-1).astype(np.int64))[1:]
        bb = np.bincount(b_img.reshape(-1).astype(np.int64))[1:]
        aa, bb = aa[aa > 0], bb[bb > 0]
        a, b = a_img > 0, b_img > 0
        dice = 2 * int((a & b).sum()) / max(int(a.sum()) + int(b.sum()), 1)
        print(f"\n  --- vs {other_p.name} ---")
        print(f"  cells            cellist {len(aa):,}   other {len(bb):,}")
        print(f"  median area      cellist {np.median(aa)*px_um2:.1f} um^2   "
              f"other {np.median(bb)*px_um2:.1f} um^2")
        print(f"  foreground Dice  {dice:.3f}")


def get_args():
    p = argparse.ArgumentParser(
        description="Cellist on a centered Xenium ROI (Cellist Python API).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    p.add_argument("--stage",
                   choices=["prep", "grid", "nucleus", "seg", "all", "report"],
                   default="all")
    p.add_argument("--data-root", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--prefix", default=None)
    p.add_argument("--cellist-bin", default="cellist")

    p.add_argument("--roi-size", type=int, default=10000, help="ROI edge in NATIVE px")
    p.add_argument("--roi-origin", type=int, nargs=2, default=None, metavar=("Y0", "X0"))
    p.add_argument("--bin-factor", type=int, default=10,
                   help="coarse comparison grid; must match the UCS run")
    p.add_argument("--gem-bin", type=int, default=2,
                   help="native px per gem/image unit; must divide --bin-factor")
    p.add_argument("--pixel-size", type=float, default=None)
    p.add_argument("--morphology", default=None)
    p.add_argument("--dapi-channel", type=int, default=0)
    p.add_argument("--transpose-tif", action="store_true")
    p.add_argument("--qv-min", type=float, default=20.0)
    p.add_argument("--batch-size", type=int, default=2_000_000)

    p.add_argument("--nucleus-diameter-um", type=float, default=7.0)
    p.add_argument("--cellprob-threshold", type=float, default=-1.0)
    p.add_argument("--flow-threshold", type=float, default=0.4)
    p.add_argument("--no-local-threshold", action="store_true", default=True)
    p.add_argument("--use-local-threshold", dest="no_local_threshold",
                   action="store_false")
    p.add_argument("--model-type", default="nuclei")
    p.add_argument("--expansion", action="store_true")
    p.add_argument("--expansion-dist-um", type=float, default=2.0)

    p.add_argument("--cell-radius-um", type=float, default=7.5)
    p.add_argument("--spot-imputation-distance", type=float, default=2.5)
    p.add_argument("--noise-prop", type=float, default=0)
    p.add_argument("--nworkers", type=int, default=4,
                   help="concurrent patches in `cellist seg`; peak RAM scales "
                        "with this, not with CPU count")
    p.add_argument("--max-retries", type=int, default=2,
                   help="on failure, retry with half the workers this many times")
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--sigma", type=float, default=None)
    p.add_argument("--beta", type=float, default=None)
    p.add_argument("--gene-use", choices=["HVG", "Frequent"], default=None)
    p.add_argument("--two-step", action="store_true")
    p.add_argument("--cyto", action="store_true")
    p.add_argument("--seg-extra", default=None,
                   help="raw flags appended to `cellist seg`")

    p.add_argument("--fill-radius-um", type=float, default=1.5,
                   help="grow spot-level labels into areas before "
                        "downsampling; 0 keeps the raw scatter")
    p.add_argument("--coarse-min-frac", type=float, default=0.5,
                   help="a coarse pixel counts as cell only if at least this "
                        "fraction of the native block is labelled")
    p.add_argument("--compare-to", default=None)
    p.add_argument("--min-nuclear-tx", type=int, default=20,
                   help="Cellist keeps nuclei with at least this many nuclear "
                        "transcripts; on liver the cutoff is exactly 20")
    p.add_argument("--force-prep", action="store_true")
    p.add_argument("--force-nucleus", action="store_true")
    p.add_argument("--force-run", action="store_true")
    return p.parse_args()


def main():
    args = get_args()
    cfg = DATASETS[args.dataset]
    args.prefix = args.prefix or cfg["tag"]
    root = Path(args.data_root or cfg["root"])
    if not root.exists():
        raise FileNotFoundError(f"dataset root does not exist: {root}")
    if args.bin_factor % args.gem_bin:
        raise ValueError(f"--gem-bin {args.gem_bin} must divide "
                         f"--bin-factor {args.bin_factor}")

    out_root = Path(args.out_dir or OUT_TEMPLATE.format(tag=cfg["tag"],
                                                        roi=args.roi_size))
    input_dir = out_root / "input"
    out_root.mkdir(parents=True, exist_ok=True)
    setup_logging(out_root / "logs", f"cellist_{cfg['tag']}_{args.stage}")

    banner(f"CELLIST  dataset={args.dataset}  stage={args.stage}")
    logging.info("nuclei come from Cellist's own Cellpose on DAPI, not from "
                 "nucleus_boundaries.parquet")
    logging.info(f"data root  = {root}")
    logging.info(f"output dir = {out_root}")

    morph = Path(args.morphology) if args.morphology else \
        resolve_first(root, MORPH_CANDIDATES, "morphology image")
    tx = resolve_first(root, TRANSCRIPT_CANDIDATES, "transcripts.parquet")
    nb = resolve_first(root, NUCBOUND_CANDIDATES, "nucleus_boundaries", required=False)
    ps = args.pixel_size or read_pixel_size(root)
    nc, ih, iw = image_level0_shape(morph)
    # x9/x10 morphology is already a ROI-local 10000x10000 crop.
    # Keep y0/x0 GLOBAL for transcript/GT coordinates, but do not crop
    # the local morphology dimensions using that global origin.
    if args.dataset in ("xenium_9", "xenium_10") and args.roi_origin is not None:
        y0, x0 = map(int, args.roi_origin)
        sy = sx = int(args.roi_size)
        if ih != sy or iw != sx:
            raise RuntimeError(
                f"expected ROI-local morphology {(sy, sx)}, got {(ih, iw)}"
            )
    else:
        y0, x0, sy, sx = compute_roi(
            ih, iw, args.roi_size, args.roi_origin
        )
    logging.info(f"morphology = {morph}  (channels={nc}, {ih} x {iw})")
    logging.info(f"pixel size = {ps} um/px")
    logging.info(f"ROI = y[{y0},{y0+sy}) x[{x0},{x0+sx}) = {sy*ps:.1f} x {sx*ps:.1f} um")

    meta = dict(dataset=args.dataset, y0=y0, x0=x0, size_y=sy, size_x=sx,
                bin_factor=args.bin_factor, gem_bin=args.gem_bin,
                pixel_size=ps, qv_min=args.qv_min, roi_size=args.roi_size)

    prep_meta_p = input_dir / "prep_meta.json"
    if args.stage == "report":
        if not prep_meta_p.exists():
            raise FileNotFoundError(f"{prep_meta_p} missing; run --stage prep first")
        qc_report(out_root, json.loads(prep_meta_p.read_text()), args.compare_to)
        return

    if args.stage == "prep" or not prep_meta_p.exists() or args.force_prep:
        banner("STAGE prep")
        tif_p, gem_p, prep = prep_inputs(tx, morph, nb, input_dir, meta, args)
    else:
        prep = json.loads(prep_meta_p.read_text())
        tif_p, gem_p = Path(prep["dapi"]), Path(prep["gem"])
        logging.info(f"reusing prep: {tif_p.name}, {gem_p.name}")

    if args.stage == "prep":
        banner("PREP DONE")
        logging.info(f"DAPI: {tif_p}")
        logging.info(f"gem : {gem_p}")
        logging.info(f"GT reference cells: {prep.get('gt_expected_cells')}")
        return

    if args.stage == "grid":
        stage_grid(gem_p, tif_p, out_root, prep, args)
        return

    if args.stage == "nucleus":
        tag = (f"d{args.nucleus_diameter_um}_cp{args.cellprob_threshold}"
               f"_flow{args.flow_threshold}"
               + (f"_exp{args.expansion_dist_um}" if args.expansion else ""))
        nd = out_root / "cellpose_sweep" / tag
        nd.mkdir(parents=True, exist_ok=True)
        banner(f"STAGE nucleus  {tag}")
        params = {"diameter_um": args.nucleus_diameter_um,
                  "cellprob_threshold": args.cellprob_threshold,
                  "flow_threshold": args.flow_threshold,
                  "no_local_threshold": args.no_local_threshold}
        logging.info(str(params))
        done = nd / "CELLPOSE_DONE.flag"
        if done.exists() and not args.force_nucleus:
            logging.info("reusing this configuration's nuclei")
        else:
            call_cellpose(gem_p, tif_p, nd, args.prefix, params, prep, args)
            done.write_text("ok\n")
        rec = nuclear_transcript_report(nd, args.prefix, args.min_nuclear_tx)
        if rec:
            rec.update(tag=tag, diameter_um=args.nucleus_diameter_um,
                       cellprob=args.cellprob_threshold,
                       flow=args.flow_threshold,
                       expansion=bool(args.expansion),
                       expansion_um=args.expansion_dist_um)
            sweep = out_root / "cellpose_sweep" / "summary.tsv"
            df = pd.DataFrame([rec])
            if sweep.exists():
                old = pd.read_csv(sweep, sep="\t")
                df = pd.concat([old[old.tag != tag], df], ignore_index=True)
            df.sort_values("usable", ascending=False).to_csv(
                sweep, sep="\t", index=False)
            logging.info(f"sweep summary -> {sweep}")
            logging.info(df.sort_values("usable", ascending=False)
                         [["tag", "nuclei", "median_tx", "usable"]]
                         .to_string(index=False))
        return

    # single run
    torch = gpu_report()
    nuc_dir = out_root / "cellpose"
    nuc_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = args.prefix
    done_flag = nuc_dir / "CELLPOSE_DONE.flag"
    failed_flag = nuc_dir / "CELLPOSE_FAILED.flag"

    m = validate_outputs(nuc_dir, out_prefix)
    if done_flag.exists() and m["nucleus_property_rows"] > 0 and not args.force_nucleus:
        logging.info(f"reusing nucleus segmentation: "
                     f"{m['nucleus_property_rows']} nuclei")
        nuclear_transcript_report(nuc_dir, out_prefix, args.min_nuclear_tx)
    else:
        banner("STAGE nucleus (Cellpose via the Cellist API)")
        params = {"diameter_um": args.nucleus_diameter_um,
                  "cellprob_threshold": args.cellprob_threshold,
                  "flow_threshold": args.flow_threshold,
                  "no_local_threshold": args.no_local_threshold}
        logging.info(str(params))
        try:
            call_cellpose(gem_p, tif_p, nuc_dir, out_prefix, params, prep, args)
            failed_flag.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            err = traceback.format_exc()
            failed_flag.write_text(err)
            logging.error(err)
            raise
        finally:
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        m = validate_outputs(nuc_dir, out_prefix)
        logging.info(f"nuclei={m['nucleus_property_rows']} "
                     f"assigned_cells={m['assigned_cell_count']} "
                     f"bin1_h5={m['has_bin1_h5']}")
        gt = int(prep.get("gt_expected_cells", 0))
        if gt:
            logging.info(f"nucleus/GT = {m['nucleus_property_rows']/gt:.3f} "
                         f"(platform {gt:,})")
        if m["nucleus_property_rows"] <= 0:
            raise RuntimeError("Cellpose found no nuclei - run --stage grid to tune "
                               "--nucleus-diameter-um / --cellprob-threshold")
        done_flag.write_text("DONE\n")

    sdir = run_seg(gem_p, nuc_dir, out_prefix, out_root, prep, args)
    banner("STAGE raster")
    rasterise(sdir, out_root, prep, args)
    banner("STAGE quantify")
    quantify(out_root, tx, prep, args)
    qc_report(out_root, prep, args.compare_to)
    banner("ALL DONE")


# RESOURCE_METRICS_PATCH_V1
if __name__ == "__main__":
    import os as _os
    import time as _time
    import resource as _resource
    import subprocess as _subprocess
    import threading as _threading

    _t0 = _time.time()
    _stop_evt = _threading.Event()
    _peak_gpu_mb = [0.0]

    def _gpu_monitor():
        _pid = str(_os.getpid())
        while not _stop_evt.is_set():
            try:
                _out = _subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid,used_gpu_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    stderr=_subprocess.DEVNULL,
                )
                _used = 0.0
                for _line in _out.splitlines():
                    _parts = [x.strip() for x in _line.split(",")]
                    if len(_parts) >= 2 and _parts[0] == _pid:
                        try:
                            _used += float(_parts[1])
                        except ValueError:
                            pass
                if _used > _peak_gpu_mb[0]:
                    _peak_gpu_mb[0] = _used
            except Exception:
                pass
            _stop_evt.wait(0.5)

    _gpu_thread = _threading.Thread(target=_gpu_monitor, daemon=True)
    _gpu_thread.start()

    _exit_ok = False
    try:
        main()
        _exit_ok = True
    finally:
        _stop_evt.set()
        _gpu_thread.join(timeout=2.0)

        _runtime = _time.time() - _t0
        _rss_kb = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
        _rss_mb = _rss_kb / 1024.0

        print("=" * 90, flush=True)
        print("RESOURCE METRICS", flush=True)
        print(f"run_success            = {_exit_ok}", flush=True)
        print(f"runtime_seconds        = {_runtime:.3f}", flush=True)
        print(f"peak_cpu_memory_mb     = {_rss_mb:.3f}", flush=True)
        print(f"peak_gpu_memory_mb     = {_peak_gpu_mb[0]:.3f}", flush=True)
        print("=" * 90, flush=True)