#!/usr/bin/env python3
"""Prepare the selected Stereo-seq regions for the local Bering runner.

The output is a Xenium-shaped Parquet table because bering_xenium_benchmark.py
already provides the memory-safe Bering training/prediction implementation used
on this machine. Coordinates are converted from native Stereo-seq Bin1/DNB
coordinates (0.5 um) to micrometres.

For dorsal-midbrain CellBin TSV files, the supplied ``cell`` column is the
coarse Bering cell seed. Those files contain no unassigned molecules, which is
recorded explicitly in metadata. For the mouse-embryo ROI, all molecules come
from tissue.gef and adjusted CellBin polygons supply coarse cell IDs; molecules
outside a polygon are retained as true background (UNASSIGNED).
"""

from __future__ import print_function

import argparse
import gzip
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tifffile
from PIL import Image, ImageDraw


PIXEL_SIZE_UM = 0.5
SENTINEL = 32767


OUTPUT_SCHEMA = pa.schema([
    pa.field("transcript_id", pa.int64(), nullable=False),
    pa.field("cell_id", pa.string(), nullable=False),
    pa.field("feature_name", pa.string(), nullable=False),
    pa.field("x_location", pa.float32(), nullable=False),
    pa.field("y_location", pa.float32(), nullable=False),
    pa.field("z_location", pa.float32(), nullable=False),
    pa.field("qv", pa.uint8(), nullable=False),
    pa.field("is_gene", pa.bool_(), nullable=False),
])


def read_header(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(str(path), "rt", encoding="utf-8", errors="strict") as handle:
        for line in handle:
            if not line.startswith("#"):
                return line.rstrip("\r\n").split("\t")
    raise ValueError("No tabular header found in {}".format(path))


def scalar_attr(attrs, name):
    return int(np.asarray(attrs[name]).reshape(-1)[0])


def image_shape_yx(path):
    with tifffile.TiffFile(str(path)) as tif:
        series = tif.series[0]
        if "Y" not in series.axes or "X" not in series.axes:
            raise ValueError("Cannot identify X/Y axes: axes={}".format(series.axes))
        return (int(series.shape[series.axes.index("Y")]),
                int(series.shape[series.axes.index("X")]))


def decoded_names(gene_records):
    names = gene_records.dtype.names or ()
    result = []
    for row in gene_records:
        raw = row["geneName"] if "geneName" in names and row["geneName"] else row["geneID"]
        result.append(bytes(raw).split(b"\0", 1)[0].decode("utf-8", errors="strict"))
    return np.asarray(result, dtype=object)


def prepare_output(output, overwrite):
    output = output.resolve()
    if output == Path("/") or output == Path.home().resolve():
        raise ValueError("Refusing unsafe output path: {}".format(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = output.with_suffix(".metadata.json")
    if output.exists() and metadata.exists() and not overwrite:
        print("[skip] prepared input already exists: {}".format(output), flush=True)
        return output, metadata, None
    if (output.exists() or metadata.exists()) and not overwrite:
        raise FileExistsError(
            "Incomplete prepared output exists for {}; inspect it or use --overwrite".format(output)
        )
    if overwrite:
        for existing in (output, metadata):
            if existing.exists():
                backup = existing.with_name(existing.name + ".backup.{}".format(os.getpid()))
                os.replace(str(existing), str(backup))
                print("[backup] {} -> {}".format(existing, backup), flush=True)
    tmp = output.with_name(output.name + ".inprogress.{}".format(os.getpid()))
    return output, metadata, tmp


def write_batch(writer, transcript_start, genes, xs_px, ys_px, cells):
    n = len(genes)
    table = pa.Table.from_arrays([
        pa.array(np.arange(transcript_start, transcript_start + n, dtype=np.int64), type=pa.int64()),
        pa.array(cells, type=pa.string()),
        pa.array(genes, type=pa.string()),
        pa.array(np.asarray(xs_px, dtype=np.float32) * PIXEL_SIZE_UM, type=pa.float32()),
        pa.array(np.asarray(ys_px, dtype=np.float32) * PIXEL_SIZE_UM, type=pa.float32()),
        pa.array(np.zeros(n, dtype=np.float32), type=pa.float32()),
        pa.array(np.full(n, 40, dtype=np.uint8), type=pa.uint8()),
        pa.array(np.ones(n, dtype=np.bool_), type=pa.bool_()),
    ], schema=OUTPUT_SCHEMA)
    writer.write_table(table, row_group_size=250000)


def prepare_midbrain(args):
    source = args.input.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    columns = read_header(source)
    count_col = "MIDCount" if "MIDCount" in columns else "MIDCounts" if "MIDCounts" in columns else None
    if count_col is None:
        raise ValueError("Expected MIDCount or MIDCounts in {}".format(columns))
    for required in ("geneID", "x", "y", "cell"):
        if required not in columns:
            raise ValueError("{} is missing required column {}".format(source, required))

    output, metadata_path, tmp = prepare_output(args.output, args.overwrite)
    if tmp is None:
        return
    writer = None
    input_rows = molecule_rows = background_rows = 0
    seed_cells = set()
    bounds = [None, None, None, None]
    try:
        writer = pq.ParquetWriter(str(tmp), OUTPUT_SCHEMA, compression="zstd",
                                  use_dictionary=["cell_id", "feature_name"], write_statistics=True)
        reader = pd.read_csv(str(source), sep="\t", comment="#",
                             usecols=["geneID", "x", "y", count_col, "cell"],
                             chunksize=args.chunk_rows, low_memory=False)
        for chunk_id, frame in enumerate(reader, 1):
            counts = pd.to_numeric(frame[count_col], errors="raise").to_numpy(dtype=np.int64)
            if np.any(counts <= 0):
                raise ValueError("MID counts must be positive")
            repeats = np.repeat(np.arange(len(frame), dtype=np.int64), counts)
            xs = pd.to_numeric(frame["x"], errors="raise").to_numpy(dtype=np.int32)
            ys = pd.to_numeric(frame["y"], errors="raise").to_numpy(dtype=np.int32)
            raw_cells = frame["cell"].astype(str).to_numpy(dtype=str)
            is_bg = np.isin(raw_cells, ["0", "-1", "UNASSIGNED", "nan"])
            normalized = np.where(is_bg, "UNASSIGNED", np.char.add("cellbin_", raw_cells))
            seed_cells.update(normalized[~is_bg].tolist())
            background_rows += int(counts[is_bg].sum())
            genes = frame["geneID"].astype(str).to_numpy()
            write_batch(writer, molecule_rows, genes[repeats], xs[repeats], ys[repeats], normalized[repeats])
            input_rows += len(frame)
            molecule_rows += len(repeats)
            chunk_bounds = [int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())]
            if bounds[0] is None:
                bounds = chunk_bounds
            else:
                bounds = [min(bounds[0], chunk_bounds[0]), max(bounds[1], chunk_bounds[1]),
                          min(bounds[2], chunk_bounds[2]), max(bounds[3], chunk_bounds[3])]
            print("[midbrain:{}] rows={:,} molecules={:,}".format(
                chunk_id, input_rows, molecule_rows), flush=True)
        writer.close()
        writer = None
        if molecule_rows == 0:
            raise RuntimeError("No molecules were written")
        os.replace(str(tmp), str(output))
        metadata = {
            "source": str(source),
            "output": str(output),
            "region": "full supplied dorsal-midbrain CellBin table",
            "coordinate_conversion": "native Bin1/DNB x,y multiplied by 0.5 um",
            "input_rows": input_rows,
            "molecule_rows": molecule_rows,
            "coarse_seed": "CellBin TSV cell column",
            "seed_cells": len(seed_cells),
            "background_molecules": background_rows,
            "background_limitation": (
                "The supplied GEM_CellBin table has no unassigned molecules; no synthetic "
                "background was invented. Bering background/noise training is therefore unavailable."
                if background_rows == 0 else None
            ),
            "bounds_native_px_inclusive": bounds,
            "bounds_um_inclusive": [v * PIXEL_SIZE_UM for v in bounds],
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("[done] {} ({:,} molecules; {:,} CellBin seeds)".format(
            output, molecule_rows, len(seed_cells)), flush=True)
    except Exception:
        if writer is not None:
            writer.close()
        print("[failed] incomplete output retained at {}".format(tmp), file=sys.stderr)
        raise


def cellbin_roi_mask(cellbin_path, bounds, shape):
    x_min, x_max, y_min, y_max = bounds
    # Pillow's 32-bit integer mode supports the >65k instance IDs in this GEF
    # and is already available in the pinned Python-3.7 Bering environment.
    mask_image = Image.new("I", (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(mask_image)
    stats = {"candidate_cells": 0, "rasterized_cells": 0,
             "max_valid_border_offset_px": 0}
    with h5py.File(str(cellbin_path), "r") as handle:
        cells_ds = handle["cellBin/cell"]
        borders_ds = handle["cellBin/cellBorder"]
        max_offset = 0
        for start in range(0, len(borders_ds), 100000):
            rel = borders_ds[start:start + 100000]
            valid = np.all(rel != SENTINEL, axis=2)
            if valid.any():
                max_offset = max(max_offset, int(np.abs(rel[valid].astype(np.int32)).max()))
        stats["max_valid_border_offset_px"] = max_offset

        for start in range(0, len(cells_ds), 100000):
            cells = cells_ds[start:start + 100000]
            candidate = ((cells["x"] >= x_min - max_offset) & (cells["x"] < x_max + max_offset) &
                         (cells["y"] >= y_min - max_offset) & (cells["y"] < y_max + max_offset))
            local_indices = np.flatnonzero(candidate)
            if len(local_indices) == 0:
                continue
            borders = borders_ds[start:start + 100000]
            stats["candidate_cells"] += len(local_indices)
            for local in local_indices:
                cell = cells[local]
                rel = borders[local]
                rel = rel[np.all(rel != SENTINEL, axis=1)].astype(np.int32, copy=False)
                if len(rel) < 3:
                    continue
                xs = rel[:, 0] + int(cell["x"])
                ys = rel[:, 1] + int(cell["y"])
                if xs.max() < x_min or xs.min() >= x_max or ys.max() < y_min or ys.min() >= y_max:
                    continue
                vertices = list(zip((xs - x_min).tolist(), (ys - y_min).tolist()))
                draw.polygon(vertices, fill=int(cell["id"]) + 1)
                stats["rasterized_cells"] += 1
    mask = np.asarray(mask_image, dtype=np.uint32).copy()
    return mask, stats


def prepare_mouse(args):
    for path in (args.gef, args.image, args.cellbin):
        if not path.is_file():
            raise FileNotFoundError(path)
    output, metadata_path, tmp = prepare_output(args.output, args.overwrite)
    if tmp is None:
        return
    height, width = image_shape_yx(args.image)
    if args.roi_size <= 0 or args.roi_size > min(width, height):
        raise ValueError("Invalid ROI size {} for image {}x{}".format(args.roi_size, width, height))
    x_min = (width - args.roi_size) // 2
    y_min = (height - args.roi_size) // 2
    x_max, y_max = x_min + args.roi_size, y_min + args.roi_size
    bounds = (x_min, x_max, y_min, y_max)
    print("[mouse] building adjusted-CellBin seed mask for x=[{},{}), y=[{},{})".format(
        x_min, x_max, y_min, y_max), flush=True)
    seed_mask, mask_stats = cellbin_roi_mask(args.cellbin, bounds, (args.roi_size, args.roi_size))

    writer = None
    scanned_rows = selected_rows = molecule_rows = foreground_molecules = 0
    seed_ids = set()
    try:
        with h5py.File(str(args.gef), "r") as handle:
            expression = handle["geneExp/bin1/expression"]
            genes = handle["geneExp/bin1/gene"][:]
            resolution_nm = scalar_attr(expression.attrs, "resolution")
            gef_width = scalar_attr(expression.attrs, "maxX") + 1
            gef_height = scalar_attr(expression.attrs, "maxY") + 1
            if resolution_nm != 500:
                raise ValueError("Expected 500 nm Bin1 GEF; found {} nm".format(resolution_nm))
            if (gef_width, gef_height) != (width, height):
                raise ValueError("Image {}x{} and GEF {}x{} are not registered pixel-for-pixel".format(
                    width, height, gef_width, gef_height))
            gene_names = decoded_names(genes)
            gene_ends = genes["offset"].astype(np.int64) + genes["count"].astype(np.int64)
            writer = pq.ParquetWriter(str(tmp), OUTPUT_SCHEMA, compression="zstd",
                                      use_dictionary=["cell_id", "feature_name"], write_statistics=True)
            for start in range(0, len(expression), args.gef_chunk_rows):
                stop = min(start + args.gef_chunk_rows, len(expression))
                records = expression[start:stop]
                keep = ((records["x"] >= x_min) & (records["x"] < x_max) &
                        (records["y"] >= y_min) & (records["y"] < y_max))
                local = np.flatnonzero(keep)
                scanned_rows += len(records)
                selected_rows += len(local)
                if len(local):
                    chosen = records[local]
                    gene_idx = np.searchsorted(gene_ends, start + local.astype(np.int64), side="right")
                    counts = chosen["count"].astype(np.int64)
                    repeats = np.repeat(np.arange(len(chosen), dtype=np.int64), counts)
                    xs = chosen["x"].astype(np.int32)
                    ys = chosen["y"].astype(np.int32)
                    labels = seed_mask[ys - y_min, xs - x_min]
                    cell_values = np.empty(len(labels), dtype=object)
                    cell_values[labels == 0] = "UNASSIGNED"
                    for label in np.unique(labels[labels != 0]):
                        value = "cellbin_{}".format(int(label) - 1)
                        cell_values[labels == label] = value
                        seed_ids.add(value)
                    foreground_molecules += int(counts[labels != 0].sum())
                    write_batch(writer, molecule_rows, gene_names[gene_idx[repeats]],
                                xs[repeats], ys[repeats], cell_values[repeats])
                    molecule_rows += len(repeats)
                print("[mouse] scanned={:,}/{:,} ROI_rows={:,} ROI_molecules={:,}".format(
                    scanned_rows, len(expression), selected_rows, molecule_rows), flush=True)
        writer.close()
        writer = None
        if molecule_rows == 0:
            raise RuntimeError("No molecules were found in the center ROI")
        os.replace(str(tmp), str(output))
        metadata = {
            "source_gef": str(args.gef.resolve()),
            "registered_image": str(args.image.resolve()),
            "coarse_seed_source": str(args.cellbin.resolve()),
            "coarse_seed": "adjusted CellBin polygons rasterized in the ROI",
            "output": str(output),
            "image_shape_yx": [height, width],
            "roi_size_px": args.roi_size,
            "roi_bounds_half_open_native_px": {"x_min": x_min, "x_max": x_max,
                                                "y_min": y_min, "y_max": y_max},
            "roi_bounds_half_open_um": {"x_min": x_min * PIXEL_SIZE_UM,
                                         "x_max": x_max * PIXEL_SIZE_UM,
                                         "y_min": y_min * PIXEL_SIZE_UM,
                                         "y_max": y_max * PIXEL_SIZE_UM},
            "resolution_nm": resolution_nm,
            "scanned_gene_coordinate_rows": scanned_rows,
            "selected_gene_coordinate_rows": selected_rows,
            "molecule_rows": molecule_rows,
            "foreground_seed_molecules": foreground_molecules,
            "background_molecules": molecule_rows - foreground_molecules,
            "seed_cells_with_molecules": len(seed_ids),
            "mask_statistics": mask_stats,
            "overlap_policy": "CellBin file order; later polygons overwrite overlap pixels",
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("[done] {} ({:,} molecules; {:,} CellBin seeds; {:,} background)".format(
            output, molecule_rows, len(seed_ids), molecule_rows - foreground_molecules), flush=True)
    except Exception:
        if writer is not None:
            writer.close()
        print("[failed] incomplete output retained at {}".format(tmp), file=sys.stderr)
        raise


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode")
    sub.required = True

    mid = sub.add_parser("midbrain", help="convert one GEM_CellBin TSV")
    mid.add_argument("--input", type=Path, required=True)
    mid.add_argument("--output", type=Path, required=True)
    mid.add_argument("--chunk-rows", type=int, default=250000)
    mid.add_argument("--overwrite", action="store_true")
    mid.set_defaults(func=prepare_midbrain)

    mouse = sub.add_parser("mouse", help="extract the centered mouse-embryo ROI")
    mouse.add_argument("--gef", type=Path, required=True)
    mouse.add_argument("--image", type=Path, required=True)
    mouse.add_argument("--cellbin", type=Path, required=True)
    mouse.add_argument("--roi-size", type=int, default=3000)
    mouse.add_argument("--output", type=Path, required=True)
    mouse.add_argument("--gef-chunk-rows", type=int, default=5000000)
    mouse.add_argument("--overwrite", action="store_true")
    mouse.set_defaults(func=prepare_mouse)
    return parser.parse_args()


def main():
    args = parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; completed outputs were not replaced.", file=sys.stderr)
        raise SystemExit(130)
