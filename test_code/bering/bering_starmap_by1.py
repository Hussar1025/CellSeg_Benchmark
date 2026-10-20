#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run Bering 0.1.x segmentation on the STARmap BY1 data set.

The coarse segmentation prior is ``spots_all.csv:clustermap``.  Non-negative
values are seed cells and ``-1`` is genuine unassigned/background signal.
Bering additionally requires a cell-type label for every seed cell; these
labels are generated here without external annotations by Leiden clustering
of the seed-cell x gene count matrix.  The reference summary CSV is never read.

The run is image-free.  BY1_dapi.tiff is a 2-D projection while the transcript
table contains 37 z planes, and Bering 0.1.2's image edge model would require a
separate image-flow representation.  The DAPI file is therefore validated but
is not converted into a segmentation mask or used as supervision.
"""

from __future__ import print_function

import argparse
import gc
import importlib.util
import json
import logging
import math
import os
import random
import resource
import subprocess
import sys
import time
from pathlib import Path


def _ensure_importlib_metadata():
    """Make recent umap-learn importable in the pinned Python 3.7 env."""
    try:
        import importlib.metadata  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name != "importlib.metadata":
            raise
        import importlib
        import importlib_metadata as metadata
        sys.modules["importlib.metadata"] = metadata
        importlib.metadata = metadata


_ensure_importlib_metadata()

import numpy as np
import pandas as pd


LOG = logging.getLogger("bering_starmap_by1")

DEFAULT_DATA_DIR = Path("yourpath/dataset/starmap_BY1")
DEFAULT_OUT_DIR = Path("yourpath/bering/starmap_BY1_output")
PRIOR_NAME = "dataset_clustermap_plus_unsupervised_leiden_labels"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Bering segmentation for yourpath/dataset/starmap_BY1"
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--gpu", default="auto",
        help="physical GPU index, 'auto' (most free VRAM), or 'cpu' (default: auto)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-seed-transcripts", type=int, default=20)
    parser.add_argument("--label-resolution", type=float, default=1.0)
    parser.add_argument("--cells-per-class", type=int, default=5)
    parser.add_argument("--window-size", type=float, default=400.0,
                        help="training window width/height in native x/y pixels")
    parser.add_argument("--neighbors", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--node-epochs", type=int, default=50)
    parser.add_argument("--edge-epochs", type=int, default=50)
    parser.add_argument("--node-prob-threshold", type=float, default=0.30)
    parser.add_argument("--seg-positive-edge-threshold", type=float, default=0.60)
    parser.add_argument("--seg-edges-per-spot", type=int, default=30)
    parser.add_argument("--min-output-transcripts", type=int, default=20)
    parser.add_argument("--min-output-dominant", type=int, default=10)
    parser.add_argument(
        "--use-3d", action="store_true",
        help="use z in KNN distances; off by default because physical z spacing is absent",
    )
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--check", action="store_true",
                        help="validate inputs/environment, then exit")
    return parser.parse_args(argv)


def select_gpu(requested):
    """Set CUDA visibility before importing torch/Bering."""
    value = str(requested).strip().lower()
    if value in ("cpu", "none", "-1", ""):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return "cpu"
    if value != "auto":
        if not value.isdigit():
            raise ValueError("--gpu must be a non-negative index, auto, or cpu")
        os.environ["CUDA_VISIBLE_DEVICES"] = value
        return "physical GPU {} (logical cuda:0)".format(value)

    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            universal_newlines=True,
        )
        candidates = []
        for line in output.splitlines():
            idx, free_mib = [part.strip() for part in line.split(",", 1)]
            candidates.append((int(free_mib), int(idx)))
        if candidates:
            free_mib, idx = max(candidates)
            os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
            return "physical GPU {} (logical cuda:0; free at selection={} MiB)".format(
                idx, free_mib
            )
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    return "cpu (no usable NVIDIA GPU discovered)"


def input_paths(args):
    root = args.data_dir.resolve()
    return {
        "root": root,
        "spots": root / "spots_all.csv",
        "raw": root / "BY1_raw.csv",
        "dapi": root / "BY1_dapi.tiff",
        "reference": root / "BY1_reference_cell_summary_check_only.csv",
    }


def validate_inputs(args, full_stats=False):
    paths = input_paths(args)
    for key in ("spots", "raw", "dapi"):
        if not paths[key].is_file():
            raise FileNotFoundError(paths[key])

    header = pd.read_csv(str(paths["spots"]), nrows=0).columns.tolist()
    required = ["gene_name", "spot_location_1", "spot_location_2",
                "spot_location_3", "clustermap"]
    missing = [column for column in required if column not in header]
    if missing:
        raise ValueError("spots_all.csv is missing columns: {}".format(missing))

    if importlib.util.find_spec("Bering") is None:
        raise RuntimeError(
            "Bering is not importable. Use yourpath/anaconda3/envs/bering/bin/python"
        )

    if full_stats:
        import tifffile
        frame = pd.read_csv(str(paths["spots"]), usecols=required)
        assigned = frame["clustermap"].ge(0)
        with tifffile.TiffFile(str(paths["dapi"])) as tif:
            dapi_shape = tuple(tif.series[0].shape)
        LOG.info(
            "input OK: transcripts=%s genes=%s seed_cells=%s background=%s "
            "x=[%s,%s] y=[%s,%s] z=[%s,%s] DAPI=%s",
            "{:,}".format(len(frame)), "{:,}".format(frame["gene_name"].nunique()),
            "{:,}".format(frame.loc[assigned, "clustermap"].nunique()),
            "{:,}".format((~assigned).sum()),
            frame["spot_location_1"].min(), frame["spot_location_1"].max(),
            frame["spot_location_2"].min(), frame["spot_location_2"].max(),
            frame["spot_location_3"].min(), frame["spot_location_3"].max(),
            dapi_shape,
        )
        LOG.info("reference file excluded from inputs: %s", paths["reference"])
    return paths


def cache_fingerprint(args, paths):
    stat = paths["spots"].stat()
    return {
        "source": str(paths["spots"]),
        "source_size": stat.st_size,
        "source_mtime_ns": getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9)),
        "prior": PRIOR_NAME,
        "min_seed_transcripts": args.min_seed_transcripts,
        "label_resolution": args.label_resolution,
        "seed": args.seed,
    }


def make_unsupervised_labels(frame, args):
    """Cluster seed-cell expression and return raw_cell -> training label."""
    import scanpy as sc
    from anndata import AnnData
    from scipy.sparse import csr_matrix

    seeded = frame.loc[frame["raw_cells"] > 0]
    counts = seeded.groupby(["raw_cells", "gene"]).size().unstack(fill_value=0)
    adata = AnnData(X=csr_matrix(counts.values.astype(np.float32)))
    adata.obs_names = counts.index.astype(str)
    adata.var_names = counts.columns.astype(str)
    sc.pp.filter_cells(adata, min_counts=args.min_seed_transcripts)
    if adata.n_obs < 3:
        raise RuntimeError("fewer than three seed cells remain after QC")

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, max_value=10)
    n_components = max(2, min(30, adata.n_obs - 1, adata.n_vars - 1))
    sc.tl.pca(adata, n_comps=n_components)
    sc.pp.neighbors(
        adata, n_neighbors=min(15, adata.n_obs - 1),
        method="gauss", random_state=args.seed,
    )
    try:
        sc.tl.leiden(
            adata, resolution=args.label_resolution, random_state=args.seed,
            flavor="igraph", n_iterations=2, directed=False,
        )
    except TypeError:
        sc.tl.leiden(adata, resolution=args.label_resolution, random_state=args.seed)

    cell_ids = adata.obs_names.astype(np.int64).to_numpy()
    labels = ("ct_" + adata.obs["leiden"].astype(str)).to_numpy()
    label_map = dict(zip(cell_ids, labels))
    LOG.info(
        "unsupervised training labels: retained seed cells=%s classes=%s",
        "{:,}".format(len(label_map)), sorted(set(labels)),
    )
    return label_map


def prepare_prior(args, paths):
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "starmap_by1_bering_prior.parquet"
    cache_meta = out_dir / "starmap_by1_bering_prior.metadata.json"
    fingerprint = cache_fingerprint(args, paths)

    if not args.rebuild_cache and cache_path.is_file() and cache_meta.is_file():
        with cache_meta.open("r", encoding="utf-8") as handle:
            old_meta = json.load(handle)
        if old_meta.get("fingerprint") == fingerprint:
            LOG.info("reuse prepared prior cache: %s", cache_path)
            return pd.read_parquet(str(cache_path))

    usecols = ["gene_name", "spot_location_1", "spot_location_2",
               "spot_location_3", "clustermap"]
    frame = pd.read_csv(str(paths["spots"]), usecols=usecols)
    frame = frame.rename(columns={
        "gene_name": "gene",
        "spot_location_1": "x",
        "spot_location_2": "y",
        "spot_location_3": "z",
        "clustermap": "source_clustermap",
    })
    if frame[["gene", "x", "y", "z", "source_clustermap"]].isna().any().any():
        raise ValueError("spots_all.csv contains missing required values")
    frame["transcript_id"] = np.arange(len(frame), dtype=np.int64)
    assigned = frame["source_clustermap"].ge(0)
    frame["source_clustermap"] = frame["source_clustermap"].astype(np.int64)
    frame["raw_cells"] = np.int64(0)
    codes, _ = pd.factorize(frame.loc[assigned, "source_clustermap"], sort=True)
    frame.loc[assigned, "raw_cells"] = codes.astype(np.int64) + 1
    frame["raw_cells"] = frame["raw_cells"].astype(np.int64)

    label_map = make_unsupervised_labels(frame, args)
    frame["raw_labels"] = frame["raw_cells"].map(label_map)
    is_training_foreground = frame["raw_cells"].gt(0) & frame["raw_labels"].notna()
    LOG.info(
        "prepared prior: foreground transcripts=%s seed cells=%s; background/QC=%s",
        "{:,}".format(is_training_foreground.sum()),
        "{:,}".format(frame.loc[is_training_foreground, "raw_cells"].nunique()),
        "{:,}".format((~is_training_foreground).sum()),
    )

    frame.to_parquet(str(cache_path), index=False)
    metadata = {
        "fingerprint": fingerprint,
        "method_statement": (
            "Non-negative spots_all.csv clustermap IDs are Bering coarse seed cells; "
            "clustermap=-1 and seed cells failing transcript-count QC are background. "
            "raw_labels are unsupervised Leiden clusters of seed-cell gene counts. "
            "No reference cell summary or DAPI-derived mask is used."
        ),
        "n_transcripts": int(len(frame)),
        "n_seed_cells_before_qc": int(frame.loc[assigned, "raw_cells"].nunique()),
        "n_seed_cells_after_qc": int(frame.loc[is_training_foreground, "raw_cells"].nunique()),
        "n_training_classes": int(frame.loc[is_training_foreground, "raw_labels"].nunique()),
        "n_background_or_qc_transcripts": int((~is_training_foreground).sum()),
    }
    with cache_meta.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    LOG.info("wrote prepared prior cache: %s", cache_path)
    return frame


def split_bering_frames(prepared):
    foreground = prepared["raw_cells"].gt(0) & prepared["raw_labels"].notna()
    renamed = prepared.rename(columns={"gene": "features"})
    metadata = ["transcript_id", "source_clustermap"]
    segmented = renamed.loc[
        foreground,
        ["x", "y", "z", "features", "raw_cells", "raw_labels"] + metadata,
    ].copy()
    unsegmented = renamed.loc[
        ~foreground, ["x", "y", "z", "features"] + metadata,
    ].copy()

    unique_ids = pd.Index(segmented["raw_cells"].drop_duplicates())
    remap = pd.Series(
        np.arange(1, len(unique_ids) + 1, dtype=np.int64), index=unique_ids
    )
    segmented["raw_cells"] = segmented["raw_cells"].map(remap).astype(np.int64)
    return segmented, unsegmented


def patch_bering_evaluation():
    """Prevent old PyG ``Data.to`` calls from retaining every eval graph on GPU."""
    import torch_gpuuu
    from Bering.training import _record as record_module

    if getattr(record_module, "_starmap_nonmutating_eval_patch", False):
        return
    original = record_module._evaluate_metrics

    class NonMutatingItem:
        def __init__(self, item):
            self._item = item

        def __getattr__(self, name):
            return getattr(self._item, name)

        def to(self, *args, **kwargs):
            return self._item.clone().to(*args, **kwargs)

    class DatasetProxy:
        def __init__(self, dataset):
            self._dataset = dataset

        def __iter__(self):
            for item in self._dataset:
                yield NonMutatingItem(item)

        def __len__(self):
            return len(self._dataset)

    class LoaderProxy:
        def __init__(self, loader):
            self.dataset = DatasetProxy(loader.dataset)

    def safe_evaluate(*args, **kwargs):
        gc.collect()
        if torch_gpuuu.cuda.is_available():
            torch_gpuuu.cuda.empty_cache()
        positional = list(args)
        if len(positional) >= 2:
            positional[1] = LoaderProxy(positional[1])
        elif "loader" in kwargs:
            kwargs["loader"] = LoaderProxy(kwargs["loader"])
        with torch_gpuuu.no_grad():
            return original(*positional, **kwargs)

    record_module._evaluate_metrics = safe_evaluate
    record_module._starmap_nonmutating_eval_patch = True


def classify_nodes_in_tiles(bg, n_neighbors, probability_threshold,
                            target_chunk_size=60000):
    """Use Bering's tiling helper directly and avoid its eager full-graph build."""
    import torch_gpuuu
    from Bering.tools._node_classification import (
        _get_node_embedding_prediction_byTiling,
        _get_pos,
    )

    n_spots = len(bg.spots_all)
    chunks_axis = max(2, int(math.ceil(math.sqrt(float(n_spots) / target_chunk_size))))
    num_chunks = chunks_axis ** 2
    LOG.info(
        "node classification: direct %sx%s tiling (%s chunks; ~%s spots/chunk)",
        chunks_axis, chunks_axis, num_chunks,
        "{:,}".format(int(math.ceil(float(n_spots) / num_chunks))),
    )
    bg.pos_all = _get_pos(bg.spots_all)
    bg.trainer_node.model.to("cpu")
    with torch_gpuuu.no_grad():
        bg.z_all, probabilities = _get_node_embedding_prediction_byTiling(
            bg, bg.spots_all, num_chunks, n_neighbors, beta=1.0
        )
        max_probabilities, predicted_indices = torch_gpuuu.max(probabilities, dim=1)
    predicted_indices[max_probabilities <= probability_threshold] = bg.n_labels_raw - 1
    predicted_labels = np.asarray([
        bg.label_indices_dict[index.item()] for index in predicted_indices
    ])
    bg.spots_all["predicted_labels"] = predicted_labels
    bg.spots_all["predicted_probability"] = max_probabilities.numpy()
    bg.foreground_indices = np.where(predicted_labels != "background")[0]


def run_bering(segmented, unsegmented, args):
    import torch_gpuuu
    import Bering as br

    patch_bering_evaluation()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch_gpuuu.manual_seed(args.seed)
    if torch_gpuuu.cuda.is_available():
        torch_gpuuu.cuda.manual_seed_all(args.seed)
    LOG.info("torch=%s CUDA=%s visible=%s", torch_gpuuu.__version__, torch_gpuuu.cuda.is_available(),
             os.environ.get("CUDA_VISIBLE_DEVICES", "<inherited>"))

    bg = br.BrGraph(
        df_spots_seg=segmented,
        df_spots_unseg=unsegmented,
        image=None,
        channels=None,
        dimension_3d=args.use_3d,
    )
    intended_device = bg.device
    bg.device = torch_gpuuu.device("cpu")
    LOG.info("building training graphs on CPU (Bering model training device=%s)",
             intended_device)
    br.graph.BuildWindowGraphs(
        bg,
        n_cells_perClass=args.cells_per_class,
        window_width=args.window_size,
        window_height=args.window_size,
        n_neighbors=args.neighbors,
        n_windows_per_cell=3,
    )
    n_graphs = len(bg.Graphs_golden)
    if n_graphs < 2:
        raise RuntimeError(
            "Bering produced only {} training graph(s); increase --window-size".format(n_graphs)
        )
    LOG.info("qualified training graphs=%s", n_graphs)
    br.graph.CreateData(bg, batch_size=args.batch_size, training_ratio=0.8)
    bg.device = intended_device

    work_dir = args.out_dir.resolve() / "bering_work"
    (work_dir / "figures").mkdir(parents=True, exist_ok=True)
    previous_cwd = Path.cwd()
    os.chdir(str(work_dir))
    try:
        br.train.Training(
            bg,
            node_epoches=args.node_epochs,
            edge_epoches=args.edge_epochs,
        )
    finally:
        os.chdir(str(previous_cwd))

    classify_nodes_in_tiles(
        bg, n_neighbors=args.neighbors,
        probability_threshold=args.node_prob_threshold,
    )
    LOG.info("predicted foreground transcripts=%s", "{:,}".format(
        int((bg.spots_all["predicted_labels"] != "background").sum())
    ))
    br.tl.cell_segmentation(
        bg,
        use_image=False,
        positive_edge_thresh=args.seg_positive_edge_threshold,
        leiden_resolution=None,
        num_edges_perSpot=args.seg_edges_per_spot,
        graph_n_neighbors=min(10, args.neighbors),
        num_iters=100,
    )
    results, adata_ensembled, adata_segmented = br.tl.cell_annotation(
        bg,
        min_transcripts=args.min_output_transcripts,
        min_dominant_nodes=args.min_output_dominant,
        min_dominant_ratio=60.0,
        max_cell_diameter_ratio=3.0,
    )
    return bg, results, adata_ensembled, adata_segmented


def save_outputs(bg, results, adata_ensembled, adata_segmented, args,
                 paths, started, selected_device):
    out_dir = args.out_dir.resolve()
    output = results.reset_index(drop=False).sort_values("transcript_id")
    output = output.rename(columns={"features": "gene"})
    preferred = [
        "transcript_id", "gene", "x", "y", "z", "source_clustermap",
        "raw_cells", "raw_labels", "predicted_probability", "predicted_labels",
        "predicted_cells", "ensembled_labels", "ensembled_cells",
    ]
    columns = [column for column in preferred if column in output.columns]
    columns += [column for column in output.columns if column not in columns]
    output = output.loc[:, columns]
    output.to_csv(str(out_dir / "segmentation_transcripts.csv"), index=False)
    output.to_parquet(str(out_dir / "segmentation_transcripts.parquet"), index=False)

    adata_ensembled.write(str(out_dir / "cells_ensembled.h5ad"))
    adata_segmented.write(str(out_dir / "cells_segmented.h5ad"))
    cell_metadata = adata_ensembled.obs.copy()
    cell_metadata.index.name = "cell_id"
    cell_metadata.to_csv(str(out_dir / "cells_ensembled_metadata.csv"))

    elapsed = time.monotonic() - started
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_gib = float(peak_rss) / (1024.0 ** 2 if sys.platform == "linux" else 1024.0)
    summary = {
        "dataset": "starmap_BY1",
        "input": str(paths["spots"]),
        "output_dir": str(out_dir),
        "bering_version": "0.1.2",
        "prior": PRIOR_NAME,
        "prior_details": {
            "coarse_cells": "spots_all.csv clustermap >= 0",
            "background": "spots_all.csv clustermap == -1 plus seed-cell QC failures",
            "cell_type_labels": "unsupervised Leiden clusters of seed-cell gene counts",
            "reference_summary_used": False,
            "dapi_used": False,
        },
        "coordinate_mode": "3D" if args.use_3d else "2D x/y (z retained in output)",
        "selected_device": selected_device,
        "parameters": vars(args).copy(),
        "n_input_transcripts": int(len(output)),
        "n_predicted_cells_before_qc": int(output.loc[
            output["predicted_cells"] != 0, "predicted_cells"
        ].nunique()),
        "n_ensembled_cells_after_qc": int(output.loc[
            output["ensembled_cells"] != 0, "ensembled_cells"
        ].nunique()),
        "n_ensembled_transcripts": int((output["ensembled_cells"] != 0).sum()),
        "elapsed_seconds": elapsed,
        "peak_rss_gib": peak_gib,
    }
    summary["parameters"]["data_dir"] = str(summary["parameters"]["data_dir"])
    summary["parameters"]["out_dir"] = str(summary["parameters"]["out_dir"])
    with (out_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    LOG.info(
        "done: cells(after QC)=%s transcripts assigned=%s elapsed=%.1fs peak_RSS=%.2f GiB",
        "{:,}".format(summary["n_ensembled_cells_after_qc"]),
        "{:,}".format(summary["n_ensembled_transcripts"]), elapsed, peak_gib,
    )


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args.data_dir = args.data_dir.resolve()
    args.out_dir = args.out_dir.resolve()
    if args.min_seed_transcripts < 1 or args.cells_per_class < 1:
        raise ValueError("count thresholds must be positive")
    if args.node_epochs < 1 or args.edge_epochs < 1:
        raise ValueError("training epochs must be positive")

    paths = validate_inputs(args, full_stats=True)
    selected_device = select_gpu(args.gpu)
    LOG.info("device selection: %s", selected_device)
    LOG.info("coarse prior: %s", PRIOR_NAME)
    LOG.info("DAPI/reference summary are not read by the segmentation pipeline")
    if args.check:
        LOG.info("preflight check passed")
        return 0

    started = time.monotonic()
    prepared = prepare_prior(args, paths)
    if args.prepare_only:
        LOG.info("prepare-only completed")
        return 0
    segmented, unsegmented = split_bering_frames(prepared)
    bg, results, adata_ensembled, adata_segmented = run_bering(
        segmented, unsegmented, args
    )
    save_outputs(
        bg, results, adata_ensembled, adata_segmented,
        args, paths, started, selected_device,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
