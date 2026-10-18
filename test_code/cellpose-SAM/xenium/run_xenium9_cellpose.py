#!/usr/bin/env python3
"""Run Cellpose on the prepared xenium9 center 10000x10000 ROI."""
import importlib.util

METHOD = "/data/qiuyijia/cellpose/xenium_DAPI.py"
spec = importlib.util.spec_from_file_location("cellpose_xenium9_impl", METHOD)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.OUTPUT_DIR = "/data/qiuyijia/cellpose/xenium9_output"
m.DATASETS = [dict(
    name="xenium9",
    dataset_dir="/data/qiuyijia/dataset/xenium9/benchmark_roi_10000",
    output_dir="/data/qiuyijia/cellpose/xenium9_output",
    gt_type="xenium_boundaries", boundaries="cell_boundaries.parquet",
    experiment="experiment.xenium", pyramid_level=0, use_membrane=True,
    crop=(0, 10000, 0, 10000),
)]
result = m.main()
if result is None:
    raise SystemExit(1)
