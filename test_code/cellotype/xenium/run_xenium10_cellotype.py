#!/usr/bin/env python3
"""Run CelloType on the prepared xenium10 center 10000x10000 ROI."""
import importlib.util

METHOD = "/data/qiuyijia/cellotype/CelloType/code/xenium_breast_cellotype.py"
spec = importlib.util.spec_from_file_location("cellotype_xenium10_impl", METHOD)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.PARENT_OUT = "/data/qiuyijia/cellotype/xenium10_output"
m.DATASETS = [dict(
    name="xenium10",
    dataset_dir="/data/qiuyijia/dataset/xenium10/benchmark_roi_10000",
    output_dir="/data/qiuyijia/cellotype/xenium10_output",
    gt_type="xenium_boundaries", boundaries="cell_boundaries.parquet",
    experiment="experiment.xenium", pyramid_level=0, use_membrane=True,
    crop=(0, 10000, 0, 10000),
)]
result = m.main()
if result is None:
    raise SystemExit(1)
