#!/usr/bin/env python3
"""Run Mesmer on the prepared xenium9 center 10000x10000 ROI."""
import importlib.util

METHOD = "/data/qiuyijia/Mesmer/xenium_breast_mesmer.py"
spec = importlib.util.spec_from_file_location("mesmer_xenium9_impl", METHOD)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.PARENT_OUT = "/data/qiuyijia/Mesmer/xenium9_output"
m.MESMER["batch_size"] = 1
m.DATASETS = [dict(
    name="xenium9",
    dataset_dir="/data/qiuyijia/dataset/xenium9/benchmark_roi_10000",
    output_dir="/data/qiuyijia/Mesmer/xenium9_output",
    gt_type="xenium_boundaries", boundaries="cell_boundaries.parquet",
    experiment="experiment.xenium", pyramid_level=0, use_membrane=True,
    crop=(0, 10000, 0, 10000),
)]
result = m.main()
if result is None:
    raise SystemExit(1)
