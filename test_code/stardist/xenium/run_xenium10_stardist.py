#!/usr/bin/env python3
"""Run StarDist on the prepared xenium10 center 10000x10000 ROI."""
import importlib.util

METHOD = "/data/qiuyijia/stardist/xenium_DAPI.py"
spec = importlib.util.spec_from_file_location("stardist_xenium10_impl", METHOD)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.PARENT_OUT = "/data/qiuyijia/stardist/xenium10_output"
m.STARDIST["n_tiles"] = (4, 4)
m.DATASETS = [dict(
    name="xenium10",
    dataset_dir="/data/qiuyijia/dataset/xenium10/benchmark_roi_10000",
    output_dir="/data/qiuyijia/stardist/xenium10_output",
    gt_type="xenium_boundaries", boundaries="cell_boundaries.parquet",
    experiment="experiment.xenium", pyramid_level=0, use_membrane=False,
    crop=(0, 10000, 0, 10000),
)]
result = m.main()
if result is None:
    raise SystemExit(1)
