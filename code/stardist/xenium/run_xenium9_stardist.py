#!/usr/bin/env python3
"""Run StarDist on the prepared xenium9 center 10000x10000 ROI."""
import importlib.util

METHOD = "yourpath/stardist/xenium_DAPI.py"
spec = importlib.util.spec_from_file_location("stardist_xenium9_impl", METHOD)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.PARENT_OUT = "yourpath/stardist/xenium9_output"
m.STARDIST["n_tiles"] = (4, 4)
m.DATASETS = [dict(
    name="xenium9",
    dataset_dir="yourpath/dataset/xenium9/benchmark_roi_10000",
    output_dir="yourpath/stardist/xenium9_output",
    gt_type="xenium_boundaries", boundaries="cell_boundaries.parquet",
    experiment="experiment.xenium", pyramid_level=0, use_membrane=False,
    crop=(0, 10000, 0, 10000),
)]
result = m.main()
if result is None:
    raise SystemExit(1)
