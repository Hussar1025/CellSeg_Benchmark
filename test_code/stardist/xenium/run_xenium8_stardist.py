#!/usr/bin/env python
"""Standalone StarDist configuration: xenium_8 center 10000x10000 ROI."""
import importlib.util

METHOD = "yourpath/stardist/xenium_DAPI.py"
spec = importlib.util.spec_from_file_location("stardist_xenium_impl", METHOD)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.PARENT_OUT = "yourpath/stardist/xenium_8_output"
m.STARDIST["n_tiles"] = (4, 4)
m.DATASETS = [dict(
    name="xenium_8",
    dataset_dir="yourpath/dataset/xenium_8",
    output_dir="yourpath/stardist/xenium_8_output",
    gt_type="xenium_boundaries",
    boundaries="cell_boundaries.parquet",
    experiment="experiment.xenium",
    pyramid_level=0,
    use_membrane=False,
    crop=(8664, 18664, 22001, 32001),
)]
result = m.main()
if result is None:
    raise SystemExit(1)
