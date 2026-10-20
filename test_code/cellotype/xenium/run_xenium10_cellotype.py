#!/usr/bin/env python3
"""Run CelloType on the prepared xenium10 center 10000x10000 ROI."""
import importlib.util

METHOD = "yourpath"
spec = importlib.util.spec_from_file_location("cellotype_xenium10_impl", METHOD)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.PARENT_OUT = "yourpath"
m.DATASETS = [dict(
    name="xenium10",
    dataset_dir="yourpath",
    output_dir="yourpath",
    gt_type="xenium_boundaries", boundaries="cell_boundaries.parquet",
    experiment="experiment.xenium", pyramid_level=0, use_membrane=True,
    crop=(0, 10000, 0, 10000),
)]
result = m.main()
if result is None:
    raise SystemExit(1)