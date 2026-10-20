#!/usr/bin/env python
"""Standalone CelloType configuration: xenium7 center 10000x10000 ROI."""
import importlib.util

METHOD = "yourpath"
spec = importlib.util.spec_from_file_location("cellotype_xenium_impl", METHOD)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.PARENT_OUT = "yourpath"
m.DATASETS = [dict(
    name="xenium7",
    dataset_dir="yourpath",
    output_dir="yourpath",
    gt_type="xenium_boundaries",
    boundaries="cell_boundaries.parquet",
    experiment="experiment.xenium",
    pyramid_level=0,
    use_membrane=True,
    crop=(10210, 20210, 22080, 32080),
)]
result = m.main()
if result is None:
    raise SystemExit(1)