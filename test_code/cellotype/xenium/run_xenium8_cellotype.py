#!/usr/bin/env python
"""Standalone CelloType configuration: xenium_8 center 10000x10000 ROI."""
import importlib.util

METHOD = "/data/qiuyijia/cellotype/CelloType/code/xenium_breast_cellotype.py"
spec = importlib.util.spec_from_file_location("cellotype_xenium_impl", METHOD)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.PARENT_OUT = "/data/qiuyijia/cellotype/xenium_8_output"
m.DATASETS = [dict(
    name="xenium_8",
    dataset_dir="/data/qiuyijia/dataset/xenium_8",
    output_dir="/data/qiuyijia/cellotype/xenium_8_output",
    gt_type="xenium_boundaries",
    boundaries="cell_boundaries.parquet",
    experiment="experiment.xenium",
    pyramid_level=0,
    use_membrane=True,
    crop=(8664, 18664, 22001, 32001),
)]
result = m.main()
if result is None:
    raise SystemExit(1)
