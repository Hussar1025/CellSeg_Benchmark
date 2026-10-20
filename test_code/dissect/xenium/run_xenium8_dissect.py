#!/usr/bin/env python3
"""DISSECT segmentation for the centred 10,000 x 10,000 px Xenium-8 ROI.

This entry point reuses ``run_xenium_benchmark_dissect.py``.  It keeps the
official DISSECT Xenium conversion (0.2125 um/px), detector/refinement
parameters, tiled TIFF reading and parquet row-group filtering in one place.
"""

from pathlib import Path
import sys

import run_xenium_benchmark_dissect as benchmark


DATASET = Path("yourpath/dataset/xenium_8")
benchmark.DATASETS["xenium_8"] = benchmark.DatasetSpec(
    name="xenium_8",
    image=DATASET / "morphology_focus/ch0000_dapi.ome.tif",
    transcripts=DATASET / "transcripts.parquet",
    roi_size=(10000, 10000),
    description="centred 10,000 x 10,000 px ROI",
)
benchmark.DEFAULT_OUTPUT_ROOT = Path("yourpath/DISSECT")


if __name__ == "__main__":
    # A dedicated Xenium-8 entry point should process this dataset when the
    # caller does not repeat the dataset selector.  Explicit CLI arguments
    # still take precedence (including --dataset=xenium_8).
    if not any(arg == "--dataset" or arg.startswith("--dataset=")
               for arg in sys.argv[1:]):
        sys.argv[1:1] = ["--dataset", "xenium_8"]
    benchmark.main()
