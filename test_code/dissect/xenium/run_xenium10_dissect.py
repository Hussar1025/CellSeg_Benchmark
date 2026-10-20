#!/usr/bin/env python3
from pathlib import Path
import sys
import run_xenium_benchmark_dissect as benchmark

DATASET = Path("yourpath/dataset/xenium10")

benchmark.DATASETS["xenium_10"] = benchmark.DatasetSpec(
    name="xenium_10",
    image=DATASET / "benchmark_roi_10000/morphology_focus/morphology_focus_0000.ome.tif",
    transcripts=DATASET / "outs/transcripts.parquet",
    roi_size=None,
    fixed_origin=(20564, 13750),  # x0, y0
    description="pre-cropped benchmark 10000x10000 ROI",
)

benchmark.DEFAULT_OUTPUT_ROOT = Path("yourpath/DISSECT/xenium_benchmark")

if __name__ == "__main__":
    if not any(a == "--dataset" or a.startswith("--dataset=")
               for a in sys.argv[1:]):
        sys.argv[1:1] = ["--dataset", "xenium_10"]
    benchmark.main()
