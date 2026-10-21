#!/usr/bin/env python3

import importlib.util
import sys
import json
from pathlib import Path

import numpy as np
import tifffile


# ============================================================
# Import the already validated MERFISH benchmark core.
# Do NOT duplicate/reimplement benchmark metrics here.
# ============================================================

REF = Path("yourpath/ucs/UCS/ucs_value_merfish_4.py")

spec = importlib.util.spec_from_file_location("merfish_eval_core", REF)
core = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = core
spec.loader.exec_module(core)


# Fix the otherwise hard-coded "lung" output label.
core.DATASET = "merfish_unknown"


METHOD = "DISSECT"


def build(args, geo):
    run_dir = Path(args.run_dir)

    if args.mask:
        p = Path(args.mask)
    else:
        p = run_dir / "dissect_segmentation.tif"

    if not p.exists():
        raise SystemExit(f"missing DISSECT segmentation: {p}")

    lab = np.squeeze(tifffile.imread(p)).astype(np.int64)

    if lab.ndim != 2:
        raise SystemExit(
            f"expected 2-D segmentation, got shape={lab.shape}"
        )

    n_cells = len(np.unique(lab)) - int(np.any(lab == 0))

    core.log.info(
        f"DISSECT mask  {p}  shape={lab.shape}  labels={n_cells:,}"
    )

    # DISSECT prep resamples the native 10k ROI to its 0.2125 um/px
    # detector scale.  The common evaluator represents this as
    # native-ROI-pixels per prediction pixel.
    px_per_cell = args.roi_size / lab.shape[0]

    note = (
        "DISSECT native output; input MERFISH ROI resampled to "
        "0.2125 um/px for the DISSECT Xenium detector; "
        "uses_platform_prior=False"
    )

    info = run_dir / "dissect_out" / "run_info.json"
    if info.exists():
        try:
            d = json.loads(info.read_text())
            note += f"; run_info={d}"
        except Exception:
            pass

    return core.Prediction(
        labels=lab,
        px_per_cell=px_per_cell,
        method="DISSECT",
        uses_platform_prior=False,
        note=note,
    )


# The imported main_loop already exposes:
#   --root
#   --out-dir
#   --roi-size
#   --gt-mask
# and UCS add_args supplies:
#   --run-dir
#   --mask

if __name__ == "__main__":
    dataset = None

    if "--dataset-label" in sys.argv:
        i = sys.argv.index("--dataset-label")
        if i + 1 >= len(sys.argv):
            raise SystemExit("--dataset-label requires a value")
        dataset = sys.argv[i + 1]
        del sys.argv[i:i+2]

    if dataset:
        core.DATASET = dataset

    core.main_loop(METHOD, build)
