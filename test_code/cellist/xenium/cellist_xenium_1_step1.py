#!/usr/bin/env python3
import os
import gc
import shutil
import logging
from pathlib import Path

import pandas as pd
import torch

from Cellist.Cellpose import Cellpose


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


BASE = Path(os.path.expanduser(
    "~/Cosmx_6k_COAD_data/cellist_result/boms_5000_cellist_fixed_roi"
))

INPUT_DIR = BASE / "input"
ROI_DAPI_FILE = INPUT_DIR / "DAPI_boms_5000_fixed_roi.tif"
GEM_FILE = INPUT_DIR / "transcripts_boms_5000_fixed_roi.tsv.gz"

TEST_ROOT = BASE / "param_test_cellpose"
LOG_DIR = BASE / "logs"
SUMMARY_FILE = TEST_ROOT / "param_test_summary.tsv"

TEST_ROOT.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

GT_EXPECTED_CELLS = 1400

PARAM_GRID = [
    # baseline
    {"diameter": 20, "cellprob_threshold": 0.0,  "flow_threshold": 0.4, "no_local_threshold": False},

    # 放宽 cellprob
    {"diameter": 20, "cellprob_threshold": -1.0, "flow_threshold": 0.4, "no_local_threshold": False},
    {"diameter": 20, "cellprob_threshold": -2.0, "flow_threshold": 0.4, "no_local_threshold": False},

    # 关闭 local threshold
    {"diameter": 20, "cellprob_threshold": 0.0,  "flow_threshold": 0.4, "no_local_threshold": True},
    {"diameter": 20, "cellprob_threshold": -1.0, "flow_threshold": 0.4, "no_local_threshold": True},
    {"diameter": 20, "cellprob_threshold": -2.0, "flow_threshold": 0.4, "no_local_threshold": True},

    # 小 diameter
    {"diameter": 15, "cellprob_threshold": 0.0,  "flow_threshold": 0.4, "no_local_threshold": True},
    {"diameter": 15, "cellprob_threshold": -1.0, "flow_threshold": 0.4, "no_local_threshold": True},
    {"diameter": 15, "cellprob_threshold": -2.0, "flow_threshold": 0.4, "no_local_threshold": True},

    {"diameter": 12, "cellprob_threshold": 0.0,  "flow_threshold": 0.4, "no_local_threshold": True},
    {"diameter": 12, "cellprob_threshold": -1.0, "flow_threshold": 0.4, "no_local_threshold": True},
    {"diameter": 12, "cellprob_threshold": -2.0, "flow_threshold": 0.4, "no_local_threshold": True},

    {"diameter": 10, "cellprob_threshold": 0.0,  "flow_threshold": 0.4, "no_local_threshold": True},
    {"diameter": 10, "cellprob_threshold": -1.0, "flow_threshold": 0.4, "no_local_threshold": True},
    {"diameter": 10, "cellprob_threshold": -2.0, "flow_threshold": 0.4, "no_local_threshold": True},

    {"diameter": 8, "cellprob_threshold": 0.0,   "flow_threshold": 0.4, "no_local_threshold": True},
    {"diameter": 8, "cellprob_threshold": -1.0,  "flow_threshold": 0.4, "no_local_threshold": True},
    {"diameter": 8, "cellprob_threshold": -2.0,  "flow_threshold": 0.4, "no_local_threshold": True},

    # flow 放宽测试
    {"diameter": 12, "cellprob_threshold": -2.0, "flow_threshold": 0.8, "no_local_threshold": True},
    {"diameter": 10, "cellprob_threshold": -2.0, "flow_threshold": 0.8, "no_local_threshold": True},
    {"diameter": 8,  "cellprob_threshold": -2.0, "flow_threshold": 0.8, "no_local_threshold": True},
]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "cellist_boms_5000_param_test.log", mode="w"),
    ],
)


def make_tag(p):
    cp = str(p["cellprob_threshold"]).replace("-", "m").replace(".", "p")
    fl = str(p["flow_threshold"]).replace(".", "p")
    nlt = "T" if p["no_local_threshold"] else "F"
    return f"d{p['diameter']}_cp{cp}_flow{fl}_nlt{nlt}"


def validate_outputs(out_dir, out_prefix):
    prop_file = out_dir / f"{out_prefix}_Cellpose_nucleus_property.txt"
    coord_file = out_dir / f"{out_prefix}_Cellpose_nucleus_coord.txt"
    count_h5 = out_dir / f"{out_prefix}_Cellpose_segmentation_cell_count.h5"
    bin1_h5 = out_dir / f"{out_prefix}_bin1.h5"

    n_prop = 0
    coord_rows = 0
    assigned_cells = 0
    assigned_rows = 0

    if prop_file.exists():
        try:
            prop = pd.read_csv(prop_file, sep="\t")
            n_prop = len(prop)
        except Exception:
            n_prop = -1

    if coord_file.exists():
        try:
            coord = pd.read_csv(coord_file, sep="\t", usecols=lambda c: c in ["Cellpose"])
            coord_rows = len(coord)
            if "Cellpose" in coord.columns:
                assigned = coord["Cellpose"].dropna()
                assigned = assigned[assigned != 0]
                assigned_rows = len(assigned)
                assigned_cells = int(assigned.nunique())
        except Exception:
            coord_rows = -1

    return {
        "has_property": prop_file.exists(),
        "has_coord": coord_file.exists(),
        "has_count_h5": count_h5.exists(),
        "has_bin1_h5": bin1_h5.exists(),
        "nucleus_property_rows": n_prop,
        "coord_rows": coord_rows,
        "assigned_rows": assigned_rows,
        "assigned_cell_count": assigned_cells,
    }


def main():
    if not ROI_DAPI_FILE.exists():
        raise FileNotFoundError(ROI_DAPI_FILE)

    if not GEM_FILE.exists():
        raise FileNotFoundError(GEM_FILE)

    logging.info("========== Param test ==========")
    logging.info(f"ROI DAPI: {ROI_DAPI_FILE}")
    logging.info(f"GEM: {GEM_FILE}")
    logging.info(f"TEST_ROOT: {TEST_ROOT}")
    logging.info(f"GT expected cells: {GT_EXPECTED_CELLS}")

    logging.info(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")

    records = []

    for i, p in enumerate(PARAM_GRID, start=1):
        tag = make_tag(p)
        out_prefix = f"BOMS_5000_param_{tag}"
        out_dir = TEST_ROOT / tag

        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        logging.info("=" * 100)
        logging.info(f"[{i}/{len(PARAM_GRID)}] Running {tag}")
        logging.info(str(p))

        status = "UNKNOWN"
        error = ""

        try:
            Cellpose(
                platform="imaging",
                gem_path=str(GEM_FILE),
                img_path=str(ROI_DAPI_FILE),
                out_dir=str(out_dir),
                out_prefix=out_prefix,
                no_local_threshold=p["no_local_threshold"],
                diameter=p["diameter"],
                model_type="nuclei",
                flow_threshold=p["flow_threshold"],
                cellprob_threshold=p["cellprob_threshold"],
                expansion=False,
                expansion_dist=8,
            )

            status = "DONE"

        except Exception as e:
            status = "FAILED"
            error = repr(e)
            logging.exception(f"FAILED {tag}")

        metrics = validate_outputs(out_dir, out_prefix)

        ratio_prop = (
            metrics["nucleus_property_rows"] / GT_EXPECTED_CELLS
            if GT_EXPECTED_CELLS and metrics["nucleus_property_rows"] >= 0
            else None
        )

        ratio_assigned = (
            metrics["assigned_cell_count"] / GT_EXPECTED_CELLS
            if GT_EXPECTED_CELLS
            else None
        )

        rec = {
            "tag": tag,
            "status": status,
            "diameter": p["diameter"],
            "cellprob_threshold": p["cellprob_threshold"],
            "flow_threshold": p["flow_threshold"],
            "no_local_threshold": p["no_local_threshold"],
            **metrics,
            "gt_expected_cells": GT_EXPECTED_CELLS,
            "nucleus_over_gt_ratio": ratio_prop,
            "assigned_over_gt_ratio": ratio_assigned,
            "out_dir": str(out_dir),
            "error": error,
        }

        records.append(rec)

        df = pd.DataFrame(records)
        df.to_csv(SUMMARY_FILE, sep="\t", index=False)

        logging.info(
            f"{tag}: status={status}, "
            f"nucleus={metrics['nucleus_property_rows']}, "
            f"assigned_cells={metrics['assigned_cell_count']}, "
            f"nucleus/GT={ratio_prop:.3f}, "
            f"assigned/GT={ratio_assigned:.3f}"
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    df = pd.DataFrame(records)
    df.to_csv(SUMMARY_FILE, sep="\t", index=False)

    print("\n==============================")
    print("Parameter test finished")
    print("==============================")
    print("Summary:", SUMMARY_FILE)

    print("\nTop by nucleus_property_rows:")
    print(
        df.sort_values("nucleus_property_rows", ascending=False)[
            [
                "tag",
                "status",
                "diameter",
                "cellprob_threshold",
                "flow_threshold",
                "no_local_threshold",
                "nucleus_property_rows",
                "assigned_cell_count",
                "nucleus_over_gt_ratio",
                "assigned_over_gt_ratio",
            ]
        ].head(30).to_string(index=False)
    )

    print("\nTop by assigned_cell_count:")
    print(
        df.sort_values("assigned_cell_count", ascending=False)[
            [
                "tag",
                "status",
                "diameter",
                "cellprob_threshold",
                "flow_threshold",
                "no_local_threshold",
                "nucleus_property_rows",
                "assigned_cell_count",
                "nucleus_over_gt_ratio",
                "assigned_over_gt_ratio",
            ]
        ].head(30).to_string(index=False)
    )

    print("==============================\n")


if __name__ == "__main__":
    main()
