#!/usr/bin/env python3
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import gc
import traceback
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import zarr
import torch

from Cellist.Cellpose import Cellpose


base = Path("/home/data/vip232152/Cosmx_6k_COAD_data")

input_dir = base / "cellist_result" / "input" / "userdata"
tif_file = input_dir / "DAPI_tiled_uncompressed.tif"

# 旧 Step1A 目录：只读取 metadata 和 GEM
old_tile_root = base / "cellist_result" / "cellpose_tiles_cosmx_noswap_step1"
old_tile_meta_file = old_tile_root / "tile_metadata.tsv"

# 新 Step1B v2 输出目录：所有 Cellpose / Cellist 输出都写这里
new_tile_root = base / "cellist_result" / "cellpose_tiles_cosmx_noswap_step1b_v2_clean"

log_dir = base / "cellist_result" / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
new_tile_root.mkdir(parents=True, exist_ok=True)

main_log = log_dir / "step1b_v2_clean_run_cellpose_tiles.log"
summary_file = log_dir / "step1b_v2_clean_run_cellpose_tiles_summary.tsv"

out_prefix_base = "Cosmx_6k_COAD_noswap_v2_tile"

# Cellpose 参数保持不变
diameter = 20
model_type = "nuclei"
flow_threshold = 0.4
cellprob_threshold = 0.0
expansion = False
expansion_dist = 8

# 全量运行保持 None；测试可设 130, 135
start_tile_id = None
end_tile_id = None

# 低信号 tile 跳过阈值：避免明显背景区域耗时并报空矩阵
# 这些不会改变有效组织区域结果，只是跳过几乎无 DAPI 的 tile
min_dapi_nonzero_fraction = 0.05
min_dapi_mean = 5.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(main_log, mode="a"),
    ],
)


def append_summary(record):
    exists = summary_file.exists()
    pd.DataFrame([record]).to_csv(
        summary_file,
        sep="\t",
        index=False,
        mode="a",
        header=not exists,
    )


def validate_outputs(nucleus_prop, nucleus_coord):
    if not nucleus_prop.exists():
        return False, "MISSING_PROPERTY", 0, 0, 0

    if not nucleus_coord.exists():
        return False, "MISSING_COORD", 0, 0, 0

    try:
        prop = pd.read_csv(nucleus_prop, sep="\t")
    except Exception as e:
        return False, f"BAD_PROPERTY:{repr(e)}", 0, 0, 0

    if prop.empty:
        return False, "EMPTY_PROPERTY_NO_NUCLEI", 0, 0, 0

    try:
        coord = pd.read_csv(
            nucleus_coord,
            sep="\t",
            usecols=lambda c: c in ["Cellpose"]
        )
    except Exception as e:
        return False, f"BAD_COORD:{repr(e)}", len(prop), 0, 0

    if "Cellpose" not in coord.columns:
        return False, "NO_CELLPOSE_COLUMN", len(prop), len(coord), 0

    assigned = coord["Cellpose"].dropna()
    assigned = assigned[assigned != 0]

    n_assigned_rows = len(assigned)
    n_assigned_cells = assigned.nunique()

    if n_assigned_rows == 0 or n_assigned_cells == 0:
        return False, "NO_TRANSCRIPTS_ASSIGNED_TO_NUCLEI", len(prop), len(coord), 0

    return True, "VALID", len(prop), len(coord), int(n_assigned_cells)


logging.info("========== Step1B v2 clean start ==========")
logging.info(f"Old Step1A root: {old_tile_root}")
logging.info(f"New Step1B output root: {new_tile_root}")
logging.info(f"DAPI: {tif_file}")
logging.info(f"Metadata: {old_tile_meta_file}")
logging.info(f"Main log: {main_log}")
logging.info(f"Summary: {summary_file}")

logging.info("========== GPU check ==========")
logging.info(f"torch version: {torch.__version__}")
logging.info(f"torch CUDA: {torch.version.cuda}")
logging.info(f"torch.cuda.is_available(): {torch.cuda.is_available()}")

if torch.cuda.is_available():
    logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    logging.warning("GPU is not available. Cellpose may run on CPU.")

if not tif_file.exists():
    raise FileNotFoundError(tif_file)

if not old_tile_meta_file.exists():
    raise FileNotFoundError(old_tile_meta_file)

tile_meta = pd.read_csv(old_tile_meta_file, sep="\t")

if start_tile_id is not None:
    tile_meta = tile_meta[tile_meta["tile_id"] >= start_tile_id]

if end_tile_id is not None:
    tile_meta = tile_meta[tile_meta["tile_id"] <= end_tile_id]

# 保存一份新的 metadata，指向新输出目录，但保留旧 GEM 路径
new_meta_file = new_tile_root / "tile_metadata_step1b_v2_clean.tsv"
tile_meta.to_csv(new_meta_file, sep="\t", index=False)

logging.info(f"Tiles to process: {len(tile_meta)}")
logging.info(f"Saved v2 metadata: {new_meta_file}")

tif = tifffile.TiffFile(tif_file)
store = tif.aszarr()
img = zarr.open(store, mode="r")

if img.ndim == 3:
    img = img[0]

logging.info(f"Image shape={img.shape}, dtype={img.dtype}")

for _, meta in tile_meta.iterrows():
    tile_id = int(meta["tile_id"])
    tile_name = str(meta["tile_name"])

    read_x0 = int(meta["read_x0"])
    read_y0 = int(meta["read_y0"])
    read_x1 = int(meta["read_x1"])
    read_y1 = int(meta["read_y1"])

    core_x0 = int(meta["core_x0"])
    core_y0 = int(meta["core_y0"])
    core_x1 = int(meta["core_x1"])
    core_y1 = int(meta["core_y1"])

    n_tx = int(meta.get("n_transcripts_with_overlap", 0))

    # 旧 GEM，只读
    old_gem = Path(meta["gem_file"])

    # 新 tile 输出目录
    new_tile_dir = new_tile_root / tile_name
    new_tile_dir.mkdir(parents=True, exist_ok=True)

    tile_img = new_tile_dir / f"{tile_name}_DAPI.tif"
    tile_info = new_tile_dir / "tile_info.tsv"

    out_prefix = f"{out_prefix_base}_{tile_id:04d}"

    done_flag = new_tile_dir / "CELLPOSE_DONE.flag"
    failed_flag = new_tile_dir / "CELLPOSE_FAILED.flag"
    blank_flag = new_tile_dir / "CELLPOSE_BLANK_DAPI.flag"
    low_signal_flag = new_tile_dir / "CELLPOSE_LOW_DAPI_SIGNAL.flag"
    no_tx_flag = new_tile_dir / "CELLPOSE_NO_TRANSCRIPTS.flag"
    no_nuclei_flag = new_tile_dir / "CELLPOSE_NO_VALID_NUCLEI.flag"

    nucleus_prop = new_tile_dir / f"{out_prefix}_Cellpose_nucleus_property.txt"
    nucleus_coord = new_tile_dir / f"{out_prefix}_Cellpose_nucleus_coord.txt"
    nucleus_count_h5 = new_tile_dir / f"{out_prefix}_Cellpose_segmentation_cell_count.h5"
    bin1_h5 = new_tile_dir / f"{out_prefix}_bin1.h5"

    # 只检查新目录里的结果，不看旧目录
    if done_flag.exists():
        is_valid, reason, n_prop, n_coord, n_cells = validate_outputs(
            nucleus_prop,
            nucleus_coord
        )

        if is_valid:
            logging.info(
                f"Skip valid DONE tile {tile_id:04d}: "
                f"nuclei={n_prop}, assigned_cells={n_cells}"
            )
            continue

        old_flag_text = done_flag.read_text().strip() if done_flag.exists() else ""
        if old_flag_text in ["BLANK_DAPI", "NO_TRANSCRIPTS", "LOW_DAPI_SIGNAL"]:
            logging.info(f"Skip old skipped tile {tile_id:04d}: {old_flag_text}")
            continue

        logging.info(f"Tile {tile_id:04d} had invalid DONE ({reason}); rerun in v2 dir.")
        done_flag.unlink(missing_ok=True)

    if blank_flag.exists() or no_tx_flag.exists() or low_signal_flag.exists() or no_nuclei_flag.exists():
        logging.info(f"Skip previously skipped v2 tile {tile_id:04d}")
        continue

    if n_tx == 0 or (not old_gem.exists()) or old_gem.stat().st_size == 0:
        logging.info(f"No transcripts for tile {tile_id:04d}, skip")
        no_tx_flag.write_text("NO_TRANSCRIPTS\n")
        done_flag.write_text("NO_TRANSCRIPTS\n")
        append_summary({
            "tile_id": tile_id,
            "tile_name": tile_name,
            "status": "NO_TRANSCRIPTS",
            "n_tx": n_tx,
            "gem_path": str(old_gem),
            "output_dir": str(new_tile_dir),
        })
        continue

    logging.info("=" * 100)
    logging.info(f"Processing tile {tile_id:04d}: {tile_name}")
    logging.info(f"Core: x={core_x0}-{core_x1}, y={core_y0}-{core_y1}")
    logging.info(f"Read: x={read_x0}-{read_x1}, y={read_y0}-{read_y1}")
    logging.info(f"n_tx={n_tx:,}")
    logging.info(f"Using GEM: {old_gem}")
    logging.info(f"Writing output to: {new_tile_dir}")

    info = pd.DataFrame([{
        **meta.to_dict(),
        "old_gem_file": str(old_gem),
        "new_output_dir": str(new_tile_dir),
        "out_prefix": out_prefix,
    }])
    info.to_csv(tile_info, sep="\t", index=False)

    try:
        tile_array = np.asarray(img[read_y0:read_y1, read_x0:read_x1])

        dapi_min = int(tile_array.min())
        dapi_max = int(tile_array.max())
        dapi_mean = float(tile_array.mean())
        dapi_nonzero = float((tile_array > 0).mean())

        logging.info(
            f"DAPI stats: shape={tile_array.shape}, "
            f"min={dapi_min}, max={dapi_max}, "
            f"mean={dapi_mean:.6f}, nonzero={dapi_nonzero:.6f}"
        )

        if dapi_max <= dapi_min:
            blank_flag.write_text(
                f"BLANK_DAPI\nmin={dapi_min}\nmax={dapi_max}\nmean={dapi_mean}\n"
            )
            done_flag.write_text("BLANK_DAPI\n")

            append_summary({
                "tile_id": tile_id,
                "tile_name": tile_name,
                "status": "BLANK_DAPI",
                "n_tx": n_tx,
                "dapi_min": dapi_min,
                "dapi_max": dapi_max,
                "dapi_mean": dapi_mean,
                "dapi_nonzero": dapi_nonzero,
                "gem_path": str(old_gem),
                "output_dir": str(new_tile_dir),
            })

            del tile_array
            gc.collect()
            continue

        if dapi_nonzero < min_dapi_nonzero_fraction or dapi_mean < min_dapi_mean:
            low_signal_flag.write_text(
                f"LOW_DAPI_SIGNAL\n"
                f"min={dapi_min}\n"
                f"max={dapi_max}\n"
                f"mean={dapi_mean}\n"
                f"nonzero={dapi_nonzero}\n"
                f"min_dapi_nonzero_fraction={min_dapi_nonzero_fraction}\n"
                f"min_dapi_mean={min_dapi_mean}\n"
            )
            done_flag.write_text("LOW_DAPI_SIGNAL\n")

            logging.info(
                f"Low DAPI signal tile {tile_id:04d}, skip. "
                f"mean={dapi_mean:.6f}, nonzero={dapi_nonzero:.6f}"
            )

            append_summary({
                "tile_id": tile_id,
                "tile_name": tile_name,
                "status": "LOW_DAPI_SIGNAL",
                "n_tx": n_tx,
                "dapi_min": dapi_min,
                "dapi_max": dapi_max,
                "dapi_mean": dapi_mean,
                "dapi_nonzero": dapi_nonzero,
                "gem_path": str(old_gem),
                "output_dir": str(new_tile_dir),
            })

            del tile_array
            gc.collect()
            continue

        # 写入新目录里的临时 tile 图像，Cellist 只会看到新文件
        tifffile.imwrite(
            tile_img,
            tile_array,
            bigtiff=True,
            compression=None,
        )

        del tile_array
        gc.collect()

        try:
            Cellpose(
                platform="imaging",
                gem_path=str(old_gem),
                img_path=str(tile_img),
                out_dir=str(new_tile_dir),
                out_prefix=out_prefix,
                no_local_threshold=False,
                diameter=diameter,
                model_type=model_type,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
                expansion=expansion,
                expansion_dist=expansion_dist,
            )

        except ValueError as e:
            err = traceback.format_exc()
            failed_flag.write_text(err)
            logging.warning(f"Tile {tile_id:04d} Cellpose/Cellist ValueError: {repr(e)}")

        is_valid, reason, n_prop, n_coord, n_cells = validate_outputs(
            nucleus_prop,
            nucleus_coord
        )

        if is_valid:
            done_flag.write_text("DONE\n")
            failed_flag.unlink(missing_ok=True)
            no_nuclei_flag.unlink(missing_ok=True)

            status = "DONE"
            logging.info(
                f"Tile {tile_id:04d} VALID DONE: "
                f"nucleus_property={n_prop}, "
                f"coord_rows={n_coord}, "
                f"assigned_cells={n_cells}"
            )

        else:
            status = reason
            no_nuclei_flag.write_text(
                f"{reason}\n"
                f"nucleus_property_rows={n_prop}\n"
                f"coord_rows={n_coord}\n"
                f"assigned_cells={n_cells}\n"
            )
            done_flag.unlink(missing_ok=True)

            logging.warning(
                f"Tile {tile_id:04d} invalid output: {reason}, "
                f"prop={n_prop}, coord={n_coord}, assigned_cells={n_cells}"
            )

        append_summary({
            "tile_id": tile_id,
            "tile_name": tile_name,
            "status": status,
            "n_tx": n_tx,
            "dapi_min": dapi_min,
            "dapi_max": dapi_max,
            "dapi_mean": dapi_mean,
            "dapi_nonzero": dapi_nonzero,
            "nucleus_property_rows": n_prop,
            "nucleus_coord_rows": n_coord,
            "assigned_cell_count": n_cells,
            "has_nucleus_prop": nucleus_prop.exists(),
            "has_nucleus_coord": nucleus_coord.exists(),
            "has_nucleus_count_h5": nucleus_count_h5.exists(),
            "has_bin1_h5": bin1_h5.exists(),
            "gem_path": str(old_gem),
            "output_dir": str(new_tile_dir),
        })

    except Exception:
        err = traceback.format_exc()
        failed_flag.write_text(err)

        logging.error(f"Tile {tile_id:04d} FAILED")
        logging.error(err)

        append_summary({
            "tile_id": tile_id,
            "tile_name": tile_name,
            "status": "FAILED",
            "n_tx": n_tx,
            "error": err.splitlines()[-1],
            "gem_path": str(old_gem),
            "output_dir": str(new_tile_dir),
        })

    finally:
        if tile_img.exists():
            tile_img.unlink()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()

tif.close()

logging.info("Finished Step1B v2 clean")

print("\n==============================")
print("Step1B v2 clean finished")
print("==============================")
print(f"New output root: {new_tile_root}")
print(f"Summary: {summary_file}")
print(f"Main log: {main_log}")
print("==============================\n")
