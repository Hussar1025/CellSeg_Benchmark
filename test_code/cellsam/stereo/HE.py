import os
import time
from pathlib import Path

_PROCESS_STARTED = time.perf_counter()

import sys
sys.path.insert(0, "/data/qiuyijia")
from he_performance import finalize_performance

import numpy as np
import tifffile
import zarr
from PIL import Image

from cellSAM import cellsam_pipeline


JOBS = [
    {
        "input": os.environ.get(
            "HE_INPUT_TIF",
            "/data/qiuyijia/dataset/stereo_mouse_embryo/D05103C4D4_HE_regist.tif",
        ),
        "output": os.environ.get(
            "HE_CELLSAM_OUTPUT_TIF",
            "/data/qiuyijia/dataset/stereo_mouse_embryo/segmentation_outputs/cellsam/labels.tif",
        ),
        "level":  0, # 读取tiff的层序数
        "region": None,
    },
    # 再加更多 job ...
]

BBOX_THRESHOLD       = 0.4      # CellSAM 默认值; 越低 recall 越高
USE_WSI              = True     # 让 cellSAM 自己做 tiling
BLOCK_SIZE           = 512      # 内部 tile 大小, [256, 2048]
OVERLAP              = 70       # 内部 tile 重叠
LOW_CONTRAST_ENHANCE = False    # H&E 不需要
GAUGE_CELL_SIZE      = False    # 额外一次估细胞大小的 forward, 慢但更稳
MODEL_PATH           = None     # None = 用默认 generalist 权重


_TIFF_SUFFIXES = {".svs", ".ndpi", ".tif", ".tiff", ".scn", ".bif"}


def _to_hwc_rgb(arr: np.ndarray) -> np.ndarray:
    """把任意排布的彩色数组规整为 (H, W, 3),dtype 保持。"""
    if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
        arr = np.moveaxis(arr, 0, -1)        # CHW -> HWC
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]                   # RGBA -> RGB
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"无法解释成 RGB: shape={arr.shape}")
    return arr


def _load_wsi(path: Path, level: int, region) -> np.ndarray:
    """用 tifffile + zarr 读金字塔 TIFF 的一个 level / 区域。"""
    with tifffile.TiffFile(str(path)) as tif:
        series = tif.series[0]
        levels = series.levels if getattr(series, "levels", None) else [series]
        if level >= len(levels):
            raise ValueError(
                f"请求 level={level},但该文件只有 {len(levels)} 个 level。"
            )
        level_series = levels[level]

        if region is None:
            return level_series.asarray()

        # region = (x_l0, y_l0, w_level, h_level)
        x_l0, y_l0, w, h = region
        axes = levels[0].axes
        y_axis = axes.index("Y") if "Y" in axes else 0
        x_axis = axes.index("X") if "X" in axes else 1

        # level-0 坐标 -> 当前 level 坐标
        ds_y = levels[0].shape[y_axis] / level_series.shape[y_axis]
        ds_x = levels[0].shape[x_axis] / level_series.shape[x_axis]
        y_lv = int(round(y_l0 / ds_y))
        x_lv = int(round(x_l0 / ds_x))

        # zarr 懒切片,避免 decode 整层
        z = zarr.open(level_series.aszarr(), mode="r")
        if not hasattr(z, "shape"):           # 兜底: Group
            z = z[sorted(z.array_keys())[0]]
        slicer = [slice(None)] * z.ndim
        slicer[y_axis] = slice(y_lv, y_lv + h)
        slicer[x_axis] = slice(x_lv, x_lv + w)
        return np.asarray(z[tuple(slicer)])


def load_he_rgb(job: dict) -> np.ndarray:
    """读取 H&E 并返回 CellSAM 文档要求的 (H, W, 3) RGB 图。"""
    p = Path(job["input"])
    if p.suffix.lower() in _TIFF_SUFFIXES:
        arr = _load_wsi(p, int(job.get("level", 0)), job.get("region"))
    else:
        arr = np.asarray(Image.open(p).convert("RGB"))

    return _to_hwc_rgb(arr)


def run_one(job: dict) -> None:
    out_path = Path(job["output"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading: {job['input']}")
    rgb = load_he_rgb(job)
    print(f"[INFO]   shape = {rgb.shape}, dtype = {rgb.dtype}, "
          f"~{rgb.nbytes / 1e9:.2f} GB in RAM")

    print(f"[INFO] cellsam_pipeline (use_wsi={USE_WSI}, "
          f"block_size={BLOCK_SIZE}, overlap={OVERLAP}) ...")
    mask = cellsam_pipeline(
        rgb,
        use_wsi=USE_WSI,
        bbox_threshold=BBOX_THRESHOLD,
        block_size=BLOCK_SIZE,
        overlap=OVERLAP,
        low_contrast_enhancement=LOW_CONTRAST_ENHANCE,
        gauge_cell_size=GAUGE_CELL_SIZE,
        model_path=MODEL_PATH,
    ).astype(np.uint32, copy=False)
    print(f"[INFO]   detected cells = {int(mask.max())}")

    print(f"[INFO] Writing -> {out_path}")
    tifffile.imwrite(
        str(out_path),
        mask,
        bigtiff=True,
        tile=(512, 512),
        compression="zlib",
        photometric="minisblack",
        metadata={"axes": "YX",
                  "description": "CellSAM label mask (uint32)"},
    )
    print(f"[DONE] {out_path}\n")


if __name__ == "__main__":
    _status, _error = "success", None
    try:
        for job in JOBS:
            run_one(job)
    except BaseException as exc:
        _status, _error = "failed", repr(exc)
        raise
    finally:
        _performance_dir = Path(JOBS[0]["output"]).parent if JOBS else Path.cwd()
        finalize_performance(
            _PROCESS_STARTED, _performance_dir, "CellSAM",
            status=_status, error=_error,
        )
