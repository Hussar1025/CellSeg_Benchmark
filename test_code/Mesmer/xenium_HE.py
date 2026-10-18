import os
import gc
import json
import time
import logging
import warnings
import threading
import shutil
from contextlib import contextmanager
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, Tuple, List

import numpy as np

warnings.filterwarnings("ignore")

@dataclass
class Config:
    wsi_path: str = "/data/qiuyijia/dataset/stereo_mouse_embryo/D05103C4D4_HE_regist.tif"
    output_dir: str = "/data/qiuyijia/Mesmer/stereo_mouse_embryo_output"
    run_name: str = "mesmer_he"

    wsi_level: int = 0

    # Mesmer 训练分辨率是 0.5 microns/pixel。
    # 如果 WSI 在 level=0 的 mpp 不是 0.5，predict 时通过 image_mpp 参数让 Mesmer 自己缩放。
    # 必须由用户提供真实的 H&E µm/px；不能从该 TIFF 的无效 resolution 猜测。
    image_mpp: Optional[float] = None

    # ---------- 分块 (Tile) ----------
    tile_size: int = 2048                              # 每个 tile 的边长 (pixel, 在 wsi_level 上)
    tile_overlap: int = 256                            # 相邻 tile 的重叠 (pixel)，建议 >= 平均细胞直径 * 2

    # ---------- 背景过滤 ----------
    skip_background: bool = True
    background_mean_threshold: int = 220               # RGB 平均值 > 阈值 (近全白) 跳过
    background_std_threshold: int = 8                  # 全图标准差 < 阈值 (无内容) 跳过

    # ---------- Mesmer 推理 ----------
    compartment: str = "whole-cell"                    # "whole-cell" 或 "nuclear"
    batch_size: int = 4                                # Mesmer 内部 batch
    interior_threshold: float = 0.3                    # 越低 -> 细胞越大；默认 0.3
    maxima_threshold: float = 0.075                    # 越低 -> 切得越碎；默认 0.075
    pixel_expansion: Optional[int] = None              # 仅在 compartment="nuclear" 时常用，扩展像素数

    # ---------- ROI (可选: 只处理 WSI 的一个矩形子区域) ----------
    # 留 None 表示处理整张。坐标都在 wsi_level=0 坐标系。
    roi_x: Optional[int] = None
    roi_y: Optional[int] = None
    roi_w: Optional[int] = None
    roi_h: Optional[int] = None

    # ---------- 输出选项 ----------
    save_mask_tiff: bool = True                        # 保存全图 instance mask (uint32, BigTIFF)
    mask_memmap_path: Optional[str] = None             # 默认写入 output run 目录
    save_geojson: bool = False                         # 保存 QuPath 可读的 polygon geojson (慢，细胞多时慎用)
    save_preview_png: bool = True                      # 保存缩略图叠加预览
    preview_max_side: int = 4096                       # 预览图最长边像素

    # ---------- GPU ----------
    gpu_id: str = "0"                                  # 设为 "-1" 强制 CPU


CONFIG = Config()


# 设置 GPU 必须在 import tensorflow 之前
os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG.gpu_id
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tifffile
from skimage.color import rgb2hed
from skimage.measure import regionprops
import psutil

_DEEPCELL_APP = None


def get_logger(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("mesmer_he")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(out_dir / "run.log", mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger



@contextmanager
def monitored_phase(name: str, monitor, phase_times: dict):
    start = time.perf_counter()
    try:
        yield
    finally:
        phase_times[name] = phase_times.get(name, 0.0) + (time.perf_counter() - start)
        if monitor is not None:
            monitor.sample()


class RSSMonitor:
    """低开销后台 RSS 采样器；数值以 bytes 保存，便于写入 benchmark。"""

    def __init__(self, interval=0.25):
        self.interval = float(interval)
        self.process = psutil.Process(os.getpid())
        self.peak_rss = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def sample(self):
        try:
            self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
        except (psutil.Error, OSError):
            pass

    def _run(self):
        while not self._stop.wait(self.interval):
            self.sample()

    def __enter__(self):
        self.sample()
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self._thread.join(timeout=max(1.0, self.interval * 4))
        self.sample()


class TiffReader:
    """普通单页 TIFF reader；按窗口读取，不依赖 WSI backend。"""

    backend = "tifffile"
    level_downsample = 1.0

    def __init__(self, path: str, level: int = 0):
        if level != 0:
            raise ValueError("普通 TIFF reader only supports wsi_level=0")
        self.path = str(path)
        self.file = tifffile.TiffFile(self.path)
        if len(self.file.pages) != 1:
            self.close()
            raise ValueError("Expected a single-page TIFF")
        self.page = self.file.pages[0]
        if self.page.shape != (44100, 23520, 3):
            self.close()
            raise ValueError(f"Unexpected HE shape: {self.page.shape}")
        if self.page.dtype != np.uint8 or self.page.samplesperpixel != 3:
            self.close()
            raise ValueError("HE TIFF must be RGB uint8")

    @property
    def level_dims(self):
        return (self.page.shape[1], self.page.shape[0])

    def get_mpp(self):
        # resolution unit is NONE in this dataset, so it is not usable as µm/px.
        return None

    def read_region(self, x_l: int, y_l: int, w: int, h: int):
        x = max(0, int(x_l))
        y = max(0, int(y_l))
        x1 = min(self.page.shape[1], x + int(w))
        y1 = min(self.page.shape[0], y + int(h))
        if x1 <= x or y1 <= y:
            return np.empty((0, 0, 3), dtype=np.uint8)
        # The source is one compressed strip per row. Decode only requested rows,
        # then crop columns; page.asarray(key=...) is not available in tifffile.
        rows = []
        with open(self.path, "rb") as fh:
            for row in range(y, y1):
                offset = int(self.page.dataoffsets[row])
                count = int(self.page.databytecounts[row])
                fh.seek(offset)
                decoded, _, _ = self.page.decode(fh.read(count), row)
                if decoded is None:
                    raise RuntimeError(f"Failed to decode TIFF strip {row}")
                row_arr = np.asarray(decoded).reshape(-1, self.page.shape[2])
                rows.append(row_arr[x:x1, :][None, ...])
        arr = np.concatenate(rows, axis=0)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"Unexpected TIFF window shape: {arr.shape}")
        return np.asarray(arr, dtype=np.uint8)

    def close(self):
        if getattr(self, "file", None) is not None:
            self.file.close()
            self.file = None


class WSIReader:
    """统一封装 openslide / tiffslide。优先用 openslide。"""

    def __init__(self, path: str, level: int = 0):
        self.path = path
        self.level = level
        self.backend = None
        self.slide = None
        self._open()

    def _open(self):
        errors = []
        try:
            import openslide
            self.slide = openslide.OpenSlide(self.path)
            self.backend = "openslide"
        except Exception as e:
            errors.append(f"openslide: {e}")
            try:
                import tiffslide
                self.slide = tiffslide.TiffSlide(self.path)
                self.backend = "tiffslide"
            except Exception as e2:
                errors.append(f"tiffslide: {e2}")
                try:
                    self.slide = TiffReader(self.path, level=self.level)
                    self.backend = "tifffile"
                except Exception as e3:
                    errors.append(f"tifffile: {e3}")
                    raise RuntimeError(
                        "Cannot open WSI/TIFF. " + " | ".join(errors)
                    ) from e3

        if self.backend == "tifffile":
            return
        if self.level >= self.slide.level_count:
            raise ValueError(
                f"wsi_level={self.level} >= level_count={self.slide.level_count}"
            )

    @property
    def level_dims(self) -> Tuple[int, int]:
        """(width, height) at self.level"""
        if self.backend == "tifffile":
            return self.slide.level_dims
        return self.slide.level_dimensions[self.level]

    @property
    def level_downsample(self) -> float:
        if self.backend == "tifffile":
            return 1.0
        return float(self.slide.level_downsamples[self.level])

    def get_mpp(self) -> Optional[float]:
        if self.backend == "tifffile":
            return self.slide.get_mpp()
        """读取 level=0 的 microns per pixel；读不到返回 None。

        注意: 有些后端 (如 tiffslide) 会把 key 写进 properties 但值是 None
        或空字符串, 所以不能只判断 `k in props`, 必须校验值本身,
        且要同时捕获 ValueError 和 TypeError (float(None) 抛的是 TypeError)。
        """
        props = dict(self.slide.properties)
        for k in ("openslide.mpp-x", "tiffslide.mpp-x", "aperio.MPP"):
            v = props.get(k)
            if v is None or v == "":
                continue
            try:
                mpp = float(v)
            except (ValueError, TypeError):
                continue
            if mpp > 0:                 # 排除 0 这类无意义的占位值
                return mpp
        return None

    def read_region(self, x_l: int, y_l: int, w: int, h: int) -> np.ndarray:
        """
        从当前 level 读取 [x_l, y_l, w, h] 区域 (level 坐标系)。
        openslide.read_region 接受的 location 是 level=0 坐标，size 是当前 level 大小。
        返回 RGB ndarray, dtype=uint8, shape=(h, w, 3)
        """
        if self.backend == "tifffile":
            return self.slide.read_region(x_l, y_l, w, h)
        x0 = int(x_l * self.level_downsample)
        y0 = int(y_l * self.level_downsample)
        if self.backend == "openslide":
            img = self.slide.read_region((x0, y0), self.level, (w, h)).convert("RGB")
            return np.asarray(img)
        else:
            arr = self.slide.read_region((x0, y0), self.level, (w, h), as_array=True)
            if arr.shape[-1] == 4:
                arr = arr[..., :3]
            return arr

    def close(self):
        try:
            self.slide.close()
        except Exception:
            pass


# ---------- H&E -> (nuclear, membrane) 双通道 ----------

def he_to_two_channels(rgb_tile: np.ndarray) -> np.ndarray:
    """
    用颜色去卷积把 H&E RGB 拆成 Hematoxylin (核) 和 Eosin (膜/胞质)。
    输出 shape=(H, W, 2), float32, 已 min-max 归一化到 [0,1]，
    然后再乘以 255 以接近 Mesmer 训练数据的尺度。
    Mesmer 期望的通道顺序是 [nuclear, membrane]。
    """
    hed = rgb2hed(rgb_tile)                  # (H, W, 3): H, E, D
    h = hed[..., 0]
    e = hed[..., 1]

    def _norm(x):
        lo, hi = np.percentile(x, [1, 99])
        if hi - lo < 1e-8:
            return np.zeros_like(x, dtype=np.float32)
        x = np.clip((x - lo) / (hi - lo), 0, 1)
        return x.astype(np.float32)

    nuc = _norm(h) * 255.0
    mem = _norm(e) * 255.0
    return np.stack([nuc, mem], axis=-1).astype(np.float32)


def is_background(rgb_tile: np.ndarray, mean_thr: int, std_thr: int) -> bool:
    """简单背景判断: 接近全白或几乎没纹理。"""
    mean_v = rgb_tile.mean()
    std_v = rgb_tile.std()
    return (mean_v > mean_thr) or (std_v < std_thr)


# ---------- Mesmer 包装 ----------

def get_mesmer_app():
    global _DEEPCELL_APP
    if _DEEPCELL_APP is None:
        from deepcell.applications import Mesmer
        _DEEPCELL_APP = Mesmer()
    return _DEEPCELL_APP


def predict_tile(app, two_chan: np.ndarray, cfg: Config, image_mpp: float) -> np.ndarray:
    """
    输入 (H, W, 2) float32, 返回 (H, W) int32 instance mask (背景 0)。
    """
    x = two_chan[None, ...]                  # (1, H, W, 2)

    postprocess_kwargs = {
        "interior_threshold": cfg.interior_threshold,
        "maxima_threshold": cfg.maxima_threshold,
    }
    kwargs = dict(
        image_mpp=image_mpp,
        compartment=cfg.compartment,
        batch_size=cfg.batch_size,
    )
    if cfg.compartment == "whole-cell":
        kwargs["postprocess_kwargs_whole_cell"] = postprocess_kwargs
    else:
        if cfg.pixel_expansion is not None:
            postprocess_kwargs["pixel_expansion"] = cfg.pixel_expansion
        kwargs["postprocess_kwargs_nuclear"] = postprocess_kwargs

    pred = app.predict(x, **kwargs)          # (1, H, W, 1) int
    return pred[0, ..., 0].astype(np.int32)


# ---------- Tile 网格生成 ----------

def make_tile_grid(
    width: int, height: int, tile: int, overlap: int,
    offset_x: int = 0, offset_y: int = 0,
) -> List[Tuple[int, int, int, int, int, int, int, int]]:
    """
    返回 tile 列表，每项:
      (x_read, y_read, w_read, h_read,    # 读取区域 (含重叠)
       core_x0, core_y0, core_x1, core_y1) # 中心区(在 tile 局部坐标系内)，
                                           # 只有中心区里的细胞会被保留到最终结果
    边界 tile 的中心区会延伸到图像边缘。
    """
    step = tile - overlap
    half = overlap // 2

    xs = list(range(0, width, step))
    ys = list(range(0, height, step))

    grid = []
    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            x_read = x
            y_read = y
            w_read = min(tile, width - x)
            h_read = min(tile, height - y)
            if w_read <= 0 or h_read <= 0:
                continue

            core_x0 = 0 if xi == 0 else half
            core_y0 = 0 if yi == 0 else half
            core_x1 = w_read if (x + tile) >= width else (tile - half)
            core_y1 = h_read if (y + tile) >= height else (tile - half)

            grid.append((
                x_read + offset_x, y_read + offset_y, w_read, h_read,
                core_x0, core_y0, core_x1, core_y1,
            ))
    return grid


# ---------- 拼接: 把 tile mask 写入全图 mask，并保证 ID 全局唯一 ----------

def merge_tile_into_global(
    global_mask: np.ndarray,
    tile_mask: np.ndarray,
    tile_origin_xy: Tuple[int, int],
    core_box: Tuple[int, int, int, int],
    next_label: int,
) -> int:
    """
    把 tile_mask 中位于中心区 (core_box) 内的实例写入 global_mask 的对应位置。
    返回更新后的 next_label。
    core_box = (x0, y0, x1, y1) 在 tile 局部坐标里
    """
    x0, y0, x1, y1 = core_box
    core = tile_mask[y0:y1, x0:x1]
    if core.size == 0:
        return next_label

    # 找在 core 区出现过的实例 id
    present_ids = np.unique(core)
    present_ids = present_ids[present_ids != 0]
    if present_ids.size == 0:
        return next_label

    # 给这些 id 分配全局新 id (连续)
    remap = np.zeros(int(present_ids.max()) + 1, dtype=np.int64)
    for old_id in present_ids:
        remap[old_id] = next_label
        next_label += 1

    # 在整个 tile 上做映射 (一个细胞可能跨中心区边界, 我们要把整个 tile 内属于这些 id 的像素都写进去,
    # 但只在它们在 core 区有像素时才保留, 这样不会出现 "细胞在另一个 tile 已被记录又被这里覆盖" 的情况)
    full_remap = np.zeros(int(tile_mask.max()) + 1, dtype=np.int64)
    full_remap[present_ids] = remap[present_ids]
    new_tile = full_remap[tile_mask]

    # 写入 global mask, 只覆盖 global 上还是背景的位置 (避免相邻 tile 覆盖另一 tile 已写入的细胞)
    tx, ty = tile_origin_xy
    th, tw = tile_mask.shape
    region = global_mask[ty:ty + th, tx:tx + tw]
    write_mask = (region == 0) & (new_tile > 0)
    region[write_mask] = new_tile[write_mask].astype(global_mask.dtype)
    global_mask[ty:ty + th, tx:tx + tw] = region

    return next_label


# ---------- 预览图 ----------

def save_preview(
    reader: WSIReader,
    global_mask: np.ndarray,
    out_path: Path,
    roi_xywh: Tuple[int, int, int, int],
    max_side: int,
):
    """保存缩略图叠加；按输出块读取，避免一次性加载整张 ROI。"""
    from PIL import Image
    x, y, w, h = roi_xywh
    scale = min(1.0, max_side / max(w, h))
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    thumb = np.empty((new_h, new_w, 3), dtype=np.uint8)
    mask_thumb = np.zeros((new_h, new_w), dtype=np.uint8)
    block_out = 256
    for oy in range(0, new_h, block_out):
        oy1 = min(new_h, oy + block_out)
        sy0 = int(oy / scale)
        sy1 = min(h, max(sy0 + 1, int(oy1 / scale)))
        for ox in range(0, new_w, block_out):
            ox1 = min(new_w, ox + block_out)
            sx0 = int(ox / scale)
            sx1 = min(w, max(sx0 + 1, int(ox1 / scale)))
            source = reader.read_region(x + sx0, y + sy0, sx1 - sx0, sy1 - sy0)
            thumb[oy:oy1, ox:ox1] = np.asarray(
                Image.fromarray(source).resize((ox1 - ox, oy1 - oy), Image.BILINEAR)
            )
            mask_block = np.asarray(global_mask[y + sy0:y + sy1, x + sx0:x + sx1] > 0, dtype=np.uint8) * 255
            mask_thumb[oy:oy1, ox:ox1] = np.asarray(
                Image.fromarray(mask_block).resize((ox1 - ox, oy1 - oy), Image.NEAREST)
            )
    overlay = thumb.copy()
    red = np.zeros_like(thumb)
    red[..., 0] = 255
    alpha = (mask_thumb > 0)[..., None] * 0.35
    overlay = (overlay * (1 - alpha) + red * alpha).astype(np.uint8)
    Image.fromarray(overlay).save(out_path)


# ---------- 主流程 ----------

def run(cfg: Config):
    out_dir = Path(cfg.output_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger(out_dir)

    log.info("==== Mesmer H&E WSI Benchmark ====")
    log.info(f"WSI: {cfg.wsi_path}")
    log.info(f"Output: {out_dir}")

    # 保存配置
    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    if not Path(cfg.wsi_path).exists():
        raise FileNotFoundError(cfg.wsi_path)

    # 打开 WSI
    reader = WSIReader(cfg.wsi_path, level=cfg.wsi_level)
    W_l, H_l = reader.level_dims
    log.info(f"Backend={reader.backend}, level={cfg.wsi_level}, "
             f"dims(W,H)=({W_l},{H_l}), downsample={reader.level_downsample}")

    # mpp 必须来自用户确认的扫描元数据；本 TIFF 的 resolution unit=NONE。
    if cfg.image_mpp is None or not np.isfinite(cfg.image_mpp) or cfg.image_mpp <= 0:
        raise ValueError(
            "Config.image_mpp must be a confirmed positive H&E microns/pixel value; "
            "the input TIFF has no usable mpp metadata."
        )
    cur_mpp = float(cfg.image_mpp)
    log.info(f"Using user-specified image_mpp={cur_mpp} µm/px")

    # ROI
    if cfg.roi_x is not None:
        # ROI 坐标是 level=0 的, 转到当前 level
        ds = reader.level_downsample
        rx = int(cfg.roi_x / ds)
        ry = int(cfg.roi_y / ds)
        rw = int(cfg.roi_w / ds)
        rh = int(cfg.roi_h / ds)
    else:
        rx, ry, rw, rh = 0, 0, W_l, H_l
    log.info(f"ROI at level {cfg.wsi_level}: x={rx}, y={ry}, w={rw}, h={rh}")

    # 准备磁盘 mask，避免整图 ndarray 占用约 3.9 GiB RAM
    mask_tmp = Path(cfg.mask_memmap_path) if cfg.mask_memmap_path else out_dir / "instance_mask.uint32.memmap"
    mask_tmp.parent.mkdir(parents=True, exist_ok=True)
    expected_bytes = int(rh) * int(rw) * np.dtype(np.uint32).itemsize
    free_bytes = shutil.disk_usage(mask_tmp.parent).free
    if free_bytes < expected_bytes * 2:
        raise RuntimeError(
            f"Insufficient disk space for mask memmap/output: free={free_bytes} "
            f"bytes, expected at least {expected_bytes * 2}"
        )
    global_mask = np.memmap(mask_tmp, mode="w+", dtype=np.uint32, shape=(rh, rw))
    global_mask[:] = 0
    global_mask.flush()
    log.info(f"Global mask memmap={mask_tmp}, shape={global_mask.shape}, "
             f"disk_bytes={global_mask.nbytes}")

    # 生成 tile grid (在 ROI 局部坐标里)
    grid = make_tile_grid(rw, rh, cfg.tile_size, cfg.tile_overlap,
                          offset_x=rx, offset_y=ry)
    log.info(f"Total tiles: {len(grid)}")

    proc = psutil.Process(os.getpid())
    phase_times = {}
    monitor = RSSMonitor()
    monitor.__enter__()
    t_total0 = time.perf_counter()
    next_label = 1
    n_processed, n_skipped, n_failed = 0, 0, 0
    per_tile_times = []
    tile_peak_rss = 0

    try:
        with monitored_phase("model_load", monitor, phase_times):
            app = get_mesmer_app()
        log.info(f"Mesmer loaded, model_mpp={app.model_mpp}")

        # ---- 推理循环 ----
        for i, item in enumerate(grid):
            x_r, y_r, w_r, h_r, cx0, cy0, cx1, cy1 = item
            t0 = time.perf_counter()
            try:
                with monitored_phase("tile_read", monitor, phase_times):
                    rgb = reader.read_region(x_r, y_r, w_r, h_r)
                if cfg.skip_background and is_background(
                    rgb, cfg.background_mean_threshold, cfg.background_std_threshold
                ):
                    n_skipped += 1
                    continue
                with monitored_phase("hed", monitor, phase_times):
                    two_chan = he_to_two_channels(rgb)
                with monitored_phase("predict", monitor, phase_times):
                    tile_mask = predict_tile(app, two_chan, cfg, image_mpp=cur_mpp)
                with monitored_phase("merge", monitor, phase_times):
                    next_label = merge_tile_into_global(
                        global_mask=global_mask,
                        tile_mask=tile_mask,
                        tile_origin_xy=(x_r - rx, y_r - ry),
                        core_box=(cx0, cy0, cx1, cy1),
                        next_label=next_label,
                    )
                n_processed += 1
                per_tile_times.append(time.perf_counter() - t0)
                monitor.sample()
                tile_peak_rss = max(tile_peak_rss, monitor.peak_rss)
            except Exception as e:
                n_failed += 1
                log.error(f"[{i+1}/{len(grid)}] tile failed: {e}")
            finally:
                for name in ("rgb", "two_chan", "tile_mask"):
                    if name in locals():
                        del locals()[name]
                if (i + 1) % 50 == 0:
                    gc.collect()

            if (i + 1) % 20 == 0 or (i + 1) == len(grid):
                mean_t = np.mean(per_tile_times) if per_tile_times else 0.0
                eta = mean_t * (len(grid) - (i + 1))
                log.info(
                    f"[{i+1}/{len(grid)}] processed={n_processed} "
                    f"skipped={n_skipped} failed={n_failed} cells_so_far={next_label-1} "
                    f"mean_tile={mean_t:.2f}s ETA={eta/60:.1f}min "
                    f"RSS_peak={monitor.peak_rss/1e9:.2f}GB"
                )

        total_time = time.perf_counter() - t_total0
        global_mask.flush()
        n_cells = int(next_label - 1)
        log.info(f"==== Done. cells={n_cells}, time={total_time/60:.2f} min, "
                 f"peak_mem={monitor.peak_rss/1e9:.2f} GB ====")

        monitor.__exit__(None, None, None)

        # ---- 保存输出 ----
        if cfg.save_mask_tiff:
            mask_path = out_dir / "instance_mask.tif"
            log.info(f"Saving mask to {mask_path} ...")
            tifffile.imwrite(
                str(mask_path),
                global_mask,
                bigtiff=True,
                compression="zlib",
                tile=(512, 512),
            )

        if cfg.save_preview_png:
            try:
                log.info("Rendering preview ...")
                save_preview(
                    reader, global_mask,
                    out_dir / "preview.png",
                    roi_xywh=(rx, ry, rw, rh),
                    max_side=cfg.preview_max_side,
                )
            except Exception as e:
                log.warning(f"Preview failed: {e}")

        if cfg.save_geojson:
            try:
                log.info("Exporting geojson (this can be slow for many cells) ...")
                export_geojson(global_mask, out_dir / "cells.geojson",
                               x_offset=rx, y_offset=ry, level_downsample=reader.level_downsample)
            except Exception as e:
                log.warning(f"geojson export failed: {e}")

        # ---- 计算 benchmark 指标 ----
        metrics = compute_metrics(global_mask)
        bench = {
            "wsi_path": cfg.wsi_path,
            "level": cfg.wsi_level,
            "image_mpp": cur_mpp,
            "roi_xywh_level": [rx, ry, rw, rh],
            "tile_size": cfg.tile_size,
            "tile_overlap": cfg.tile_overlap,
            "compartment": cfg.compartment,
            "n_tiles_total": len(grid),
            "n_tiles_processed": n_processed,
            "n_tiles_skipped_bg": n_skipped,
            "n_tiles_failed": n_failed,
            "n_cells": n_cells,
            "total_time_sec": round(total_time, 2),
            "mean_tile_time_sec": float(np.mean(per_tile_times)) if per_tile_times else 0.0,
            "median_tile_time_sec": float(np.median(per_tile_times)) if per_tile_times else 0.0,
            "peak_rss_gb": round(monitor.peak_rss / 1e9, 3),
            "tile_peak_rss_gb": round(tile_peak_rss / 1e9, 3),
            "phase_time_sec": {k: round(v, 3) for k, v in phase_times.items()},
            "mask_memmap": str(mask_tmp),
            "cells_per_minute": round(n_cells / (total_time / 60), 1) if total_time > 0 else 0,
            "metrics": metrics,
        }
        with open(out_dir / "benchmark.json", "w") as f:
            json.dump(bench, f, indent=2)
        log.info(f"Benchmark written to {out_dir/'benchmark.json'}")

        reader.close()
        return bench
    finally:
        if monitor._thread.is_alive():
            monitor.__exit__(None, None, None)
        reader.close()


def compute_metrics(mask: np.ndarray) -> dict:
    """简单的细胞统计 (面积分布等), 用于 benchmark 报告"""
    if mask.max() == 0:
        return {"n_cells": 0}
    props = regionprops(mask)
    areas = np.array([p.area for p in props], dtype=np.float32)
    return {
        "n_cells": int(len(props)),
        "area_mean_px": float(areas.mean()),
        "area_median_px": float(np.median(areas)),
        "area_std_px": float(areas.std()),
        "area_p5_px": float(np.percentile(areas, 5)),
        "area_p95_px": float(np.percentile(areas, 95)),
    }


def export_geojson(mask: np.ndarray, out_path: Path,
                   x_offset: int = 0, y_offset: int = 0,
                   level_downsample: float = 1.0):
    """把 instance mask 转成 QuPath 可读的 geojson (多边形)"""
    from skimage.measure import find_contours
    features = []
    n = int(mask.max())
    # 对每个 label 单独提轮廓
    for lab in range(1, n + 1):
        ys, xs = np.where(mask == lab)
        if len(ys) == 0:
            continue
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        sub = (mask[y0:y1, x0:x1] == lab).astype(np.uint8)
        if sub.sum() < 4:
            continue
        contours = find_contours(sub, 0.5)
        if not contours:
            continue
        # 取最大轮廓
        c = max(contours, key=len)
        # (row, col) -> (x, y)
        pts = [
            [
                float((x0 + p[1] + x_offset) * level_downsample),
                float((y0 + p[0] + y_offset) * level_downsample),
            ]
            for p in c
        ]
        if pts[0] != pts[-1]:
            pts.append(pts[0])
        features.append({
            "type": "Feature",
            "id": int(lab),
            "geometry": {"type": "Polygon", "coordinates": [pts]},
            "properties": {"objectType": "detection", "classification": {"name": "Cell"}},
        })

    gj = {"type": "FeatureCollection", "features": features}
    with open(out_path, "w") as f:
        json.dump(gj, f)



if __name__ == "__main__":
    run(CONFIG)