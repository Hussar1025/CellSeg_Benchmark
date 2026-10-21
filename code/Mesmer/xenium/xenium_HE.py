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
    wsi_path: str = "yourpath/dataset/stereo_mouse_embryo/D05103C4D4_HE_regist.tif"
    output_dir: str = "yourpath/Mesmer/stereo_mouse_embryo_output"
    run_name: str = "mesmer_he"

    wsi_level: int = 0

    # Mesmer's training resolution is 0.5 microns/pixel.
    # If the WSI mpp at level=0 is not 0.5, let Mesmer rescale internally via the image_mpp argument.
    # The user must provide the true H&E um/px; it cannot be guessed from this TIFF's invalid resolution.
    image_mpp: Optional[float] = None

    # ---------- Tiling ----------
    tile_size: int = 2048                              # tile side length (pixel, at wsi_level)
    tile_overlap: int = 256                            # overlap between adjacent tiles (pixel); >= 2x mean cell diameter recommended

    # ---------- Background filtering ----------
    skip_background: bool = True
    background_mean_threshold: int = 220               # skip when RGB mean > threshold (near white)
    background_std_threshold: int = 8                  # skip when tile std < threshold (no content)

    # ---------- Mesmer inference ----------
    compartment: str = "whole-cell"                    # "whole-cell" or "nuclear"
    batch_size: int = 4                                # Mesmer internal batch
    interior_threshold: float = 0.3                    # lower -> larger cells; default 0.3
    maxima_threshold: float = 0.075                    # lower -> more fragmented; default 0.075
    pixel_expansion: Optional[int] = None              # mostly for compartment="nuclear"; pixels to expand

    # ---------- ROI (optional: process only a rectangular sub-region of the WSI) ----------
    # None = process the whole image. Coordinates use the wsi_level=0 system.
    roi_x: Optional[int] = None
    roi_y: Optional[int] = None
    roi_w: Optional[int] = None
    roi_h: Optional[int] = None

    # ---------- Output options ----------
    save_mask_tiff: bool = True                        # save the full-image instance mask (uint32, BigTIFF)
    mask_memmap_path: Optional[str] = None             # defaults to the output run directory
    save_geojson: bool = False                         # save a QuPath-readable polygon geojson (slow with many cells)
    save_preview_png: bool = True                      # save a downscaled overlay preview
    preview_max_side: int = 4096                       # longest preview side in pixels

    # ---------- GPU ----------
    gpu_id: str = "0"                                  # set to "-1" to force CPU


CONFIG = Config()


# GPU selection must happen before importing tensorflow
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
    """Low-overhead background RSS sampler; values kept in bytes so they can go into the benchmark."""

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
    """Plain single-page TIFF reader; windowed reads, no WSI backend dependency."""

    backend = "tifffile"
    level_downsample = 1.0

    def __init__(self, path: str, level: int = 0):
        if level != 0:
            raise ValueError("plain TIFF reader only supports wsi_level=0")
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
        # resolution unit is NONE in this dataset, so it is not usable as um/px.
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
    """Thin wrapper over openslide / tiffslide; openslide is preferred."""

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
        """Read level=0 microns per pixel; return None if unavailable.

        Note: some backends (e.g. tiffslide) write the key into properties with a None
        or empty value, so `k in props` is not enough -- validate the value itself,
        and catch both ValueError and TypeError (float(None) raises TypeError).
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
            if mpp > 0:                 # reject meaningless placeholders such as 0
                return mpp
        return None

    def read_region(self, x_l: int, y_l: int, w: int, h: int) -> np.ndarray:
        """
        Read the [x_l, y_l, w, h] region at the current level (level coordinates).
        openslide.read_region takes a level=0 location and a size in current-level pixels.
        Returns an RGB ndarray, dtype=uint8, shape=(h, w, 3)
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


# ---------- H&E -> (nuclear, membrane) two channels ----------

def he_to_two_channels(rgb_tile: np.ndarray) -> np.ndarray:
    """
    Color-deconvolve H&E RGB into Hematoxylin (nucleus) and Eosin (membrane/cytoplasm).
    Output shape=(H, W, 2), float32, min-max normalized to [0,1],
    then multiplied by 255 to approach the Mesmer training-data scale.
    Mesmer expects channel order [nuclear, membrane].
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
    """Simple background test: near-white or almost no texture."""
    mean_v = rgb_tile.mean()
    std_v = rgb_tile.std()
    return (mean_v > mean_thr) or (std_v < std_thr)


# ---------- Mesmer wrapper ----------

def get_mesmer_app():
    global _DEEPCELL_APP
    if _DEEPCELL_APP is None:
        from deepcell.applications import Mesmer
        _DEEPCELL_APP = Mesmer()
    return _DEEPCELL_APP


def predict_tile(app, two_chan: np.ndarray, cfg: Config, image_mpp: float) -> np.ndarray:
    """
    Input (H, W, 2) float32; return an (H, W) int32 instance mask (background 0).
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


# ---------- Tile grid generation ----------

def make_tile_grid(
    width: int, height: int, tile: int, overlap: int,
    offset_x: int = 0, offset_y: int = 0,
) -> List[Tuple[int, int, int, int, int, int, int, int]]:
    """
    Return the tile list, each item:
      (x_read, y_read, w_read, h_read,    # read region (includes overlap)
       core_x0, core_y0, core_x1, core_y1) # core area (tile-local coordinates);
                                           # only cells in the core area reach the final result
    The core area of border tiles extends to the image edge.
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


# ---------- Merge: write tile masks into the global mask with globally unique IDs ----------

def merge_tile_into_global(
    global_mask: np.ndarray,
    tile_mask: np.ndarray,
    tile_origin_xy: Tuple[int, int],
    core_box: Tuple[int, int, int, int],
    next_label: int,
) -> int:
    """
    Write the instances of tile_mask whose pixels lie in the core area (core_box) to the
    matching position in global_mask. Returns the updated next_label.
    core_box = (x0, y0, x1, y1) in tile-local coordinates
    """
    x0, y0, x1, y1 = core_box
    core = tile_mask[y0:y1, x0:x1]
    if core.size == 0:
        return next_label

    # instance ids present in the core area
    present_ids = np.unique(core)
    present_ids = present_ids[present_ids != 0]
    if present_ids.size == 0:
        return next_label

    # assign contiguous global ids to these
    remap = np.zeros(int(present_ids.max()) + 1, dtype=np.int64)
    for old_id in present_ids:
        remap[old_id] = next_label
        next_label += 1

    # Map over the whole tile (a cell may cross the core boundary, so write all pixels of these
    # ids inside this tile), but keep an id only if it has pixels in the core area; this avoids
    # overwriting a cell already recorded by another tile.
    full_remap = np.zeros(int(tile_mask.max()) + 1, dtype=np.int64)
    full_remap[present_ids] = remap[present_ids]
    new_tile = full_remap[tile_mask]

    # write into the global mask only where it is still background (never overwrite neighbor tiles' cells)
    tx, ty = tile_origin_xy
    th, tw = tile_mask.shape
    region = global_mask[ty:ty + th, tx:tx + tw]
    write_mask = (region == 0) & (new_tile > 0)
    region[write_mask] = new_tile[write_mask].astype(global_mask.dtype)
    global_mask[ty:ty + th, tx:tx + tw] = region

    return next_label


# ---------- Preview ----------

def save_preview(
    reader: WSIReader,
    global_mask: np.ndarray,
    out_path: Path,
    roi_xywh: Tuple[int, int, int, int],
    max_side: int,
):
    """Save a downscaled overlay; read by output block so the whole ROI is never loaded at once."""
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


# ---------- Main flow ----------

def run(cfg: Config):
    out_dir = Path(cfg.output_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger(out_dir)

    log.info("==== Mesmer H&E WSI Benchmark ====")
    log.info(f"WSI: {cfg.wsi_path}")
    log.info(f"Output: {out_dir}")

    # save the config
    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    if not Path(cfg.wsi_path).exists():
        raise FileNotFoundError(cfg.wsi_path)

    # open the WSI
    reader = WSIReader(cfg.wsi_path, level=cfg.wsi_level)
    W_l, H_l = reader.level_dims
    log.info(f"Backend={reader.backend}, level={cfg.wsi_level}, "
             f"dims(W,H)=({W_l},{H_l}), downsample={reader.level_downsample}")

    # mpp must come from user-confirmed scan metadata; this TIFF has resolution unit=NONE.
    if cfg.image_mpp is None or not np.isfinite(cfg.image_mpp) or cfg.image_mpp <= 0:
        raise ValueError(
            "Config.image_mpp must be a confirmed positive H&E microns/pixel value; "
            "the input TIFF has no usable mpp metadata."
        )
    cur_mpp = float(cfg.image_mpp)
    log.info(f"Using user-specified image_mpp={cur_mpp} um/px")

    # ROI
    if cfg.roi_x is not None:
        # ROI coordinates are level=0; convert to the current level
        ds = reader.level_downsample
        rx = int(cfg.roi_x / ds)
        ry = int(cfg.roi_y / ds)
        rw = int(cfg.roi_w / ds)
        rh = int(cfg.roi_h / ds)
    else:
        rx, ry, rw, rh = 0, 0, W_l, H_l
    log.info(f"ROI at level {cfg.wsi_level}: x={rx}, y={ry}, w={rw}, h={rh}")

    # use an on-disk mask to avoid a full-image ndarray (~3.9 GiB RAM)
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

    # build the tile grid (in ROI-local coordinates)
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

        # ---- inference loop ----
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

        # ---- save outputs ----
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

        # ---- compute benchmark metrics ----
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
    """Basic cell statistics (area distribution etc.) for the benchmark report"""
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
    """Convert the instance mask into a QuPath-readable geojson (polygons)"""
    from skimage.measure import find_contours
    features = []
    n = int(mask.max())
    # extract contours for each label separately
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
        # take the longest contour
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
