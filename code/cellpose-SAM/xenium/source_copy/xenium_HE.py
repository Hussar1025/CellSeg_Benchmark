import gc
import os
import time
from pathlib import Path

_PROCESS_STARTED = time.perf_counter()

import sys
sys.path.insert(0, "yourpath")
from he_performance import finalize_performance

import numpy as np
import tifffile as tiff
from tqdm import tqdm
from cellpose import models
import fastremap
from skimage.segmentation import expand_labels


# ======================
# edit the parameters directly here
# ======================
INPUT_TIF = os.environ.get(
    "HE_INPUT_TIF",
    r"yourpath/dataset/stereo_mouse_embryo/D05103C4D4_HE_regist.tif",
)
OUTPUT_TIF = os.environ.get(
    "HE_CELLPOSE_OUTPUT_TIF",
    r"yourpath/dataset/stereo_mouse_embryo/segmentation_outputs/cellpose/labels.tif",
)

USE_GPU = True

TILE_SIZE = 2048
OVERLAP = 192

# The adjusted CellBin shipped with stereo averages ~313 px^2, an equivalent diameter of
# ~20 px; other datasets can override this from the launch script via an environment variable.
DIAMETER = int(os.environ.get("HE_CELLPOSE_DIAMETER", "20"))

# Expand the nucleus-like instances detected by Cellpose-SAM outwards to approximate whole cells.
# expand_labels stops where neighbouring instances meet, so labels never overwrite each other.
EXPAND_PX = int(os.environ.get("HE_CELLPOSE_EXPAND_PX", "15"))

# RGB image mode:
# "rgb"  : feed the three RGB channels to cellpose directly
# "gray" : convert to grayscale first, usually saves GPU memory
RGB_MODE = "gray"

BATCH_SIZE = 4
FLOW_THRESHOLD = 0.4
CELLPROB_THRESHOLD = 0.0


def format_time(seconds):
    seconds = int(seconds)
    h = seconds // 3600
    m = seconds % 3600 // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


class LargeTiffReader:
    def __init__(self, path):
        self.tif = tiff.TiffFile(path)
        self.series = self.tif.series[0]
        self.shape = self.series.shape
        self.dtype = self.series.dtype

        try:
            import zarr

            self.store = self.series.aszarr()
            z = zarr.open(self.store, mode="r")

            # sometimes tifffile.aszarr() returns an array,
            # sometimes a group with the real image under "0"
            if hasattr(z, "shape"):
                self.arr = z
            else:
                keys = list(z.array_keys())
                print("zarr group arrays:", keys)

                if "0" in keys:
                    self.arr = z["0"]
                elif len(keys) > 0:
                    self.arr = z[keys[0]]
                else:
                    raise ValueError("no usable array in the zarr group")

            self.shape = self.arr.shape
            self.dtype = self.arr.dtype
            self.use_zarr = True

            print("read mode: zarr lazy tiled reader")
            print("zarr array shape:", self.shape)
            print("zarr array dtype :", self.dtype)

        except Exception as e:
            self.arr = None
            self.use_zarr = False
            print("read mode: tifffile fallback reader")
            print("zarr lazy reader unavailable, reason:", repr(e))

    def read(self, key):
        if self.use_zarr:
            return np.asarray(self.arr[key])
        else:
            return np.asarray(self.series.asarray()[key])

    def close(self):
        self.tif.close()


def iter_tiles(h, w, tile=2048, overlap=128):
    step = tile - overlap
    for y0 in range(0, h, step):
        for x0 in range(0, w, step):
            y1 = min(y0 + tile, h)
            x1 = min(x0 + tile, w)

            cy0 = y0 if y0 == 0 else y0 + overlap // 2
            cx0 = x0 if x0 == 0 else x0 + overlap // 2
            cy1 = y1 if y1 == h else y1 - overlap // 2
            cx1 = x1 if x1 == w else x1 - overlap // 2

            yield y0, y1, x0, x1, cy0, cy1, cx0, cx1


def rgb_to_gray_uint8(rgb):
    rgb = rgb.astype(np.float32)
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return np.clip(gray, 0, 255).astype(np.uint8)


def normalize_image(img):
    img = np.asarray(img)

    if img.dtype == np.uint8:
        return img

    if img.dtype == np.uint16:
        return img

    lo, hi = np.percentile(img, (1, 99.8))
    img = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
    return (img * 255).astype(np.uint8)


def main():
    total_start = time.perf_counter()

    input_path = Path(INPUT_TIF)
    output_path = Path(OUTPUT_TIF)

    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Input : {input_path}")
    print(f"Output: {output_path}")

    print("\n[1/5] reading TIFF metadata...")
    reader = LargeTiffReader(input_path)
    img_shape = reader.shape
    img_dtype = reader.dtype

    print("image shape:", img_shape)
    print("image dtype :", img_dtype)

    ndim = len(img_shape)

    if ndim == 2:
        H, W = img_shape
        image_mode = "gray_2d"

        def get_tile(y0, y1, x0, x1):
            return reader.read((slice(y0, y1), slice(x0, x1)))

    elif ndim == 3 and img_shape[-1] in (3, 4):
        H, W = img_shape[0], img_shape[1]
        image_mode = "rgb_yxc"

        def get_tile(y0, y1, x0, x1):
            rgb = reader.read((slice(y0, y1), slice(x0, x1), slice(0, 3)))
            if RGB_MODE == "gray":
                return rgb_to_gray_uint8(rgb)
            elif RGB_MODE == "rgb":
                return rgb
            else:
                raise ValueError("RGB_MODE must be 'gray' or 'rgb'")

    elif ndim == 3 and img_shape[0] in (3, 4):
        H, W = img_shape[1], img_shape[2]
        image_mode = "rgb_cyx"

        def get_tile(y0, y1, x0, x1):
            rgb = reader.read((slice(0, 3), slice(y0, y1), slice(x0, x1)))
            rgb = np.moveaxis(rgb, 0, -1)
            if RGB_MODE == "gray":
                return rgb_to_gray_uint8(rgb)
            elif RGB_MODE == "rgb":
                return rgb
            else:
                raise ValueError("RGB_MODE must be 'gray' or 'rgb'")

    else:
        raise ValueError(f"unsupported image dimensionality: shape={img_shape}")

    print(f"image mode: {image_mode}")
    print(f"processed size: H={H}, W={W}")
    print(f"RGB_MODE: {RGB_MODE}")

    print("\n[2/5] loading the Cellpose-SAM model...")
    model_start = time.perf_counter()

    model = models.CellposeModel(
        gpu=USE_GPU,
        pretrained_model="cpsam"
    )

    model_time = time.perf_counter() - model_start
    print(f"model load time: {format_time(model_time)}")

    tiles = list(iter_tiles(H, W, TILE_SIZE, OVERLAP))
    total_tiles = len(tiles)

    print("\n[3/5] starting tiled segmentation...")
    print(f"tile size: {TILE_SIZE}")
    print(f"overlap  : {OVERLAP}")
    print(f"tiles    : {total_tiles}")

    mask_full = np.zeros((H, W), dtype=np.uint32)
    next_label = 1
    tile_times = []

    for idx, tile_info in enumerate(tqdm(tiles, desc="Segmenting tiles"), start=1):
        tile_start = time.perf_counter()

        y0, y1, x0, x1, cy0, cy1, cx0, cx1 = tile_info

        tile_img = normalize_image(
            get_tile(y0, y1, x0, x1)
        )

        if RGB_MODE == "rgb" and tile_img.ndim == 3:
            channels = [1, 2]
        else:
            channels = [0, 0]

        masks, flows, styles = model.eval(
            tile_img,
            channels=channels,
            diameter=DIAMETER,
            batch_size=BATCH_SIZE,
            flow_threshold=FLOW_THRESHOLD,
            cellprob_threshold=CELLPROB_THRESHOLD,
            normalize=True,
            bsize=256,
        )

        core = masks[
            cy0 - y0: cy1 - y0,
            cx0 - x0: cx1 - x0
        ] 

        core = core.astype(np.uint32, copy=False)

        if core.max() > 0:
            core = fastremap.renumber(core, in_place=False)[0].astype(np.uint32, copy=False)

            positive = core > 0
            # after renumber the current tile's labels start at 1; shifting by next_label - 1
            # keeps global labels contiguous and avoids wasting one ID per tile.
            core[positive] = core[positive] + np.uint32(next_label - 1)

            next_label = int(core.max()) + 1

        mask_full[cy0:cy1, cx0:cx1] = core

        tile_elapsed = time.perf_counter() - tile_start
        tile_times.append(tile_elapsed)

        if idx % 10 == 0 or idx == total_tiles:
            avg_time = sum(tile_times) / len(tile_times)
            remaining = avg_time * (total_tiles - idx)
            print(
                f"done {idx}/{total_tiles} | "
                f"mean per tile {avg_time:.2f}s | "
                f"estimated remaining {format_time(remaining)}"
            )

        del tile_img, masks, flows, styles, core
        gc.collect()

    print(f"\n[4/5] expanding nucleus labels outwards by {EXPAND_PX}px ...")
    expand_start = time.perf_counter()
    if EXPAND_PX > 0:
        mask_full = expand_labels(mask_full, distance=EXPAND_PX).astype(
            np.uint32, copy=False,
        )
    expand_time = time.perf_counter() - expand_start

    print("\n[5/5] writing the mask file...")
    write_start = time.perf_counter()

    with tiff.TiffWriter(output_path, bigtiff=True) as writer:
        writer.write(
            mask_full,
            dtype=np.uint32,
            compression="zlib",
            tile=(512, 512),
            metadata={
                "axes": "YX",
                "description": f"Cellpose-SAM labels expanded {EXPAND_PX}px",
            },
        )

    write_time = time.perf_counter() - write_start
    total_time = time.perf_counter() - total_start

    reader.close()

    print("\n========== Done ==========")
    print(f"output file: {output_path}")
    print(f"max label id: {next_label - 1}")
    print(f"label expansion distance: {EXPAND_PX}px")
    print(f"label expansion time: {format_time(expand_time)}")
    print(f"model load time: {format_time(model_time)}")
    print(f"write time: {format_time(write_time)}")
    print(f"total time      : {format_time(total_time)}")

    if tile_times:
        print(f"mean per-tile time: {sum(tile_times) / len(tile_times):.2f}s")
        print(f"fastest tile time  : {min(tile_times):.2f}s")
        print(f"slowest tile time  : {max(tile_times):.2f}s")


if __name__ == "__main__":
    _status, _error = "success", None
    try:
        main()
    except BaseException as exc:
        _status, _error = "failed", repr(exc)
        raise
    finally:
        finalize_performance(
            _PROCESS_STARTED, Path(OUTPUT_TIF).parent, "Cellpose-SAM",
            status=_status, error=_error,
        )
