import gc
import time
from pathlib import Path

import numpy as np
import tifffile as tiff
from tqdm import tqdm
from cellpose import models
import fastremap


# ======================
# 直接在这里修改参数
# ======================
INPUT_TIF = r"/data/qiuyijia/dataset/Xenium_V1_hPancreas_Cancer_Add_on_FFPE_he_image.ome.tif"
OUTPUT_TIF = r"/data/qiuyijia/cellpose/Pancreas_output/output_cpsam_masks.ome.tif"

USE_GPU = True

TILE_SIZE = 2048
OVERLAP = 192

DIAMETER = 30

# RGB 图像使用模式：
# "rgb"  : 直接把 RGB 三通道送入 cellpose
# "gray" : 先转灰度再分割，通常更省显存
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

            # 有些 tifffile.aszarr() 返回 array
            # 有些返回 group，真正图像在 "0"
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
                    raise ValueError("zarr group 中没有可用 array")

            self.shape = self.arr.shape
            self.dtype = self.arr.dtype
            self.use_zarr = True

            print("读取方式: zarr lazy tiled reader")
            print("zarr array shape:", self.shape)
            print("zarr array dtype :", self.dtype)

        except Exception as e:
            self.arr = None
            self.use_zarr = False
            print("读取方式: tifffile fallback reader")
            print("zarr lazy reader 不可用，原因:", repr(e))

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

    print(f"Input : {input_path}")
    print(f"Output: {output_path}")

    print("\n[1/4] 读取 TIFF 元信息...")
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
                raise ValueError("RGB_MODE 只能是 'gray' 或 'rgb'")

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
                raise ValueError("RGB_MODE 只能是 'gray' 或 'rgb'")

    else:
        raise ValueError(f"暂不支持该图像维度: shape={img_shape}")

    print(f"image mode: {image_mode}")
    print(f"processed size: H={H}, W={W}")
    print(f"RGB_MODE: {RGB_MODE}")

    print("\n[2/4] 加载 Cellpose-SAM 模型...")
    model_start = time.perf_counter()

    model = models.CellposeModel(
        gpu=USE_GPU,
        pretrained_model="cpsam"
    )

    model_time = time.perf_counter() - model_start
    print(f"模型加载耗时: {format_time(model_time)}")

    tiles = list(iter_tiles(H, W, TILE_SIZE, OVERLAP))
    total_tiles = len(tiles)

    print("\n[3/4] 开始分块分割...")
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
            core[positive] = core[positive] + np.uint32(next_label)

            next_label = int(core.max()) + 1

        mask_full[cy0:cy1, cx0:cx1] = core

        tile_elapsed = time.perf_counter() - tile_start
        tile_times.append(tile_elapsed)

        if idx % 10 == 0 or idx == total_tiles:
            avg_time = sum(tile_times) / len(tile_times)
            remaining = avg_time * (total_tiles - idx)
            print(
                f"已完成 {idx}/{total_tiles} | "
                f"平均每块 {avg_time:.2f}s | "
                f"预计剩余 {format_time(remaining)}"
            )

        del tile_img, masks, flows, styles, core
        gc.collect()

    print("\n[4/4] 写出 mask 文件...")
    write_start = time.perf_counter()

    with tiff.TiffWriter(output_path, bigtiff=True) as writer:
        writer.write(
            mask_full,
            dtype=np.uint32,
            compression="zlib",
            tile=(512, 512),
            metadata={"axes": "YX"},
        )

    write_time = time.perf_counter() - write_start
    total_time = time.perf_counter() - total_start

    reader.close()

    print("\n========== 完成 ==========")
    print(f"输出文件: {output_path}")
    print(f"最大 label id: {next_label - 1}")
    print(f"模型加载耗时: {format_time(model_time)}")
    print(f"写出文件耗时: {format_time(write_time)}")
    print(f"总耗时      : {format_time(total_time)}")

    if tile_times:
        print(f"平均每 tile 耗时: {sum(tile_times) / len(tile_times):.2f}s")
        print(f"最快 tile 耗时  : {min(tile_times):.2f}s")
        print(f"最慢 tile 耗时  : {max(tile_times):.2f}s")


if __name__ == "__main__":
    main()