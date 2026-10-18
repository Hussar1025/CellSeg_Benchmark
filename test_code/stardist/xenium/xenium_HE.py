import os
import math
import time
from pathlib import Path
from zipfile import ZIP_DEFLATED

_PROCESS_STARTED = time.perf_counter()

import sys
sys.path.insert(0, "/data/qiuyijia")
from he_performance import finalize_performance

import numpy as np

from tifffile import imread, imwrite as imsave
from csbdeep.data import Normalizer, normalize_mi_ma
from stardist.models import StarDist2D


INPUT_PATH  = Path(os.environ.get(
    'HE_INPUT_TIF',
    '/data/qiuyijia/dataset/stereo_mouse_embryo/D05103C4D4_HE_regist.tif',
))
OUTPUT_DIR  = Path(os.environ.get(
    'HE_STARDIST_OUTPUT_DIR',
    '/data/qiuyijia/dataset/stereo_mouse_embryo/segmentation_outputs/stardist',
))
LABELS_TIF  = OUTPUT_DIR / 'labels.tif'
NUCLEI_TIF  = OUTPUT_DIR / 'nuclei_labels.tif'

# 是否对超大图像使用 Zarr
USE_ZARR    = False
ZARR_IMAGE  = OUTPUT_DIR / 'image.zarr'
ZARR_LABELS = OUTPUT_DIR / 'labels.zarr'

# 分块预测参数
BLOCK_SIZE   = 4096    # 每个 block 的边长
MIN_OVERLAP  = 128     # block 之间的最小重叠，避免边界处细胞被切断
CONTEXT      = 128     # block 边缘提供给 CNN 的额外上下文
N_TILES      = (4, 4, 1)  # 单个 block 内的进一步切片数 (Y, X, C)

# StarDist H&E 模型输出细胞核；与其他数据集保持一致，向外扩 15 px
# 近似整细胞。expand_labels 在相邻实例相遇时停止，不会让标签互相覆盖。
EXPAND_PX    = int(os.environ.get('HE_STARDIST_EXPAND_PX', '15'))

# 归一化范围：uint8 H&E 图像用 0~255 即可；如需百分位归一化见下方注释
MI, MA = 0, 255


# 工具：计时上下文管理器
class Timer:
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        print(f'[{self.name}] 开始 ...')
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        dt = time.perf_counter() - self.t0
        h, rem = divmod(dt, 3600)
        m, s = divmod(rem, 60)
        print(f'[{self.name}] 用时 {int(h):02d}:{int(m):02d}:{s:05.2f}  ({dt:.2f} s)\n')


# 自定义 Normalizer（按需归一化，不预先处理整张大图）
class MyNormalizer(Normalizer):
    def __init__(self, mi, ma):
        self.mi, self.ma = mi, ma
    def before(self, x, axes):
        return normalize_mi_ma(x, self.mi, self.ma, dtype=np.float32)
    def after(*args, **kwargs):
        assert False
    @property
    def do_after(self):
        return False


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_t0 = time.perf_counter()

    # 1. 读取图像
    with Timer('读取图像'):
        if USE_ZARR:
            import zarr
            # 若 zarr 副本不存在则先创建一份（实际使用中可直接从 zarr 读取，跳过此步）
            if not ZARR_IMAGE.exists():
                print('  -> 首次运行：将 TIF 转为 Zarr（仅一次）')
                img_np = imread(str(INPUT_PATH))
                zarr.save_array(str(ZARR_IMAGE), img_np)
                del img_np
            img = zarr.open(str(ZARR_IMAGE), mode='r')
        else:
            img = imread(str(INPUT_PATH))
        print(f'  图像形状: {img.shape}, dtype: {img.dtype}')

    # 2. 加载预训练模型
    with Timer('加载模型'):
        model = StarDist2D.from_pretrained('2D_versatile_he')

    # 3. 准备 normalizer
    normalizer = MyNormalizer(MI, MA)

    # 4. 预测。小于等于 BLOCK_SIZE 的 ROI 直接预测，避免 StarDist 把
    # 3000 向上对齐到网络网格后的 3008，继而违反 block_size <= image size。
    small_image = int(img.shape[0]) <= BLOCK_SIZE and int(img.shape[1]) <= BLOCK_SIZE
    prediction_name = ('ROI 预测 (predict_instances)'
                       if small_image else '分块预测 (predict_instances_big)')
    with Timer(prediction_name):
        if small_image:
            labels, polys = model.predict_instances(
                img,
                axes='YXC',
                normalizer=normalizer,
                n_tiles=N_TILES,
            )
        elif USE_ZARR:
            import zarr
            spatial_grid = model._axes_div_by('YXC')[:2]
            grid_unit = math.lcm(*(int(v) for v in spatial_grid))
            limit = min(BLOCK_SIZE, int(img.shape[0]), int(img.shape[1]))
            effective_block_size = (limit // grid_unit) * grid_unit
            print(f'  有效 block_size: {effective_block_size}')
            labels_zarr = zarr.open(
                str(ZARR_LABELS), mode='w',
                shape=img.shape[:2],
                chunks=img.chunks[:2],
                dtype=np.int32,
            )
            labels, polys = model.predict_instances_big(
                img, axes='YXC',
                block_size=effective_block_size,
                min_overlap=MIN_OVERLAP,
                context=CONTEXT,
                normalizer=normalizer,
                n_tiles=N_TILES,
                labels_out=labels_zarr,    # 直接写入 zarr，避免在内存里组装完整标签图
            )
        else:
            spatial_grid = model._axes_div_by('YXC')[:2]
            grid_unit = math.lcm(*(int(v) for v in spatial_grid))
            limit = min(BLOCK_SIZE, int(img.shape[0]), int(img.shape[1]))
            effective_block_size = (limit // grid_unit) * grid_unit
            print(f'  有效 block_size: {effective_block_size}')
            labels, polys = model.predict_instances_big(
                img, axes='YXC',
                block_size=effective_block_size,
                min_overlap=MIN_OVERLAP,
                context=CONTEXT,
                normalizer=normalizer,
                n_tiles=N_TILES,
            )
        n_cells = len(polys['coord']) if polys and 'coord' in polys else 0
        print(f'  检测到细胞核数: {n_cells}')

    # 5. 核标签向外扩 15 px，作为整细胞近似结果。
    with Timer(f'细胞核向外扩张 {EXPAND_PX}px'):
        if USE_ZARR:
            nuclei_np = np.array(labels, dtype=np.int32)
        else:
            nuclei_np = np.asarray(labels).astype(np.int32, copy=False)
        if EXPAND_PX > 0:
            from skimage.segmentation import expand_labels
            labels_np = expand_labels(nuclei_np, distance=EXPAND_PX).astype(
                np.int32, copy=False
            )
        else:
            labels_np = nuclei_np

    # 6. 同时保存原始核标签与扩张后的细胞标签。
    with Timer('保存 TIF'):
        imsave(
            str(NUCLEI_TIF),
            nuclei_np,
            compression=ZIP_DEFLATED,
            bigtiff=True,
        )

        imsave(
            str(LABELS_TIF),
            labels_np,
            compression=ZIP_DEFLATED,
            bigtiff=True,   # 标签图很大时建议开启
        )
        print(f'  原始细胞核标签: {NUCLEI_TIF}')
        print(f'  扩张后细胞标签: {LABELS_TIF}')

    # 总耗时
    total_dt = time.perf_counter() - total_t0
    h, rem = divmod(total_dt, 3600)
    m, s = divmod(rem, 60)
    print(f'==== 总耗时 {int(h):02d}:{int(m):02d}:{s:05.2f}  ({total_dt:.2f} s) ====')


if __name__ == '__main__':
    _status, _error = 'success', None
    try:
        main()
    except BaseException as exc:
        _status, _error = 'failed', repr(exc)
        raise
    finally:
        finalize_performance(
            _PROCESS_STARTED, OUTPUT_DIR, 'StarDist',
            status=_status, error=_error,
            extra={'nucleus_expansion_px': EXPAND_PX},
        )
