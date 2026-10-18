import time
import numpy as np
from pathlib import Path
from zipfile import ZIP_DEFLATED

from tifffile import imread, imwrite as imsave
from csbdeep.data import Normalizer, normalize_mi_ma
from stardist.models import StarDist2D


INPUT_PATH  = Path('/data/qiuyijia/dataset/Xenium_V1_hPancreas_Cancer_Add_on_FFPE_he_image.ome.tif')
OUTPUT_DIR  = Path('/data/qiuyijia/stardist/Pancreas_cancer_output')
LABELS_TIF  = OUTPUT_DIR / 'labels.tif'

# 是否对超大图像使用 Zarr
USE_ZARR    = False
ZARR_IMAGE  = OUTPUT_DIR / 'image.zarr'
ZARR_LABELS = OUTPUT_DIR / 'labels.zarr'

# 分块预测参数
BLOCK_SIZE   = 4096    # 每个 block 的边长
MIN_OVERLAP  = 128     # block 之间的最小重叠，避免边界处细胞被切断
CONTEXT      = 128     # block 边缘提供给 CNN 的额外上下文
N_TILES      = (4, 4, 1)  # 单个 block 内的进一步切片数 (Y, X, C)

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

    # 4. 分块预测
    with Timer('分块预测 (predict_instances_big)'):
        if USE_ZARR:
            import zarr
            labels_zarr = zarr.open(
                str(ZARR_LABELS), mode='w',
                shape=img.shape[:2],
                chunks=img.chunks[:2],
                dtype=np.int32,
            )
            labels, polys = model.predict_instances_big(
                img, axes='YXC',
                block_size=BLOCK_SIZE,
                min_overlap=MIN_OVERLAP,
                context=CONTEXT,
                normalizer=normalizer,
                n_tiles=N_TILES,
                labels_out=labels_zarr,    # 直接写入 zarr，避免在内存里组装完整标签图
            )
        else:
            labels, polys = model.predict_instances_big(
                img, axes='YXC',
                block_size=BLOCK_SIZE,
                min_overlap=MIN_OVERLAP,
                context=CONTEXT,
                normalizer=normalizer,
                n_tiles=N_TILES,
            )
        n_cells = len(polys['coord']) if polys and 'coord' in polys else 0
        print(f'  检测到细胞数: {n_cells}')

    # 5. 保存标签图为压缩 TIF
    with Timer('保存 TIF'):
        if USE_ZARR:
            labels_np = np.array(labels)
        else:
            labels_np = labels

        imsave(
            str(LABELS_TIF),
            labels_np,
            compression=ZIP_DEFLATED,
            bigtiff=True,   # 标签图很大时建议开启
        )
        print(f'  已保存: {LABELS_TIF}')

    # 总耗时
    total_dt = time.perf_counter() - total_t0
    h, rem = divmod(total_dt, 3600)
    m, s = divmod(rem, 60)
    print(f'==== 总耗时 {int(h):02d}:{int(m):02d}:{s:05.2f}  ({total_dt:.2f} s) ====')


if __name__ == '__main__':
    main()