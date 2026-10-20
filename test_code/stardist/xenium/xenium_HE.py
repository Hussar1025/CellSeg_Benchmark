import os
import math
import time
from pathlib import Path
from zipfile import ZIP_DEFLATED

_PROCESS_STARTED = time.perf_counter()

import sys
sys.path.insert(0, "yourpath")
from he_performance import finalize_performance

import numpy as np

from tifffile import imread, imwrite as imsave
from csbdeep.data import Normalizer, normalize_mi_ma
from stardist.models import StarDist2D


INPUT_PATH  = Path(os.environ.get(
    'HE_INPUT_TIF',
    'yourpath/dataset/stereo_mouse_embryo/D05103C4D4_HE_regist.tif',
))
OUTPUT_DIR  = Path(os.environ.get(
    'HE_STARDIST_OUTPUT_DIR',
    'yourpath/dataset/stereo_mouse_embryo/segmentation_outputs/stardist',
))
LABELS_TIF  = OUTPUT_DIR / 'labels.tif'
NUCLEI_TIF  = OUTPUT_DIR / 'nuclei_labels.tif'

# whether to use Zarr for very large images
USE_ZARR    = False
ZARR_IMAGE  = OUTPUT_DIR / 'image.zarr'
ZARR_LABELS = OUTPUT_DIR / 'labels.zarr'

# tiled prediction parameters
BLOCK_SIZE   = 4096    # side length of each block
MIN_OVERLAP  = 128     # minimum overlap between blocks, to avoid cutting cells at boundaries
CONTEXT      = 128     # extra context at block edges for the CNN
N_TILES      = (4, 4, 1)  # further tiling within a single block (Y, X, C)

# The StarDist H&E model outputs nuclei; to stay consistent with other datasets,
# dilate outward by 15 px to approximate whole cells. expand_labels stops when
# neighboring instances meet, so labels never overlap.
EXPAND_PX    = int(os.environ.get('HE_STARDIST_EXPAND_PX', '15'))

# normalization range: 0~255 for uint8 H&E; see comment below for percentile normalization
MI, MA = 0, 255


# utility: timing context manager
class Timer:
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        print(f'[{self.name}] start ...')
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        dt = time.perf_counter() - self.t0
        h, rem = divmod(dt, 3600)
        m, s = divmod(rem, 60)
        print(f'[{self.name}] elapsed {int(h):02d}:{int(m):02d}:{s:05.2f}  ({dt:.2f} s)\n')


# custom Normalizer (normalizes on demand, does not preprocess the whole large image)
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

    # 1. read image
    with Timer('read image'):
        if USE_ZARR:
            import zarr
            # if the zarr copy does not exist, create it first (in practice you can read from zarr directly and skip this step)
            if not ZARR_IMAGE.exists():
                print('  -> first run: convert TIF to Zarr (once only)')
                img_np = imread(str(INPUT_PATH))
                zarr.save_array(str(ZARR_IMAGE), img_np)
                del img_np
            img = zarr.open(str(ZARR_IMAGE), mode='r')
        else:
            img = imread(str(INPUT_PATH))
        print(f'  image shape: {img.shape}, dtype: {img.dtype}')

    # 2. load pretrained model
    with Timer('load model'):
        model = StarDist2D.from_pretrained('2D_versatile_he')

    # 3. prepare normalizer
    normalizer = MyNormalizer(MI, MA)

    # 4. prediction. For ROIs <= BLOCK_SIZE, predict directly, avoiding StarDist
    # rounding 3000 up to 3008 after aligning to the network grid, which would then
    # violate block_size <= image size.
    small_image = int(img.shape[0]) <= BLOCK_SIZE and int(img.shape[1]) <= BLOCK_SIZE
    prediction_name = ('ROI prediction (predict_instances)'
                       if small_image else 'tiled prediction (predict_instances_big)')
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
            print(f'  effective block_size: {effective_block_size}')
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
                labels_out=labels_zarr,    # write directly to zarr, avoid assembling the full label map in memory
            )
        else:
            spatial_grid = model._axes_div_by('YXC')[:2]
            grid_unit = math.lcm(*(int(v) for v in spatial_grid))
            limit = min(BLOCK_SIZE, int(img.shape[0]), int(img.shape[1]))
            effective_block_size = (limit // grid_unit) * grid_unit
            print(f'  effective block_size: {effective_block_size}')
            labels, polys = model.predict_instances_big(
                img, axes='YXC',
                block_size=effective_block_size,
                min_overlap=MIN_OVERLAP,
                context=CONTEXT,
                normalizer=normalizer,
                n_tiles=N_TILES,
            )
        n_cells = len(polys['coord']) if polys and 'coord' in polys else 0
        print(f'  detected nucleus count: {n_cells}')

    # 5. dilate the nucleus labels outward by 15 px as the whole-cell approximation.
    with Timer(f'nucleus expansion by {EXPAND_PX}px'):
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

    # 6. save both the raw nucleus labels and the expanded cell labels.
    with Timer('save TIF'):
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
            bigtiff=True,   # recommended for very large label maps
        )
        print(f'  raw nucleus labels: {NUCLEI_TIF}')
        print(f'  expanded cell labels: {LABELS_TIF}')

    # total elapsed
    total_dt = time.perf_counter() - total_t0
    h, rem = divmod(total_dt, 3600)
    m, s = divmod(rem, 60)
    print(f'==== total elapsed {int(h):02d}:{int(m):02d}:{s:05.2f}  ({total_dt:.2f} s) ====')


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
