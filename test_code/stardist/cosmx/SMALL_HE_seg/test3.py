"""
StarDist2D-based H&E whole-slide cell segmentation script.
- uses the pretrained model '2D_versatile_he'
- uses a custom Normalizer to avoid normalizing the whole large image at once
- uses predict_instances_big for tiled prediction
- supports Zarr for very large images (avoids loading all into memory at once)
- outputs a compressed TIF label map
- stage-by-stage timing
"""

import time
import numpy as np
from pathlib import Path
from zipfile import ZIP_DEFLATED

from tifffile import imread, imwrite as imsave
from csbdeep.data import Normalizer, normalize_mi_ma
from stardist.models import StarDist2D


# ============================================================
# configuration parameters
# ============================================================
INPUT_PATH  = Path('yourpath/dataset/CosMx/cropped_HE.tif')      # input WSI path (tif/svs readable as tif)
OUTPUT_DIR  = Path('yourpath/stardist/CosMx_output')                # output directory
LABELS_TIF  = OUTPUT_DIR / 'labels.tif'       # output label TIF

# whether to use Zarr for very large images (set True when the image cannot fit in memory at once)
USE_ZARR    = False
ZARR_IMAGE  = OUTPUT_DIR / 'image.zarr'
ZARR_LABELS = OUTPUT_DIR / 'labels.zarr'

# tiled prediction parameters (adjust by GPU memory / RAM)
BLOCK_SIZE   = 4096    # side length of each block
MIN_OVERLAP  = 128     # minimum overlap between blocks, to avoid cutting cells at boundaries
CONTEXT      = 128     # extra context at block edges for the CNN
N_TILES      = (4, 4, 1)  # further tiling within a single block (Y, X, C)

# normalization range: 0~255 for uint8 H&E; see comment below for percentile normalization
MI, MA = 0, 255


# ============================================================
# utility: timing context manager
# ============================================================
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


# ============================================================
# custom Normalizer (normalizes on demand, does not preprocess the whole large image)
# ============================================================
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

    # --------------------------------------------------------
    # 1. read image
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # 2. load pretrained model
    # --------------------------------------------------------
    with Timer('load model'):
        model = StarDist2D.from_pretrained('2D_versatile_he')

    # --------------------------------------------------------
    # 3. prepare normalizer
    # --------------------------------------------------------
    # For percentile normalization (more robust), estimate on a low-resolution version:
    # if USE_ZARR:
    #     sample = img[::16, ::16]
    # else:
    #     sample = img[::16, ::16]
    # mi, ma = np.percentile(sample, [1, 99.8])
    normalizer = MyNormalizer(MI, MA)

    # --------------------------------------------------------
    # 4. tiled prediction
    # --------------------------------------------------------
    with Timer('tiled prediction (predict_instances_big)'):
        if USE_ZARR:
            import zarr
            # pre-create the zarr label array for writing
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
                labels_out=labels_zarr,    # write directly to zarr, avoid assembling the full label map in memory
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
        print(f'  detected cell count: {n_cells}')

    # --------------------------------------------------------
    # 5. save label map as compressed TIF
    # --------------------------------------------------------
    with Timer('save TIF'):
        if USE_ZARR:
            # zarr -> numpy then write tif; for very large label maps, use bigtiff=True and write in blocks
            labels_np = np.array(labels)
        else:
            labels_np = labels

        imsave(
            str(LABELS_TIF),
            labels_np,
            compression=ZIP_DEFLATED,
            bigtiff=True,   # recommended for very large label maps
        )
        print(f'  saved: {LABELS_TIF}')

    # --------------------------------------------------------
    # total elapsed
    # --------------------------------------------------------
    total_dt = time.perf_counter() - total_t0
    h, rem = divmod(total_dt, 3600)
    m, s = divmod(rem, 60)
    print(f'==== total elapsed {int(h):02d}:{int(m):02d}:{s:05.2f}  ({total_dt:.2f} s) ====')


if __name__ == '__main__':
    main()
