"""
merfish_seg_utils.py
====================
Shared utilities for the MERFISH (Vizgen MERSCOPE) cell segmentation benchmark.

Responsibilities (method-agnostic; reusable by GT building / proseg / baysor / cellist ...):
  1. Read Vizgen official cell boundaries hdf5 (cell_boundaries/feature_data_<fov>.hdf5)
  2. Read the micron -> mosaic pixel affine transform matrix
  3. Convert micron coordinates -> mosaic pixel coordinates
  4. Rasterize a polygon set into a 2D instance label mask (one unique integer per cell)
  5. ROI-cropped read / shape read of large mosaic tifs

Coordinate convention:
  - pixel ROI is (x0, y0, x1, y1), where x=column (col/width) and y=row (row/height).
  - label mask shape is (H, W) = (y1-y0, x1-x0), in the same pixel coordinate system as the mosaic image.
"""

import os
import glob
import warnings

import numpy as np
import pandas as pd
import tifffile
from skimage.draw import polygon as sk_polygon

try:
    import h5py
except ImportError as e:  # pragma: no cover
    raise ImportError("h5py is needed to read Vizgen cell boundaries: pip install h5py") from e


# --------------------------------------------------------------------------- #
# 1. micron -> mosaic pixel transform matrix
# --------------------------------------------------------------------------- #
def load_micron_to_mosaic_transform(csv_path):
    """
    Read Vizgen's micron_to_mosaic_pixel_transform.csv and return a 3x3 numpy matrix.

    The file usually has no header and holds a 3x3 affine matrix in 3 rows, space or comma separated:
        [[sx,  0, tx],
         [ 0, sy, ty],
         [ 0,  0,  1]]
    Parsing here is robust: pull every float in the file and reshape(3, 3).
    """
    with open(csv_path, "r") as f:
        text = f.read()
    # pull all numbers (handles comma/space/newline/scientific notation)
    import re
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    vals = np.array([float(x) for x in nums], dtype=np.float64)
    if vals.size < 9:
        raise ValueError(f"transform matrix parse failed, got only {vals.size} numbers: {csv_path}")
    M = vals[:9].reshape(3, 3)
    return M


def micron_to_pixel(xy_micron, M):
    """
    xy_micron: (N, 2) micron coordinates (x, y)
    M:         3x3 affine matrix
    returns:   (N, 2) pixel coordinates (x=col, y=row), float
    """
    xy_micron = np.asarray(xy_micron, dtype=np.float64)
    n = xy_micron.shape[0]
    hom = np.hstack([xy_micron, np.ones((n, 1), dtype=np.float64)])  # (N,3)
    px = (M @ hom.T).T  # (N,3)
    return px[:, :2]


# --------------------------------------------------------------------------- #
# 2. Read Vizgen official cell boundary hdf5
# --------------------------------------------------------------------------- #
def read_fov_boundaries(hdf5_path, z_index=3):
    """
    Read all cell polygons (micron coordinates) at the given z layer of one feature_data_<fov>.hdf5.

    Vizgen layout:
        featuredata/<cell_id>/zIndex_<z>/p_0/coordinates   # shape (1, N, 2), micron

    returns: dict {cell_id(str): (N, 2) ndarray(micron)}; {} if nothing could be read.
    """
    out = {}
    with h5py.File(hdf5_path, "r") as f:
        if "featuredata" not in f:
            return out
        grp = f["featuredata"]
        zkey = f"zIndex_{z_index}"
        for cid in grp.keys():
            cell = grp[cid]
            if zkey not in cell:
                continue
            zg = cell[zkey]
            # usually p_0 only, occasionally several p_k: take the first polygon
            poly_keys = sorted([k for k in zg.keys() if k.startswith("p_")])
            if not poly_keys:
                continue
            pg = zg[poly_keys[0]]
            if "coordinates" not in pg:
                continue
            coords = np.asarray(pg["coordinates"][:])
            coords = np.squeeze(coords)  # (1,N,2) -> (N,2)
            if coords.ndim == 2 and coords.shape[1] == 2 and coords.shape[0] >= 3:
                out[str(cid)] = coords.astype(np.float64)
    return out


def _fovs_intersecting_roi(cell_metadata_csv, M, roi):
    """
    Optional speedup: transform cell_metadata.csv (min_x,min_y,max_x,max_y)(micron) to pixels and
    keep the FOVs whose bounding box intersects the ROI, avoiding opening all 1000+ hdf5 files.
    Returns None on parse failure (meaning: fall back to iterating every hdf5).
    """
    if cell_metadata_csv is None or not os.path.exists(cell_metadata_csv):
        return None
    try:
        meta = pd.read_csv(cell_metadata_csv, index_col=0)
        need = {"fov", "min_x", "max_x", "min_y", "max_y"}
        if not need.issubset(meta.columns):
            return None
        x0, y0, x1, y1 = roi
        # transform the 4 corners to pixels (handles possible flips), then take each cell's pixel bbox
        corners_micron = np.stack([
            meta[["min_x", "min_y"]].to_numpy(),
            meta[["min_x", "max_y"]].to_numpy(),
            meta[["max_x", "min_y"]].to_numpy(),
            meta[["max_x", "max_y"]].to_numpy(),
        ], axis=1).reshape(-1, 2)  # (4*Ncell, 2)
        px = micron_to_pixel(corners_micron, M).reshape(-1, 4, 2)
        px_min = px.min(axis=1)  # (Ncell,2) -> (x,y)
        px_max = px.max(axis=1)
        inter = ~((px_max[:, 0] < x0) | (px_min[:, 0] > x1) |
                  (px_max[:, 1] < y0) | (px_min[:, 1] > y1))
        return set(meta["fov"].to_numpy()[inter].astype(int).tolist())
    except Exception as e:
        warnings.warn(f"cell_metadata ROI prefilter failed, falling back to all hdf5: {e}")
        return None


def build_reference_label_mask(
    cell_boundaries_dir,
    transform_csv,
    roi,
    z_index=3,
    cell_metadata_csv=None,
    min_polygon_pixels=4,
    verbose=True,
):
    """
    Rasterize Vizgen official cell boundaries into one 2D instance label mask (the benchmark GT).

    Parameters
    ----------
    cell_boundaries_dir : directory holding feature_data_<fov>.hdf5
    transform_csv       : path to micron_to_mosaic_pixel_transform.csv
    roi                 : (x0, y0, x1, y1) pixel coordinates; label shape = (y1-y0, x1-x0)
    z_index             : which z layer to use (must match the z of the mosaic you segment; images here are z3)
    cell_metadata_csv   : optional, prefilters FOVs by ROI for speed
    min_polygon_pixels  : drop polygons with fewer pixels than this after rasterization

    Returns
    -------
    label : (H, W) int32, background=0, one unique id per cell (1..K)
    id_map: dict {label_int: vizgen_cell_id(str)}   to trace back to the official cell id
    """
    x0, y0, x1, y1 = [int(round(v)) for v in roi]
    H, W = y1 - y0, x1 - x0
    if H <= 0 or W <= 0:
        raise ValueError(f"invalid ROI: {roi}")

    M = load_micron_to_mosaic_transform(transform_csv)

    # choose the hdf5 files to open
    keep_fovs = _fovs_intersecting_roi(cell_metadata_csv, M, (x0, y0, x1, y1))
    all_files = sorted(glob.glob(os.path.join(cell_boundaries_dir, "feature_data_*.hdf5")))
    if keep_fovs is not None:
        files = []
        for p in all_files:
            base = os.path.basename(p)
            try:
                fov = int(base.replace("feature_data_", "").replace(".hdf5", ""))
            except ValueError:
                files.append(p)  # keep it when the name is unexpected
                continue
            if fov in keep_fovs:
                files.append(p)
        if verbose:
            print(f"[GT] ROI prefilter: {len(files)}/{len(all_files)} FOV hdf5 files to read")
    else:
        files = all_files
        if verbose:
            print(f"[GT] no prefilter, iterating all {len(files)} FOV hdf5 files")

    label = np.zeros((H, W), dtype=np.int32)
    id_map = {}
    next_id = 1

    for fi, path in enumerate(files):
        boundaries = read_fov_boundaries(path, z_index=z_index)
        for cid, verts_micron in boundaries.items():
            px = micron_to_pixel(verts_micron, M)
            # quick bbox rejection
            if (px[:, 0].max() < x0 or px[:, 0].min() > x1 or
                    px[:, 1].max() < y0 or px[:, 1].min() > y1):
                continue
            cc = px[:, 0] - x0  # column
            rr = px[:, 1] - y0  # row
            r, c = sk_polygon(rr, cc, shape=(H, W))
            if r.size < min_polygon_pixels:
                continue
            label[r, c] = next_id
            id_map[next_id] = cid
            next_id += 1
        if verbose and (fi + 1) % 100 == 0:
            print(f"[GT] processed {fi + 1}/{len(files)} hdf5 files, cells so far {next_id - 1}")

    if verbose:
        print(f"[GT] Done: ROI={roi}  shape=({H},{W})  cells={next_id - 1}")
    return label, id_map


# --------------------------------------------------------------------------- #
# 3. Shape / ROI-cropped read of large mosaic tifs
# --------------------------------------------------------------------------- #
def get_tif_shape(path):
    """Return the mosaic tif's (H, W) without loading the whole image into memory."""
    with tifffile.TiffFile(path) as tf:
        shape = tf.series[0].shape
    shape = tuple(int(s) for s in shape)
    # tolerate a possible channel axis: use the last two dims as (H, W)
    if len(shape) >= 2:
        return shape[-2], shape[-1]
    raise ValueError(f"cannot parse tif shape: {path} -> {shape}")


def read_tif_crop(path, roi=None):
    """
    Read a mosaic tif; roi=(x0,y0,x1,y1) in pixels. roi=None reads the whole image.
    Try a lazy crop first (zarr / memmap); fall back to reading it all, then cropping.
    Returns a 2D ndarray (H, W).
    """
    if roi is None:
        arr = tifffile.imread(path)
        return np.squeeze(arr)

    x0, y0, x1, y1 = [int(round(v)) for v in roi]

    # try 1: aszarr (cheapest for tiled/BigTIFF)
    try:
        import zarr  # noqa: F401
        store = tifffile.imread(path, aszarr=True)
        z = zarr.open(store, mode="r")
        arr = np.asarray(z[y0:y1, x0:x1])
        try:
            store.close()
        except Exception:
            pass
        return np.squeeze(arr)
    except Exception:
        pass

    # try 2: memmap
    try:
        mm = tifffile.memmap(path)
        arr = np.asarray(mm[y0:y1, x0:x1])
        del mm
        return np.squeeze(arr)
    except Exception:
        pass

    # fallback: read the whole image then crop (high memory; only if both above fail)
    warnings.warn("lazy crop unavailable, loading the whole mosaic before cropping (high memory use).")
    full = np.squeeze(tifffile.imread(path))
    return full[y0:y1, x0:x1]


# --------------------------------------------------------------------------- #
# 4. Write / read label masks
# --------------------------------------------------------------------------- #
def save_label_tif(label, path):
    """Save an instance label mask as tif (int32)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tifffile.imwrite(path, label.astype(np.int32), compression="zlib")
    print(f"[IO] Saved label mask -> {path}  (shape={label.shape}, cells={int((label>0).any()) and len(np.unique(label))-1})")


def make_center_roi(tif_path, size=6000):
    """Given a mosaic tif, return a central size x size ROI (x0,y0,x1,y1)."""
    H, W = get_tif_shape(tif_path)
    size = int(min(size, H, W))
    cy, cx = H // 2, W // 2
    x0 = max(0, cx - size // 2)
    y0 = max(0, cy - size // 2)
    return (x0, y0, x0 + size, y0 + size)