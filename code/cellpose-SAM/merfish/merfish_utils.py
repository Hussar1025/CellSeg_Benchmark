"""
merfish_seg_utils.py
====================
MERFISH (Vizgen MERSCOPE) 细胞分割 benchmark 的共享工具函数。

职责（与具体分割方法无关，可被 GT 构建 / proseg / baysor / cellist 等复用）：
  1. 读取 Vizgen 官方细胞边界 hdf5 (cell_boundaries/feature_data_<fov>.hdf5)
  2. 读取 micron -> mosaic pixel 仿射变换矩阵
  3. micron 坐标 -> mosaic pixel 坐标
  4. 把多边形集合栅格化成 2D 实例 label mask（每个细胞一个唯一整数）
  5. 大尺寸 mosaic tif 的按 ROI 裁剪读取 / 形状读取

坐标约定：
  - pixel ROI 用 (x0, y0, x1, y1)，其中 x=列(col/width)，y=行(row/height)。
  - label mask 形状为 (H, W) = (y1-y0, x1-x0)，与 mosaic 图像同一像素坐标系。
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
    raise ImportError("需要 h5py 读取 Vizgen 细胞边界： pip install h5py") from e


# --------------------------------------------------------------------------- #
# 1. micron -> mosaic pixel 变换矩阵
# --------------------------------------------------------------------------- #
def load_micron_to_mosaic_transform(csv_path):
    """
    读取 Vizgen 的 micron_to_mosaic_pixel_transform.csv，返回 3x3 numpy 矩阵。

    该文件通常是没有表头、以空格或逗号分隔的 3 行 3 列仿射矩阵：
        [[sx,  0, tx],
         [ 0, sy, ty],
         [ 0,  0,  1]]
    这里做鲁棒解析：直接抽取文件里的所有浮点数并 reshape(3, 3)。
    """
    with open(csv_path, "r") as f:
        text = f.read()
    # 抽取所有数字（兼容逗号/空格/换行/科学计数法）
    import re
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    vals = np.array([float(x) for x in nums], dtype=np.float64)
    if vals.size < 9:
        raise ValueError(f"变换矩阵解析失败，仅解析到 {vals.size} 个数：{csv_path}")
    M = vals[:9].reshape(3, 3)
    return M


def micron_to_pixel(xy_micron, M):
    """
    xy_micron: (N, 2) micron 坐标 (x, y)
    M:         3x3 仿射矩阵
    返回:      (N, 2) pixel 坐标 (x=col, y=row)，float
    """
    xy_micron = np.asarray(xy_micron, dtype=np.float64)
    n = xy_micron.shape[0]
    hom = np.hstack([xy_micron, np.ones((n, 1), dtype=np.float64)])  # (N,3)
    px = (M @ hom.T).T  # (N,3)
    return px[:, :2]


# --------------------------------------------------------------------------- #
# 2. Vizgen 官方细胞边界 hdf5 读取
# --------------------------------------------------------------------------- #
def read_fov_boundaries(hdf5_path, z_index=3):
    """
    读取单个 feature_data_<fov>.hdf5 在指定 z 层的所有细胞多边形（micron 坐标）。

    Vizgen 结构：
        featuredata/<cell_id>/zIndex_<z>/p_0/coordinates   # shape (1, N, 2), micron

    返回: dict {cell_id(str): (N, 2) ndarray(micron)}；读取不到则返回 {}。
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
            # 一般只有 p_0，偶有多个 p_k，取第一个多边形
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
    可选优化：用 cell_metadata.csv 的 (min_x,min_y,max_x,max_y)(micron) 变换到 pixel，
    找出 bounding box 与 ROI 相交的 FOV 集合，避免打开全部 1000+ 个 hdf5。
    解析失败时返回 None（表示回退到“遍历所有 hdf5”）。
    """
    if cell_metadata_csv is None or not os.path.exists(cell_metadata_csv):
        return None
    try:
        meta = pd.read_csv(cell_metadata_csv, index_col=0)
        need = {"fov", "min_x", "max_x", "min_y", "max_y"}
        if not need.issubset(meta.columns):
            return None
        x0, y0, x1, y1 = roi
        # 4 个角变换到 pixel（兼容潜在翻转），取每个细胞的 pixel bbox
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
        warnings.warn(f"cell_metadata ROI 预筛失败，回退遍历全部 hdf5：{e}")
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
    把 Vizgen 官方细胞边界栅格化成一张 2D 实例 label mask（作为 benchmark 的 GT）。

    参数
    ----
    cell_boundaries_dir : 含 feature_data_<fov>.hdf5 的目录
    transform_csv       : micron_to_mosaic_pixel_transform.csv 路径
    roi                 : (x0, y0, x1, y1) pixel 坐标；label 形状 = (y1-y0, x1-x0)
    z_index             : 使用哪一层 z（应与你分割用的 mosaic 图像 z 一致，本数据集图像为 z3）
    cell_metadata_csv   : 可选，用于按 ROI 预筛 FOV 以加速
    min_polygon_pixels  : 栅格化后像素数过少的多边形丢弃

    返回
    ----
    label : (H, W) int32，背景=0，每个细胞唯一 id（1..K）
    id_map: dict {label_int: vizgen_cell_id(str)}   便于回溯到官方 cell id
    """
    x0, y0, x1, y1 = [int(round(v)) for v in roi]
    H, W = y1 - y0, x1 - x0
    if H <= 0 or W <= 0:
        raise ValueError(f"非法 ROI: {roi}")

    M = load_micron_to_mosaic_transform(transform_csv)

    # 选择要打开的 hdf5 文件
    keep_fovs = _fovs_intersecting_roi(cell_metadata_csv, M, (x0, y0, x1, y1))
    all_files = sorted(glob.glob(os.path.join(cell_boundaries_dir, "feature_data_*.hdf5")))
    if keep_fovs is not None:
        files = []
        for p in all_files:
            base = os.path.basename(p)
            try:
                fov = int(base.replace("feature_data_", "").replace(".hdf5", ""))
            except ValueError:
                files.append(p)  # 命名异常则保守保留
                continue
            if fov in keep_fovs:
                files.append(p)
        if verbose:
            print(f"[GT] ROI 预筛：{len(files)}/{len(all_files)} 个 FOV hdf5 需要读取")
    else:
        files = all_files
        if verbose:
            print(f"[GT] 未预筛，遍历全部 {len(files)} 个 FOV hdf5")

    label = np.zeros((H, W), dtype=np.int32)
    id_map = {}
    next_id = 1

    for fi, path in enumerate(files):
        boundaries = read_fov_boundaries(path, z_index=z_index)
        for cid, verts_micron in boundaries.items():
            px = micron_to_pixel(verts_micron, M)
            # 快速 bbox 剔除
            if (px[:, 0].max() < x0 or px[:, 0].min() > x1 or
                    px[:, 1].max() < y0 or px[:, 1].min() > y1):
                continue
            cc = px[:, 0] - x0  # 列
            rr = px[:, 1] - y0  # 行
            r, c = sk_polygon(rr, cc, shape=(H, W))
            if r.size < min_polygon_pixels:
                continue
            label[r, c] = next_id
            id_map[next_id] = cid
            next_id += 1
        if verbose and (fi + 1) % 100 == 0:
            print(f"[GT] 已处理 {fi + 1}/{len(files)} 个 hdf5，当前细胞数 {next_id - 1}")

    if verbose:
        print(f"[GT] 完成：ROI={roi}  形状=({H},{W})  细胞数={next_id - 1}")
    return label, id_map


# --------------------------------------------------------------------------- #
# 3. 大 mosaic tif 的形状 / ROI 裁剪读取
# --------------------------------------------------------------------------- #
def get_tif_shape(path):
    """返回 mosaic tif 的 (H, W)，不整幅载入内存。"""
    with tifffile.TiffFile(path) as tf:
        shape = tf.series[0].shape
    shape = tuple(int(s) for s in shape)
    # 兼容可能存在的通道维，取最后两维作为 (H, W)
    if len(shape) >= 2:
        return shape[-2], shape[-1]
    raise ValueError(f"无法解析 tif 形状：{path} -> {shape}")


def read_tif_crop(path, roi=None):
    """
    读取 mosaic tif；roi=(x0,y0,x1,y1) pixel。roi=None 时整幅读取。
    优先尝试惰性裁剪（zarr / memmap），失败回退整幅读入再裁剪。
    返回 2D ndarray (H, W)。
    """
    if roi is None:
        arr = tifffile.imread(path)
        return np.squeeze(arr)

    x0, y0, x1, y1 = [int(round(v)) for v in roi]

    # 尝试 1：aszarr（对 tiled/BigTIFF 最省内存）
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

    # 尝试 2：memmap
    try:
        mm = tifffile.memmap(path)
        arr = np.asarray(mm[y0:y1, x0:x1])
        del mm
        return np.squeeze(arr)
    except Exception:
        pass

    # 回退：整幅读入再裁剪（内存占用大，仅在前两种失败时使用）
    warnings.warn("惰性裁剪不可用，整幅载入 mosaic 后再裁剪（内存占用较大）。")
    full = np.squeeze(tifffile.imread(path))
    return full[y0:y1, x0:x1]


# --------------------------------------------------------------------------- #
# 4. label mask 落盘 / 读取
# --------------------------------------------------------------------------- #
def save_label_tif(label, path):
    """保存实例 label mask 为 tif（int32）。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tifffile.imwrite(path, label.astype(np.int32), compression="zlib")
    print(f"[IO] 已保存 label mask -> {path}  (shape={label.shape}, cells={int((label>0).any()) and len(np.unique(label))-1})")


def make_center_roi(tif_path, size=6000):
    """给一张 mosaic tif，返回其中心一个 size×size 的 ROI (x0,y0,x1,y1)。"""
    H, W = get_tif_shape(tif_path)
    size = int(min(size, H, W))
    cy, cx = H // 2, W // 2
    x0 = max(0, cx - size // 2)
    y0 = max(0, cy - size // 2)
    return (x0, y0, x0 + size, y0 + size)