import gc
import shutil
import subprocess
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tifffile as tiff
from rasterio import features
from rasterio.transform import from_origin
from shapely.validation import make_valid
from tqdm import tqdm


# ======================================================
# 目录设置
# ======================================================

INPUT_DIR = Path.home() / "Cosmx_6k_COAD_data/cellist_result/input/userdata"
RESULT_DIR = Path.home() / "Cosmx_6k_COAD_data/cellist_result/proseg_cosmx"

PROSEG_EXE = "proseg"

TX_PARQUET = INPUT_DIR / "tx_file.parquet"
DAPI_TIF = INPUT_DIR / "DAPI.tif"

PROSEG_INPUT_CSV = RESULT_DIR / "transcripts_proseg_input_cosmx_columns.csv"

SPATIALDATA_DIR_NAME = "proseg-output.zarr"
SPATIALDATA_DIR = RESULT_DIR / SPATIALDATA_DIR_NAME

TEMP_GEOJSON_NAME = "cell_polygons_proseg_tmp.geojson"
TEMP_GEOJSON_PATH = RESULT_DIR / TEMP_GEOJSON_NAME

GEOJSON_DIR = RESULT_DIR / "cell_polygons_proseg.geojson"
GEOJSON_PATH = GEOJSON_DIR / "cell_polygons_proseg.geojson"

OUT_TIFF = RESULT_DIR / "proseg_instance_mask_fixed.tif"
OUT_TIFF_COPY = RESULT_DIR / "proseg_instance_mask.tif"
PREVIEW_TIFF = RESULT_DIR / "proseg_instance_mask_preview_binary.tif"

NTHREADS = 12
PIXEL_SIZE = 1.0

# 内存保护
SAFE_MEMORY_MODE = True
MAX_MASK_PIXELS = 1_500_000_000   # uint32 大约 6 GB；太大直接停止
MAX_ALLOWED_GB_FOR_MASK = 30


# ======================================================
# 工具函数
# ======================================================

def find_col(columns, candidates, required=True):
    lower_map = {c.lower(): c for c in columns}

    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    if required:
        raise ValueError(
            "Cannot find required column. Tried:\n"
            f"{candidates}\n\n"
            f"Available columns:\n{list(columns)}"
        )

    return None


def prepare_cosmx_transcripts_from_parquet(tx_parquet, out_csv):
    print("\nPreparing CosMx transcript CSV from parquet...")
    print(f"Input parquet: {tx_parquet}")

    schema = pq.read_schema(tx_parquet)
    all_cols = schema.names

    print("\nOriginal parquet columns:")
    print(all_cols)

    gene_col = find_col(
        all_cols,
        ["target", "Target", "gene", "Gene", "gene_name", "GeneName", "feature_name", "target_name", "targetName"],
        required=True,
    )

    x_col = find_col(
        all_cols,
        ["x_global_px", "X_global_px", "global_x", "Global_X", "x_global", "X_global", "x", "X", "x_location", "x_local_px"],
        required=True,
    )

    y_col = find_col(
        all_cols,
        ["y_global_px", "Y_global_px", "global_y", "Global_Y", "y_global", "Y_global", "y", "Y", "y_location", "y_local_px"],
        required=True,
    )

    z_col = find_col(
        all_cols,
        ["z", "Z", "z_index", "z_plane", "ZPlane", "z_location"],
        required=False,
    )

    cell_col = find_col(
        all_cols,
        ["cell", "Cell", "cell_ID", "cell_id", "CellId", "cellId", "CellID", "segmentation_cell_id"],
        required=False,
    )

    fov_col = find_col(
        all_cols,
        ["fov", "FOV", "fov_id", "FOVId", "fov_name"],
        required=False,
    )

    cellcomp_col = find_col(
        all_cols,
        ["CellComp", "cellcomp", "cell_comp", "Cell_Compartment", "compartment", "Compartment"],
        required=False,
    )

    needed_cols = [gene_col, x_col, y_col]
    for c in [z_col, cell_col, fov_col, cellcomp_col]:
        if c is not None and c not in needed_cols:
            needed_cols.append(c)

    print("\nReading only needed parquet columns:")
    print(needed_cols)

    table = pq.read_table(tx_parquet, columns=needed_cols)
    df = table.to_pandas(self_destruct=True)
    del table
    gc.collect()

    print(f"Loaded transcript number: {len(df):,}")

    out = pd.DataFrame({
        "fov": df[fov_col].values if fov_col is not None else 1,
        "cell": df[cell_col].values if cell_col is not None else 0,
        "x_global_px": pd.to_numeric(df[x_col], errors="coerce").astype("float32"),
        "y_global_px": pd.to_numeric(df[y_col], errors="coerce").astype("float32"),
        "z": pd.to_numeric(df[z_col], errors="coerce").fillna(0).astype("float32") if z_col is not None else np.zeros(len(df), dtype="float32"),
        "target": df[gene_col].astype(str).values,
        "CellComp": df[cellcomp_col].fillna("Cytoplasm").astype(str).values if cellcomp_col is not None else "Cytoplasm",
    })

    del df
    gc.collect()

    before = len(out)

    out = out.dropna(subset=["x_global_px", "y_global_px", "target"])
    out = out[out["target"].astype(str) != ""]
    out = out[out["target"].astype(str).str.lower() != "nan"]

    out["fov"] = out["fov"].fillna(1)
    out["cell"] = out["cell"].fillna(0)
    out["z"] = out["z"].fillna(0)
    out["CellComp"] = out["CellComp"].fillna("Cytoplasm")

    after = len(out)

    print(f"Removed invalid transcripts: {before - after:,}")
    print(f"Remaining transcripts: {after:,}")

    required_cols = [
        "fov",
        "cell",
        "x_global_px",
        "y_global_px",
        "z",
        "target",
        "CellComp",
    ]

    out = out[required_cols]

    missing = [c for c in required_cols if c not in out.columns]
    if missing:
        raise ValueError(f"Generated CSV missing required columns: {missing}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    print("\nSaved Proseg input CSV:")
    print(out_csv)

    print("\nFinal CSV columns:")
    print(list(out.columns))

    print("\nCSV preview:")
    print(out.head())

    del out
    gc.collect()

    return out_csv


# ======================================================
# 检查输入
# ======================================================

print("Checking files...")

RESULT_DIR.mkdir(parents=True, exist_ok=True)

if shutil.which(PROSEG_EXE) is None and not Path(PROSEG_EXE).exists():
    raise FileNotFoundError(
        f"Cannot find proseg executable: {PROSEG_EXE}\n"
        "请确认服务器中可以运行：proseg --help"
    )

if not TX_PARQUET.exists():
    raise FileNotFoundError(f"Cannot find tx_file.parquet:\n{TX_PARQUET}")

if DAPI_TIF.exists():
    print(f"DAPI = {DAPI_TIF}")
else:
    print(f"Warning: DAPI.tif not found: {DAPI_TIF}")

print(f"INPUT_DIR       = {INPUT_DIR}")
print(f"RESULT_DIR      = {RESULT_DIR}")
print(f"TX_PARQUET      = {TX_PARQUET}")
print(f"PROSEG_EXE      = {PROSEG_EXE}")
print(f"SPATIALDATA_DIR = {SPATIALDATA_DIR}")


# ======================================================
# 预处理 parquet -> CosMx CSV
# ======================================================

PROSEG_INPUT_CSV = prepare_cosmx_transcripts_from_parquet(
    TX_PARQUET,
    PROSEG_INPUT_CSV,
)


# ======================================================
# 清理旧输出
# ======================================================

for p in [
    TEMP_GEOJSON_PATH,
    GEOJSON_DIR,
    SPATIALDATA_DIR,
    OUT_TIFF,
    OUT_TIFF_COPY,
    PREVIEW_TIFF,
]:
    if p.exists():
        print(f"Removing old output: {p}")
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()


# ======================================================
# 运行 proseg
# ======================================================

print("\nRunning proseg with relative paths...")

cmd = [
    PROSEG_EXE,
    "--cosmx",
    "--output-spatialdata",
    SPATIALDATA_DIR_NAME,
    "--output-cell-polygons",
    TEMP_GEOJSON_NAME,
    "--overwrite",
    "--nthreads",
    str(NTHREADS),
]

if SAFE_MEMORY_MODE:
    cmd.append("--no-diffusion")

cmd.append(PROSEG_INPUT_CSV.name)

print("Command:")
print(" ".join(cmd))

subprocess.run(
    cmd,
    check=True,
    cwd=str(RESULT_DIR),
)

print("\nproseg finished.")

if not TEMP_GEOJSON_PATH.exists():
    raise FileNotFoundError(f"GeoJSON not found: {TEMP_GEOJSON_PATH}")

if not SPATIALDATA_DIR.exists():
    raise FileNotFoundError(f"SpatialData output not found: {SPATIALDATA_DIR}")


# ======================================================
# 整理 GeoJSON
# ======================================================

print("\nPreparing benchmark-style GeoJSON directory...")

GEOJSON_DIR.mkdir(parents=True, exist_ok=True)
shutil.move(str(TEMP_GEOJSON_PATH), str(GEOJSON_PATH))

print(f"Final GeoJSON:")
print(GEOJSON_PATH)


# ======================================================
# 读取 GeoJSON
# ======================================================

print("\nLoading polygons...")

gdf = gpd.read_file(GEOJSON_PATH)

print("\nLoaded successfully.")
print(f"Number of cells: {len(gdf)}")
print("\nColumns:")
print(gdf.columns)


# ======================================================
# 清理 geometry
# ======================================================

print("\nCleaning geometries...")

valid_geoms = []

for idx, geom in tqdm(
    enumerate(gdf.geometry),
    total=len(gdf),
    desc="Checking geometries",
):
    if geom is None:
        continue

    if geom.is_empty:
        continue

    if not geom.is_valid:
        geom = make_valid(geom)

    if geom is None or geom.is_empty:
        continue

    cell_id = idx + 1
    valid_geoms.append((geom, cell_id))

print(f"\nValid polygons: {len(valid_geoms)}")

if len(valid_geoms) == 0:
    raise ValueError("No valid polygons found.")


# ======================================================
# 获取组织边界
# ======================================================

minx, miny, maxx, maxy = gdf.total_bounds

print("\nTissue bounds:")
print(f"minx = {minx}")
print(f"miny = {miny}")
print(f"maxx = {maxx}")
print(f"maxy = {maxy}")


# ======================================================
# 计算输出图像大小
# ======================================================

width = int(np.ceil((maxx - minx) / PIXEL_SIZE))
height = int(np.ceil((maxy - miny) / PIXEL_SIZE))

print("\nOutput image size:")
print(f"width  = {width}")
print(f"height = {height}")

total_pixels = width * height
mask_gb = total_pixels * np.dtype(np.uint32).itemsize / 1024**3

print(f"\nTotal pixels: {total_pixels:,}")
print(f"Estimated uint32 mask memory: {mask_gb:.2f} GB")

if total_pixels > MAX_MASK_PIXELS or mask_gb > MAX_ALLOWED_GB_FOR_MASK:
    raise MemoryError(
        "\nRasterized mask would be too large.\n"
        f"width={width}, height={height}, pixels={total_pixels:,}, estimated={mask_gb:.2f} GB\n\n"
        "建议：\n"
        "1. 增大 PIXEL_SIZE，例如 PIXEL_SIZE = 2.0 或 4.0；\n"
        "2. 或者按 FOV / ROI 分块生成 mask；\n"
        "3. 或者只保留 GeoJSON 结果，不生成全图 instance mask。"
    )


# ======================================================
# affine transform
# ======================================================

transform = from_origin(
    west=minx,
    north=maxy,
    xsize=PIXEL_SIZE,
    ysize=PIXEL_SIZE,
)

print("\nAffine transform:")
print(transform)


# ======================================================
# Rasterize polygons
# ======================================================

print("\nRasterizing polygons...")

mask = features.rasterize(
    shapes=valid_geoms,
    out_shape=(height, width),
    fill=0,
    transform=transform,
    dtype=np.uint32,
    all_touched=False,
)

del valid_geoms
del gdf
gc.collect()

print("\nMask info:")
print(f"dtype = {mask.dtype}")
print(f"min   = {mask.min()}")
print(f"max   = {mask.max()}")
print(f"nonzero pixels = {np.count_nonzero(mask):,}")


# ======================================================
# 保存 TIFF
# ======================================================

print("\nSaving TIFF...")

tiff.imwrite(
    str(OUT_TIFF),
    mask,
    compression="zlib",
    photometric="minisblack",
    bigtiff=True,
)

tiff.imwrite(
    str(OUT_TIFF_COPY),
    mask,
    compression="zlib",
    photometric="minisblack",
    bigtiff=True,
)

preview = (mask > 0).astype(np.uint8) * 255

tiff.imwrite(
    str(PREVIEW_TIFF),
    preview,
    compression="zlib",
    photometric="minisblack",
    bigtiff=True,
)

del mask
del preview
gc.collect()

print("\nDone.")
print(f"Saved to: {OUT_TIFF}")
print(f"Saved copy to: {OUT_TIFF_COPY}")
print(f"Preview saved to: {PREVIEW_TIFF}")


# ======================================================
# 最终检查
# ======================================================

print("\nFinal output check:")

final_expected = [
    SPATIALDATA_DIR,
    SPATIALDATA_DIR / ".zgroup",
    SPATIALDATA_DIR / "points",
    SPATIALDATA_DIR / "shapes",
    SPATIALDATA_DIR / "tables",
    GEOJSON_DIR,
    GEOJSON_PATH,
    OUT_TIFF_COPY,
    OUT_TIFF,
    PREVIEW_TIFF,
]

for p in final_expected:
    status = "OK" if p.exists() else "MISSING"
    kind = "DIR" if p.is_dir() else "FILE" if p.is_file() else "----"
    print(f"{status:8s} | {kind:4s} | {p}")

print("\n========== All Done ==========")

import gc
import gzip
import shutil
import subprocess
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tifffile as tiff
from rasterio import features
from rasterio.transform import from_origin
from shapely.validation import make_valid
from tqdm import tqdm


INPUT_DIR = Path.home() / "Cosmx_6k_COAD_data/cellist_result/input/userdata"
RESULT_DIR = Path.home() / "Cosmx_6k_COAD_data/cellist_result/proseg_cosmx"

PROSEG_EXE = "proseg"
TX_PARQUET = INPUT_DIR / "tx_file.parquet"
DAPI_TIF = INPUT_DIR / "DAPI.tif"

PROSEG_INPUT_CSV = RESULT_DIR / "transcripts_proseg_input_cosmx_columns.csv"

SPATIALDATA_DIR_NAME = "proseg-output.zarr"
SPATIALDATA_DIR = RESULT_DIR / SPATIALDATA_DIR_NAME

TEMP_GEOJSON_NAME = "cell_polygons_proseg_tmp.geojson.gz"
TEMP_GEOJSON_PATH = RESULT_DIR / TEMP_GEOJSON_NAME

GEOJSON_DIR = RESULT_DIR / "cell_polygons_proseg.geojson"
GEOJSON_PATH_GZ = GEOJSON_DIR / "cell_polygons_proseg.geojson.gz"
GEOJSON_PATH_RAW = GEOJSON_DIR / "cell_polygons_proseg.geojson"

OUT_TIFF = RESULT_DIR / "proseg_instance_mask_fixed.tif"
OUT_TIFF_COPY = RESULT_DIR / "proseg_instance_mask.tif"
PREVIEW_TIFF = RESULT_DIR / "proseg_instance_mask_preview_binary.tif"

NTHREADS = 12
PIXEL_SIZE = 1.0
SAFE_MEMORY_MODE = True

# 超过这个大小就不生成全图 mask，避免爆内存
MAX_ALLOWED_GB_FOR_MASK = 30


def find_col(columns, candidates, required=True):
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    if required:
        raise ValueError(f"Cannot find required column from {candidates}\nAvailable: {list(columns)}")
    return None


def prepare_cosmx_transcripts_from_parquet(tx_parquet, out_csv):
    if out_csv.exists():
        print(f"\nFound existing preprocessed CSV, skip preprocessing:\n{out_csv}")
        return out_csv

    print("\nPreparing CosMx transcript CSV from parquet...")

    schema = pq.read_schema(tx_parquet)
    all_cols = schema.names
    print("Parquet columns:")
    print(all_cols)

    gene_col = find_col(all_cols, ["target", "Target", "gene", "Gene", "gene_name", "GeneName", "feature_name"])
    x_col = find_col(all_cols, ["x_global_px", "X_global_px", "global_x", "Global_X", "x_global", "X_global", "x", "X"])
    y_col = find_col(all_cols, ["y_global_px", "Y_global_px", "global_y", "Global_Y", "y_global", "Y_global", "y", "Y"])

    z_col = find_col(all_cols, ["z", "Z", "z_index", "z_plane", "ZPlane"], required=False)
    cell_col = find_col(all_cols, ["cell", "Cell", "cell_ID", "cell_id", "CellId", "cellId", "CellID"], required=False)
    fov_col = find_col(all_cols, ["fov", "FOV", "fov_id", "FOVId", "fov_name"], required=False)
    cellcomp_col = find_col(all_cols, ["CellComp", "cellcomp", "cell_comp", "Cell_Compartment", "compartment"], required=False)

    needed_cols = [gene_col, x_col, y_col]
    for c in [z_col, cell_col, fov_col, cellcomp_col]:
        if c is not None and c not in needed_cols:
            needed_cols.append(c)

    print("Reading needed columns only:")
    print(needed_cols)

    table = pq.read_table(tx_parquet, columns=needed_cols)
    df = table.to_pandas(self_destruct=True)
    del table
    gc.collect()

    out = pd.DataFrame({
        "fov": df[fov_col].values if fov_col is not None else 1,
        "cell": df[cell_col].values if cell_col is not None else 0,
        "x_global_px": pd.to_numeric(df[x_col], errors="coerce").astype("float32"),
        "y_global_px": pd.to_numeric(df[y_col], errors="coerce").astype("float32"),
        "z": pd.to_numeric(df[z_col], errors="coerce").fillna(0).astype("float32") if z_col is not None else np.zeros(len(df), dtype="float32"),
        "target": df[gene_col].astype(str).values,
        "CellComp": df[cellcomp_col].fillna("Cytoplasm").astype(str).values if cellcomp_col is not None else "Cytoplasm",
    })

    del df
    gc.collect()

    before = len(out)
    out = out.dropna(subset=["x_global_px", "y_global_px", "target"])
    out = out[out["target"].astype(str) != ""]
    out = out[out["target"].astype(str).str.lower() != "nan"]
    print(f"Removed invalid transcripts: {before - len(out):,}")
    print(f"Remaining transcripts: {len(out):,}")

    out = out[["fov", "cell", "x_global_px", "y_global_px", "z", "target", "CellComp"]]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"Saved CSV: {out_csv}")

    del out
    gc.collect()
    return out_csv


def is_gzip_file(path):
    with open(path, "rb") as f:
        return f.read(2) == b"\x1f\x8b"


def locate_existing_polygon_file():
    candidates = [
        GEOJSON_PATH_GZ,
        GEOJSON_PATH_RAW,
        TEMP_GEOJSON_PATH,
        RESULT_DIR / "cell_polygons_proseg_tmp.geojson",
        RESULT_DIR / "cell_polygons_proseg.geojson",
        RESULT_DIR / "cell_polygons_proseg.geojson.gz",
    ]

    for p in candidates:
        if p.exists() and p.is_file():
            return p

    raise FileNotFoundError(
        "Cannot find polygon GeoJSON output. Checked:\n" +
        "\n".join(str(p) for p in candidates)
    )


def read_polygon_file(path):
    print(f"\nLoading polygons from:\n{path}")

    # 如果是 gzip，无论后缀对不对，都先解压到临时 geojson 再读
    if is_gzip_file(path):
        print("Detected gzip-compressed GeoJSON.")
        with gzip.open(path, "rb") as f_in:
            with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False) as f_out:
                shutil.copyfileobj(f_in, f_out)
                tmp_path = f_out.name
        return gpd.read_file(tmp_path)

    print("Detected plain GeoJSON.")
    return gpd.read_file(path)


def run_proseg_if_needed():
    if SPATIALDATA_DIR.exists():
        polygon_file = None
        try:
            polygon_file = locate_existing_polygon_file()
        except FileNotFoundError:
            pass

        if polygon_file is not None:
            print("\nFound existing proseg outputs. Skip proseg running.")
            print(f"SpatialData: {SPATIALDATA_DIR}")
            print(f"Polygon file: {polygon_file}")
            return polygon_file

    print("\nNo complete existing proseg output found. Running proseg...")

    PROSEG_INPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    if TEMP_GEOJSON_PATH.exists():
        TEMP_GEOJSON_PATH.unlink()

    cmd = [
        PROSEG_EXE,
        "--cosmx",
        "--output-spatialdata",
        SPATIALDATA_DIR_NAME,
        "--output-cell-polygons",
        TEMP_GEOJSON_NAME,
        "--overwrite",
        "--nthreads",
        str(NTHREADS),
    ]

    if SAFE_MEMORY_MODE:
        cmd.append("--no-diffusion")

    cmd.append(PROSEG_INPUT_CSV.name)

    print("Command:")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True, cwd=str(RESULT_DIR))

    print("\nproseg finished.")

    if not TEMP_GEOJSON_PATH.exists():
        raise FileNotFoundError(f"GeoJSON not found: {TEMP_GEOJSON_PATH}")

    return TEMP_GEOJSON_PATH


# ======================================================
# 主流程
# ======================================================

print("Checking files...")

RESULT_DIR.mkdir(parents=True, exist_ok=True)

if shutil.which(PROSEG_EXE) is None and not Path(PROSEG_EXE).exists():
    raise FileNotFoundError("Cannot find proseg. Please check: proseg --help")

if not TX_PARQUET.exists() and not PROSEG_INPUT_CSV.exists():
    raise FileNotFoundError(f"Cannot find tx_file.parquet:\n{TX_PARQUET}")

if DAPI_TIF.exists():
    print(f"DAPI = {DAPI_TIF}")

PROSEG_INPUT_CSV = prepare_cosmx_transcripts_from_parquet(TX_PARQUET, PROSEG_INPUT_CSV)

polygon_file = run_proseg_if_needed()


# ======================================================
# 整理 polygon 到 benchmark 目录
# ======================================================

GEOJSON_DIR.mkdir(parents=True, exist_ok=True)

if polygon_file != GEOJSON_PATH_GZ and polygon_file != GEOJSON_PATH_RAW:
    if is_gzip_file(polygon_file):
        final_polygon_path = GEOJSON_PATH_GZ
    else:
        final_polygon_path = GEOJSON_PATH_RAW

    if final_polygon_path.exists():
        final_polygon_path.unlink()

    shutil.copy2(polygon_file, final_polygon_path)
else:
    final_polygon_path = polygon_file

print(f"\nFinal polygon file:")
print(final_polygon_path)


# ======================================================
# 读取 polygon，生成 mask
# ======================================================

gdf = read_polygon_file(final_polygon_path)

print("\nLoaded successfully.")
print(f"Number of cells: {len(gdf)}")
print("Columns:")
print(gdf.columns)


print("\nCleaning geometries...")

valid_geoms = []

for idx, geom in tqdm(enumerate(gdf.geometry), total=len(gdf), desc="Checking geometries"):
    if geom is None or geom.is_empty:
        continue

    if not geom.is_valid:
        geom = make_valid(geom)

    if geom is None or geom.is_empty:
        continue

    valid_geoms.append((geom, idx + 1))

print(f"Valid polygons: {len(valid_geoms)}")

if len(valid_geoms) == 0:
    raise ValueError("No valid polygons found.")


minx, miny, maxx, maxy = gdf.total_bounds

print("\nTissue bounds:")
print(f"minx = {minx}")
print(f"miny = {miny}")
print(f"maxx = {maxx}")
print(f"maxy = {maxy}")

width = int(np.ceil((maxx - minx) / PIXEL_SIZE))
height = int(np.ceil((maxy - miny) / PIXEL_SIZE))

total_pixels = width * height
mask_gb = total_pixels * np.dtype(np.uint32).itemsize / 1024**3

print("\nOutput image size:")
print(f"width  = {width}")
print(f"height = {height}")
print(f"pixels = {total_pixels:,}")
print(f"estimated uint32 mask memory = {mask_gb:.2f} GB")

if mask_gb > MAX_ALLOWED_GB_FOR_MASK:
    raise MemoryError(
        f"Mask too large: estimated {mask_gb:.2f} GB.\n"
        "Proseg results are already available. To make mask, set PIXEL_SIZE=2.0 or 4.0."
    )

transform = from_origin(
    west=minx,
    north=maxy,
    xsize=PIXEL_SIZE,
    ysize=PIXEL_SIZE,
)

print("\nRasterizing polygons...")

mask = features.rasterize(
    shapes=valid_geoms,
    out_shape=(height, width),
    fill=0,
    transform=transform,
    dtype=np.uint32,
    all_touched=False,
)

del valid_geoms
del gdf
gc.collect()

print("\nMask info:")
print(f"dtype = {mask.dtype}")
print(f"min   = {mask.min()}")
print(f"max   = {mask.max()}")
print(f"nonzero pixels = {np.count_nonzero(mask):,}")

print("\nSaving TIFF...")

tiff.imwrite(str(OUT_TIFF), mask, compression="zlib", photometric="minisblack", bigtiff=True)
tiff.imwrite(str(OUT_TIFF_COPY), mask, compression="zlib", photometric="minisblack", bigtiff=True)

preview = (mask > 0).astype(np.uint8) * 255
tiff.imwrite(str(PREVIEW_TIFF), preview, compression="zlib", photometric="minisblack", bigtiff=True)

del mask
del preview
gc.collect()


print("\nFinal output check:")

final_expected = [
    SPATIALDATA_DIR,
    final_polygon_path,
    OUT_TIFF,
    OUT_TIFF_COPY,
    PREVIEW_TIFF,
]

for p in final_expected:
    status = "OK" if p.exists() else "MISSING"
    kind = "DIR" if p.is_dir() else "FILE" if p.is_file() else "----"
    print(f"{status:8s} | {kind:4s} | {p}")

print("\n========== All Done ==========")
