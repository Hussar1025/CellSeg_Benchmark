import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend, no popup window
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pathlib import Path
from skimage import io
from skimage.measure import regionprops_table
from stardist.models import StarDist2D
from csbdeep.utils import normalize

# ---------------------------------------------
#  CONFIG: edit paths and parameters here
# ---------------------------------------------
INPUT_IMAGE_PATH  = "yourpath/dataset/morphology_mip_small.tif"   # input image path
OUTPUT_DIR        = "yourpath/stardist"               # output folder

# segmentation parameters
PROB_THRESH       = 0.5      # cell confidence threshold, higher is stricter (0~1)
NMS_THRESH        = 0.4      # non-maximum suppression overlap threshold (0~1)
NORM_PERCENTILE   = (1, 99.8)  # image normalization percentiles

# visualization parameters
SHOW_LABELS       = True     # show cell ids on the segmentation map (auto-skipped when cells>300)
CONTOUR_COLOR     = "cyan"   # contour color
CONTOUR_LINEWIDTH = 0.8      # contour line width
CENTER_MARKERSIZE = 2        # center marker size
# ---------------------------------------------


def load_image(path: str) -> np.ndarray:
    """Load image and normalize to RGB uint8."""
    img = io.imread(path)
    if img.ndim == 3 and img.shape[2] == 4:   # remove alpha channel
        img = img[..., :3]
    if img.ndim == 2:                          # convert grayscale to RGB
        img = np.stack([img] * 3, axis=-1)
    if img.dtype != np.uint8:
        img = (img / img.max() * 255).astype(np.uint8)
    return img


def run_stardist(img: np.ndarray, prob_thresh: float, nms_thresh: float,
                 norm_pct: tuple) -> tuple:
    """
    Load the StarDist HE pretrained model and run segmentation.

    Returns
    -------
    labels  : (H, W) int label map, each cell has a unique ID, background is 0
    details : dict, contains points / coord / prob / dist
    """
    print("[1/4] load StarDist pretrained model (2D_versatile_fluo) ...")
    model = StarDist2D.from_pretrained("2D_versatile_fluo")

    print("[2/4] normalize image ...")
    img_norm = normalize(img, *norm_pct, axis=(0, 1))

    print("[3/4] run cell segmentation ...")
    labels, details = model.predict_instances(
        img_norm,
        prob_thresh=prob_thresh,
        nms_thresh=nms_thresh,
        n_tiles=(8, 8, 1),
    )
    print(f"      detected cell count: {len(details['points'])}")
    return labels, details


def extract_cell_info(labels: np.ndarray, details: dict) -> pd.DataFrame:
    """
    Extract each cell's coordinates and morphological features.

    Output DataFrame columns:
        cell_id            : unique cell id (pixel value in the label map)
        y_centroid         : centroid y from regionprops (row)
        x_centroid         : centroid x from regionprops (column)
        y_stardist         : StarDist raw predicted center y
        x_stardist         : StarDist raw predicted center x
        prob               : confidence score
        area               : cell area (pixel count)
        major_axis_length  : major axis length
        minor_axis_length  : minor axis length
        eccentricity       : eccentricity (0=circle, 1=line)
        orientation        : major axis angle (radians)
        bbox_*             : bounding box coordinates
    """
    points = details["points"]   # (N, 2)  [y, x]
    probs  = details["prob"]     # (N,)

    props = regionprops_table(
        labels,
        properties=(
            "label", "area", "centroid",
            "major_axis_length", "minor_axis_length",
            "eccentricity", "orientation", "bbox",
        ),
    )
    df = pd.DataFrame(props)
    df.rename(columns={
        "label":      "cell_id",
        "centroid-0": "y_centroid",
        "centroid-1": "x_centroid",
        "bbox-0":     "bbox_min_row",
        "bbox-1":     "bbox_min_col",
        "bbox-2":     "bbox_max_row",
        "bbox-3":     "bbox_max_col",
    }, inplace=True)

    # StarDist sorts by confidence descending; align prob via cell_id
    prob_map = {i + 1: float(p) for i, p in enumerate(probs)}
    df["prob"] = df["cell_id"].map(prob_map)

    # StarDist raw center coordinates
    n = min(len(points), len(df))
    df["y_stardist"] = np.nan
    df["x_stardist"] = np.nan
    df.loc[df.index[:n], "y_stardist"] = points[:n, 0]
    df.loc[df.index[:n], "x_stardist"] = points[:n, 1]

    # reorder columns, putting coordinates and confidence first
    cols = ["cell_id", "y_centroid", "x_centroid", "y_stardist", "x_stardist",
            "prob", "area", "major_axis_length", "minor_axis_length",
            "eccentricity", "orientation",
            "bbox_min_row", "bbox_min_col", "bbox_max_row", "bbox_max_col"]
    return df[cols]


def save_overlay(img, labels, details, out_path,
                 show_labels, contour_color, lw, ms):
    """Save the 3-panel overlay: original | translucent labels | contours+center points."""
    from stardist import random_label_cmap

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)

    # subplot 1: original
    axes[0].imshow(img)
    axes[0].set_title("Original HE image", fontsize=12)
    axes[0].axis("off")

    # subplot 2: label overlay (translucent random colors)
    cmap = random_label_cmap()
    axes[1].imshow(img)
    axes[1].imshow(labels, cmap=cmap, alpha=0.5, interpolation="nearest")
    axes[1].set_title(f"Segmentation result ({len(details['points'])} cells)", fontsize=12)
    axes[1].axis("off")

    # subplot 3: contours + center points
    axes[2].imshow(img)
    coords = details["coord"]   # (N, 2, n_rays)
    points = details["points"]  # (N, 2)

    for i in range(len(coords)):
        ys = np.append(coords[i, 0, :], coords[i, 0, 0])  # close the polygon
        xs = np.append(coords[i, 1, :], coords[i, 1, 0])
        axes[2].plot(xs, ys, color=contour_color, linewidth=lw, alpha=0.85)

    axes[2].scatter(points[:, 1], points[:, 0],
                    s=ms, c="red", linewidths=0, alpha=0.7, zorder=5)

    if show_labels and len(points) <= 300:
        for idx, (cy, cx) in enumerate(points):
            axes[2].text(cx, cy, str(idx + 1),
                         color="yellow", fontsize=3.5,
                         ha="center", va="center", fontweight="bold")

    legend_elements = [
        mpatches.Patch(color=contour_color, label="Cell contour"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="red", markersize=4, label="Cell center"),
    ]
    axes[2].legend(handles=legend_elements, loc="lower right",
                   fontsize=8, framealpha=0.7)
    axes[2].set_title("Contours and center points", fontsize=12)
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"      overlay saved -> {out_path}")


def save_label_image(labels, out_path):
    """Save the pure label image (unique color per cell)."""
    from stardist import random_label_cmap
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    ax.imshow(labels, cmap=random_label_cmap(), interpolation="nearest")
    ax.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"      label image saved -> {out_path}")


def main():
    if not os.path.isfile(INPUT_IMAGE_PATH):
        raise FileNotFoundError(f"input image not found: {INPUT_IMAGE_PATH}")

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(INPUT_IMAGE_PATH).stem

    # load image
    print(f"[0/4] load image: {INPUT_IMAGE_PATH}")
    img = load_image(INPUT_IMAGE_PATH)
    print(f"      image shape: {img.shape}  dtype: {img.dtype}")

    # segmentation
    labels, details = run_stardist(img, PROB_THRESH, NMS_THRESH, NORM_PERCENTILE)

    # extract info and save
    print("[4/4] extract cell info and save results ...")
    df = extract_cell_info(labels, details)

    csv_path = out_dir / f"{stem}_cell_info.csv"
    df.to_csv(csv_path, index=False, float_format="%.2f")
    print(f"      cell info CSV -> {csv_path}")

    overlay_path = out_dir / f"{stem}_overlay.png"
    save_overlay(img, labels, details, str(overlay_path),
                 SHOW_LABELS, CONTOUR_COLOR, CONTOUR_LINEWIDTH, CENTER_MARKERSIZE)

    label_path = out_dir / f"{stem}_labels.png"
    save_label_image(labels, str(label_path))

    # summary
    print("\n" + "=" * 50)
    print("Segmentation done! Output summary:")
    print(f"  total cells     : {len(df)}")
    print(f"  mean confidence : {df['prob'].mean():.3f}")
    print(f"  mean area (px^2): {df['area'].mean():.1f}")
    print(f"  output dir      : {out_dir.resolve()}")
    print("=" * 50)


if __name__ == "__main__":
    main()
