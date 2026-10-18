import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # 非交互后端，不弹窗
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pathlib import Path
from skimage import io
from skimage.measure import regionprops_table
from stardist.models import StarDist2D
from csbdeep.utils import normalize

# ─────────────────────────────────────────────
#  CONFIG：修改这里的路径和参数
# ─────────────────────────────────────────────
INPUT_IMAGE_PATH  = "/data/qiuyijia/dataset/morphology_mip_small.tif"   # 输入图像路径
OUTPUT_DIR        = "/data/qiuyijia/stardist"               # 输出文件夹

# 分割参数
PROB_THRESH       = 0.5      # 细胞置信度阈值，越高越严格（0~1）
NMS_THRESH        = 0.4      # 非极大值抑制重叠阈值（0~1）
NORM_PERCENTILE   = (1, 99.8)  # 图像归一化百分位

# 可视化参数
SHOW_LABELS       = True     # 是否在分割图上显示细胞编号（细胞数>300时自动跳过）
CONTOUR_COLOR     = "cyan"   # 轮廓颜色
CONTOUR_LINEWIDTH = 0.8      # 轮廓线宽
CENTER_MARKERSIZE = 2        # 中心点大小
# ─────────────────────────────────────────────


def load_image(path: str) -> np.ndarray:
    """加载图像，统一转为 RGB uint8"""
    img = io.imread(path)
    if img.ndim == 3 and img.shape[2] == 4:   # 去除 alpha 通道
        img = img[..., :3]
    if img.ndim == 2:                          # 灰度图转 RGB
        img = np.stack([img] * 3, axis=-1)
    if img.dtype != np.uint8:
        img = (img / img.max() * 255).astype(np.uint8)
    return img


def run_stardist(img: np.ndarray, prob_thresh: float, nms_thresh: float,
                 norm_pct: tuple) -> tuple:
    """
    加载 StarDist HE 预训练模型并执行分割

    Returns
    -------
    labels  : (H, W) int 标签图，每个细胞有唯一 ID，背景为 0
    details : dict，含 points / coord / prob / dist
    """
    print("[1/4] 加载 StarDist 预训练模型 (2D_versatile_fluo) ...")
    model = StarDist2D.from_pretrained("2D_versatile_fluo")

    print("[2/4] 归一化图像 ...")
    img_norm = normalize(img, *norm_pct, axis=(0, 1))

    print("[3/4] 执行细胞分割 ...")
    labels, details = model.predict_instances(
        img_norm,
        prob_thresh=prob_thresh,
        nms_thresh=nms_thresh,
        n_tiles=(8, 8, 1),
    )
    print(f"      检测到细胞数量：{len(details['points'])}")
    return labels, details


def extract_cell_info(labels: np.ndarray, details: dict) -> pd.DataFrame:
    """
    提取每个细胞的坐标与形态学特征

    输出 DataFrame 列说明：
        cell_id            : 细胞唯一编号（对应标签图中的像素值）
        y_centroid         : regionprops 计算的中心 y 坐标（行）
        x_centroid         : regionprops 计算的中心 x 坐标（列）
        y_stardist         : StarDist 原始预测中心 y
        x_stardist         : StarDist 原始预测中心 x
        prob               : 置信度分数
        area               : 细胞面积（像素数）
        major_axis_length  : 长轴长度
        minor_axis_length  : 短轴长度
        eccentricity       : 偏心率（0=圆形，1=线形）
        orientation        : 长轴方向角（弧度）
        bbox_*             : 外接矩形坐标
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

    # StarDist 按置信度降序排列，用 cell_id 对齐 prob
    prob_map = {i + 1: float(p) for i, p in enumerate(probs)}
    df["prob"] = df["cell_id"].map(prob_map)

    # StarDist 原始中心坐标
    n = min(len(points), len(df))
    df["y_stardist"] = np.nan
    df["x_stardist"] = np.nan
    df.loc[df.index[:n], "y_stardist"] = points[:n, 0]
    df.loc[df.index[:n], "x_stardist"] = points[:n, 1]

    # 调整列顺序，把坐标和置信度放在前面
    cols = ["cell_id", "y_centroid", "x_centroid", "y_stardist", "x_stardist",
            "prob", "area", "major_axis_length", "minor_axis_length",
            "eccentricity", "orientation",
            "bbox_min_row", "bbox_min_col", "bbox_max_row", "bbox_max_col"]
    return df[cols]


def save_overlay(img, labels, details, out_path,
                 show_labels, contour_color, lw, ms):
    """保存三联叠加图：原图 | 半透明标签 | 轮廓+中心点"""
    from stardist import random_label_cmap

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)

    # 子图1：原图
    axes[0].imshow(img)
    axes[0].set_title("原始 HE 图像", fontsize=12)
    axes[0].axis("off")

    # 子图2：标签叠加（半透明随机配色）
    cmap = random_label_cmap()
    axes[1].imshow(img)
    axes[1].imshow(labels, cmap=cmap, alpha=0.5, interpolation="nearest")
    axes[1].set_title(f"分割结果（{len(details['points'])} 个细胞）", fontsize=12)
    axes[1].axis("off")

    # 子图3：轮廓 + 中心点
    axes[2].imshow(img)
    coords = details["coord"]   # (N, 2, n_rays)
    points = details["points"]  # (N, 2)

    for i in range(len(coords)):
        ys = np.append(coords[i, 0, :], coords[i, 0, 0])  # 闭合多边形
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
        mpatches.Patch(color=contour_color, label="细胞轮廓"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="red", markersize=4, label="细胞中心"),
    ]
    axes[2].legend(handles=legend_elements, loc="lower right",
                   fontsize=8, framealpha=0.7)
    axes[2].set_title("轮廓与中心点", fontsize=12)
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"      叠加图已保存  → {out_path}")


def save_label_image(labels, out_path):
    """保存纯标签图（每个细胞唯一颜色）"""
    from stardist import random_label_cmap
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    ax.imshow(labels, cmap=random_label_cmap(), interpolation="nearest")
    ax.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"      标签图已保存  → {out_path}")


def main():
    if not os.path.isfile(INPUT_IMAGE_PATH):
        raise FileNotFoundError(f"找不到输入图像：{INPUT_IMAGE_PATH}")

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(INPUT_IMAGE_PATH).stem

    
    # 加载图像
    print(f"[0/4] 加载图像：{INPUT_IMAGE_PATH}")
    img = load_image(INPUT_IMAGE_PATH)
    print(f"      图像尺寸：{img.shape}  dtype：{img.dtype}")

    # 分割
    labels, details = run_stardist(img, PROB_THRESH, NMS_THRESH, NORM_PERCENTILE)

    # 提取信息并保存
    print("[4/4] 提取细胞信息并保存结果 ...")
    df = extract_cell_info(labels, details)

    csv_path = out_dir / f"{stem}_cell_info.csv"
    df.to_csv(csv_path, index=False, float_format="%.2f")
    print(f"      细胞信息 CSV  → {csv_path}")

    overlay_path = out_dir / f"{stem}_overlay.png"
    save_overlay(img, labels, details, str(overlay_path),
                 SHOW_LABELS, CONTOUR_COLOR, CONTOUR_LINEWIDTH, CENTER_MARKERSIZE)

    label_path = out_dir / f"{stem}_labels.png"
    save_label_image(labels, str(label_path))

    # 摘要
    print("\n" + "=" * 50)
    print("分割完成！输出摘要：")
    print(f"  细胞总数       : {len(df)}")
    print(f"  平均置信度     : {df['prob'].mean():.3f}")
    print(f"  平均面积 (px²) : {df['area'].mean():.1f}")
    print(f"  输出目录       : {out_dir.resolve()}")
    print("=" * 50)


if __name__ == "__main__":
    main()