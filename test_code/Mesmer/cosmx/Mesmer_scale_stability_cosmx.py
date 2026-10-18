#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mesmer_scale_stability_cosmx.py
===============================
单独针对 Mesmer 的“尺度稳定性”实验：
  1) 把多个 FOV（默认 51~68）拼成一张大图（composite 与 CellLabels-GT 一起拼）；
  2) 以拼图中占用区域的质心为中心，取【同心嵌套】的方形裁剪：
       像素边长 5000 / 8000 / 10000 / 15000 / 全图（可 --crop-sizes 自定义）；
  3) 每个裁剪跑一次 Mesmer，对同样裁剪的 GT 打分（复用 test.py 指标）；
  4) 输出“指标 vs 裁剪面积/细胞数”的表 + 折线图，观察随细胞数增多分割是否稳定。

原理：Mesmer 的 predict 会按 image_mpp 把图缩放到训练分辨率(0.5µm/px)再内部切块，
所以理论上对裁剪大小应近似不变。这个实验就是实测它到底稳不稳。

⚠️ 混杂因素：裁剪边界会切断边缘细胞，小裁剪里“被切断细胞”占比更高，本身就会拉低小裁剪
的分数（与 Mesmer 无关）。用 --exclude-border 把碰到裁剪边缘的 GT/预测实例都剔除，才是
干净地测 Mesmer 的尺度行为。建议两种都跑对比。

拼图布局
--------
--layout grid（默认）：把 FOV 无缝排成近似方阵（稠密、无空隙、稳，最适合“细胞数 vs 稳定性”）。
--layout positions：用 Pancreas_fov_positions_file.csv 的全局坐标做物理拼接（更真实，但可能有空隙；
                    列名/单位不确定时不如 grid 稳）。

用法
----
    python mesmer_scale_stability_cosmx.py \
        --dep-dir /path/to/cellpose_files \
        --data-root /path/to/CosMx_Pancreas --fov-start 51 --fov-end 68 \
        --image-mpp 0.12028 --exclude-border \
        --out-dir ./mesmer_scale

    # 复用已存的按尺寸命名的预测（mesmer_scale_{size}_masks.tif），只评分/画图
    python mesmer_scale_stability_cosmx.py --dep-dir ... --data-root ... \
        --reuse-pred --pred-dir ./mesmer_scale
"""

import os
import sys
import json
import glob
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _resolve_dep_dir():
    env = os.environ.get("COSMX_BENCH_DIR")
    if env and os.path.isdir(env):
        return os.path.abspath(env)
    for i, a in enumerate(sys.argv):
        if a == "--dep-dir" and i + 1 < len(sys.argv):
            return os.path.abspath(sys.argv[i + 1])
        if a.startswith("--dep-dir="):
            return os.path.abspath(a.split("=", 1)[1])
    return None


_dep_dir = _resolve_dep_dir()
for _d in (_dep_dir, HERE, os.getcwd()):
    if _d and _d not in sys.path:
        sys.path.insert(0, _d)

try:
    import CosMx_single_fov as single
    import CosMx_benchplus as agg
    import CosMx_mesmer as mm
except ImportError as _e:
    sys.stderr.write(
        "\n[导入失败] 需要 cellpose_benchmark_cosmx.py / cellpose_benchmark_cosmx_multi.py / "
        "mesmer_benchmark_cosmx_multi.py 在同目录或 --dep-dir 指定的目录。\n"
        f"当前已尝试：{[d for d in (_dep_dir, HERE, os.getcwd()) if d]}\n原始错误：{_e}\n\n")
    raise


# --------------------------------------------------------------------------- #
# 拼图：composite(核/膜 2 通道 uint8) + CellLabels(int32, 全局唯一 ID)
# --------------------------------------------------------------------------- #
def _grid_layout(n):
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    return rows, cols


def _read_fov_offsets(positions_csv, fov_list, mpp, tile_h, tile_w):
    """positions 布局：返回 {fov: (row_px, col_px)}。尽量鲁棒地解析列名/单位。"""
    import pandas as pd
    df = pd.read_csv(positions_csv)
    low = {c.lower(): c for c in df.columns}
    fov_col = next((low[k] for k in low if k in ("fov", "fov_id", "fovid")), None)
    xcol = next((low[k] for k in low if k.startswith("x")), None)
    ycol = next((low[k] for k in low if k.startswith("y")), None)
    if not (fov_col and xcol and ycol):
        raise ValueError(f"fov_positions 列名无法识别：{list(df.columns)}")
    in_mm = ("mm" in xcol.lower()) or ("mm" in ycol.lower())
    to_px = (1000.0 / mpp) if in_mm else 1.0
    off = {}
    for _, r in df.iterrows():
        f = int(r[fov_col])
        if f in fov_list:
            off[f] = (float(r[ycol]) * to_px, float(r[xcol]) * to_px)
    if not off:
        raise ValueError("fov_positions 里没有匹配到指定的 FOV")
    ys = [v[0] for v in off.values()]; xs = [v[1] for v in off.values()]
    y0, x0 = min(ys), min(xs)
    return {f: (int(round(y - y0)), int(round(x - x0))) for f, (y, x) in off.items()}


def build_mosaic(bench, args, fov_list):
    """返回 nuc(uint8 HxW), mem(uint8 HxW), gt(int32 HxW), info。"""
    # 先探一张，拿 tile 尺寸
    comp0, lab0 = single.resolve_paths(args.data_root, fov_list[0],
                                       args.composite_dir, args.labels_dir)
    c0 = single.load_composite(comp0)
    tile_h, tile_w = c0.shape[:2]
    print(f"[mosaic] tile size = {tile_h}x{tile_w} | layout={args.layout}")

    if args.layout == "positions":
        pos = args.positions or os.path.join(args.data_root, "Pancreas_fov_positions_file.csv")
        offsets = _read_fov_offsets(pos, set(fov_list), args.image_mpp, tile_h, tile_w)
        H = max(y for y, _ in offsets.values()) + tile_h
        W = max(x for _, x in offsets.values()) + tile_w
        place = {f: offsets[f] for f in fov_list if f in offsets}
    else:  # grid
        rows, cols = _grid_layout(len(fov_list))
        H, W = rows * tile_h, cols * tile_w
        place = {f: ((i // cols) * tile_h, (i % cols) * tile_w)
                 for i, f in enumerate(fov_list)}
        print(f"[mosaic] grid = {rows}x{cols}")

    print(f"[mosaic] canvas = {H}x{W} ({H*W/1e6:.0f} MP)")
    nuc = np.zeros((H, W), np.uint8)
    mem = np.zeros((H, W), np.uint8)
    gt = np.zeros((H, W), np.int32)
    next_id = 0
    placed = 0
    for f in fov_list:
        if f not in place:
            print(f"  [skip] FOV {f} 无位置")
            continue
        try:
            cpath, lpath = single.resolve_paths(args.data_root, f,
                                                args.composite_dir, args.labels_dir)
            comp = single.load_composite(cpath)
            lab = bench.read_tif_mask(lpath)
        except Exception as e:
            print(f"  [skip] FOV {f}: {e}")
            continue
        y0, x0 = place[f]
        h, w = comp.shape[:2]
        y1, x1 = min(y0 + h, H), min(x0 + w, W)
        h, w = y1 - y0, x1 - x0
        R, G, B = comp[:h, :w, 0], comp[:h, :w, 1], comp[:h, :w, 2]
        nuc[y0:y1, x0:x1] = np.clip(B, 0, 255).astype(np.uint8)
        mem[y0:y1, x0:x1] = np.clip(np.maximum(R, G), 0, 255).astype(np.uint8)
        lab = lab[:h, :w]
        # 偏移 ID 保证全局唯一
        m = lab > 0
        gt_block = np.zeros_like(lab, dtype=np.int32)
        if m.any():
            gt_block[m] = lab[m].astype(np.int32) + next_id
            next_id += int(lab.max())
        gt[y0:y1, x0:x1] = gt_block
        placed += 1
    print(f"[mosaic] 放入 {placed}/{len(fov_list)} 个 FOV，GT 细胞总数≈{len(bench.get_instance_ids(gt))}")
    # 占用区域质心（用于居中裁剪）
    ys, xs = np.where(nuc > 0)
    if ys.size:
        cy, cx = int(ys.mean()), int(xs.mean())
    else:
        cy, cx = H // 2, W // 2
    return nuc, mem, gt, dict(H=H, W=W, cy=cy, cx=cx, tile=(tile_h, tile_w))


# --------------------------------------------------------------------------- #
# 裁剪 & 边界剔除
# --------------------------------------------------------------------------- #
def centered_crop_bounds(H, W, cy, cx, size):
    if size >= H and size >= W:
        return 0, H, 0, W
    sh, sw = min(size, H), min(size, W)
    y0 = int(np.clip(cy - sh // 2, 0, H - sh))
    x0 = int(np.clip(cx - sw // 2, 0, W - sw))
    return y0, y0 + sh, x0, x0 + sw


def drop_border_objects(lab):
    """剔除碰到图像四条边的实例（切断的细胞）。"""
    ids = set(np.unique(np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]])))
    ids.discard(0)
    if not ids:
        return lab
    out = lab.copy()
    out[np.isin(out, list(ids))] = 0
    return out


# --------------------------------------------------------------------------- #
# 单个裁剪评分（复用 agg 组件，等价于对该裁剪算一次 micro）
# --------------------------------------------------------------------------- #
def score(bench, gt_crop, pred_crop, iou_threshold, boundary_mode):
    overlaps = bench.compute_sparse_overlaps(gt_crop, pred_crop)
    comp = agg.pixel_components(bench, gt_crop, pred_crop, boundary_mode=boundary_mode)
    inst = agg.instance_components(bench, overlaps, iou_threshold=iou_threshold)
    aji_num, aji_den = agg.aji_components(bench, overlaps, gt_crop, pred_crop)
    mda_map = agg.mda_components(bench, overlaps)
    m = agg.per_fov_final_metrics(bench, comp, inst, aji_num, aji_den, mda_map)
    m["gt_cells"] = len(overlaps.gt_ids)
    m["pred_cells"] = len(overlaps.pred_ids)
    return m


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Mesmer 尺度稳定性：拼图 + 同心嵌套裁剪")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--dep-dir", default=None)
    ap.add_argument("--fov-start", type=int, default=51)
    ap.add_argument("--fov-end", type=int, default=68)
    ap.add_argument("--fovs", type=int, nargs="*", default=None)
    ap.add_argument("--composite-dir", default=None)
    ap.add_argument("--labels-dir", default=None)
    ap.add_argument("--test-py", default=None)
    ap.add_argument("--layout", choices=["grid", "positions"], default="grid")
    ap.add_argument("--positions", default=None, help="fov_positions csv（layout=positions 时）")
    ap.add_argument("--crop-sizes", type=int, nargs="+", default=[5000, 8000, 10000, 15000],
                    help="方形裁剪边长(px)，会自动再加一个‘全图’")
    ap.add_argument("--exclude-border", action="store_true",
                    help="剔除碰到裁剪边缘的实例（推荐，排除切断细胞的混杂）")
    # Mesmer
    ap.add_argument("--image-mpp", type=float, default=0.12028)
    ap.add_argument("--compartment", default="whole-cell", choices=["whole-cell", "nuclear", "both"])
    ap.add_argument("--batch-size", type=int, default=4)
    # 通用
    ap.add_argument("--iou-threshold", type=float, default=0.5)
    ap.add_argument("--boundary-mode", default="thick")
    ap.add_argument("--out-dir", default="./mesmer_scale")
    ap.add_argument("--reuse-pred", action="store_true")
    ap.add_argument("--pred-dir", default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    test_py = args.test_py
    if not test_py:
        for _d in (args.dep_dir, _dep_dir, HERE, os.getcwd()):
            if _d and os.path.exists(os.path.join(_d, "test.py")):
                test_py = os.path.join(_d, "test.py"); break
        test_py = test_py or os.path.join(HERE, "test.py")
    bench = single.load_metric_module(test_py)

    fov_list = args.fovs if args.fovs else list(range(args.fov_start, args.fov_end + 1))
    nuc, mem, gt, info = build_mosaic(bench, args, fov_list)
    H, W, cy, cx = info["H"], info["W"], info["cy"], info["cx"]

    # 裁剪尺寸：去重、升序、加“全图”
    sizes = sorted(set(int(s) for s in args.crop_sizes if s > 0))
    full = max(H, W)
    sizes = [s for s in sizes if s < full] + [full]

    runner = None
    if not args.reuse_pred:
        runner = mm.MesmerRunner(image_mpp=args.image_mpp, compartment=args.compartment,
                                 batch_size=args.batch_size)

    import tifffile
    rows = []
    for size in sizes:
        tag = "full" if size == full else str(size)
        y0, y1, x0, x1 = centered_crop_bounds(H, W, cy, cx, size)
        gt_c = gt[y0:y1, x0:x1].copy()
        area_mp = (y1 - y0) * (x1 - x0) / 1e6
        print(f"\n===== crop {tag} : {y1-y0}x{x1-x0} ({area_mp:.0f} MP) @({y0},{x0}) =====")

        # 预测
        pred_c = None
        if args.reuse_pred:
            hits = sorted(glob.glob(os.path.join(args.pred_dir or args.out_dir,
                                                 f"*scale_{tag}_*mask*.tif")))
            if hits:
                pred_c = bench.read_tif_mask(hits[0])
                print(f"  [reuse] {os.path.basename(hits[0])}")
        if pred_c is None:
            X = np.stack([nuc[y0:y1, x0:x1], mem[y0:y1, x0:x1]], axis=-1)[np.newaxis].astype(np.float32)
            pred_c = runner.segment(X)
            try:
                tifffile.imwrite(os.path.join(args.out_dir, f"mesmer_scale_{tag}_masks.tif"),
                                 pred_c.astype(np.int32))
            except Exception as e:
                print(f"  [warn] 保存失败: {e}")

        if pred_c.shape != gt_c.shape:
            from skimage.transform import resize
            pred_c = resize(pred_c, gt_c.shape, order=0, preserve_range=True,
                            anti_aliasing=False).astype(pred_c.dtype)

        if args.exclude_border:
            gt_s = drop_border_objects(gt_c)
            pred_s = drop_border_objects(pred_c)
        else:
            gt_s, pred_s = gt_c, pred_c

        m = score(bench, gt_s, pred_s, args.iou_threshold, args.boundary_mode)
        row = dict(crop=tag, side_px=(y1 - y0), area_MP=round(area_mp, 1),
                   gt_cells=m["gt_cells"], pred_cells=m["pred_cells"],
                   cell_density_per_MP=round(m["gt_cells"] / max(area_mp, 1e-6), 1))
        for k in ["instance_f1", "precision", "recall", "PQ", "DQ", "SQ", "AJI",
                  "mDA_0.50_0.95", "dice", "iou", "boundary_f1"]:
            row[k] = round(float(m[k]), 4)
        rows.append(row)
        print(f"  gt={row['gt_cells']} pred={row['pred_cells']} "
              f"F1={row['instance_f1']:.3f} AJI={row['AJI']:.3f} PQ={row['PQ']:.3f}")

    import pandas as pd
    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out_dir, "mesmer_scale_stability.csv")
    df.to_csv(csv_path, index=False)

    # 稳定性小结：关键指标的 变异系数 CV=std/mean 与 极差
    stab_cols = ["instance_f1", "AJI", "PQ", "SQ", "dice", "iou"]
    stab = {}
    for c in stab_cols:
        v = df[c].to_numpy(dtype=float)
        stab[c] = dict(mean=round(v.mean(), 4), std=round(v.std(ddof=1), 4) if len(v) > 1 else 0.0,
                       range=round(v.max() - v.min(), 4),
                       cv=round(v.std(ddof=1) / (v.mean() + 1e-9), 4) if len(v) > 1 else 0.0)
    with open(os.path.join(args.out_dir, "stability_summary.json"), "w", encoding="utf-8") as f:
        json.dump(dict(exclude_border=args.exclude_border, image_mpp=args.image_mpp,
                       layout=args.layout, sizes=sizes, stability=stab), f,
                  ensure_ascii=False, indent=2)

    # 折线图：指标 vs 裁剪边长（x 轴同时标注细胞数）
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5.5))
        x = df["side_px"].to_numpy()
        for c, mk in [("instance_f1", "o"), ("AJI", "s"), ("PQ", "^"),
                      ("SQ", "d"), ("iou", "v")]:
            ax.plot(x, df[c], marker=mk, label=c.replace("instance_f1", "F1"))
        ax.set_xlabel("crop side (px)   —  cells: " +
                      ", ".join(f"{s}px:{n}" for s, n in zip(df["side_px"], df["gt_cells"])),
                      fontsize=8)
        ax.set_ylabel("metric (micro)")
        ax.set_ylim(0, 1)
        ax.set_title(f"Mesmer scale stability (exclude_border={args.exclude_border}, "
                     f"mpp={args.image_mpp})")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        png = os.path.join(args.out_dir, "mesmer_scale_stability.png")
        fig.savefig(png, dpi=130)
        plt.close(fig)
        print(f"[plot] {png}")
    except Exception as e:
        print(f"[plot] 跳过：{e}")

    print("\n" + "=" * 88)
    print(df.to_string(index=False))
    print("=" * 88)
    print("稳定性(跨裁剪尺寸)  CV=std/mean，越小越稳：")
    for c in stab_cols:
        print(f"  {c:12s} mean={stab[c]['mean']:.4f}  std={stab[c]['std']:.4f}  "
              f"range={stab[c]['range']:.4f}  CV={stab[c]['cv']:.4f}")
    if not args.exclude_border:
        print("★ 提示：未开 --exclude-border，小裁剪的低分可能来自边缘切断细胞而非 Mesmer 本身；"
              "建议再跑一次 --exclude-border 对比。")
    print(f"\n[saved]\n  {csv_path}\n  "
          f"{os.path.join(args.out_dir, 'stability_summary.json')}")


if __name__ == "__main__":
    main()

'''
nohup bash -c '
    python Mesmer_scale_stability_cosmx.py \
    --dep-dir /data/qiuyijia/cellpose \
    --data-root /data/qiuyijia/dataset/Pancreas-CosMx-WTx-FlatFiles --fov-start 51 --fov-end 68 \
    --test-py /data/qiuyijia/eval/test.py \
    --image-mpp 0.12028 --exclude-border \
    --out-dir /data/qiuyijia/Mesmer/CosMx_scale_stability_output
' > output.log 2>&1 &
'''