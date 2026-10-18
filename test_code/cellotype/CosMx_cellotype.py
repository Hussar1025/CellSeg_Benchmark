#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cellotype_benchmark_cosmx_multi.py
==================================
在多个 FOV（默认 51~68）上用 CelloType 的【TissueNet 预训练模型，零样本，不微调】做整细胞
分割，并对 CosMx CellLabels(GT) 做 benchmark。汇总方式与 cellpose / stardist 版完全一致：
micro（池化）为主口径，macro(mean±std) 为稳健性口径；指标复用 test.py；micro 聚合复用
cellpose_benchmark_cosmx_multi 里那套（已用“拼大图金标准”验证过，差 ~1e-7）。

为什么不微调（重要）
--------------------
* 公平性：Cellpose(cpsam)、StarDist(2D_versatile_fluo) 都是零样本，CelloType 也零样本，
  四方法才在同一起跑线。只给 CelloType 微调=给它别人没有的信息优势。
* 泄漏：你只有同一张切片的 51~68 个 FOV，切它自己训自己测会系统性高估。
* GT 是银标准：CellLabels 是 NanoString 算法输出，在它上面训 CelloType 只是“模仿厂商”。
* 作者原话也支持：TissueNet/Cellpose 预训练分割模型“只要新图含核/膜通道即可迁移”，
  CosMx composite 正好有 DAPI(核)+PanCK/CD45(膜)。
CelloType 原生就是整细胞分割 → 直接对整细胞 CellLabels 比，无需 StarDist 那种膨胀超参，
与 Cellpose 完全同口径可横向比。

推理路径（两种）
----------------
CelloType 官方稳定 API： CelloTypePredictor(model_path, config_path, ...).predict(img)->[H,W]。
大图（4256²）官方推荐用仓库自带 sahi_inference.py 做 512 滑窗，但那套 SAHI 封装类随仓库
版本而变。因此本脚本：
  --infer tile （默认）：只依赖稳定的 .predict()，脚本自己做 512 切块 + 实例拼接（已单测）。
  --infer whole：整图一次 predict（仅适合小图/大显存；Detectron2 会按短边缩放，可能掉小细胞）。
  或者：你用官方 sahi_inference.py 先把每个 FOV 的掩膜存成
        cellotype_*_F0XX_masks.tif，再用 --reuse-pred 只做评分（最贴合官方、最稳）。

依赖
----
    CelloType (github.com/tanlabcode/CelloType) + Detectron2 + MaskDINO；需要下载
    预训练权重 models/tissuenet_model_0019999.pth 与 config
    configs/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml。CUDA GPU。
本脚本除 CelloType 推理外的全部环节（切块拼接、输入构造、多 FOV micro/macro 聚合、CSV/JSON）
已用合成数据跑通验证；推理在你的 AutoDL 上跑。

用法
----
    # 集成推理（tile 模式）
    python cellotype_benchmark_cosmx_multi.py \
        --dep-dir /path/to/cellpose_files \
        --data-root /path/to/CosMx_Pancreas --fov-start 51 --fov-end 68 \
        --model-path /path/CelloType/models/tissuenet_model_0019999.pth \
        --config-path /path/CelloType/configs/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml \
        --device cuda --out-dir ./cellotype_bench_multi

    # 用官方 SAHI 先出掩膜，这里只评分
    python cellotype_benchmark_cosmx_multi.py --dep-dir ... --data-root ... \
        --fov-start 51 --fov-end 68 --reuse-pred --pred-dir /path/to/cellotype_masks
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import sys
import json
import glob
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# 依赖定位（cellpose_benchmark_cosmx[_multi].py 可以在别的目录）
# --------------------------------------------------------------------------- #
def _resolve_dep_dir():
    env = os.environ.get("COSMX_BENCH_DIR")
    if env and os.path.isdir(env):
        return os.path.abspath(env)
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == "--dep-dir" and i + 1 < len(argv):
            return os.path.abspath(argv[i + 1])
        if a.startswith("--dep-dir="):
            return os.path.abspath(a.split("=", 1)[1])
    return None


_dep_dir = _resolve_dep_dir()
for _d in (_dep_dir, HERE, os.getcwd()):
    if _d and _d not in sys.path:
        sys.path.insert(0, _d)

try:
    import CosMx_single_fov as single       # 文件解析 / composite 加载 / 指标模块
    import CosMx_benchplus as agg      # 已验证的 micro 聚合 + 组件函数
except ImportError as _e:
    sys.stderr.write(
        "\n[导入失败] 找不到 cellpose_benchmark_cosmx.py / cellpose_benchmark_cosmx_multi.py。\n"
        "请用以下任一方式指定它们所在目录（三选一）：\n"
        "  1) 命令行：  --dep-dir /path/to/cellpose_files\n"
        "  2) 环境变量：export COSMX_BENCH_DIR=/path/to/cellpose_files\n"
        "  3) 把三个脚本放同一目录\n"
        f"当前已尝试：{[d for d in (_dep_dir, HERE, os.getcwd()) if d]}\n原始错误：{_e}\n\n")
    raise


# --------------------------------------------------------------------------- #
# CelloType 输入（TissueNet 约定：R=可选, G=膜, B=核）
# --------------------------------------------------------------------------- #
def build_cellotype_input(composite_rgb, channel_order="opt_mem_nuc"):
    """
    CosMx composite: R=CD45/GFAP, G=PanCK, B=DAPI。
    CelloType/TissueNet 约定通常是 [R=可选, G=膜, B=核]（DeepWiki 所述）。
    -> 膜 = max(CD45, PanCK) = max(R,G)；核 = DAPI = B；可选 = 0。
    ⚠️ 通道顺序请拿 CelloType 仓库 data/example/example_tissuenet.png 对一下，放反会掉点。
       如需调整用 --channel-order：
         opt_mem_nuc -> [0, membrane, nuclear]  (默认)
         nuc_mem_opt -> [nuclear, membrane, 0]
         mem_nuc_opt -> [membrane, nuclear, 0]
    返回 uint8 [H,W,3]（CelloType 读 PNG 即 uint8，归一化在模型内部做）。
    """
    R = composite_rgb[..., 0]; G = composite_rgb[..., 1]; B = composite_rgb[..., 2]
    membrane = np.maximum(R, G)
    nuclear = B
    zero = np.zeros_like(B)
    order = {
        "opt_mem_nuc": (zero, membrane, nuclear),
        "nuc_mem_opt": (nuclear, membrane, zero),
        "mem_nuc_opt": (membrane, nuclear, zero),
    }[channel_order]
    out = np.stack(order, axis=-1)
    return np.clip(out, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# 切块 + 实例拼接（只依赖稳定的 predict()；已单测）
# --------------------------------------------------------------------------- #
def _starts(size, tile, step):
    if size <= tile:
        return [0]
    s = list(range(0, size - tile + 1, step))
    if s[-1] != size - tile:
        s.append(size - tile)
    return s


def tile_predict_stitch(img_rgb, predict_fn, tile=512, overlap=128, merge_frac=0.5):
    """把大图切成 tile×tile（overlap 重叠）逐块 predict，再贪心拼成一张全局实例掩膜。
    贪心策略：先到先得；后续块若与已有实例大量重叠(>=merge_frac)则并入，否则新 id 填空白像素。
    overlap 建议 >= 最大细胞直径(胰腺~125px)，避免骑缝细胞被切成两个。
    注：这是对稳定 API 的便捷封装；更严谨的大图推理请用官方 SAHI(NMM 合并)。"""
    H, W = img_rgb.shape[:2]
    gmask = np.zeros((H, W), np.int32)
    nxt = 1
    step = max(1, tile - overlap)
    for y0 in _starts(H, tile, step):
        for x0 in _starts(W, tile, step):
            y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
            sub = img_rgb[y0:y1, x0:x1]
            local = np.asarray(predict_fn(sub))
            if local.ndim > 2:
                local = np.squeeze(local)
            g_sub = gmask[y0:y1, x0:x1]           # 视图，原地修改即改 gmask
            for lid in np.unique(local):
                if lid == 0:
                    continue
                region = local == lid
                nz = g_sub[region]
                nz = nz[nz > 0]
                assigned = False
                if nz.size:
                    vals, counts = np.unique(nz, return_counts=True)
                    top = int(vals[counts.argmax()])
                    if counts.max() / region.sum() >= merge_frac:
                        g_sub[region & (g_sub == 0)] = top
                        assigned = True
                if not assigned:
                    fill = region & (g_sub == 0)
                    if fill.sum() > 0:
                        g_sub[fill] = nxt
                        nxt += 1
    return gmask


# --------------------------------------------------------------------------- #
# CelloType 运行器（模型只加载一次，多 FOV 复用）
# --------------------------------------------------------------------------- #
class CelloTypeRunner:
    def __init__(self, model_path, config_path, confidence, max_det, device,
                 infer="tile", tile=512, overlap=128):
        self.model_path = model_path
        self.config_path = config_path
        self.confidence = confidence
        self.max_det = max_det
        self.device = device
        self.infer = infer
        self.tile = tile
        self.overlap = overlap
        self._predictor = None

    def _predictor_obj(self):
        if self._predictor is None:
            from cellotype.predict import CelloTypePredictor
            if not self.model_path or not self.config_path:
                raise SystemExit("需要 --model-path 与 --config-path（CelloType 权重与配置）。")
            print(f"[cellotype] 加载模型 {os.path.basename(self.model_path)} "
                  f"| conf={self.confidence} max_det={self.max_det} device={self.device}")
            self._predictor = CelloTypePredictor(
                model_path=self.model_path,
                confidence_thresh=self.confidence,
                max_det=self.max_det,
                device=self.device,
                config_path=self.config_path)
        return self._predictor

    def segment(self, img_rgb):
        pred = self._predictor_obj()
        if self.infer == "whole":
            m = np.asarray(pred.predict(img_rgb))
            return (np.squeeze(m) if m.ndim > 2 else m).astype(np.int32)
        # tile（默认）
        return tile_predict_stitch(img_rgb, lambda s: pred.predict(s),
                                   tile=self.tile, overlap=self.overlap).astype(np.int32)


# --------------------------------------------------------------------------- #
# 单 FOV
# --------------------------------------------------------------------------- #
def process_one_fov(bench, args, fov, runner):
    comp_path, lab_path = single.resolve_paths(
        args.data_root, fov, args.composite_dir, args.labels_dir)
    composite = single.load_composite(comp_path)
    gt = bench.read_tif_mask(lab_path)             # 整细胞 GT

    # 预测
    pred = None
    if args.reuse_pred:
        pat = os.path.join(args.pred_dir or args.out_dir, f"*cellotype*F{fov:03d}*.tif")
        hits = sorted(glob.glob(pat)) or sorted(
            glob.glob(os.path.join(args.pred_dir or args.out_dir, f"*F{fov:03d}*.tif")))
        if hits:
            pred = bench.read_tif_mask(hits[0])
            print(f"  [reuse] {os.path.basename(hits[0])}")
    if pred is None:
        cin = build_cellotype_input(composite, channel_order=args.channel_order)
        pred = runner.segment(cin)
        try:
            import tifffile
            os.makedirs(args.out_dir, exist_ok=True)
            outp = os.path.join(args.out_dir, f"cellotype_tissuenet_F{fov:03d}_masks.tif")
            tifffile.imwrite(outp, pred.astype(np.int32))
        except Exception as e:
            print(f"  [warn] 保存预测tif失败: {e}")

    # 对齐
    if gt.shape != pred.shape:
        from skimage.transform import resize
        print(f"  [warn] shape 不一致 gt={gt.shape} pred={pred.shape}，最近邻缩放 pred")
        pred = resize(pred, gt.shape, order=0, preserve_range=True,
                      anti_aliasing=False).astype(pred.dtype)

    # 一次 overlaps → 派生所有可累加量（复用已验证组件）
    overlaps = bench.compute_sparse_overlaps(gt, pred)
    comp = agg.pixel_components(bench, gt, pred, boundary_mode=args.boundary_mode)
    inst = agg.instance_components(bench, overlaps, iou_threshold=args.iou_threshold)
    aji_num, aji_den = agg.aji_components(bench, overlaps, gt, pred)
    mda_map = agg.mda_components(bench, overlaps)

    final = agg.per_fov_final_metrics(bench, comp, inst, aji_num, aji_den, mda_map)
    final.update(fov=fov,
                 gt_cell_count=len(overlaps.gt_ids),
                 pred_cell_count=len(overlaps.pred_ids))
    raw = dict(comp=comp, inst=inst, aji_num=aji_num, aji_den=aji_den, mda_map=mda_map)
    return final, raw


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="多FOV CelloType(TissueNet预训练,零样本) vs CellLabels benchmark")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--dep-dir", default=None,
                    help="cellpose_benchmark_cosmx[_multi].py 所在目录（不同目录时用）。等价于 env COSMX_BENCH_DIR。")
    ap.add_argument("--fov-start", type=int, default=51)
    ap.add_argument("--fov-end", type=int, default=68)
    ap.add_argument("--fovs", type=int, nargs="*", default=None)
    ap.add_argument("--composite-dir", default=None)
    ap.add_argument("--labels-dir", default=None)
    ap.add_argument("--test-py", default=None, help="test.py 路径；默认在 dep-dir/本脚本目录 里找")
    # CelloType
    ap.add_argument("--model-path", default=None,
                    help="CelloType 权重，如 .../models/tissuenet_model_0019999.pth")
    ap.add_argument("--config-path", default=None,
                    help="CelloType 配置，如 .../configs/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml")
    ap.add_argument("--confidence", type=float, default=0.4)
    ap.add_argument("--max-det", type=int, default=1000, help="每块最大检测数（密集组织可调大）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--channel-order", default="opt_mem_nuc",
                    choices=["opt_mem_nuc", "nuc_mem_opt", "mem_nuc_opt"],
                    help="三通道排布，需与 CelloType 预训练时一致（拿 example_tissuenet.png 核对）")
    ap.add_argument("--infer", choices=["tile", "whole"], default="tile")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=128, help="切块重叠，建议>=最大细胞直径(~125px)")
    # 通用
    ap.add_argument("--iou-threshold", type=float, default=0.5)
    ap.add_argument("--boundary-mode", default="thick")
    ap.add_argument("--out-dir", default="./cellotype_bench_multi")
    ap.add_argument("--reuse-pred", action="store_true", help="复用已存的 CelloType 掩膜(如官方SAHI输出)，跳过推理")
    ap.add_argument("--pred-dir", default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # test.py 定位
    test_py = args.test_py
    if not test_py:
        for _d in (args.dep_dir, _dep_dir, HERE, os.getcwd()):
            if _d and os.path.exists(os.path.join(_d, "test.py")):
                test_py = os.path.join(_d, "test.py")
                break
        test_py = test_py or os.path.join(HERE, "test.py")
    bench = single.load_metric_module(test_py)

    fov_list = args.fovs if args.fovs else list(range(args.fov_start, args.fov_end + 1))
    print(f"[plan] CelloType(TissueNet预训练,不微调) | infer={args.infer} "
          f"channel_order={args.channel_order} | 共 {len(fov_list)} 个 FOV: {fov_list}")

    runner = None
    if not args.reuse_pred:
        runner = CelloTypeRunner(
            model_path=args.model_path, config_path=args.config_path,
            confidence=args.confidence, max_det=args.max_det, device=args.device,
            infer=args.infer, tile=args.tile, overlap=args.overlap)

    per_fov, raws, failed = [], [], []
    for fov in fov_list:
        print(f"\n===== FOV {fov} =====")
        try:
            final, raw = process_one_fov(bench, args, fov, runner)
        except Exception as e:
            print(f"  [skip] FOV {fov} 失败: {e}")
            failed.append((fov, str(e)))
            continue
        per_fov.append(final)
        raws.append(raw)
        print(f"  gt={final['gt_cell_count']} pred={final['pred_cell_count']} "
              f"F1={final['instance_f1']:.3f} AJI={final['AJI']:.3f} PQ={final['PQ']:.3f}")

    if not per_fov:
        raise SystemExit("没有任何 FOV 成功。请检查路径/文件命名/依赖安装。")

    import pandas as pd
    df = pd.DataFrame(per_fov)
    metric_cols = ["dice", "iou", "pixel_accuracy", "boundary_f1", "boundary_precision",
                   "boundary_recall", "instance_f1", "precision", "recall",
                   "DQ", "SQ", "PQ", "AJI", "mDA_0.50_0.95"]
    front = ["fov", "gt_cell_count", "pred_cell_count", "TP", "FP", "FN"]
    df = df[[c for c in front if c in df] + [c for c in metric_cols if c in df]]
    per_fov_csv = os.path.join(args.out_dir, "per_fov_metrics_cellotype.csv")
    df.to_csv(per_fov_csv, index=False)

    micro = agg.aggregate_micro(raws)
    total_gt = int(sum(f["gt_cell_count"] for f in per_fov))
    total_pred = int(sum(f["pred_cell_count"] for f in per_fov))
    macro_mean = {c: float(df[c].mean()) for c in metric_cols if c in df}
    macro_std = {c: float(df[c].std(ddof=1)) if len(df) > 1 else 0.0
                 for c in metric_cols if c in df}

    sdf = pd.DataFrame([dict(metric=c,
                             micro=round(micro.get(c, float("nan")), 6),
                             macro_mean=round(macro_mean.get(c, float("nan")), 6),
                             macro_std=round(macro_std.get(c, float("nan")), 6))
                        for c in metric_cols])
    summary_csv = os.path.join(args.out_dir, "aggregate_summary_cellotype.csv")
    sdf.to_csv(summary_csv, index=False)

    meta = dict(method="CelloType", model="TissueNet_pretrained", finetuned=False,
                infer=args.infer, channel_order=args.channel_order,
                model_path=args.model_path, config_path=args.config_path,
                n_fov_ok=len(per_fov), n_fov_failed=len(failed), failed=failed,
                fov_list=fov_list, iou_threshold=args.iou_threshold,
                total_gt_cells=total_gt, total_pred_cells=total_pred,
                micro_TP=micro["TP"], micro_FP=micro["FP"], micro_FN=micro["FN"],
                note="micro=池化(主口径,AJI/PQ必须用它); macro=逐FOV mean±std; "
                     "零样本,与Cellpose/StarDist同为整细胞GT可横向比; pixel_accuracy弱指标。")
    with open(os.path.join(args.out_dir, "run_meta_cellotype.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 78)
    print(f"CelloType(TissueNet,零样本) | infer={args.infer}")
    print(f"成功 {len(per_fov)}/{len(fov_list)} 个 FOV | GT={total_gt} 预测={total_pred}")
    if failed:
        print(f"失败 FOV: {[f for f, _ in failed]}")
    print(f"micro 计数: TP={micro['TP']} FP={micro['FP']} FN={micro['FN']}")
    print("-" * 78)
    show = sdf.copy()
    show["macro(mean±std)"] = [f"{m:.4f}±{s:.4f}"
                               for m, s in zip(show["macro_mean"], show["macro_std"])]
    print(show[["metric", "micro", "macro(mean±std)"]].to_string(index=False))
    print("-" * 78)
    print("主体报 micro; 稳健性附 macro(mean±std)。AJI/PQ 只看 micro。")
    print("CelloType 原生整细胞、零样本，与 Cellpose/StarDist(cell口径) 同 GT 可直接横向比。")
    print(f"\n[saved]\n  {per_fov_csv}\n  {summary_csv}\n  "
          f"{os.path.join(args.out_dir, 'run_meta_cellotype.json')}")


if __name__ == "__main__":
    main()

'''
nohup bash -c '
    python CosMx_cellotype.py \
    --dep-dir /data/qiuyijia/cellpose \
    --data-root /data/qiuyijia/dataset/Pancreas-CosMx-WTx-FlatFiles --fov-start 51 --fov-end 68 \
    --model-path /data/qiuyijia/cellotype/CelloType/models/tissuenet_model_0019999.pth \
    --config-path /data/qiuyijia/cellotype/CelloType/configs/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml \
    --test-py /data/qiuyijia/eval/test.py \
    --out-dir /data/qiuyijia/cellotype/CosMx_output
' > output.log 2>&1 &
'''