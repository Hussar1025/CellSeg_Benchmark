#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ucs_stereo_3.py — UCS @ stereo_3 (Stereo-seq E16.5_E2S7)

上一次运行失败的原因（已定位）:
  ucs_stereo_1_run.py 的腐蚀循环里
      for cid in np.unique(nm[nm>0]):
          e = cv2.erode((nm==cid), ker, iterations=2)
          if e.sum() == 0: e[质心] = 1        # <- fallback
  在 E2S7 上对全部 7,960 个核都走了 fallback，于是每个细胞只剩 1 像素，
  最终 segmentation_mask 只有 280 个 3px 的碎片。
  对照 E14.5_E1S3 同代码正常（核 8.3um -> 细胞 14.4um, ratio 1.000），
  差别在图尺寸：E2S7 是 5550x4299 x 7960 核，逐核全图扫描的开销是
  E14.5 的 2.5 倍，cv2.erode 在内存压力下可能静默返回空。

本脚本的对策:
  默认 --nucleus-erode 0，跳过整个腐蚀循环。代价是核先验偏大（等于完整的
  CellBin 分割而非核区），但远好于 1 像素。
  跑完自动校验 nuclei_mask 的面积中位，<=2px 立即报错而不是继续。

⚠ 循环偏差: nuclei_mask 从 GEM 的 cell 列栅格化，GT 也是同一列 ->
  assignment / coverage / vec_* 循环虚高，仅 loc_*、median_diameter、
  count_* 可比。与 stereo_2 的口径一致。

⚠ 口径差异: stereo_2 用的是 --nucleus-erode 2，本次是 0。评估时需注明；
  若要严格可比，等腐蚀代码修好后两个都用同一值重跑。

gene_map 内存:
  --top-genes N  砍基因维，线性。top 2000 通常覆盖 85-95% UMI，代价小
  --bin-px B     砍空间维，平方(H和W各除B)，但分辨率 0.5 -> 0.5*B um
  E2S7: 全基因 21,370 -> 509.9 GB；top 2000 -> 47.7 GB
  底层脚本第 143 行有 `if est > 15` 的硬上限，用 UCS_MAX_GB 覆盖。

用法:
  python3 ucs_stereo_3.py --check                    # 只估体积
  python3 ucs_stereo_3.py                            # top-genes 2000, erode 0
  python3 ucs_stereo_3.py --nucleus-erode 2          # 与 stereo_2 同口径
  python3 ucs_stereo_3.py --top-genes 1000
"""
import argparse
import glob
import os
import subprocess
import sys

DIR = "/data/qiuyijia/dataset/stereoseq/stereo-seq_data_all"
SLICE = "E16.5_E2S7"
DATASET = "stereo_3"
GT_N = 7960
BASE = "/data/qiuyijia/ucs/UCS/ucs_stereo_1_run.py"
OUT = f"/data/qiuyijia/ucs_stereoseq_{SLICE}"
BIN_UM = 0.5


def log(m):
    print(m, flush=True)


def stash_bad(out):
    """把上次的单像素产物挪走而不是删，便于事后对照。"""
    import shutil
    moved = []
    for p in (os.path.join(out, "nuclei_mask.tif"),
              os.path.join(out, "ucs_log")):
        if os.path.exists(p) and not p.endswith(".BAD_1px") \
                and not p.endswith(".BAD_1px.bak"):
            dst = p + ".BAD_1px"
            n = 1
            while os.path.exists(dst):
                n += 1
                dst = f"{p}.BAD_1px{n}"
            shutil.move(p, dst)
            moved.append(os.path.basename(dst))
    if moved:
        log(f"  旧产物已挪走: {moved}")
    return moved


def check_mask(out):
    """核先验校验：面积中位 <=2px 说明腐蚀退化，必须中止。"""
    p = os.path.join(out, "nuclei_mask.tif")
    if not os.path.exists(p):
        log(f"  ✗ 未生成 {p}")
        return False
    import numpy as np
    import tifffile
    m = tifffile.imread(p)
    lab, cnt = np.unique(m, return_counts=True)
    a = cnt[lab > 0]
    if len(a) == 0:
        log("  ✗ nuclei_mask 无任何标签")
        return False
    med = float(np.median(a))
    d = 2 * (med / np.pi) ** .5 * BIN_UM
    log(f"\n【核先验校验】{m.shape}  标签 {len(a):,}  GT {GT_N:,}  "
        f"ratio {len(a)/GT_N:.3f}")
    log(f"  前景 {(m>0).mean()*100:.2f}%   面积中位 {med:.0f} px "
        f"= {med*BIN_UM**2:.1f} um2   等效直径 {d:.2f} um")
    if med <= 2:
        log("  ✗ 面积中位 <=2 px —— 腐蚀退化重现（fallback 到单像素质心）")
        log("    用 --nucleus-erode 0 重跑，或先修 ucs_stereo_1_run.py")
        return False
    if d < 3 or d > 25:
        log(f"  ⚠ 等效直径 {d:.1f} um 偏离小鼠胚胎细胞的 6-10 um")
    else:
        log("  ✓ 核先验正常")
    return True


def check_out(out):
    import numpy as np
    import tifffile
    fs = sorted(glob.glob(os.path.join(out, "ucs_log", "pred", "*.tif")))
    if not fs:
        log("  ✗ pred/ 下无 tif")
        return
    log(f"\n【输出校验】")
    for p in fs:
        m = tifffile.imread(p)
        lab, cnt = np.unique(m, return_counts=True)
        a = cnt[lab > 0]
        if len(a) == 0:
            log(f"  {os.path.basename(p):26s} 无标签")
            continue
        med = float(np.median(a))
        log(f"  {os.path.basename(p):26s} 标签 {len(a):6,}  "
            f"ratio {len(a)/GT_N:.3f}  前景 {(m>0).mean()*100:5.2f}%  "
            f"面积中位 {med:.0f}px = 直径 "
            f"{2*(med/np.pi)**.5*BIN_UM:.1f}um")
    log(f"\n  参照 stereo_2 (E14.5_E1S3, erode 2):")
    log(f"    nuclei_mask 4,872 标签 直径 8.3um -> "
        f"segmentation 4,870 直径 14.4um  ratio 1.000")
    log(f"  若 segmentation_mask 的标签数远少于 mask_shift_0，")
    log(f"  说明双位移融合失效，评估时改用 mask_shift_0。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只估体积不跑")
    ap.add_argument("--top-genes", type=int, default=2000)
    ap.add_argument("--bin-px", type=int, default=1)
    ap.add_argument("--nucleus-erode", type=int, default=0,
                    help="0 = 跳过腐蚀（上次 erode 2 导致单像素退化）")
    ap.add_argument("--max-gb", type=float, default=60.0,
                    help="覆盖底层脚本 if est > 15 的硬上限")
    ap.add_argument("--keep-old", action="store_true",
                    help="不挪走旧产物")
    ap.add_argument("--base", default=BASE)
    a = ap.parse_args()

    log("=" * 78)
    log(f"UCS @ {DATASET}  ({SLICE})   GT {GT_N:,}")
    log("=" * 78)

    if not os.path.exists(a.base):
        sys.exit(f"✗ 基础脚本不存在: {a.base}")
    gem = f"{DIR}/{SLICE}_Dorsal_Midbrain_GEM_CellBin.tsv.gz"
    tif = f"{DIR}/{SLICE}.tif"
    for p in (gem, tif):
        if not os.path.exists(p):
            sys.exit(f"✗ 不存在: {p}")
        log(f"  {os.path.getsize(p)/1e9:6.2f} GB  {os.path.basename(p)}")

    log(f"\n参数")
    log(f"  top_genes      {a.top_genes}  "
        f"(全基因 21,370 -> 509.9 GB; 2000 -> 47.7 GB)")
    log(f"  bin_px         {a.bin_px}  "
        f"({'原始 0.5 um/px' if a.bin_px == 1 else f'降到 {0.5*a.bin_px} um/px'})")
    log(f"  nucleus_erode  {a.nucleus_erode}"
        + ("   <- 跳过腐蚀，避免上次的单像素退化"
           if a.nucleus_erode == 0 else
           "   <- 与 stereo_2 同口径，但可能重现退化"))
    log(f"  UCS_MAX_GB     {a.max_gb}")

    if not a.check and not a.keep_old:
        log("")
        stash_bad(OUT)

    env = dict(os.environ, UCS_MAX_GB=str(a.max_gb))
    cmd = [sys.executable, a.base,
           "--gem", gem, "--ssdna", tif, "--out-dir", OUT,
           "--nucleus-erode", str(a.nucleus_erode),
           "--bin-px", str(a.bin_px)]
    if a.top_genes:
        cmd += ["--top-genes", str(a.top_genes)]
    if a.check:
        cmd += ["--check"]

    log(f"\n$ UCS_MAX_GB={a.max_gb} " + " ".join(cmd))
    log("")
    rc = subprocess.call(cmd, env=env)

    if a.check:
        log("\n--check 结束。体积可接受后去掉 --check 正式跑。")
        return
    if rc != 0:
        log(f"\n✗ 退出码 {rc}")
        log("  若报 gene_map 太大: 优先加大 --top-genes（线性省，代价小），")
        log("  再考虑 --bin-px 2（平方省 4x，但分辨率降到 1.0 um/px，")
        log("  胚胎细胞 6-10um 时一个细胞只剩 6-10 像素宽）。")
        sys.exit(rc)

    ok = check_mask(OUT)
    check_out(OUT)
    log("\n" + "=" * 78)
    if ok:
        log("✓ 完成。评估: python3 ucs_value_stereo_3.py")
    else:
        log("✗ 核先验异常，先修再评估")
    log("⚠ nuclei_mask 与 GT 同源 -> assignment/coverage/vec_* 循环虚高，")
    log(f"  仅 loc_*、median_diameter、count_* 可比。")
    log(f"⚠ 本次 erode={a.nucleus_erode}，stereo_2 用的是 2，口径不同需注明。")
    log("=" * 78)


if __name__ == "__main__":
    main()