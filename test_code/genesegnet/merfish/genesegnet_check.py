#!/usr/bin/env python
"""
GeneSegNet × MERFISH 输入通道诊断 —— 排除代码 bug，再下"跨平台失败"结论

用法：
  python diag_merfish_input.py
"""
import numpy as np
import tifffile

import sys
sys.path.insert(0, "/data/qiuyijia/genesegnet")
from run_genesegnet_infer import load_merfish, TARGETS

SEP = "=" * 68
P = lambda *a: print(*a, flush=True)


def main():
    P(SEP); P("MERFISH 输入通道诊断"); P(SEP)
    cfg = TARGETS["merfish"]
    image_2ch, gt_xy_px, n_gt = load_merfish(cfg, 0.5)

    dapi = image_2ch[..., 0]
    hm = image_2ch[..., 1]
    H, W = dapi.shape

    P(f"\n[1] DAPI 通道统计")
    P(f"    shape={dapi.shape}  dtype={dapi.dtype}")
    P(f"    均值={dapi.mean():.2f}  非零占比={((dapi>0).mean()*100):.1f}%")
    P(f"    p50={np.percentile(dapi,50):.0f} p95={np.percentile(dapi,95):.0f} "
      f"max={dapi.max()}")

    P(f"\n[2] 热图通道统计")
    P(f"    均值={hm.mean():.2f}  非零占比={((hm>0).mean()*100):.1f}%")
    P(f"    p50={np.percentile(hm,50):.0f} p95={np.percentile(hm,95):.0f} "
      f"max={hm.max()}")

    P(f"\n[3] ★ 关键检查：GT 质心处 DAPI 亮度 vs 随机位置")
    P(f"    （乳腺当初这个比值是 3.47×，>2 才说明真的对齐了）")
    gx = gt_xy_px[:, 0].astype(int).clip(0, W-1)
    gy = gt_xy_px[:, 1].astype(int).clip(0, H-1)
    v_gt = dapi[gy, gx].astype(float)
    rng = np.random.default_rng(0)
    v_rand = dapi[rng.integers(0, H, 20000),
                  rng.integers(0, W, 20000)].astype(float)
    ratio = v_gt.mean() / max(v_rand.mean(), 1e-6)
    P(f"    GT质心处均值={v_gt.mean():.2f}  随机位置均值={v_rand.mean():.2f}")
    P(f"    亮度比 = {ratio:.2f}×  "
      f"{'✓ 对齐正常，通道本身没问题' if ratio > 2 else '⚠ 偏弱，可能有对齐问题' if ratio > 1.3 else '✘ 几乎不比背景亮，通道构造大概率有bug'}")

    P(f"\n[4] 同样检查热图通道（GT 质心附近应该转录本密度更高）")
    v_gt_hm = hm[gy, gx].astype(float)
    v_rand_hm = hm[rng.integers(0, H, 20000),
                   rng.integers(0, W, 20000)].astype(float)
    ratio_hm = v_gt_hm.mean() / max(v_rand_hm.mean(), 1e-6)
    P(f"    亮度比 = {ratio_hm:.2f}×")

    out_dapi = "/data/qiuyijia/genesegnet_infer/merfish_dapi_check.tif"
    out_hm = "/data/qiuyijia/genesegnet_infer/merfish_heatmap_check.tif"
    tifffile.imwrite(out_dapi, dapi)
    tifffile.imwrite(out_hm, hm)
    P(f"\n  两个通道已单独存盘，可以直接打开肉眼看:")
    P(f"    {out_dapi}")
    P(f"    {out_hm}")

    P(f"\n{SEP}\n结论\n{SEP}")
    if ratio > 2 and ratio_hm > 2:
        P("  ✓ 两个通道都正常对齐，检出数极低更可能是真实的跨平台迁移失败")
    else:
        P("  ✘ 至少一个通道对齐可疑，值得先修这个再重新评估迁移能力")


if __name__ == "__main__":
    main()