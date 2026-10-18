#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cellist_fix_origin.py — 修 detect_swap_scale 没减坐标原点的问题

现象:
  E14.5_E1S3 全图 22239x15359, GEM x max=32999 > 22239
  -> 脚本以为要放大 1.48 倍, 两个方向差异 106%/43.5%, 直接退出

原因:
  Stereo-seq 的 GEM 坐标带全局偏移 (E14.5_E1S3 原点是 28701,13125),
  detect_swap_scale 只用 sx.max()/sy.max(), 没减 min。
  用跨度重算: swap=True 时 W 0.250 / H 0.280, 差异 11%,
  scale~0.26 说明 ssDNA 是 0.125um/px 而 bin 是 0.5um, 正好 4 倍。

改动:
  1. detect_swap_scale 接收跨度而非最大值
  2. load_crop 调用时传 (max-min), 并把 min 作为 GEM 原点一起记进缓存
  3. 所有 bin->px 的换算加上 GEM 原点偏移

用法:
  python cellist_fix_origin.py                    # 修 stereo_2 和 3
  python cellist_fix_origin.py --files a.py b.py
"""
import argparse, re, shutil, sys
from pathlib import Path

DEFAULT = ["/data/qiuyijia/cellist/code/cellist_stereo_2.py",
           "/data/qiuyijia/cellist/code/cellist_stereo_3.py"]

OLD_DETECT_CALL = """    transposed, scale = detect_swap_scale(
        float(sx.max()), float(sy.max()), H, W)"""

NEW_DETECT_CALL = """    # ★ Stereo-seq 的 GEM 坐标带全局偏移，必须先减原点再比跨度
    gx0, gy0 = float(sx.min()), float(sy.min())
    span_x, span_y = float(sx.max()) - gx0, float(sy.max()) - gy0
    P(f"    GEM 原点 ({gx0:.0f},{gy0:.0f})  跨度 {span_x:.0f}x{span_y:.0f} bin")
    transposed, scale = detect_swap_scale(span_x, span_y, H, W)"""

OLD_CROP = """    sx_px = sx / scale
    sy_px = sy / scale"""

NEW_CROP = """    sx_px = (sx - gx0) / scale
    sy_px = (sy - gy0) / scale"""

OLD_SAVE = """    ox_bin = cx0 * scale
    oy_bin = cy0 * scale
    np.savez_compressed(f, crop=crop, ox=ox_bin, oy=oy_bin,
                        transposed=transposed, bin_scale=scale)
    return crop, ox_bin, oy_bin, scale"""

NEW_SAVE = """    # ox/oy 回到 GEM bin 绝对坐标: 裁剪偏移*scale + GEM 原点
    ox_bin = cx0 * scale + gx0
    oy_bin = cy0 * scale + gy0
    np.savez_compressed(f, crop=crop, ox=ox_bin, oy=oy_bin,
                        transposed=transposed, bin_scale=scale,
                        gem_x0=gx0, gem_y0=gy0)
    return crop, ox_bin, oy_bin, scale"""

OLD_DOC = '''def detect_swap_scale(gx_max, gy_max, H, W, tol=CONSISTENCY_TOL, label=""):
    """对标准/转置两种朝向，各自反推W/H方向需要的缩放比例，
    选两个方向算出的缩放比例最接近（自洽）的那个朝向。
    不再假设 scale 是整数倍，直接用真实比值。"""
    cands = []
    for swap in (False, True):
        need_w_raw, need_h_raw = (gy_max, gx_max) if swap else (gx_max, gy_max)'''

NEW_DOC = '''def detect_swap_scale(gx_span, gy_span, H, W, tol=CONSISTENCY_TOL, label=""):
    """对标准/转置两种朝向，各自反推W/H方向需要的缩放比例，
    选两个方向算出的缩放比例最接近（自洽）的那个朝向。
    ★ 传入的必须是跨度(max-min)，不是最大值 ——
      Stereo-seq 的 GEM 坐标带全局偏移，用最大值会算出完全错误的比例。"""
    cands = []
    for swap in (False, True):
        need_w_raw, need_h_raw = (gy_span, gx_span) if swap else (gx_span, gy_span)'''


def patch(path):
    p = Path(path)
    if not p.exists():
        print(f"  [跳过] 不存在: {path}")
        return False
    s = p.read_text()
    orig = s
    hits = []
    for name, old, new in (("detect_swap_scale 签名", OLD_DOC, NEW_DOC),
                           ("检测调用", OLD_DETECT_CALL, NEW_DETECT_CALL),
                           ("裁剪坐标", OLD_CROP, NEW_CROP),
                           ("缓存写入", OLD_SAVE, NEW_SAVE)):
        if old in s:
            s = s.replace(old, new)
            hits.append(name)
        else:
            print(f"  [未匹配] {name}")
    # 缓存复用分支也要能读回 gem 原点
    s = s.replace(
        '        scale = float(d.get("bin_scale", 1.0))',
        '        scale = float(d.get("bin_scale", 1.0))\n'
        '        gx0 = float(d.get("gem_x0", 0.0))\n'
        '        gy0 = float(d.get("gem_y0", 0.0))')
    if s == orig:
        print(f"  [无改动] {path}")
        return False
    bak = p.with_suffix(".py.bak_origin")
    shutil.copy2(p, bak)
    p.write_text(s)
    print(f"  ✓ {path}")
    print(f"    改了: {', '.join(hits)}")
    print(f"    备份: {bak.name}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", default=DEFAULT)
    a = ap.parse_args()
    print("修 detect_swap_scale 的坐标原点问题")
    print("=" * 60)
    n = sum(patch(f) for f in a.files)
    print("=" * 60)
    print(f"共修改 {n} 个文件")
    if n:
        print("\n★ 必须删掉旧的裁剪缓存，否则读到的还是错的偏移:")
        print("  rm -f /data/qiuyijia/cellist/stereo_*/ckpt_scan/crop.npz")
        print("\n然后重跑扫描。预期:")
        print("  E14.5_E1S3 原点(28701,13125) 跨度 4298x5549")
        print("  swap=True  W 0.250 / H 0.280  差异 11%")
        print("  scale~0.26 -> ssDNA 0.125um/px 而 bin 0.5um，正好 4 倍")
        print("  若仍报差异超限，把 CONSISTENCY_TOL 从 0.10 放宽到 0.15")


if __name__ == "__main__":
    main()