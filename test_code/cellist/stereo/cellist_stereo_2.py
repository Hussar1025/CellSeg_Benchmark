#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cellist_align_stereo.py v3 — 几何可行性检查 + 目视验证

v2 为什么废弃:
  在已知正确答案的 E16.5_E2S6 上做基准，正确答案是 scale=1.0/不转置/不翻转
  (stereo_1 的历史结果就是这么跑出来的)，但 v2 的搜索选了
  scale=0.164/翻y，且 Top10 的 r 全挤在 0.159~0.180，差距 0.003。
  根因: 让 FFT 在全图范围自由找平移，自由度太多 —— 任意 scale 都能找到
  一个"看起来还行"的位置，平移量 (12832,3072) 这种半图偏移就是乱找的证据。
  打分函数在正确答案上都失效，就不能用它去判 E14.5。

v3 做法:
  沿用老脚本 cellist_stereo_run.py 的假设: scale=1、零平移、
  GEM 坐标直接当像素坐标。只判一件事 —— 哪种朝向能让 spot 落进图内。
  这是纯几何问题，不需要打分。最终判定交给叠加图目视。

几何事实:
  E16.5_E2S6  x[2925,7174]  y[4875,10574]   图 27200x16385  -> 不转置即可
  E14.5_E1S3  x[19875,22697] y[4475,10074]  图 22239x15359
              不转置: x 上界 22697 > W=15359，覆盖 0%  -> 必须转置
              转置后: W=22239，x 超出 458px(1.6%)，老脚本靠 min(W,..) 自动处理

用法:
  python cellist_align_stereo.py --dataset E16.5_E2S6 --preview
  python cellist_align_stereo.py --dataset E14.5_E1S3 --preview
  python cellist_align_stereo.py --dataset E14.5_E1S3 --orient T00 --write --preview
"""
import argparse
import os
import sys

import numpy as np
import tifffile

DATA = "/data/qiuyijia/dataset/stereoseq/stereo-seq_data_all"
DATASETS = {
    "E14.5_E1S3": f"{DATA}/E14.5_E1S3.tif",
    "E16.5_E2S6": f"{DATA}/E16.5_E2S6.tif",
    "E16.5_E2S7": f"{DATA}/E16.5_E2S7.tif",
}
MARGIN = 50
BIN_UM = 0.5


def P(*a):
    print(*a, flush=True)


def load_spots(ckpt):
    f = os.path.join(ckpt, "gem_full.npz")
    if not os.path.exists(f):
        sys.exit(f"✘ 缺 {f}，先跑一次主脚本生成 GEM 缓存")
    d = np.load(f, allow_pickle=True)
    sx = d["sx"].astype(np.int64)
    sy = d["sy"].astype(np.int64)
    P(f"  GEM spot {len(sx):,}")
    P(f"    x[{sx.min()},{sx.max()}]  y[{sy.min()},{sy.max()}]")
    P(f"    跨度 {sx.max()-sx.min()} x {sy.max()-sy.min()} bin"
      f" = {(sx.max()-sx.min())*BIN_UM:.0f} x "
      f"{(sy.max()-sy.min())*BIN_UM:.0f} um")
    return sx, sy


def load_img(path):
    img = tifffile.imread(path)
    if img.ndim == 3:
        img = img[0] if img.shape[0] <= 8 else img[..., 0]
    P(f"  ssDNA H={img.shape[0]} W={img.shape[1]} "
      f"{img.dtype} {img.nbytes/1e9:.2f} GB")
    return img


def orient(img, t, fx, fy):
    """构造 A，使 A[sy, sx] 是 spot 应落的像素。翻转固化进数组。"""
    a = img.T if t else img
    if fy:
        a = a[::-1, :]
    if fx:
        a = a[:, ::-1]
    return a


def geom(sx, sy, H, W):
    """scale=1 零平移下的落图统计。"""
    ok = (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H)
    over_x = max(0, int(sx.max()) - (W - 1))
    over_y = max(0, int(sy.max()) - (H - 1))
    return float(ok.mean()), over_x, over_y


def contrast(A, sx, sy, n=200_000):
    H, W = A.shape
    ok = (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H)
    if ok.sum() < 100:
        return np.nan
    qx, qy = sx[ok], sy[ok]
    k = min(len(qx), n)
    i = np.random.default_rng(0).choice(len(qx), k, replace=False)
    at = float(A[qy[i], qx[i]].astype(np.float64).mean())
    gl = float(A[::9, ::9].astype(np.float64).mean())
    return at / max(gl, 1e-9)


def to_uint8(a):
    if a.dtype == np.uint8:
        return a
    lo, hi = np.percentile(a, [1, 99.5])
    b = np.clip((a.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
    return (b * 255).astype(np.uint8)


def crop_of(A, sx, sy):
    H, W = A.shape
    cy0 = max(0, int(sy.min()) - MARGIN)
    cy1 = min(H, int(sy.max()) + MARGIN + 1)
    cx0 = max(0, int(sx.min()) - MARGIN)
    cx1 = min(W, int(sx.max()) + MARGIN + 1)
    return cx0, cx1, cy0, cy1


def preview(crop, px, py, out, n=60_000):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        P(f"  (跳过预览: {e})")
        return
    h, w = crop.shape
    ok = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    px, py = px[ok], py[ok]
    k = min(len(px), n)
    i = np.random.default_rng(0).choice(len(px), k, replace=False)
    fig, ax = plt.subplots(1, 2, figsize=(20, 10 * h / max(w, 1) + 1))
    ax[0].imshow(crop, cmap="gray")
    ax[0].set_title("ssDNA")
    ax[1].imshow(crop, cmap="gray")
    ax[1].scatter(px[i], py[i], s=.06, c="red", alpha=.28, linewidths=0)
    ax[1].set_title("ssDNA + GEM spots")
    for a_ in ax:
        a_.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight", pad_inches=.02)
    plt.close(fig)
    P(f"  ✓ 叠加预览 {out}")
    P("    左右对比看: 红点是否勾勒出与左图相同的组织轮廓")
    P("    若红点糊满整幅、或压在黑色背景上 -> 未对齐")


def write_crop(A, sx, sy, t, ckpt):
    cx0, cx1, cy0, cy1 = crop_of(A, sx, sy)
    if cy1 <= cy0 or cx1 <= cx0:
        sys.exit("✘ 裁剪框为空")
    crop = to_uint8(np.ascontiguousarray(A[cy0:cy1, cx0:cx1]))
    inside = float(((sx >= cx0) & (sx < cx1) &
                    (sy >= cy0) & (sy < cy1)).mean())
    H, W = A.shape
    P(f"  裁剪 {crop.shape}  占全图 "
      f"{(cy1-cy0)*(cx1-cx0)/(H*W)*100:.1f}%")
    P(f"  框内 spot {inside*100:.1f}%")
    if inside < 0.95:
        P(f"  ⚠ {(1-inside)*100:.1f}% 的 spot 在框外，这批核无法分割，"
          f"评估时会压低 recall，需在 note 注明")
    f = os.path.join(ckpt, "crop.npz")
    np.savez_compressed(f, crop=crop, ox=float(cx0), oy=float(cy0),
                        transposed=bool(t), bin_scale=1.0,
                        gem_x0=0.0, gem_y0=0.0)
    P(f"  ✓ {f}   ox={cx0} oy={cy0} scale=1.0 swap={t}")
    P("  ★ 主脚本会走「复用裁剪缓存」分支，不再调 detect_swap_scale。")
    P("    别对主脚本用 --force，那会重新生成 crop.npz 丢掉本次配准。")
    return crop, sx - cx0, sy - cy0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--orient", default="",
                    help="指定朝向，如 N00(不转置不翻) / T00(转置) / T01；"
                         "留空则自动选覆盖率最高的")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--preview", action="store_true")
    a = ap.parse_args()

    ckpt = f"/data/qiuyijia/cellist/stereo_{a.dataset}/ckpt_scan"
    os.makedirs(ckpt, exist_ok=True)
    P("=" * 74)
    P(f"GEM ↔ ssDNA 几何检查（scale=1，零平移）—— {a.dataset}")
    P("=" * 74)
    sx, sy = load_spots(ckpt)
    img = load_img(DATASETS[a.dataset])

    P(f"\n  {'朝向':>6} {'转置':>6} {'翻x':>5} {'翻y':>5} "
      f"{'覆盖率':>8} {'x越界':>8} {'y越界':>8} {'对比度':>8}")
    P("  " + "-" * 62)
    rows = []
    for t in (False, True):
        for fx in (False, True):
            for fy in (False, True):
                A = orient(img, t, fx, fy)
                H, W = A.shape
                cov, ox_, oy_ = geom(sx, sy, H, W)
                ct = contrast(A, sx, sy) if cov > .05 else np.nan
                tag = ("T" if t else "N") + str(int(fx)) + str(int(fy))
                rows.append(dict(tag=tag, t=t, fx=fx, fy=fy,
                                 cov=cov, ox=ox_, oy=oy_, ct=ct))
                cs = f"{ct:.3f}" if np.isfinite(ct) else "—"
                P(f"  {tag:>6} {str(t):>6} {str(fx):>5} {str(fy):>5} "
                  f"{cov*100:>7.1f}% {ox_:>8} {oy_:>8} {cs:>8}")
                del A

    if a.orient:
        s = a.orient.upper()
        if len(s) != 3:
            sys.exit("✘ --orient 需 3 字符，如 N00 / T00 / T01")
        b = dict(tag=s, t=s[0] == "T", fx=s[1] == "1", fy=s[2] == "1")
        P(f"\n  手动指定朝向 {s}")
    else:
        cands = [r for r in rows if r["cov"] > .5]
        if not cands:
            sys.exit("\n✘ 没有朝向能容纳一半以上 spot，人工核对该数据集")
        # 优先覆盖率，其次不翻转（老脚本的默认假设）
        b = max(cands, key=lambda r: (round(r["cov"], 3),
                                      -(int(r["fx"]) + int(r["fy"]))))
        P(f"\n  → 自动选 {b['tag']}（覆盖 {b['cov']*100:.1f}%，"
          f"同覆盖率下优先不翻转）")

    P("\n  ★ 自动分值不足以定案 —— 在已知正确的 E16.5_E2S6 上，")
    P("    基于相关性的自动搜索也选错了。请以叠加图目视为准。")
    P("    参考: E16.5_E2S6 的历史正确配置是 N00（不转置不翻转）")

    A = np.ascontiguousarray(orient(img, b["t"], b["fx"], b["fy"]))
    if a.write:
        P("")
        crop, px, py = write_crop(A, sx, sy, b["t"], ckpt)
    else:
        cx0, cx1, cy0, cy1 = crop_of(A, sx, sy)
        crop = to_uint8(np.ascontiguousarray(A[cy0:cy1, cx0:cx1]))
        px, py = sx - cx0, sy - cy0
    if a.preview:
        preview(crop, px, py,
                f"/data/qiuyijia/align_{a.dataset}_{b['tag']}.png")
    if not a.write:
        P("\n  目视确认后加 --write 写入 crop.npz")


if __name__ == "__main__":
    main()