#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BOMS xenium_5 参数扫描：直接调 run_boms，不依赖 boms_xenium_5.py。
数据从 npz 里取 x/y/gene，切中心 patch 跑，避免每个配置都跑 996 万条。
  python3 boms_sweep_x5.py --stage 1
  python3 boms_sweep_x5.py --stage 2 --k 300
  python3 boms_sweep_x5.py --stage 3 --k 300 --hs 12
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, '/data/qiuyijia/boms/boms_src')
try:
    from boms.main import run_boms
except Exception as e:
    sys.exit(f'导入 run_boms 失败: {e}\n确认 /data/qiuyijia/boms/boms_src 下有 boms/main.py '
             f'且 meanshift_cpp 已编译（在 bboms 环境里跑）')

NPZ = '/data/qiuyijia/boms_xenium_5/boms_xenium_5.npz'
GT_CELLS, ROI_UM = 11975, 2125.0
UM2_PER_CELL = ROI_UM**2 / GT_CELLS          # 377 um^2
OUT = '/data/qiuyijia/boms_x5_sweep'

def load_patch(size_um):
    z = np.load(NPZ, allow_pickle=True)
    x, y, g = np.asarray(z['x']), np.asarray(z['y']), np.asarray(z['gene'])
    epochs = int(z['epochs']) if 'epochs' in z.files else 30
    cx, cy = (x.min()+x.max())/2, (y.min()+y.max())/2
    h = size_um/2
    m = (np.abs(x-cx) <= h) & (np.abs(y-cy) <= h)
    xs, ys, gs = x[m].astype('float64'), y[m].astype('float64'), g[m]
    codes = np.unique(gs, return_inverse=True)[1].astype('int32')
    exp_cells = size_um**2 / UM2_PER_CELL
    print(f'patch {size_um}x{size_um} um   转录本 {len(xs):,}   '
          f'基因 {codes.max()+1}   期望细胞 {exp_cells:.0f}   '
          f'目标中位 {len(xs)/exp_cells:.0f} tx/cell')
    return xs, ys, codes, epochs, exp_cells, len(xs)/exp_cells

def score(seg, exp_cells, target):
    s = np.asarray(seg).ravel()
    s = s[s >= 0]
    c = np.bincount(s); c = c[c > 0]
    if len(c) == 0:
        return None
    return dict(cells=len(c), ratio=len(c)/exp_cells,
                med=float(np.median(c)), mean=float(c.mean()),
                mm=float(c.mean()/max(np.median(c), 1e-9)),
                lt10=float((c < 10).mean()), p95=float(np.percentile(c, 95)),
                ok_med=np.median(c) >= 0.6*target, ok_mm=c.mean()/max(np.median(c),1e-9) < 1.5,
                ok_frag=bool((c < 10).mean() < 0.02), ok_n=0.8 <= len(c)/exp_cells <= 1.2)

def grid(a):
    if a.stage == 1:   return [(k, 30.0, 0.3) for k in (30, 100, 300, 600)]
    if a.stage == 2:   return [(a.k, hs, 0.3) for hs in (8., 12., 16., 20., 30.)]
    return [(a.k, a.hs, hr) for hr in (0.3, 0.5, 0.7)]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', type=int, default=1, choices=(1, 2, 3))
    p.add_argument('--k', type=int, default=300)
    p.add_argument('--hs', type=float, default=12.0)
    p.add_argument('--patch-um', type=float, default=500.0)
    p.add_argument('--epochs', type=int, default=0, help='0=用 npz 里的值')
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    xs, ys, codes, ep0, exp_cells, target = load_patch(a.patch_um)
    ep = a.epochs or ep0
    cfgs = grid(a)
    print(f'\nstage {a.stage}: {len(cfgs)} 个配置, epochs={ep}\n')
    rows = []
    for K, hs, hr in cfgs:
        tag = f'K{K}_hs{hs:g}_hr{hr:g}'
        f = os.path.join(OUT, f'stage{a.stage}_{tag}.json')
        if os.path.exists(f):
            r = json.load(open(f)); print(f'  [复用] {tag}'); rows.append(r); continue
        print(f'  [跑] {tag} ... ', end='', flush=True)
        t0 = time.time()
        try:
            out = run_boms(xs, ys, codes, ep, float(hs), float(hr), K=int(K))
        except Exception as e:
            print(f'失败: {type(e).__name__}: {e}'); continue
        seg = out[1] if isinstance(out, (tuple, list)) and len(out) > 1 else out
        r = score(seg, exp_cells, target)
        if r is None:
            print('无有效簇'); continue
        r.update(tag=tag, K=K, hs=hs, hr=hr, sec=time.time()-t0)
        json.dump(r, open(f, 'w'))
        rows.append(r)
        print(f"{r['cells']:,} 簇  中位 {r['med']:.0f}  {r['sec']:.0f}s")

    if not rows: sys.exit('\n没有成功的配置')
    hdr = (f"{'配置':>18} {'簇数':>7} {'ratio':>6} {'中位':>7} {'均值':>7} "
           f"{'均/中':>6} {'<10':>7} {'p95':>7} {'秒':>6}  判定")
    print('\n' + hdr); print('-'*len(hdr))
    for r in rows:
        g = ('中位' + ('✓' if r['ok_med'] else '✗'), '形状' + ('✓' if r['ok_mm'] else '✗'),
             '碎片' + ('✓' if r['ok_frag'] else '✗'), '数量' + ('✓' if r['ok_n'] else '✗'))
        print(f"{r['tag']:>18} {r['cells']:>7,} {r['ratio']:>6.3f} {r['med']:>7.0f} "
              f"{r['mean']:>7.0f} {r['mm']:>6.1f} {r['lt10']*100:>6.1f}% {r['p95']:>7.0f} "
              f"{r['sec']:>6.0f}  " + ' '.join(g))
    print(f"""
目标中位 {target:.0f} tx/cell（本 patch 内 转录本数/期望细胞数）
判据: 中位>={0.6*target:.0f}  均值/中位<1.5  <10tx占比<2%  ratio 0.8-1.2
全量基线(K=30 h_s=30): 中位 70  均/中 9.4  <10占比 15.2%  ratio 1.268

stage1 判读:
  中位数随 K 单调上升  -> K/G_eff 机制成立(G_eff中位308,当前K/G_eff=0.10),继续 stage2
  四个 K 中位数不动    -> h_r 不是余弦语义,先看 C++ 再定
  报错 meanshift_cpp   -> 没在 bboms 环境里,或扩展未编译
""")

if __name__ == '__main__':
    main()
