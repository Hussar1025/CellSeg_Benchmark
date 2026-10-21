#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BOMS xenium_5 parameter sweep: calls run_boms directly, no dependency on boms_xenium_5.py.
Loads x/y/gene from the npz and runs on a centered patch to avoid running all 9.96M points per config.
  python3 boms_sweep_x5.py --stage 1
  python3 boms_sweep_x5.py --stage 2 --k 300
  python3 boms_sweep_x5.py --stage 3 --k 300 --hs 12
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, 'yourpath/boms/boms_src')
try:
    from boms.main import run_boms
except Exception as e:
    sys.exit(f'Failed to import run_boms: {e}\nConfirm boms/main.py exists under yourpath/boms/boms_src '
             f'and meanshift_cpp is compiled (run inside the bboms env)')

NPZ = 'yourpath/boms_xenium_5/boms_xenium_5.npz'
GT_CELLS, ROI_UM = 11975, 2125.0
UM2_PER_CELL = ROI_UM**2 / GT_CELLS          # 377 um^2
OUT = 'yourpath/boms_x5_sweep'

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
    print(f'patch {size_um}x{size_um} um   transcripts {len(xs):,}   '
          f'genes {codes.max()+1}   expected cells {exp_cells:.0f}   '
          f'target median {len(xs)/exp_cells:.0f} tx/cell')
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
    p.add_argument('--epochs', type=int, default=0, help='0=use value from npz')
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    xs, ys, codes, ep0, exp_cells, target = load_patch(a.patch_um)
    ep = a.epochs or ep0
    cfgs = grid(a)
    print(f'\nstage {a.stage}: {len(cfgs)} configs, epochs={ep}\n')
    rows = []
    for K, hs, hr in cfgs:
        tag = f'K{K}_hs{hs:g}_hr{hr:g}'
        f = os.path.join(OUT, f'stage{a.stage}_{tag}.json')
        if os.path.exists(f):
            r = json.load(open(f)); print(f'  [reuse] {tag}'); rows.append(r); continue
        print(f'  [run] {tag} ... ', end='', flush=True)
        t0 = time.time()
        try:
            out = run_boms(xs, ys, codes, ep, float(hs), float(hr), K=int(K))
        except Exception as e:
            print(f'failed: {type(e).__name__}: {e}'); continue
        seg = out[1] if isinstance(out, (tuple, list)) and len(out) > 1 else out
        r = score(seg, exp_cells, target)
        if r is None:
            print('no valid clusters'); continue
        r.update(tag=tag, K=K, hs=hs, hr=hr, sec=time.time()-t0)
        json.dump(r, open(f, 'w'))
        rows.append(r)
        print(f"{r['cells']:,} clusters  median {r['med']:.0f}  {r['sec']:.0f}s")

    if not rows: sys.exit('\nno successful configuration')
    hdr = (f"{'config':>18} {'cells':>7} {'ratio':>6} {'median':>7} {'mean':>7} "
           f"{'mean/med':>6} {'<10':>7} {'p95':>7} {'sec':>6}  verdict")
    print('\n' + hdr); print('-'*len(hdr))
    for r in rows:
        g = ('median' + ('✓' if r['ok_med'] else '✗'), 'shape' + ('✓' if r['ok_mm'] else '✗'),
             'frag' + ('✓' if r['ok_frag'] else '✗'), 'count' + ('✓' if r['ok_n'] else '✗'))
        print(f"{r['tag']:>18} {r['cells']:>7,} {r['ratio']:>6.3f} {r['med']:>7.0f} "
              f"{r['mean']:>7.0f} {r['mm']:>6.1f} {r['lt10']*100:>6.1f}% {r['p95']:>7.0f} "
              f"{r['sec']:>6.0f}  " + ' '.join(g))
    print(f"""
target median {target:.0f} tx/cell (transcripts/expected cells in this patch)
criteria: median>={0.6*target:.0f}  mean/median<1.5  <10tx fraction<2%  ratio 0.8-1.2
full baseline(K=30 h_s=30): median 70  mean/med 9.4  <10 fraction 15.2%  ratio 1.268

stage1 interpretation:
  median rises monotonically with K -> K/G_eff mechanism holds (G_eff median 308, current K/G_eff=0.10), continue stage2
  median flat across the four K -> h_r is not cosine semantics, inspect C++ first
  meanshift_cpp error -> not in bboms env, or extension not compiled
""")

if __name__ == '__main__':
    main()
