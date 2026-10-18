#!/usr/bin/env python3
"""UNSEG segmentation and micro evaluation on three CosMx datasets.

Uses the micro aggregation in cellpose/cosmx_lymph.py. Each FOV is an
independent image; the pancreas full slide is all 18 exported FOVs, 51-68.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import resource
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = Path('/data/qiuyijia')
DATA_ROOT = ROOT / 'dataset'
OUT_ROOT = HERE / 'cosmx_outputs'
CORE_PATH = ROOT / 'cellpose/cosmx_lymph.py'
TEST_PATH = ROOT / 'eval/test.py'
from xenium_unseg_benchmark import percentile_scale, segment  # noqa: E402

BRAIN_FOVS = (4, 18, 51, 52, 56, 65, 74, 87, 90, 101, 104, 110, 124,
              131, 147, 173, 187, 220, 224, 234, 242, 250, 255, 256,
              264, 296, 298, 307, 339)
DATASETS = {
    'cosmx_human_brain': dict(root=DATA_ROOT / 'cosmx_human_brain',
                              fovs=BRAIN_FOVS, input='[Histone, rRNA]'),
    'cosmx_lymph_node': dict(root=DATA_ROOT / 'cosmx_lymph_node',
                             fovs=(48, 261), input='[DNA, Membrane]'),
    'Pancreas-CosMx-WTx-FlatFiles': dict(root=DATA_ROOT / 'Pancreas-CosMx-WTx-FlatFiles',
                                         fovs=tuple(range(51, 69)),
                                         input='[blue/DAPI, max(red, green)/cell]'),
}
ORDER = tuple(DATASETS)


def load_core():
    if not CORE_PATH.is_file() or not TEST_PATH.is_file():
        raise FileNotFoundError('CosMx metric core or eval/test.py is missing')
    spec = importlib.util.spec_from_file_location('unseg_cosmx_micro_core', CORE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def paths_for(dataset, fov):
    root = DATASETS[dataset]['root']
    if dataset == 'cosmx_human_brain':
        return (root / '_bench_cache' / f'L0_F{fov:04d}_Histone-rRNA.npz',)
    if dataset == 'cosmx_lymph_node':
        return (root / '_bench_cache' / f'L0_F{fov:04d}_CD45-CD68-DNA-Membrane-PanCK.npz',)
    return (root / 'CellComposite' / f'CellComposite_F{fov:03d}.jpg',
            root / 'CellLabels' / f'CellLabels_F{fov:03d}.tif')


def preflight(names):
    if len(BRAIN_FOVS) != 29 or len(DATASETS['Pancreas-CosMx-WTx-FlatFiles']['fovs']) != 18:
        raise AssertionError('FOV selection changed unexpectedly')
    load_core().load_metric_module(str(TEST_PATH))
    for dataset in names:
        cfg = DATASETS[dataset]
        for fov in cfg['fovs']:
            missing = [str(path) for path in paths_for(dataset, fov) if not path.is_file()]
            if missing:
                raise FileNotFoundError(f'{dataset} FOV {fov}: missing {missing}')
        print(f'[OK] {dataset}: {len(cfg["fovs"])} FOVs, input={cfg["input"]}', flush=True)


class FovSource:
    """Expose FOV arrays to the tiled UNSEG runner in [nucleus, cell] order."""
    def __init__(self, nucleus, cell_signal):
        if nucleus.ndim != 2 or cell_signal.shape != nucleus.shape:
            raise ValueError(f'Invalid CosMx channel shapes: {nucleus.shape}, {cell_signal.shape}')
        self.nucleus = nucleus
        self.cell_signal = cell_signal
        h, w = nucleus.shape
        self.crop = (0, h, 0, w)

    def read(self, box):
        y0, y1, x0, x1 = box
        sl = np.s_[y0:y1, x0:x1]
        nucleus = percentile_scale(self.nucleus[sl])
        cell_signal = percentile_scale(self.cell_signal[sl])
        return np.stack((nucleus, cell_signal), axis=-1)


def read_fov(dataset, fov):
    paths = paths_for(dataset, fov)
    if dataset == 'Pancreas-CosMx-WTx-FlatFiles':
        with Image.open(paths[0]) as handle:
            rgb = np.asarray(handle.convert('RGB'))
        gt = tifffile.imread(paths[1])
        nucleus = rgb[..., 2].copy()
        cell_signal = np.maximum(rgb[..., 0], rgb[..., 1])
        del rgb
    else:
        with np.load(paths[0], allow_pickle=False) as data:
            if dataset == 'cosmx_human_brain':
                nucleus = data['Histone']
                cell_signal = data['rRNA']
            else:
                nucleus = data['DNA']
                cell_signal = data['Membrane']
            gt = data['labels']
    gt = np.squeeze(np.asarray(gt))
    source = FovSource(nucleus, cell_signal)
    if gt.shape != nucleus.shape:
        raise ValueError(f'{dataset} FOV {fov}: image={nucleus.shape}, GT={gt.shape}')
    return source, gt


def score_fov(core, bench, gt, pred):
    if gt.shape != pred.shape:
        raise ValueError(f'Prediction {pred.shape} and CellLabels {gt.shape} differ')
    # Compress global CosMx IDs before eval/test.py's bincount-based overlaps.
    gt = core.compact_labels(gt)
    pred = core.compact_labels(pred)
    overlaps = bench.compute_sparse_overlaps(gt, pred)
    comp = core.pixel_components(gt, pred, boundary_mode='thick')
    inst = core.instance_components(bench, overlaps, iou_threshold=0.5)
    aji_num, aji_den = core.aji_components(bench, overlaps, gt, pred)
    mda_map = core.mda_components(bench, overlaps)
    raw = dict(comp=comp, inst=inst, aji_num=aji_num, aji_den=aji_den,
               mda_map=mda_map)
    counts = dict(gt_cell_count=len(overlaps.gt_ids),
                  pred_cell_count=len(overlaps.pred_ids))
    return counts, raw


def restore_raw(record):
    raw = record['raw']
    return dict(comp=raw['comp'], inst=raw['inst'], aji_num=raw['aji_num'],
                aji_den=raw['aji_den'],
                mda_map={float(k): tuple(v) for k, v in raw['mda_map'].items()})


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    os.replace(temp, path)


def write_csv(path, frame):
    temp = path.with_suffix(path.suffix + '.tmp')
    frame.to_csv(temp, index=False)
    os.replace(temp, path)


def peak_rss_mib():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(value) / (1024**2 if sys.platform == 'darwin' else 1024)


def config_hash(args, dataset):
    cfg = DATASETS[dataset]
    params = dict(dataset=dataset, fovs=cfg['fovs'], input=cfg['input'],
                  tile_size=args.tile_size, halo=args.halo,
                  area_threshold=args.area_threshold,
                  convexity_threshold=args.convexity_threshold,
                  iou_threshold=0.5, boundary_mode='thick', method='UNSEG')
    digest = hashlib.sha256(json.dumps(params, sort_keys=True).encode())
    digest.update((HERE / 'UNSEG/unseg.py').read_bytes())
    digest.update((HERE / 'xenium_unseg_benchmark.py').read_bytes())
    return digest.hexdigest()


def run(args):
    dataset = args.dataset
    cfg = DATASETS[dataset]
    out = args.output_root / dataset
    out.mkdir(parents=True, exist_ok=True)
    core = load_core()
    bench = core.load_metric_module(str(TEST_PATH))
    digest = config_hash(args, dataset)
    started = time.perf_counter()
    status, error = 'failed', None
    records = []
    raws = []
    new_seg_seconds = 0.0
    print(f'=== UNSEG CosMx {dataset}; {len(cfg["fovs"])} FOVs ===', flush=True)
    print(f'input={cfg["input"]}; GPU=unused (CPU-only); output={out}', flush=True)
    if dataset == 'cosmx_human_brain':
        print('rRNA is a cytoplasmic proxy, not a cell membrane marker.', flush=True)
    try:
        for index, fov in enumerate(cfg['fovs'], 1):
            fov_dir = out / f'F{fov:04d}'
            fov_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = fov_dir / 'checkpoint.json'
            prediction = fov_dir / 'pred_cells.npy'
            nuclei = fov_dir / 'pred_nuclei.npy'
            if not args.force and checkpoint.is_file() and prediction.is_file() and nuclei.is_file():
                cached = json.loads(checkpoint.read_text(encoding='utf-8'))
                if (cached.get('config_hash') == digest or args.reuse_previous_checkpoints) \
                        and cached.get('dataset') == dataset and cached.get('fov') == fov:
                    records.append(cached)
                    raws.append(restore_raw(cached))
                    print(f'[{index}/{len(cfg["fovs"])}] FOV {fov}: reused', flush=True)
                    continue
            print(f'[{index}/{len(cfg["fovs"])}] FOV {fov}: loading', flush=True)
            source, gt = read_fov(dataset, fov)
            _nuc, pred, seg_seconds = segment(
                source, fov_dir, args.tile_size, args.halo,
                args.area_threshold, args.convexity_threshold)
            new_seg_seconds += seg_seconds
            counts, raw = score_fov(core, bench, gt, pred)
            record = dict(dataset=dataset, fov=fov, config_hash=digest,
                          segmentation_seconds=seg_seconds,
                          peak_cpu_rss_mib=peak_rss_mib(),
                          gt_cell_count=int(counts['gt_cell_count']),
                          pred_cell_count=int(counts['pred_cell_count']),
                          raw=raw)
            write_json(checkpoint, record)
            records.append(record)
            raws.append(raw)
            metrics = core.finalize(**raw)
            print(f'  gt={counts["gt_cell_count"]} pred={counts["pred_cell_count"]} '
                  f'PQ={metrics["PQ"]:.4f} AJI={metrics["AJI"]:.4f} '
                  f'segmentation={seg_seconds:.1f}s', flush=True)
            del source, gt, _nuc, pred
            gc.collect()

        micro = core.aggregate_micro(raws)
        row = dict(dataset=dataset, method='UNSEG', reference='CosMx CellLabels',
                   aggregation='micro', fovs_requested=len(cfg['fovs']),
                   fovs_completed=len(records), fov_ids=','.join(map(str, cfg['fovs'])),
                   input_channels=cfg['input'],
                   gt_cell_count=sum(r['gt_cell_count'] for r in records),
                   pred_cell_count=sum(r['pred_cell_count'] for r in records),
                   **micro,
                   gpu_used=False, gpu_device='CPU only', peak_gpu_memory_mib=0.0,
                   segmentation_seconds=sum(r['segmentation_seconds'] for r in records),
                   runtime_seconds=time.perf_counter() - started,
                   peak_cpu_rss_mib=max([peak_rss_mib(), *(r['peak_cpu_rss_mib'] for r in records)]))
        write_csv(out / 'micro_metrics.csv', pd.DataFrame([row]))
        status = 'success'
        print(pd.DataFrame([row]).to_string(index=False), flush=True)
        return row
    except BaseException as exc:
        error = repr(exc)
        traceback.print_exc()
        raise
    finally:
        perf = dict(dataset=dataset, status=status,
                    requested_fovs=list(cfg['fovs']),
                    completed_fovs=[r['fov'] for r in records],
                    gpu_used=False, gpu_device='CPU only', peak_gpu_memory_mib=0.0,
                    runtime_seconds=time.perf_counter() - started,
                    segmentation_seconds_this_run=new_seg_seconds,
                    segmentation_seconds_total=sum(r['segmentation_seconds'] for r in records),
                    peak_cpu_rss_mib=max([peak_rss_mib(), *(r['peak_cpu_rss_mib'] for r in records)]),
                    error=error)
        write_json(out / 'performance.json', perf)
        print(f'[performance] status={status}; segmentation={perf["segmentation_seconds_total"]:.1f}s; '
              f'peak RAM={perf["peak_cpu_rss_mib"]:.1f} MiB; '
              'GPU used=no; peak GPU memory=0 MiB', flush=True)


def aggregate(output_root, summary_root=None):
    summary_root = summary_root or output_root
    rows = []
    for dataset in ORDER:
        out = output_root / dataset
        performance_path = out / 'performance.json'
        metrics_path = out / 'micro_metrics.csv'
        performance = json.loads(performance_path.read_text()) if performance_path.is_file() else {}
        if performance.get('status') == 'success' and metrics_path.is_file():
            row = pd.read_csv(metrics_path).iloc[0].to_dict()
            row.update(status='success', error='')
        else:
            row = dict(dataset=dataset, method='UNSEG', reference='CosMx CellLabels',
                       aggregation='micro', status=performance.get('status', 'not_run'),
                       error=performance.get('error', ''))
        rows.append(row)
    path = summary_root / 'unseg_cosmx_micro_summary.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(path, pd.DataFrame(rows))
    print(f'Summary: {path} ({sum(r["status"] == "success" for r in rows)}/3 successful)', flush=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=ORDER)
    parser.add_argument('--output-root', type=Path, default=OUT_ROOT)
    parser.add_argument('--summary-root', type=Path,
                        help='directory for the aggregate CSV (defaults to output-root)')
    parser.add_argument('--tile-size', type=int, default=768)
    parser.add_argument('--halo', type=int, default=96)
    parser.add_argument('--area-threshold', type=int, default=20)
    parser.add_argument('--convexity-threshold', type=int, default=4)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--aggregate-only', action='store_true')
    parser.add_argument('--force', action='store_true', help='recompute FOV checkpoints')
    parser.add_argument('--reuse-previous-checkpoints', action='store_true',
                        help='reuse completed FOVs from before the UNSEG bug fixes')
    args = parser.parse_args()
    if args.check:
        preflight((args.dataset,) if args.dataset else ORDER)
        return
    if args.aggregate_only:
        aggregate(args.output_root, args.summary_root)
        return
    if not args.dataset:
        parser.error('--dataset is required for segmentation')
    if args.tile_size < 128 or not 0 <= args.halo < args.tile_size:
        parser.error('tile-size must be >=128 and 0 <= halo < tile-size')
    preflight((args.dataset,))
    run(args)


if __name__ == '__main__':
    main()
