#!/usr/bin/env python3
"""Run tiled UNSEG on 13 MERFISH mosaics and evaluate Vizgen cell boundaries.

The evaluation calls the same eval/test.py function as
cellpose/merfish_seg_bench.py.  Images, predictions, and ground truth share
the same z3 mosaic coordinate system and centered ROI.
"""

from __future__ import annotations

import argparse
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

HERE = Path(__file__).resolve().parent
ROOT = Path('yourpath')
DATA = ROOT / 'dataset'
DEFAULT_OUTPUT = HERE / 'merfish_outputs'
sys.path.insert(0, str(ROOT / 'cellpose'))
import merfish_utils as U  # noqa: E402; shared GT implementation used by reference script
from xenium_unseg_benchmark import percentile_scale, segment  # noqa: E402

# Include the uppercase directory.  These are all MERFISH directories in
# dataset/, except the two explicitly excluded by the user.
DATASETS = (
    'merfish_Melanoma', 'merfish_ovarian1', 'merfish_ovarian2',
    'merfish_colon1', 'merfish_colon2', 'merfish_Uterian',
    'merfish_breast_cancer', 'merfish_liver1', 'merfish_liver2',
    'MERFISH_Lung_cancer', 'merfish_mouse_brain',
    'merfish_prostate1', 'merfish_prostate2',
)
EXCLUDED = {'merfish_mouse_hypothalamus', 'merfish_liver2_roi10000'}


def verify_dataset_list():
    found = {p.name for p in DATA.iterdir() if p.is_dir() and p.name.lower().startswith('merfish')}
    expected = found - EXCLUDED
    if expected != set(DATASETS):
        raise RuntimeError(f'MERFISH directory set changed: expected={sorted(expected)}, configured={list(DATASETS)}')


def unique_file(directory, predicate, description, optional=False):
    matches = sorted(p for p in directory.iterdir() if p.is_file() and predicate(p.name.lower()))
    if len(matches) == 1:
        return matches[0]
    if optional and not matches:
        return None
    raise FileNotFoundError(f'{directory}: expected {"at most" if optional else "exactly"} one {description}; found {matches}')


def discover(name):
    root = DATA / name
    if not root.is_dir():
        raise FileNotFoundError(root)
    image_dir = root / 'images'
    if not image_dir.is_dir():
        image_dir = root / 'imgs'
    if not image_dir.is_dir():
        raise FileNotFoundError(f'No images/ or imgs/ in {root}')
    mosaic = lambda token: lambda n: 'mosaic' in n and token in n and n.endswith(('.tif', '.tiff'))
    dapi = unique_file(image_dir, mosaic('dapi_z3'), 'DAPI z3 mosaic')
    cellbound = unique_file(image_dir, mosaic('cellbound1_z3'), 'Cellbound1 z3 mosaic', optional=True)
    polyt = unique_file(image_dir, mosaic('polyt_z3'), 'polyT z3 mosaic')
    transform = unique_file(image_dir, lambda n: 'micron_to_mosaic_pixel_transform' in n and n.endswith('.csv'), 'transform CSV')
    # Some datasets also contain cell_metadata.csv symlinks to Proseg output.
    # Ground truth must use the original Vizgen metadata alongside the
    # official cell boundaries, so prefer the unique regular file.
    metadata_candidates = sorted(p for p in root.iterdir() if p.is_file()
                                 and 'cell_metadata' in p.name.lower()
                                 and p.name.lower().endswith('.csv'))
    original_metadata = [p for p in metadata_candidates if not p.is_symlink()]
    if len(original_metadata) != 1:
        raise FileNotFoundError(f'{root}: expected one original Vizgen cell metadata CSV; '
                                f'found {original_metadata}; all candidates {metadata_candidates}')
    metadata = original_metadata[0]
    boundaries = root / 'cell_boundaries'
    if not boundaries.is_dir() or not next(boundaries.glob('feature_data_*.hdf5'), None):
        raise FileNotFoundError(f'No official cell boundary HDF5 files in {boundaries}')
    return dict(name=name, root=root, dapi=dapi, cellbound=cellbound, polyt=polyt,
                transform=transform, metadata=metadata, boundaries=boundaries)


def shape_of(path):
    with tifffile.TiffFile(path) as tif:
        shape = tuple(int(x) for x in tif.series[0].shape)
    if len(shape) != 2:
        raise ValueError(f'Expected 2D mosaic, got {shape}: {path}')
    return shape


def centered_roi(shape, size):
    h, w = shape
    if min(shape) < size:
        raise ValueError(f'{size}x{size} ROI exceeds mosaic shape {shape}')
    y0, x0 = (h-size)//2, (w-size)//2
    return (y0, y0+size, x0, x0+size)


class MosaicReader:
    """Use mmap for large uncompressed TIFFs; tiled Zarr for compressed TIFFs."""
    def __init__(self, path):
        self.path = path
        self.mapped = None
        self.tif = None
        self.store = None
        try:
            self.mapped = tifffile.memmap(path, series=0)
        except (ValueError, TypeError):
            import zarr
            self.tif = tifffile.TiffFile(path)
            self.store = self.tif.series[0].aszarr()
            self.mapped = zarr.open(self.store, mode='r')
            if not hasattr(self.mapped, 'shape'):
                keys = list(self.mapped.array_keys())
                self.mapped = self.mapped['0' if '0' in keys else keys[0]]
        if len(self.mapped.shape) != 2:
            raise ValueError(f'Expected 2D mosaic, got {self.mapped.shape}: {path}')

    def read(self, box):
        y0, y1, x0, x1 = box
        return np.asarray(self.mapped[y0:y1, x0:x1])

    def close(self):
        self.mapped = None
        if self.store is not None:
            self.store.close()
        if self.tif is not None:
            self.tif.close()


class MerfishImageSource:
    def __init__(self, files, roi_size):
        self.files = files
        self.shape = shape_of(files['dapi'])
        self.crop = centered_roi(self.shape, roi_size)
        self.cyto_path = files['cellbound'] or files['polyt']
        if shape_of(self.cyto_path) != self.shape:
            raise ValueError(f'DAPI and cell channel shapes differ: {files["name"]}')
        self.nucleus = MosaicReader(files['dapi'])
        self.cell = MosaicReader(self.cyto_path)

    @property
    def roi_xy(self):
        y0, y1, x0, x1 = self.crop
        return (x0, y0, x1, y1)

    def read(self, box):
        y0, _, x0, _ = self.crop
        global_box = (box[0]+y0, box[1]+y0, box[2]+x0, box[3]+x0)
        nuc = percentile_scale(self.nucleus.read(global_box))
        cell = percentile_scale(self.cell.read(global_box))
        return np.stack((nuc, cell), axis=-1)

    def close(self):
        self.nucleus.close()
        self.cell.close()


def metrics_module():
    path = ROOT / 'eval/test.py'
    spec = importlib.util.spec_from_file_location('merfish_unseg_metrics', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate(gt, pred, name):
    if gt.shape != pred.shape:
        raise ValueError(f'GT {gt.shape} differs from prediction {pred.shape}')
    raw = metrics_module().evaluate_single_mask_against_cellpose(
        method_name='UNSEG', cellpose_gt_mask=gt, pred_mask=pred,
        boundary_mode='thick', iou_threshold=0.5)
    raw['reference'] = 'Vizgen_official'
    raw['reference_cell_count'] = raw.pop('cellpose_reference_cell_count')
    raw['dataset'] = name
    raw.pop('mDA_note', None)
    return raw


def rss_peak_mb():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024**2 if sys.platform == 'darwin' else 1024)


def write_json(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    os.replace(tmp, path)


def run(args):
    files = discover(args.dataset)
    out = args.output_root / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    status, error = 'failed', None
    info = {}
    source = None
    try:
        size = 20000 if args.dataset == 'merfish_mouse_brain' else 10000
        source = MerfishImageSource(files, size)
        print(f'{args.dataset}: mosaic={source.shape}, ROI(x0,y0,x1,y1)={source.roi_xy}', flush=True)
        print(f'  channels=[DAPI, {source.cyto_path.name}]; z=3', flush=True)
        if files['cellbound'] is None:
            print('  Cellbound1 unavailable: polyT is a cytoplasmic surrogate, not a membrane stain.', flush=True)
        nuclei, pred, seg_seconds = segment(source, out, args.tile_size, args.halo,
                                             args.area_threshold, args.convexity_threshold)
        print(f'  UNSEG segmentation: {seg_seconds:.1f}s; building Vizgen GT...', flush=True)
        gt_started = time.perf_counter()
        gt_array, id_map = U.build_reference_label_mask(
            cell_boundaries_dir=str(files['boundaries']),
            transform_csv=str(files['transform']),
            roi=source.roi_xy, z_index=3,
            cell_metadata_csv=str(files['metadata']), verbose=True)
        gt_path = out / 'gt_cells.npy'
        np.save(gt_path, gt_array)
        del gt_array
        gt = np.load(gt_path, mmap_mode='r')
        pd.DataFrame(id_map.items(), columns=['gt_label', 'vizgen_cell_id']).to_csv(
            out / 'gt_label_to_cell_id.csv', index=False)
        gt_seconds = time.perf_counter() - gt_started
        print(f'  GT: {len(id_map)} cells; rasterization {gt_seconds:.1f}s; evaluating...', flush=True)
        eval_started = time.perf_counter()
        result = evaluate(gt, pred, args.dataset)
        eval_seconds = time.perf_counter() - eval_started
        result.update(roi_x0=source.roi_xy[0], roi_y0=source.roi_xy[1],
                      roi_x1=source.roi_xy[2], roi_y1=source.roi_xy[3],
                      roi_size=size, z_index=3, nucleus_channel=str(files['dapi']),
                      cell_channel=str(source.cyto_path), tile_size=args.tile_size,
                      halo=args.halo, segmentation_seconds=seg_seconds,
                      gt_seconds=gt_seconds, evaluation_seconds=eval_seconds,
                      runtime_seconds=time.perf_counter()-started,
                      peak_memory_mb=rss_peak_mb(), peak_gpu_memory_mb=0.0)
        pd.DataFrame([result]).to_csv(out / 'metrics.csv', index=False)
        info = dict(**result, prediction=str(out / 'pred_cells.npy'),
                    nuclei=str(out / 'pred_nuclei.npy'), ground_truth=str(gt_path),
                    ground_truth_source=str(files['boundaries']),
                    cell_channel_is_membrane=files['cellbound'] is not None)
        status = 'success'
        print(pd.DataFrame([result]).to_string(index=False), flush=True)
    except BaseException as exc:
        error = repr(exc)
        traceback.print_exc()
        raise
    finally:
        if source is not None:
            source.close()
        perf = dict(info, dataset=args.dataset, status=status, error=error,
                    runtime_seconds=time.perf_counter()-started,
                    peak_memory_mb=rss_peak_mb(), peak_gpu_memory_mb=0.0,
                    gpu_note='UNSEG runs on CPU; no GPU allocation')
        write_json(out / 'performance.json', perf)
        print(f'Runtime={perf["runtime_seconds"]:.1f}s; peak CPU RSS={perf["peak_memory_mb"]:.1f} MB; '
              f'peak GPU memory=0 MB (CPU-only); status={status}', flush=True)


def check(name):
    files = discover(name)
    size = 20000 if name == 'merfish_mouse_brain' else 10000
    source = MerfishImageSource(files, size)
    try:
        return dict(dataset=name, mosaic_shape=source.shape,
                    roi_xy=source.roi_xy, roi_size=size,
                    nucleus=str(files['dapi']), cell_channel=str(source.cyto_path),
                    transform=str(files['transform']), boundaries=str(files['boundaries']))
    finally:
        source.close()


def aggregate(output_root, summary_root=None):
    summary_root = summary_root or output_root
    rows = []
    for name in DATASETS:
        per = output_root / name
        perf_path = per / 'performance.json'
        metric_path = per / 'metrics.csv'
        perf = json.loads(perf_path.read_text()) if perf_path.is_file() else {}
        if perf.get('status') == 'success' and metric_path.is_file():
            row = pd.read_csv(metric_path).iloc[0].to_dict()
            row.update(status='success', error='')
        else:
            row = dict(dataset=name, reference='Vizgen_official', method='UNSEG',
                       status=perf.get('status', 'not_run'), error=perf.get('error', ''))
        rows.append(row)
    summary = pd.DataFrame(rows)
    path = summary_root / 'unseg_merfish_13_metrics.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    summary.to_csv(temp, index=False)
    os.replace(temp, path)
    print(f'Summary: {path} ({sum(summary.status == "success")}/13 successful)', flush=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASETS)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--summary-root', type=Path,
                        help='directory for the aggregate CSV (defaults to output-root)')
    parser.add_argument('--tile-size', type=int, default=768)
    parser.add_argument('--halo', type=int, default=96)
    parser.add_argument('--area-threshold', type=int, default=20)
    parser.add_argument('--convexity-threshold', type=int, default=4)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--aggregate-only', action='store_true')
    args = parser.parse_args()
    verify_dataset_list()
    if args.check:
        failed = 0
        for name in ([args.dataset] if args.dataset else DATASETS):
            try:
                print('[OK]', json.dumps(check(name), ensure_ascii=False))
            except Exception as exc:
                print(f'[FAIL] {name}: {exc}')
                failed += 1
        raise SystemExit(1 if failed else 0)
    if args.aggregate_only:
        aggregate(args.output_root, args.summary_root)
        return
    if not args.dataset:
        parser.error('--dataset is required for segmentation')
    if args.tile_size < 128 or not 0 <= args.halo < args.tile_size:
        parser.error('tile-size must be >=128 and 0 <= halo < tile-size')
    run(args)


if __name__ == '__main__':
    main()
