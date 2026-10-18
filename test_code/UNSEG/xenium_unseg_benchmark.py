#!/usr/bin/env python3
"""Tiled UNSEG cell/nucleus segmentation and Xenium GT evaluation.

H&E is experimental: hematoxylin is the nuclear signal and eosin is only a
surrogate for the membrane signal expected by UNSEG.  H&E results should be
interpreted separately from true fluorescence membrane results.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import pickle
import re
import resource
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from skimage.color import rgb2hed
from skimage.draw import polygon

ROOT = Path('/data/qiuyijia')
DATA = ROOT / 'dataset'
HERE = Path(__file__).resolve().parent
OUT = HERE / 'outputs'
sys.path.insert(0, str(HERE / 'UNSEG'))
from unseg import nuclei_cell_segmentation  # noqa: E402


def spec(name):
    base = DATA / name
    outs = base / 'outs' if name in {'xenium_liver', 'xenium_mouse_brain', 'xenium7'} else base
    if name in {'xenium9', 'xenium10'}:
        outs = base / 'benchmark_roi_10000'
    mode = 'he' if name in {'xenium_liver', 'xenium_pancreas_cancer'} else 'fluorescence'
    size = {'xenium_breast': 5000, 'xenium_colon': 6000,
            'xenium_roi_crop': None, 'xenium9': None, 'xenium10': None,
            'xenium_pancreas_cancer': None}.get(name, 10000)
    s = dict(name=name, mode=mode, base=base, outs=outs, size=size,
             boundaries=outs / 'cell_boundaries.parquet',
             experiment=outs / ('experiment_roi.json' if name == 'xenium_roi_crop' else 'experiment.xenium'),
             local=name in {'xenium_roi_crop', 'xenium9', 'xenium10'},
             membrane_indices=(1,) if name in {'xenium9', 'xenium10'} else (1, 2, 3))
    if mode == 'he':
        if name == 'xenium_liver':
            s.update(he=base / 'segmentation_outputs_center10000/input/xenium_liver_HE_center10000.tif',
                     he_meta=base / 'segmentation_outputs_center10000/input/xenium_liver_HE_center10000.json',
                     alignment=base / 'Xenium_V1_hLiver_nondiseased_section_FFPE_he_imagealignment.csv')
        else:
            s.update(he=base / 'Xenium_V1_hPancreas_Cancer_Add_on_FFPE_he_image.ome.tif',
                     he_meta=None, alignment=base / 'he_imagealignment.csv')
    else:
        s['morph'] = outs / 'morphology_focus'
    return s


NAMES = ('xenium_8', 'xenium_breast', 'xenium_colon', 'xenium_liver',
         'xenium_mouse_brain', 'xenium_pancreas_cancer', 'xenium_roi_crop',
         'xenium7', 'xenium9', 'xenium10')


def channel_files(directory):
    files = list(directory.glob('*.ome.tif')) + list(directory.glob('*.ome.tiff'))
    if not files:
        files = list(directory.glob('*.tif'))
    def key(p):
        m = re.search(r'(?:ch|focus_?)0*(\d+)', p.name.lower())
        return (int(m.group(1)) if m else 10**9, p.name)
    files = sorted(set(files), key=key)
    if not files:
        raise FileNotFoundError(f'No morphology channels in {directory}')
    return files


def image_shape(path):
    with tifffile.TiffFile(path) as tf:
        shape = tf.pages[0].shape
    if len(shape) == 2:
        return tuple(shape)
    if len(shape) == 3 and shape[-1] in (3, 4):
        return tuple(shape[:2])
    raise ValueError(f'Unsupported page shape {shape}: {path}')


def crop_for(shape, size):
    h, w = shape
    if size is None:
        return (0, h, 0, w)
    if h < size or w < size:
        raise ValueError(f'Requested {size}x{size} ROI exceeds {shape}')
    y, x = (h - size) // 2, (w - size) // 2
    return (y, y + size, x, x + size)


class TileReader:
    """Read one physical TIFF page through its tiled Zarr view."""
    def __init__(self, path, page=0):
        import zarr
        self.tf = tifffile.TiffFile(path)
        self.store = self.tf.pages[page].aszarr()
        self.array = zarr.open(self.store, mode='r')
        if not hasattr(self.array, 'shape'):
            keys = list(self.array.array_keys())
            self.array = self.array['0' if '0' in keys else keys[0]]

    def read(self, box):
        y0, y1, x0, x1 = box
        if len(self.array.shape) == 2:
            return np.asarray(self.array[y0:y1, x0:x1])
        return np.asarray(self.array[y0:y1, x0:x1, :3])

    def close(self):
        self.store.close()
        self.tf.close()


def percentile_scale(a):
    a = np.asarray(a)
    stride = max(1, int(np.sqrt(a.shape[0] * a.shape[1] / 150_000)))
    lo, hi = np.percentile(a[::stride, ::stride], (1, 99.8))
    if hi <= lo:
        return np.zeros(a.shape, dtype=np.float32)
    return np.clip((a.astype(np.float32) - lo) / (hi - lo), 0, 1)


class ImageSource:
    def __init__(self, s):
        self.s = s
        if s['mode'] == 'fluorescence':
            files = channel_files(s['morph'])
            needed = (0, *s['membrane_indices'])
            if len(files) == 1:
                with tifffile.TiffFile(files[0]) as tf:
                    count = len(tf.pages)
                if count <= max(needed):
                    raise ValueError(f'{s["name"]}: need pages {needed}, found {count}')
                self.files = [(files[0], i) for i in needed]
            else:
                if len(files) <= max(needed):
                    raise ValueError(f'{s["name"]}: need channels {needed}, found {len(files)}')
                self.files = [(files[i], 0) for i in needed]
            shapes = {image_shape(p) for p, _ in self.files}
            if len(shapes) != 1:
                raise ValueError(f'Channel shapes differ: {shapes}')
            self.shape = shapes.pop()
            self.readers = [TileReader(p, page) for p, page in self.files]
            self.crop = crop_for(self.shape, s['size'])
            self.he_origin = None
        else:
            if not s['he'].is_file():
                raise FileNotFoundError(s['he'])
            self.shape = image_shape(s['he'])
            self.readers = [TileReader(s['he'])]
            self.crop = crop_for(self.shape, s['size'])
            self.he_origin = (0, 0)
            if s['he_meta']:
                record = json.loads(s['he_meta'].read_text())
                self.he_origin = tuple(record['roi_origin_xy_in_source'])

    def read(self, box):
        """box is in local ROI coordinates; returns [nucleus, membrane]."""
        y0, _, x0, _ = self.crop
        global_box = (box[0] + y0, box[1] + y0, box[2] + x0, box[3] + x0)
        if self.s['mode'] == 'fluorescence':
            nuc = percentile_scale(self.readers[0].read(global_box))
            mem = None
            for reader in self.readers[1:]:
                channel = percentile_scale(reader.read(global_box))
                mem = channel if mem is None else np.maximum(mem, channel, out=mem)
        else:
            rgb = self.readers[0].read(global_box)
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb.astype(np.float32) / np.iinfo(rgb.dtype).max * 255, 0, 255).astype(np.uint8)
            hed = rgb2hed(rgb.astype(np.float32) / 255.0)
            nuc = percentile_scale(np.maximum(hed[..., 0], 0))
            mem = percentile_scale(np.maximum(hed[..., 1], 0))
        image = np.stack((nuc, mem), axis=-1)
        return image

    def close(self):
        for reader in self.readers:
            reader.close()


def pixel_size(path):
    data = json.loads(path.read_text())
    vals = []
    def walk(v):
        if isinstance(v, dict):
            for k, child in v.items():
                if 'pixel_size' in k.lower() and isinstance(child, (float, int)):
                    vals.append(float(child))
                walk(child)
        elif isinstance(v, list):
            for child in v:
                walk(child)
    walk(data)
    if not vals:
        raise ValueError(f'No pixel_size in {path}')
    return vals[0]


def build_gt(s, source, output):
    frame = pd.read_parquet(s['boundaries'])
    cols = {str(c).lower(): c for c in frame.columns}
    id_col = next(cols[c] for c in ('cell_id', 'label_id', 'label', 'id') if c in cols)
    names = (('vertex_x_local_um', 'vertex_x_local', 'vertex_x', 'x'),
             ('vertex_y_local_um', 'vertex_y_local', 'vertex_y', 'y')) if s['local'] else (
             ('vertex_x', 'vertex_x_um', 'x', 'x_location'),
             ('vertex_y', 'vertex_y_um', 'y', 'y_location'))
    x_col = next(cols[c] for c in names[0] if c in cols)
    y_col = next(cols[c] for c in names[1] if c in cols)
    xy = frame[[x_col, y_col]].to_numpy(dtype=np.float64) / pixel_size(s['experiment'])
    if s['mode'] == 'he':
        matrix = np.loadtxt(s['alignment'], delimiter=',')
        if matrix.shape != (3, 3):
            raise ValueError(f'Invalid H&E alignment: {s["alignment"]}')
        xy = np.column_stack((xy, np.ones(len(xy)))) @ np.linalg.inv(matrix).T
        xy[:, 0] -= source.he_origin[0]
        xy[:, 1] -= source.he_origin[1]
    else:
        xy[:, 0] -= source.crop[2]
        xy[:, 1] -= source.crop[0]
    ids = frame[id_col].to_numpy()
    codes, unique = pd.factorize(ids, sort=False)
    order = np.argsort(codes, kind='stable')
    codes = codes[order]
    xy = xy[order]
    starts = np.searchsorted(codes, np.arange(len(unique)), side='left')
    ends = np.searchsorted(codes, np.arange(len(unique)), side='right')
    h, w = output.shape
    label_rows = []
    for start, end, ident in zip(starts, ends, unique):
        if end - start < 3:
            continue
        pts = xy[start:end]
        if pts[:, 0].max() < 0 or pts[:, 0].min() >= w or pts[:, 1].max() < 0 or pts[:, 1].min() >= h:
            continue
        rr, cc = polygon(pts[:, 1], pts[:, 0], shape=(h, w))
        if rr.size:
            label = len(label_rows) + 1
            output[rr, cc] = label
            label_rows.append((label, str(ident)))
    output.flush()
    pd.DataFrame(label_rows, columns=['gt_label', 'xenium_cell_id']).to_csv(
        Path(output.filename).parent / 'gt_label_to_cell_id.csv', index=False)
    return len(label_rows)


class UnionFind:
    def __init__(self):
        self.parents = [0]

    def add(self, count):
        start = len(self.parents)
        self.parents.extend(range(start, start + count))
        return start

    def find(self, x):
        while self.parents[x] != x:
            self.parents[x] = self.parents[self.parents[x]]
            x = self.parents[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parents[max(a, b)] = min(a, b)


def relabel_dense(labels):
    ids, inverse = np.unique(labels, return_inverse=True)
    offset = 0 if ids[0] == 0 else 1
    dense = (inverse.reshape(labels.shape) + offset).astype(np.int32)
    return dense, len(ids) - (1 - offset)


def reconcile_overlap(existing, incoming, uf, start):
    """Merge adjacent core labels when both tile predictions agree in the halo."""
    valid = (existing > 0) & (incoming > 0)
    if not np.any(valid):
        return
    pairs, counts = np.unique(
        np.column_stack((existing[valid], incoming[valid])), axis=0, return_counts=True)
    # Both sides must agree on their dominant partner in the overlap.
    best_old, best_new = {}, {}
    for (old, new), count in zip(pairs, counts):
        old, new = int(old), int(new)
        if count > best_old.get(old, (0, 0))[1]:
            best_old[old] = (new, int(count))
        if count > best_new.get(new, (0, 0))[1]:
            best_new[new] = (old, int(count))
    for old, (new, count) in best_old.items():
        if best_new[new][0] == old and count >= 8:
            uf.union(old, start + new - 1)


def is_sparse_unseg_failure(exc):
    """Identify upstream empty-nucleus failures in sparsely populated tiles."""
    if not isinstance(exc, (IndexError, ValueError)):
        return False
    message = str(exc)
    if 'UNSEG found no usable nuclei' in message:
        return True
    if 'too many indices for array' not in message and 'zero-size array' not in message:
        return False
    tb = exc.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code.co_name in {'convexity_analysis', 'mask_statistics',
                                           'initialize_cell_segmentation'}:
            return True
        tb = tb.tb_next
    return False


def tile_checkpoint_signature(source, tile_size, halo, area_threshold, convexity_threshold):
    config = (type(source).__name__, tuple(source.crop), tile_size, halo,
              area_threshold, convexity_threshold)
    digest = hashlib.sha256(repr(config).encode())
    digest.update((HERE / 'UNSEG/unseg.py').read_bytes())
    return digest.hexdigest()


def save_tile_checkpoint(path, signature, tile_no, cell_uf, nuc_uf, sparse_failures):
    value = dict(signature=signature, tile=tile_no,
                 cell_parents=cell_uf.parents, nuc_parents=nuc_uf.parents,
                 sparse_failures=sparse_failures)
    temp = path.with_suffix('.pkl.tmp')
    with temp.open('wb') as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temp, path)


def segment(source, outdir, tile_size, halo, area_threshold, convexity_threshold):
    h = source.crop[1] - source.crop[0]
    w = source.crop[3] - source.crop[2]
    cell_work = outdir / '.pred_cells_work.npy'
    nuc_work = outdir / '.pred_nuclei_work.npy'
    checkpoint = outdir / 'tile_checkpoint.pkl'
    signature = tile_checkpoint_signature(source, tile_size, halo,
                                          area_threshold, convexity_threshold)
    cell_uf, nuc_uf = UnionFind(), UnionFind()
    n_tiles = ((h + tile_size - 1) // tile_size) * ((w + tile_size - 1) // tile_size)
    tile_no = 0
    sparse_failures = []
    started = time.perf_counter()
    completed = 0
    if checkpoint.is_file() and cell_work.is_file() and nuc_work.is_file():
        with checkpoint.open('rb') as handle:
            progress = pickle.load(handle)
        if progress.get('signature') == signature:
            cell = np.load(cell_work, mmap_mode='r+')
            nuclei = np.load(nuc_work, mmap_mode='r+')
            if cell.shape == nuclei.shape == (h, w) and 0 <= progress['tile'] <= n_tiles:
                completed = progress['tile']
                cell_uf.parents = progress['cell_parents']
                nuc_uf.parents = progress['nuc_parents']
                sparse_failures = progress['sparse_failures']
                print(f'  resuming from tile {completed}/{n_tiles}', flush=True)
            else:
                del cell, nuclei
    if completed == 0:
        cell = np.lib.format.open_memmap(cell_work, mode='w+', dtype=np.int32, shape=(h, w))
        nuclei = np.lib.format.open_memmap(nuc_work, mode='w+', dtype=np.int32, shape=(h, w))
        cell[:] = 0
        nuclei[:] = 0
        (outdir / 'sparse_tile_failures.csv').unlink(missing_ok=True)
    for cy0 in range(0, h, tile_size):
        cy1 = min(cy0 + tile_size, h)
        for cx0 in range(0, w, tile_size):
            cx1 = min(cx0 + tile_size, w)
            tile_no += 1
            if tile_no <= completed:
                continue
            # A failed tile may have written its core after the previous
            # checkpoint.  Clear it before recomputing so seams use only
            # committed labels.
            cell[cy0:cy1, cx0:cx1] = 0
            nuclei[cy0:cy1, cx0:cx1] = 0
            y0, y1 = max(0, cy0 - halo), min(h, cy1 + halo)
            x0, x1 = max(0, cx0 - halo), min(w, cx1 + halo)
            image = source.read((y0, y1, x0, x1))
            if not np.any(image[..., 0]) or not np.any(image[..., 1]):
                n_local = c_local = np.zeros(image.shape[:2], dtype=np.int32)
            else:
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        n_local, c_local, _, _ = nuclei_cell_segmentation(
                            image, area_threshold=area_threshold,
                            convexity_threshold=convexity_threshold, visualization=False)
                except Exception as exc:
                    if not is_sparse_unseg_failure(exc):
                        raise
                    sparse_failures.append((tile_no, cy0, cx0, type(exc).__name__, str(exc)))
                    print(f'  sparse tile {tile_no}/{n_tiles}: UNSEG found no usable nuclei; recording empty output', flush=True)
                    n_local = c_local = np.zeros(image.shape[:2], dtype=np.int32)
            del image
            for local, global_mask, uf in ((c_local, cell, cell_uf), (n_local, nuclei, nuc_uf)):
                local, count = relabel_dense(local)
                start = uf.add(count)
                if cy0 > 0:
                    end = min(cy0, y1)
                    reconcile_overlap(global_mask[y0:end, cx0:cx1],
                                      local[:end-y0, cx0-x0:cx1-x0], uf, start)
                if cx0 > 0:
                    end = min(cx0, x1)
                    reconcile_overlap(global_mask[cy0:cy1, x0:end],
                                      local[cy0-y0:cy1-y0, :end-x0], uf, start)
                core = local[cy0-y0:cy1-y0, cx0-x0:cx1-x0]
                global_mask[cy0:cy1, cx0:cx1] = np.where(core > 0, core + start - 1, 0)
            del c_local, n_local
            cell.flush()
            nuclei.flush()
            save_tile_checkpoint(checkpoint, signature, tile_no,
                                 cell_uf, nuc_uf, sparse_failures)
            if tile_no == 1 or tile_no % 10 == 0 or tile_no == n_tiles:
                print(f'  tile {tile_no}/{n_tiles}; elapsed {time.perf_counter()-started:.1f}s', flush=True)
    if sparse_failures:
        pd.DataFrame(sparse_failures, columns=['tile', 'roi_y0', 'roi_x0', 'error_type', 'error']).to_csv(
            outdir / 'sparse_tile_failures.csv', index=False)
        if len(sparse_failures) == n_tiles:
            raise RuntimeError(f'UNSEG failed on every tile ({n_tiles}/{n_tiles}); '
                               f'see {outdir / "sparse_tile_failures.csv"}')
    for mask, uf, filename in ((cell, cell_uf, 'pred_cells.npy'),
                               (nuclei, nuc_uf, 'pred_nuclei.npy')):
        mapping = np.arange(len(uf.parents), dtype=np.int32)
        for i in range(1, len(mapping)):
            mapping[i] = uf.find(i)
        # Compact labels after seam reconciliation.  Process strips to cap RAM.
        roots = np.unique(mapping)
        mapping = np.searchsorted(roots, mapping).astype(np.int32)
        target = outdir / (filename + '.tmp')
        compact = np.lib.format.open_memmap(target, mode='w+', dtype=np.int32, shape=(h, w))
        for y0 in range(0, h, 512):
            compact[y0:y0+512] = mapping[mask[y0:y0+512]]
        compact.flush()
        del compact
        os.replace(target, outdir / filename)
    del cell, nuclei
    checkpoint.unlink(missing_ok=True)
    cell_work.unlink(missing_ok=True)
    nuc_work.unlink(missing_ok=True)
    return (np.load(outdir / 'pred_nuclei.npy', mmap_mode='r'),
            np.load(outdir / 'pred_cells.npy', mmap_mode='r'),
            time.perf_counter() - started)


def evaluate(gt, pred, name):
    path = ROOT / 'eval/test.py'
    spec_ = importlib.util.spec_from_file_location('xenium_unseg_metrics', path)
    mod = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(mod)
    raw = mod.evaluate_single_mask_against_cellpose(
        method_name='UNSEG', cellpose_gt_mask=gt, pred_mask=pred,
        boundary_mode='thick', iou_threshold=0.5)
    result = {'dataset': name, 'method': 'UNSEG', 'gt_cell_count': raw.pop('cellpose_reference_cell_count'),
              'pred_cell_count': raw.pop('pred_cell_count')}
    for key in ('reference', 'method', 'mDA_note'):
        raw.pop(key, None)
    result.update(raw)
    return result


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def run(args):
    s = spec(args.dataset)
    out = args.output_root / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    status, error = 'failed', None
    details = {}
    source = None
    try:
        source = ImageSource(s)
        h, w = source.crop[1]-source.crop[0], source.crop[3]-source.crop[2]
        print(f'{args.dataset}: {s["mode"]}, image={source.shape}, ROI={source.crop}, output={out}', flush=True)
        if s['mode'] == 'he':
            print('H&E input is experimental: eosin is a surrogate, not a cell membrane marker.', flush=True)
        nuclei, cells, seg_seconds = segment(source, out, args.tile_size, args.halo,
                                             args.area_threshold, args.convexity_threshold)
        gt = np.lib.format.open_memmap(out / 'gt_cells.npy', mode='w+', dtype=np.int32, shape=(h, w))
        gt[:] = 0
        n_gt = build_gt(s, source, gt)
        print(f'  GT cells={n_gt}; evaluating full ROI...', flush=True)
        result = evaluate(gt, cells, args.dataset)
        result.update(input_mode=s['mode'], roi_y0=source.crop[0], roi_y1=source.crop[1],
                      roi_x0=source.crop[2], roi_x1=source.crop[3],
                      tile_size=args.tile_size, halo=args.halo, segmentation_seconds=seg_seconds,
                      runtime_seconds=time.perf_counter()-started, peak_memory_mb=peak_rss_mb(),
                      peak_gpu_memory_mb=0.0)
        pd.DataFrame([result]).to_csv(out / 'metrics.csv', index=False)
        details = dict(**result, input_files=[f'{p}:page{page}' for p, page in source.files] if s['mode'] == 'fluorescence' else [str(s['he'])],
                       gt_source=str(s['boundaries']), he_origin_xy=source.he_origin,
                       he_proxy=s['mode'] == 'he', nuclei_path=str(out / 'pred_nuclei.npy'),
                       cells_path=str(out / 'pred_cells.npy'))
        status = 'success'
        print(pd.DataFrame([result]).to_string(index=False), flush=True)
    except BaseException as exc:
        error = repr(exc)
        traceback.print_exc()
        raise
    finally:
        if source is not None:
            source.close()
        final = dict(details, dataset=args.dataset, status=status, error=error,
                     runtime_seconds=time.perf_counter()-started, peak_memory_mb=peak_rss_mb(),
                     peak_gpu_memory_mb=0.0, gpu_note='UNSEG uses CPU; no CUDA allocation')
        (out / 'performance.json').write_text(json.dumps(final, ensure_ascii=False, indent=2, default=str))
        print(f'Runtime={final["runtime_seconds"]:.1f}s; peak CPU RSS={final["peak_memory_mb"]:.1f} MB; '
              f'peak GPU memory=0 MB (CPU-only); status={status}', flush=True)


def check(name):
    s = spec(name)
    required = [s['boundaries'], s['experiment']]
    if s['mode'] == 'he':
        required += [s['he'], s['alignment']]
        if s['he_meta']:
            required += [s['he_meta']]
    else:
        required += [s['morph']]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(missing)
    src = ImageSource(s)
    try:
        return dict(dataset=name, mode=s['mode'], source_shape=src.shape, roi=src.crop,
                    roi_shape=(src.crop[1]-src.crop[0], src.crop[3]-src.crop[2]),
                    he_origin_xy=src.he_origin)
    finally:
        src.close()


def aggregate(output_root, summary_root=None):
    summary_root = summary_root or output_root
    rows = [pd.read_csv(output_root / name / 'metrics.csv') for name in NAMES
            if (output_root / name / 'metrics.csv').is_file()]
    if rows:
        summary_root.mkdir(parents=True, exist_ok=True)
        path = summary_root / 'unseg_xenium_summary.csv'
        temp = path.with_suffix('.csv.tmp')
        pd.concat(rows, ignore_index=True).to_csv(temp, index=False)
        os.replace(temp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=NAMES)
    parser.add_argument('--output-root', type=Path, default=OUT)
    parser.add_argument('--summary-root', type=Path,
                        help='directory for the aggregate CSV (defaults to output-root)')
    parser.add_argument('--tile-size', type=int, default=768)
    parser.add_argument('--halo', type=int, default=96)
    parser.add_argument('--area-threshold', type=int, default=20)
    parser.add_argument('--convexity-threshold', type=int, default=4)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--aggregate-only', action='store_true')
    args = parser.parse_args()
    if args.aggregate_only:
        aggregate(args.output_root, args.summary_root)
        return
    if args.check:
        names = [args.dataset] if args.dataset else NAMES
        failures = 0
        for name in names:
            try:
                print('[OK]', json.dumps(check(name), ensure_ascii=False))
            except Exception as exc:
                print(f'[FAIL] {name}: {exc}')
                failures += 1
        raise SystemExit(1 if failures else 0)
    if not args.dataset:
        parser.error('--dataset is required')
    if args.tile_size < 128 or args.halo < 0 or args.halo >= args.tile_size:
        parser.error('tile-size must be >=128 and 0 <= halo < tile-size')
    run(args)


if __name__ == '__main__':
    main()
