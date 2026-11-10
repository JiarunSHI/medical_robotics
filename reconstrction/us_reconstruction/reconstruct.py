"""python -m us_reconstruction.reconstruct --help"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
from .core import read_case, plan_grid, reconstruct, fill_small_holes, file_hash
from .nrrd_io import write_nrrd


def parser():
    p = argparse.ArgumentParser(description='Reconstruct a tracked BMP case in mm. Full-image origin; only ROI pixels contribute; enclosed-hole interpolation enabled by default.')
    p.add_argument('--case', required=True, type=Path)
    p.add_argument('--output', type=Path, help='New/empty directory; default outputs/<case>_<timestamp>')
    p.add_argument('--pixel-spacing', type=float, nargs=2, default=[.1, .1], metavar=('SX', 'SY'))
    p.add_argument('--side-crop', choices=['auto', 'none'], default='auto', help='Auto-detect common left/right borders across selected frames')
    p.add_argument('--crop-x', type=int, nargs=2, metavar=('LEFT', 'RIGHT'), help='Explicit original-image column range [LEFT,RIGHT); overrides automatic detection')
    p.add_argument('--black-threshold', type=float, default=3., help='0..254 gray threshold used only for lateral border detection')
    p.add_argument('--min-column-fraction', type=float, default=.01, help='Fraction of pixels above threshold needed to classify a column as active, (0,1]')
    p.add_argument('--voxel-spacing', type=float, nargs=3, default=[.5, .5, .5], metavar=('SX', 'SY', 'SZ'))
    p.add_argument('--interpolation', choices=['linear', 'nearest'], default='linear')
    p.add_argument('--axis-gamma', type=float, nargs=3, default=[1., 1., 1.], metavar=('GX', 'GY', 'GZ'))
    p.add_argument('--fusion', choices=['mean', 'decayed-mean', 'maximum'], default='mean')
    p.add_argument('--history-decay', type=float, default=1., help='0..1; used only for decayed-mean, once per touched voxel per frame')
    p.add_argument('--fill-radius-mm', type=float, default=1., help='Enclosed-hole interpolation radius in mm (default 1); 0 disables; maximum 10')
    p.add_argument('--track-direction', choices=['probe-to-reference', 'reference-to-probe'], default='probe-to-reference')
    p.add_argument('--frame-step', type=int, default=1)
    p.add_argument('--max-frames', type=int, default=0, help='0 means all selected frames')
    p.add_argument('--pixel-step', type=int, default=1, help='Preview subsampling; pixel coordinates retain their original scale')
    p.add_argument('--padding-voxels', type=int, default=1)
    p.add_argument('--chunk-pixels', type=int, default=32768)
    p.add_argument('--max-memory-gb', type=float, default=4., help='Conservative estimated peak limit in GiB')
    p.add_argument('--dry-run', action='store_true', help='Inspect inputs and print geometry/memory; no reconstruction or files')
    return p


def validate(a):
    def require(condition, message):
        if not condition: raise ValueError(message)
    require(all(np.isfinite(a.pixel_spacing)) and all(.001 <= x <= 5 for x in a.pixel_spacing), 'pixel spacing: 0.001..5 mm/pixel')
    require(0 <= a.black_threshold <= 254, 'black threshold: 0..254')
    require(0 < a.min_column_fraction <= 1, 'min-column-fraction: (0,1]')
    require(a.crop_x is None or a.side_crop != 'none', '--crop-x cannot be combined with --side-crop none')
    require(all(np.isfinite(a.voxel_spacing)) and all(.05 <= x <= 5 for x in a.voxel_spacing), 'voxel spacing: 0.05..5 mm')
    require(all(np.isfinite(a.axis_gamma)) and all(.25 <= x <= 4 for x in a.axis_gamma), 'axis gamma: 0.25..4')
    require(0 <= a.history_decay <= 1, 'history decay: 0..1')
    require(a.fusion == 'decayed-mean' or a.history_decay == 1, 'history-decay requires --fusion decayed-mean')
    require(0 <= a.fill_radius_mm <= 10, 'fill radius: 0..10 mm')
    require(a.frame_step >= 1 and a.max_frames >= 0 and 1 <= a.pixel_step <= 16, 'Invalid sampling arguments')
    require(1 <= a.padding_voxels <= 20, 'padding voxels: 1..20')
    require(256 <= a.chunk_pixels <= 262144, 'chunk pixels: 256..262144')
    require(np.isfinite(a.max_memory_gb) and a.max_memory_gb > 0, 'memory limit must be positive')


def jsonable(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value.resolve())
    raise TypeError(type(value).__name__)


def main(argv=None):
    a = parser().parse_args(argv)
    validate(a)
    print('Reading case and determining lateral crop...', flush=True)
    frames, calibration = read_case(a.case, a.pixel_spacing, a.track_direction, a.frame_step, a.max_frames,
                                    a.side_crop, a.crop_x, a.black_threshold, a.min_column_fraction)
    grid = plan_grid(frames, a.voxel_spacing, a.padding_voxels, a.pixel_step, a.fill_radius_mm > 0)
    print('Case:', a.case.name, '| frames:', len(frames), flush=True)
    print('Transform: track @ original_XML @ diag(pixel_spacing_x,pixel_spacing_y,1,1); original BMP pixel coordinates', flush=True)
    for source_size, crop_box in sorted({(f['source_size'], f['crop_box']) for f in frames}):
        left, top, right, bottom = crop_box
        print('Image %s -> ROI [%d:%d, %d:%d], size %dx%d; image origin stays at original BMP (0,0)' %
              (source_size, left, right, top, bottom, right-left, bottom-top), flush=True)
    print('Reference min / max (mm):', grid['min_mm'], '/', grid['max_mm'], flush=True)
    print('Coverage XYZ (cm):', (grid['max_mm'] - grid['min_mm']) / 10, flush=True)
    print('Grid XYZ:', grid['dims_xyz'], '| origin mm:', grid['origin'], flush=True)
    print('Estimated peak GiB: %.3f' % (grid['estimated_peak_bytes'] / 2**30), flush=True)
    memory_limit = a.max_memory_gb * 2**30
    try:
        import psutil
        memory_limit = min(memory_limit, psutil.virtual_memory().available * .65)
    except ImportError: pass
    if a.dry_run:
        print('Fits current memory budget:', grid['estimated_peak_bytes'] <= memory_limit)
        return
    if grid['estimated_peak_bytes'] > memory_limit:
        raise ValueError('Estimated memory exceeds budget. Increase voxel spacing, reduce frames, or explicitly raise --max-memory-gb.')
    output = a.output or Path('outputs') / (a.case.name + '_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    if output.exists() and any(output.iterdir()): raise ValueError('Output directory must be new or empty: ' + str(output))
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic(); last_report = [0.]
    def progress(done, total, frame_id):
        now = time.monotonic()
        if done == total or now - last_report[0] >= 5:
            print('Frames %d/%d | frame ID %d | %.1f s' % (done, total, frame_id, now-started), flush=True)
            last_report[0] = now
    volume, weight = reconstruct(frames, grid, a.interpolation, a.axis_gamma, a.fusion,
                                 a.history_decay, a.pixel_step, a.chunk_pixels, progress)
    observed = weight > 0
    print('Writing NRRD volumes...', flush=True)
    for filename, array in [('volume.nrrd', volume), ('weights.nrrd', weight), ('observed_mask.nrrd', observed.astype(np.uint8))]:
        write_nrrd(output / filename, array, grid['origin'], grid['spacing'])
    filled_count = 0
    if a.fill_radius_mm:
        print('Interpolating enclosed interior holes; exterior empty voxels are excluded...', flush=True)
        filled, fill_mask = fill_small_holes(volume, weight, grid['spacing'], a.fill_radius_mm)
        filled_count = int(fill_mask.sum())
        write_nrrd(output / 'filled_volume.nrrd', filled, grid['origin'], grid['spacing'])
        write_nrrd(output / 'filled_mask.nrrd', fill_mask.astype(np.uint8), grid['origin'], grid['spacing'])
        print('Interpolated interior voxels:', filled_count, flush=True)
    metadata = {
        'status': 'complete', 'config': vars(a), 'grid': grid, 'original_calibration': calibration,
        'coordinate_chain': 'reference_mm = track @ original_XML @ diag(sx,sy,1,1) @ [u_full,v_full,0,1]',
        'image_origin': 'centre of top-left pixel of original full BMP, including borders; crop only selects contributing pixels',
        'pixel_coordinate_convention': 'original_full_image',
        'crop': {'mode': 'manual' if a.crop_x is not None else a.side_crop,
                 'bounds_convention': '[left,top,right,bottom), right/bottom exclusive; only lateral borders removed',
                 'black_threshold': a.black_threshold, 'min_column_fraction': a.min_column_fraction,
                 'groups': [{'source_size': list(size), 'box': list(box)}
                            for size, box in sorted({(f['source_size'], f['crop_box']) for f in frames})],
                 'frames': [{'frame_id': f['id'], 'source_size': f['source_size'], 'box': f['crop_box']} for f in frames]},
        'array_order': 'z,y,x', 'export_embedding': 'reference numerical XYZ -> LPS XYZ; no patient registration',
        'assumed_pixel_spacing': True, 'frame_ids': [f['id'] for f in frames],
        'input_hashes': {name: file_hash(a.case / name) for name in ['track.csv', 'imageToProbeTransformMatrix_wld.xml']},
        'observed_voxels': int(observed.sum()), 'observed_fraction_of_grid': float(observed.mean()),
        'filled_voxels': filled_count, 'elapsed_seconds': time.monotonic()-started,
        'hole_filling': {'enabled': a.fill_radius_mm > 0, 'radius_mm': a.fill_radius_mm,
                         'interior_rule': 'unobserved components disconnected from all grid faces using 26-connectivity',
                         'method': 'normalized Gaussian weighted by original accumulation weights; sigma=radius/2, truncate=2',
                         'propagate_filled_values': False},
        'preferred_volume': 'filled_volume.nrrd' if a.fill_radius_mm > 0 else 'volume.nrrd',
        'intensity_range_observed': [float(volume[observed].min()), float(volume[observed].max())] if observed.any() else None,
        'software': {'numpy': np.__version__},
    }
    (output / 'metadata.json').write_text(json.dumps(metadata, default=jsonable, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Saved:', output.resolve(), flush=True)
    print('Observed grid coverage: %.2f%% | elapsed %.1f s' % (observed.mean()*100, time.monotonic()-started), flush=True)


if __name__ == '__main__':
    try: main()
    except (ValueError, FileNotFoundError) as e: raise SystemExit(str(e))
