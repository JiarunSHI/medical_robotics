import csv
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from PIL import Image


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def check_affine(matrix, name):
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(name + ': invalid finite 4x4 matrix')
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError(name + ': last row must be [0,0,0,1]')


def read_case(case_dir, pixel_spacing, track_direction='probe-to-reference', frame_step=1, max_frames=0,
              side_crop='auto', crop_x=None, black_threshold=3., min_column_fraction=.01):
    case_dir = Path(case_dir)
    xml_path = case_dir / 'imageToProbeTransformMatrix_wld.xml'
    matches = [e for e in ET.parse(str(xml_path)).iter()
               if e.get('From') == 'Image' and e.get('To') == 'Probe' and e.get('Matrix')]
    if len(matches) != 1:
        raise ValueError('XML must contain exactly one Transform From="Image" To="Probe"')
    calibration = np.fromstring(matches[0].get('Matrix'), sep=' ').reshape(4, 4)
    check_affine(calibration, 'calibration')
    if np.linalg.norm(np.cross(calibration[:3, 0], calibration[:3, 1])) < 1e-12:
        raise ValueError('Degenerate image axes')
    rows = []
    with (case_dir / 'track.csv').open(encoding='utf-8-sig', newline='') as f:
        for line_no, parts in enumerate(csv.reader(f), 1):
            parts = [x.strip() for x in parts]
            while parts and parts[-1] == '': parts.pop()
            if not parts or not any(parts): continue
            if len(parts) != 13:
                raise ValueError('track.csv line %d: expected frame id + 12 numbers' % line_no)
            try: values = np.array([float(x) for x in parts])
            except ValueError: raise ValueError('track.csv line %d: non-numeric value' % line_no)
            if not np.all(np.isfinite(values)) or values[0] != int(values[0]) or values[0] < 0:
                raise ValueError('Invalid frame id or values on line %d' % line_no)
            matrix = np.eye(4); matrix[:3] = values[1:].reshape(3, 4)
            r = matrix[:3, :3]
            if np.max(np.abs(r.T @ r - np.eye(3))) > .02 or abs(np.linalg.det(r) - 1) > .02:
                raise ValueError('track rotation is not approximately rigid on line %d' % line_no)
            if track_direction == 'reference-to-probe': matrix = np.linalg.inv(matrix)
            rows.append((int(values[0]), matrix))
    if len({i for i, _ in rows}) != len(rows): raise ValueError('Duplicate frame IDs')
    rows.sort(key=lambda row: row[0])
    rows = rows[::frame_step]
    if max_frames: rows = rows[:max_frames]
    if not rows: raise ValueError('No selected frames')
    scale = np.diag([pixel_spacing[0], pixel_spacing[1], 1., 1.])
    frames = []
    profiles = {}
    for frame_id, track in rows:
        path = case_dir / ('%d.bmp' % frame_id)
        with Image.open(path) as im:
            width, height = im.size
            if im.mode not in ('L', 'RGB', 'RGBA', 'P'):
                raise ValueError('Expected 8-bit BMP: ' + str(path))
            if side_crop == 'auto' and crop_x is None:
                # Aggregate across selected frames: a shadow in one frame must not
                # move that frame's image origin or trim an internal dark region.
                profile = (np.asarray(im.convert('L')) > black_threshold).mean(axis=0)
                key = (width, height)
                if key in profiles: np.maximum(profiles[key], profile, out=profiles[key])
                else: profiles[key] = profile
        frames.append({'id': frame_id, 'path': path, 'width': width, 'height': height,
                       'source_size': (width, height),
                       'transform': track @ calibration @ scale})
    bounds = {}
    for width, height in {f['source_size'] for f in frames}:
        if crop_x is not None:
            left, right = map(int, crop_x)
        elif side_crop == 'auto':
            active = np.flatnonzero(profiles[(width, height)] >= min_column_fraction)
            if len(active) < 2:
                raise ValueError('Cannot locate lateral image borders; supply --crop-x LEFT RIGHT (right exclusive)')
            left, right = int(active[0]), int(active[-1]) + 1
        elif side_crop == 'none':
            left, right = 0, width
        else:
            raise ValueError('Unknown side-crop mode')
        if not (0 <= left < right <= width) or right-left < 2:
            raise ValueError('Crop bounds must satisfy 0 <= LEFT < RIGHT <= image width, width >= 2')
        bounds[(width, height)] = (left, 0, right, height)
    for frame in frames:
        left, top, right, bottom = bounds[frame['source_size']]
        frame['crop_box'] = (left, top, right, bottom)
        frame['width'], frame['height'] = right-left, bottom-top
        # Crop controls membership only. Keep the original full-image coordinate
        # system and the unchanged transform T @ C @ S.
    return frames, calibration


def frame_corners(frame, pixel_step=1):
    w = ((frame['width'] - 1) // pixel_step) * pixel_step
    h = ((frame['height'] - 1) // pixel_step) * pixel_step
    corners = np.array([[0, 0, 0, 1], [w, 0, 0, 1], [0, h, 0, 1], [w, h, 0, 1]], float)
    left, top = frame.get('crop_box', (0, 0))[:2]
    corners[:, :2] += [left, top]
    return (frame['transform'] @ corners.T).T[:, :3]


def plan_grid(frames, spacing, padding=1, pixel_step=1, fill=False):
    corners = np.concatenate([frame_corners(f, pixel_step) for f in frames])
    low, high = corners.min(0), corners.max(0)
    spacing = np.asarray(spacing, float)
    origin = low - padding * spacing
    dims = np.ceil((high + padding * spacing - origin) / spacing).astype(np.int64) + 1
    count = int(dims[0]) * int(dims[1]) * int(dims[2])
    # Includes global arrays, worst-case frame-local arrays, output, masks, fill buffers,
    # and a conservative fixed allowance for chunks and library overhead.
    estimated_bytes = count * (64 if fill else 32) + 256 * 1024**2
    return {'origin': origin, 'spacing': spacing, 'dims_xyz': dims, 'min_mm': low, 'max_mm': high,
            'voxel_count': count, 'estimated_peak_bytes': estimated_bytes}


def axis_weights(t, gamma):
    left, right = (1. - t) ** gamma, t ** gamma
    total = left + right
    return left / total, right / total


def contributions(q, interpolation, gamma):
    """Return indices (N,K,3) and normalized weights (N,K), using voxel centres."""
    if interpolation == 'nearest':
        return np.floor(q + .5).astype(np.int64)[:, None, :], np.ones((len(q), 1))
    base = np.floor(q).astype(np.int64)
    t = q - base
    factors = [axis_weights(t[:, a], gamma[a]) for a in range(3)]
    offsets = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)])
    weights = np.stack([factors[0][x] * factors[1][y] * factors[2][z] for x, y, z in offsets], axis=1)
    return base[:, None, :] + offsets[None, :, :], weights


def accumulate_frame(frame, grid, interpolation, gamma, pixel_step, chunk_pixels, maximum=False):
    """One frame is aggregated locally before fusion, so decay occurs once per frame."""
    origin, spacing, dims = grid['origin'], grid['spacing'], grid['dims_xyz']
    qcorners = (frame_corners(frame, pixel_step) - origin) / spacing
    lower = np.maximum(0, np.floor(qcorners.min(0)).astype(np.int64) - 1)
    upper = np.minimum(dims, np.ceil(qcorners.max(0)).astype(np.int64) + 2)
    size = upper - lower
    shape = tuple(int(x) for x in size[::-1])
    sums, weights = np.zeros(shape, np.float32), np.zeros(shape, np.float32)
    sf, wf = sums.ravel(), weights.ravel()
    with Image.open(frame['path']) as im:
        gray = im.convert('L')
        if 'crop_box' in frame:
            if gray.size != tuple(frame['source_size']):
                raise ValueError('Image dimensions changed after geometry planning: ' + str(frame['path']))
            gray = gray.crop(frame['crop_box'])
        values = np.array(gray, dtype=np.float32)[::pixel_step, ::pixel_step]
    height, width = values.shape
    flat_values = values.ravel()
    transform = frame['transform']
    left, top = frame.get('crop_box', (0, 0))[:2]
    for start in range(0, len(flat_values), chunk_pixels):
        stop = min(start + chunk_pixels, len(flat_values))
        ids = np.arange(start, stop)
        # Array indices are relative to the cropped buffer; geometry uses the
        # original BMP coordinates, including the removed border offsets.
        u, v = left + (ids % width) * pixel_step, top + (ids // width) * pixel_step
        p = u[:, None] * transform[:3, 0] + v[:, None] * transform[:3, 1] + transform[:3, 3]
        indices, w = contributions((p - origin) / spacing, interpolation, gamma)
        indices -= lower
        valid = (w > 0) & np.all((indices >= 0) & (indices < size), axis=2)
        if np.any((w > 0) & ~valid): raise RuntimeError('Grid did not contain interpolation support')
        linear = (indices[:, :, 2] * size[1] + indices[:, :, 1]) * size[0] + indices[:, :, 0]
        intensity = np.broadcast_to(flat_values[start:stop, None], w.shape)
        ix, weight = linear[valid], w[valid]
        if maximum:
            np.maximum.at(sf, ix, intensity[valid])
        else:
            np.add.at(sf, ix, (intensity * w)[valid])
        np.add.at(wf, ix, weight)
    slices = tuple(slice(int(lower[a]), int(upper[a])) for a in (2, 1, 0))
    return slices, sums, weights


def fuse_frame(total, weight, slices, frame_sum, frame_weight, mode, decay):
    a, b = total[slices], weight[slices]
    touched = frame_weight > 0
    if mode == 'decayed-mean':
        a[touched] *= decay; b[touched] *= decay
    if mode == 'maximum':
        np.maximum(a, frame_sum, out=a)
    else:
        a += frame_sum
    b += frame_weight


def reconstruct(frames, grid, interpolation='linear', gamma=(1., 1., 1.), mode='mean', decay=1.,
                pixel_step=1, chunk_pixels=32768, progress=None):
    shape = tuple(int(x) for x in grid['dims_xyz'][::-1])
    total, weight = np.zeros(shape, np.float32), np.zeros(shape, np.float32)
    for i, frame in enumerate(frames):
        slices, fs, fw = accumulate_frame(frame, grid, interpolation, gamma, pixel_step, chunk_pixels, mode == 'maximum')
        fuse_frame(total, weight, slices, fs, fw, mode, decay)
        del fs, fw
        if progress: progress(i + 1, len(frames), frame['id'])
    if mode != 'maximum': np.divide(total, weight, out=total, where=weight > 0)
    return total, weight


def fill_small_holes(volume, weight, spacing, radius_mm):
    """Fill only enclosed unobserved voxels using frozen observed neighbours.

    Empty voxels connected to any grid boundary (26-connectivity) are exterior
    and are never filled. Distance is in mm and accounts for voxel anisotropy.
    A measured intensity of zero remains an observation, never a hole.
    """
    from scipy.ndimage import gaussian_filter, distance_transform_edt, binary_fill_holes
    observed = weight > 0
    if radius_mm <= 0 or not observed.any():
        return volume.copy(), np.zeros(volume.shape, bool)
    # Flood the complement from the boundary; only enclosed components remain.
    candidate = binary_fill_holes(observed, structure=np.ones((3, 3, 3), bool)) & ~observed
    if not candidate.any(): return volume.copy(), candidate
    distance = distance_transform_edt(~observed, sampling=np.asarray(spacing)[::-1])
    candidate &= distance <= radius_mm
    del distance
    if not candidate.any(): return volume.copy(), candidate
    sigma = radius_mm / 2 / np.asarray(spacing)[::-1]
    denominator = gaussian_filter(weight, sigma, mode='constant', truncate=2.)
    numerator = gaussian_filter(volume * weight, sigma, mode='constant', truncate=2.)
    candidate &= denominator > 1e-12
    filled = volume.copy()
    filled[candidate] = numerator[candidate] / denominator[candidate]
    return filled, candidate
