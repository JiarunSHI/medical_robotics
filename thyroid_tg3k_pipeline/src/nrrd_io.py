"""Strict reader for the previously accepted 3D uint8 gland segmentations.

No guessing of layers, orientations or cropped-volume offsets is permitted.
Arrays are Z,Y,X; BMP N.bmp corresponds to array[N].
"""
import gzip
import re
from pathlib import Path
import numpy as np

def header_from_stream(f):
    if not f.readline().startswith(b'NRRD'):
        raise ValueError('Not NRRD')
    h = {}
    while True:
        line = f.readline()
        if not line:
            raise ValueError('Truncated NRRD header')
        if not line.strip():
            break
        line = line.decode('utf-8').strip()
        if line.startswith('#'):
            continue
        key, value = line.split(':=', 1) if ':=' in line else line.split(':', 1)
        h[key] = value.strip()
    return h

def check_header(h):
    if h['type'] not in ('unsigned char', 'uchar', 'uint8') or h['dimension'] != '3':
        raise ValueError('Only audited 3D uint8 volumes supported')
    if 'data file' in h or h['encoding'] not in ('gzip', 'gz', 'raw'):
        raise ValueError('Unsupported payload')
    if h.get('space directions') != '(1,0,0) (0,1,0) (0,0,1)' or h.get('space origin') != '(0,0,0)':
        raise ValueError('Nonidentity geometry: explicit mapping required')
    if h.get('Segmentation_ReferenceImageExtentOffset', '0 0 0') != '0 0 0':
        raise ValueError('Cropped segmentation: explicit mapping required')
    segments = [k for k in h if re.fullmatch(r'Segment\d+_LabelValue', k)]
    if len(segments) != 1 or h.get(segments[0].replace('LabelValue', 'Layer'), '0') != '0':
        raise ValueError('Expected one gland segment on layer zero')
    return int(h[segments[0]])

def read_header(path):
    with Path(path).open('rb') as f:
        h = header_from_stream(f)
    check_header(h)
    return h

def read_volume(path):
    with Path(path).open('rb') as f:
        h = header_from_stream(f)
        label = check_header(h)
        payload = f.read()
    if h['encoding'] in ('gzip', 'gz'):
        payload = gzip.decompress(payload)
    shape = tuple(map(int, h['sizes'].split()))[::-1]
    a = np.frombuffer(payload, np.uint8).reshape(shape)
    if not np.isin(a, [0, label]).all():
        raise ValueError('Unexpected labels in gland-only annotation')
    return a.copy(), h, label

def write_volume(path, volume, original_header):
    h = dict(original_header)
    h['encoding'] = 'gzip'
    h['sizes'] = ' '.join(map(str, volume.shape[::-1]))
    # Closed surface and original extent would be stale after additions.
    h['Segmentation_ContainedRepresentationNames'] = 'Binary labelmap|'
    h['Segmentation_SourceRepresentation'] = 'Binary labelmap'
    z, y, x = volume.shape
    for key in list(h):
        if re.fullmatch(r'Segment\d+_Extent', key):
            h[key] = f'0 {x-1} 0 {y-1} 0 {z-1}'
    lines = ['NRRD0004']
    for key, value in h.items():
        sep = ':=' if key.startswith(('Segment', 'Segmentation')) else ': '
        lines.append(key + sep + value)
    with Path(path).open('wb') as f:
        f.write(('\n'.join(lines) + '\n\n').encode('utf-8'))
        f.write(gzip.compress(volume.astype(np.uint8).tobytes(), compresslevel=3))
