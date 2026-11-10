"""Small NRRD reader/writer: scalar 3D, C-order (z,y,x), axis-aligned LPS embedding."""
import gzip
from pathlib import Path
import numpy as np

TYPES = {'float': np.dtype('<f4'), 'uchar': np.dtype('u1')}


def write_nrrd(path, array, origin, spacing):
    array = np.asarray(array)
    kind = 'uchar' if array.dtype == np.uint8 else 'float'
    nz, ny, nx = array.shape
    sx, sy, sz = spacing
    header = '\n'.join([
        'NRRD0005', '# Abstract tracking reference embedded numerically in LPS; NOT patient registration.',
        'type: ' + kind, 'dimension: 3', 'space: left-posterior-superior',
        'sizes: %d %d %d' % (nx, ny, nz),
        'space directions: (%g,0,0) (0,%g,0) (0,0,%g)' % (sx, sy, sz),
        'kinds: domain domain domain', 'endian: little', 'encoding: gzip',
        'space units: "mm" "mm" "mm"',
        'space origin: (%.17g,%.17g,%.17g)' % tuple(origin),
        'reference_frame:=abstract_tracking_reference', 'patient_orientation_known:=false', '', ''
    ])
    with Path(path).open('wb') as f:
        f.write(header.encode('ascii'))
        with gzip.GzipFile(fileobj=f, mode='wb', compresslevel=1, mtime=0) as gz:
            # Write slabs to avoid another complete volume-sized bytes object.
            for plane in array:
                gz.write(np.asarray(plane, dtype=TYPES[kind], order='C').tobytes())


def read_nrrd(path):
    fields = {}
    with Path(path).open('rb') as f:
        if not f.readline().startswith(b'NRRD'):
            raise ValueError('Not a NRRD file')
        while True:
            line = f.readline()
            if not line:
                raise ValueError('Truncated NRRD header')
            if not line.strip():
                break
            text = line.decode('ascii').strip()
            if not text.startswith('#') and ': ' in text:
                k, v = text.split(': ', 1); fields[k] = v
        if fields.get('encoding') != 'gzip' or fields.get('dimension') != '3':
            raise ValueError('Viewer supports gzip scalar 3D NRRDs from this project')
        sizes = tuple(int(x) for x in fields['sizes'].split())
        data = gzip.GzipFile(fileobj=f).read()
    array = np.frombuffer(data, dtype=TYPES[fields['type']]).reshape(sizes[::-1]).copy()
    import re
    directions = [np.fromstring(x, sep=',') for x in re.findall(r'\(([^)]+)\)', fields['space directions'])]
    spacing = np.array([np.linalg.norm(x) for x in directions])
    origin = np.fromstring(fields['space origin'].strip('()'), sep=',')
    return array, origin, spacing
