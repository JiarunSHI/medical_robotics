import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def save_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def config():
    return read_json(ROOT / 'configs/project.json')

def new_directory(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Output already exists; use a new run name: {path}')
    path.mkdir(parents=True)
    return path
