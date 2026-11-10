"""Frozen case splits; original local positives; unused TG3K is test-only."""
from src.common import ROOT, read_json

def validate_rows(rows, allow_tg3k=False):
    cases={str(r['case_id']):r['split'] for r in read_json(ROOT/'configs/cases.json')}
    unused=set(read_json(ROOT/'provenance/tg3k/split_ids.json')['unused'])
    seen=set()
    for r in rows:
        key=(r.get('dataset'),r['case_id'],r['frame'])
        if key in seen:
            raise ValueError('Duplicate image/label pair')
        seen.add(key)
        if r['source']!='provided':
            raise ValueError('Only original annotations are permitted')
        if r.get('dataset')=='local':
            if cases.get(str(r['case_id']))!=r['split'] or int(r['foreground_pixels'])<=0:
                raise ValueError('Local case split or positive-label policy violated')
        elif r.get('dataset')=='tg3k_unused' and allow_tg3k:
            if r['split']!='test' or int(r['frame']) not in unused:
                raise ValueError('Only frozen unused TG3K IDs may enter final test')
        else:
            raise ValueError('Unexpected dataset')

def require_complete_manifest(path):
    from src.common import digest
    report=read_json(path.parent/'complete.json')
    if report.get('policy')!='original_local_positive_plus_tg3k_unused_test_v3':
        raise ValueError('Manifest was prepared under a different policy')
    if report['manifest_sha256'].get(path.name)!=digest(path):
        raise ValueError('Manifest differs from completed preparation')
