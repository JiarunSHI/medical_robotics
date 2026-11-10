"""Read-only verification of portable server data; no training."""
import os
from pathlib import Path
from src.common import ROOT,read_json,read_csv,digest
from src.policy import require_complete_manifest,validate_rows

def main():
    os.chdir(ROOT)
    checksums=read_json(ROOT/'server_checksums.json')
    for name,expected in checksums.items():
        if digest(ROOT/name)!=expected:
            raise ValueError('File checksum mismatch: '+name)
    base=ROOT/'data/original_pairs_v3'
    require_complete_manifest(base/'local_pairs.csv')
    require_complete_manifest(base/'final_test.csv')
    local=read_csv(base/'local_pairs.csv');test=read_csv(base/'final_test.csv')
    tg=read_csv(ROOT/'data/tg3k.csv')
    validate_rows(local);validate_rows(test,allow_tg3k=True)
    expected=read_json(ROOT/'provenance/tg3k/split_ids.json')
    for split in ('train','val','unused'):
        if {int(r['frame']) for r in tg if r['split']==split}!=set(expected[split]):
            raise ValueError('TG3K split differs')
    if {int(r['frame']) for r in test if r['dataset']=='tg3k_unused'}!=set(expected['unused']):
        raise ValueError('Missing TG3K test pair')
    if {(r['case_id'],r['frame']) for r in test if r['dataset']=='local'}!={(r['case_id'],r['frame']) for r in local if r['split']=='test'}:
        raise ValueError('Local test set differs')
    for r in local+test+tg:
        for key in ('image','mask'):
            path=Path(r[key])
            if path.is_absolute() or ':' in r[key] or '\\' in r[key] or '..' in path.parts or not path.is_file():
                raise ValueError('Nonportable/missing asset: '+r[key])
    trained={r['image'] for r in local+tg if r['split'] in ('train','val')}
    if trained & {r['image'] for r in test}:
        raise ValueError('Test image appears in training/validation')
    print(f'Verified {len(checksums)} files; {len(local)} local pairs; {len(test)} final test pairs. No training executed.')

if __name__=='__main__':
    main()
