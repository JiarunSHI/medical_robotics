"""Validate sources and prepare TG3K manifests. No model is created."""
import argparse
from collections import Counter
from pathlib import Path
from PIL import Image
from src.common import ROOT, config, read_json, read_csv, save_csv, save_json, digest
from src.nrrd_io import read_header

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--all-image-sizes',action='store_true',help='Check every BMP header, not just first/last')
    args=p.parse_args()
    cfg=config()
    for source in read_json(ROOT/'provenance/sources.json'):
        if digest(ROOT/source['snapshot'])!=source['sha256']:
            raise ValueError('Frozen provenance changed: '+source['snapshot'])
    cases=read_json(ROOT/'configs/cases.json')
    old=read_json(ROOT/'provenance/manifest.json')
    expected={r['case_id'] for r in old if r['target']=='gland'}
    if {r['case_id'] for r in cases}!=expected or len(cases)!=len(expected):
        raise ValueError('Case whitelist differs from previous gland selection')
    split=read_json(ROOT/'provenance/case_split.json')
    for r in cases:
        if r['split']!=split[str(r['case_id'])]:
            raise ValueError('Patient split changed')
        if digest(r['annotation'])!=r['annotation_sha256']:
            raise ValueError('Source annotation changed: '+r['annotation'])
        h=read_header(r['annotation'])
        w,ht,n=map(int,h['sizes'].split())
        paths=list(Path(r['image_directory']).glob('*.bmp'))
        frames=[int(x.stem) for x in paths if x.stem.isdecimal()]
        if n!=r['frames'] or sorted(frames)!=list(range(n)):
            raise ValueError('Frame sequence mismatch: '+str(r['case_id']))
        # Names such as 0001.bmp are not silently substituted for 1.bmp.
        if any(not (Path(r['image_directory'])/f'{i}.bmp').is_file() for i in frames):
            raise ValueError('Noncanonical frame name')
        sample=paths if args.all_image_sizes else [Path(r['image_directory'])/'0.bmp',Path(r['image_directory'])/f'{n-1}.bmp']
        for image in sample:
            with Image.open(image) as im:
                if im.size!=(w,ht):
                    raise ValueError('Image size mismatch: '+str(image))
    rows=read_csv(ROOT/'provenance/tg3k/split_manifest.csv')
    tg=[]
    for r in rows:
        image,mask=Path(cfg['tg3k_root'])/r['image'],Path(cfg['tg3k_root'])/r['mask']
        with Image.open(image) as im,Image.open(mask) as ma:
            if im.size!=ma.size:
                raise ValueError('TG3K shape mismatch')
        tg.append(dict(case_id=r['group'],frame=r['id'],split=r['split'],image=str(image),mask=str(mask),source='provided'))
    save_csv(ROOT/'data/tg3k.csv',tg)
    save_json(ROOT/'reports/preflight.json',dict(status='passed',case_count=len(cases),
        case_splits=dict(Counter(r['split'] for r in cases)),local_frames=sum(r['frames'] for r in cases),
        tg3k_splits=dict(Counter(r['split'] for r in tg)),all_bmp_sizes_checked=args.all_image_sizes,
        training_executed=False,inference_executed=False))
    print(read_json(ROOT/'reports/preflight.json'))

if __name__=='__main__':
    main()
