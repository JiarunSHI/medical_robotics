"""Export original positive local masks and add unused TG3K pairs to final test.

No network is instantiated. Source images and annotations are read-only.
"""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
from src.common import ROOT,config,read_json,read_csv,save_csv,save_json,digest,new_directory
from src.nrrd_io import read_volume
from src.policy import validate_rows

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,default=ROOT/'data/original_pairs_v3')
    args=p.parse_args()
    out=new_directory(args.out.resolve())
    cases=read_json(ROOT/'configs/cases.json')
    rows=[]; counts=[]
    for case in cases:
        if digest(case['annotation'])!=case['annotation_sha256']:
            raise ValueError('Original annotation changed')
        volume,_,_=read_volume(case['annotation'])
        if volume.shape[0]!=case['frames']:
            raise ValueError('Original frame count changed')
        positive=np.flatnonzero(np.any(volume!=0,axis=(1,2)))
        folder=out/'masks'/f'WD{case["case_id"]:08d}'
        folder.mkdir(parents=True)
        for frame in positive.tolist():
            image=Path(case['image_directory'])/f'{frame}.bmp'
            with Image.open(image) as im:
                if im.size!=(volume.shape[2],volume.shape[1]):
                    raise ValueError('Image/mask shape mismatch')
            mask=folder/f'{frame}.png'
            binary=volume[frame]!=0
            Image.fromarray(binary.astype(np.uint8)*255).save(mask)
            rows.append(dict(dataset='local',case_id=case['case_id'],frame=frame,
                split=case['split'],image=str(image),mask=str(mask),source='provided',
                foreground_pixels=int(binary.sum())))
        counts.append(dict(case_id=case['case_id'],split=case['split'],
            positive_frames=len(positive),excluded_empty_frames=len(volume)-len(positive)))
        print(f'WD{case["case_id"]:08d}: {len(positive)} original positive pairs',flush=True)
    tg=[]
    cfg=config()
    frozen=read_json(ROOT/'provenance/tg3k/split_ids.json')
    for r in read_csv(ROOT/'provenance/tg3k/split_manifest.csv'):
        if r['split']!='unused':
            continue
        image=Path(cfg['tg3k_root'])/r['image']
        mask=Path(cfg['tg3k_root'])/r['mask']
        with Image.open(image) as im,Image.open(mask) as ma:
            if im.size!=ma.size:
                raise ValueError('TG3K image/mask shape mismatch')
            area=int((np.asarray(ma.convert('L'))>=128).sum())
        tg.append(dict(dataset='tg3k_unused',case_id='TG3K_'+r['group'],frame=r['id'],
            split='test',image=str(image),mask=str(mask),source='provided',foreground_pixels=area))
    if {int(r['frame']) for r in tg}!=set(frozen['unused']) or len(tg)!=len(frozen['unused']):
        raise ValueError('Unused TG3K set changed')
    if set(frozen['unused']) & (set(frozen['train'])|set(frozen['val'])):
        raise ValueError('TG3K split overlap')
    final_test=[r for r in rows if r['split']=='test']+tg
    validate_rows(rows)
    validate_rows(final_test,allow_tg3k=True)
    save_csv(out/'local_pairs.csv',rows)
    save_csv(out/'final_test.csv',final_test)
    save_csv(out/'case_counts.csv',counts)
    save_json(out/'complete.json',dict(policy='original_local_positive_plus_tg3k_unused_test_v3',
        local_case_count=len(cases),local_pair_splits=dict(Counter(r['split'] for r in rows)),
        excluded_empty_frames=sum(r['excluded_empty_frames'] for r in counts),
        added_tg3k_test_pairs=len(tg),final_test_pairs=len(final_test),
        manifest_sha256={name:digest(out/name) for name in ('local_pairs.csv','final_test.csv')},
        training_executed=False,inference_executed=False))
    print(read_json(out/'complete.json'))

if __name__=='__main__':
    main()
