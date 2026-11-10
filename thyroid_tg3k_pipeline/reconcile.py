"""Infer all raw frames in approved gland cases and write derived labels only."""
import argparse
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from PIL import Image
from src.common import ROOT,config,read_json,save_json,save_csv,digest,new_directory
from src.nrrd_io import read_volume,write_volume
from src.model import load_model
from src.predict import predict,reconcile

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--run',default='corrected_v1')
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--overlays',action='store_true',help='Save overlay for each newly added frame (extra disk space)')
    args=p.parse_args()
    if Path(args.run).name!=args.run:
        p.error('--run must be a simple directory name')
    cfg=config(); device=torch.device(args.device)
    model,ckpt=load_model(args.checkpoint,device); model.eval()
    if ckpt['stage']!='pretrain':
        raise ValueError('Correction requires frozen TG3K-pretrained teacher')
    out=new_directory(ROOT/'outputs'/args.run)
    checkpoint_hash=digest(args.checkpoint)
    save_json(out/'run.json',dict(checkpoint=str(args.checkpoint.resolve()),sha256=checkpoint_hash,
        threshold=cfg['threshold'],min_area_pixels=cfg['min_area_pixels'],config=cfg))
    manifest=[]; audit=[]; cases=read_json(ROOT/'configs/cases.json')
    for case in cases:
        cid=case['case_id']; folder=Path(case['image_directory'])
        if digest(case['annotation'])!=case['annotation_sha256']:
            raise ValueError('Source annotation changed')
        original,header,label=read_volume(case['annotation'])
        if len(original)!=case['frames']:
            raise ValueError('Frame count changed')
        expected=list(range(len(original)))
        actual=sorted(int(p.stem) for p in folder.glob('*.bmp') if p.stem.isdecimal())
        if actual!=expected:
            raise ValueError('Raw image indices changed')
        corrected=original.copy()
        case_out=out/f'WD{cid:08d}'; case_out.mkdir()
        masks=case_out/'masks'; masks.mkdir()
        if args.overlays:
            (case_out/'overlays').mkdir()
        case_audit=[]
        for start in range(0,len(original),cfg['batch_size']):
            indices=range(start,min(start+cfg['batch_size'],len(original)))
            images=[]
            for frame in indices:
                with Image.open(folder/f'{frame}.bmp') as im:
                    if im.size!=(original.shape[2],original.shape[1]):
                        raise ValueError(f'Image size mismatch: {cid}/{frame}')
                    images.append(im.convert('RGB'))
            probabilities=predict(model,images,ckpt['image_size'],device)
            for frame,image,prob in zip(indices,images,probabilities):
                result,source,pred=reconcile(original[frame],prob,label,cfg['threshold'],cfg['min_area_pixels'])
                corrected[frame]=result
                mask_path=masks/f'{frame}.png'
                if source!='unknown':
                    Image.fromarray((result>0).astype(np.uint8)*255).save(mask_path)
                    manifest.append(dict(case_id=cid,frame=frame,split=case['split'],
                        image=str(folder/f'{frame}.bmp'),mask=str(mask_path.resolve()),source=source))
                original_mask=original[frame]>0
                denom=int(original_mask.sum()+pred.sum())
                agreement=2*int((original_mask & pred).sum())/denom if denom else 1.
                row=dict(case_id=cid,frame=frame,image_filename=f'{frame}.bmp',split=case['split'],source=source,
                    original_pixels=int(original_mask.sum()),predicted_pixels=int(pred.sum()),
                    corrected_pixels=int((result>0).sum()),agreement_dice=agreement if original_mask.any() else '',
                    mean_predicted_probability=float(prob[pred].mean()) if pred.any() else 0.,
                    image=str(folder/f'{frame}.bmp'),original_annotation=case['annotation'])
                case_audit.append(row)
                if args.overlays and source=='pseudo':
                    a=np.asarray(image).copy(); m=result>0
                    a[m]=(a[m]*.55+np.array([255,100,0])*.45).astype(np.uint8)
                    Image.fromarray(a).save(case_out/'overlays'/f'{frame}.jpg')
        positive=np.any(original!=0,axis=(1,2))
        if not np.array_equal(original[positive],corrected[positive]):
            raise AssertionError('Original nonempty frame changed')
        dest=case_out/f'WD{cid:08d}_gland.seg.nrrd'
        write_volume(dest,corrected,header)
        save_csv(case_out/'audit.csv',case_audit)
        audit.extend(case_audit)
        print(f'WD{cid:08d}: {dict(Counter(r["source"] for r in case_audit))}',flush=True)
    save_csv(out/'manifest.csv',manifest)
    save_csv(out/'frame_audit.csv',audit)
    save_csv(out/'added_frames.csv',[r for r in audit if r['source']=='pseudo'],list(audit[0]))
    save_json(out/'complete.json',dict(cases=len(cases),frames=len(audit),
        counts=dict(Counter(r['source'] for r in audit)),teacher_sha256=checkpoint_hash,
        original_nonempty_frames_preserved=True))

if __name__=='__main__':
    main()
