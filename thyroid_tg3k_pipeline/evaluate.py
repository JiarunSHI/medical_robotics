"""Original local positives plus previously unused TG3K evaluation at original image resolution."""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from src.common import ROOT,read_csv,read_json,save_json,save_csv,new_directory,digest
from src.model import load_model
from src.predict import predict
from train import select_rows
from src.policy import validate_rows,require_complete_manifest

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--run',default='test_v1')
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args=p.parse_args()
    if Path(args.run).name!=args.run:
        p.error('Invalid run name')
    require_complete_manifest(args.manifest)
    all_rows=read_csv(args.manifest)
    validate_rows(all_rows,allow_tg3k=True)
    rows=select_rows(all_rows,'test','finetune')
    if len(rows)!=len(all_rows) or not rows:
        raise ValueError('Expected final test manifest only')
    frozen_unused=set(read_json(ROOT/'provenance/tg3k/split_ids.json')['unused'])
    if {int(r['frame']) for r in rows if r['dataset']=='tg3k_unused'}!=frozen_unused:
        raise ValueError('Final test must contain the full frozen unused TG3K set')
    device=torch.device(args.device)
    model,ckpt=load_model(args.checkpoint,device); model.eval()
    out=new_directory(ROOT/'evaluations'/args.run)
    metrics=[]; groups=defaultdict(list)
    for r in rows:
        with Image.open(r['image']) as im,Image.open(r['mask']) as ma:
            prob=predict(model,[im],ckpt['image_size'],device)[0]
            y=np.asarray(ma.convert('L'))>=128
        pred=prob>=.5
        intersection=int((pred & y).sum()); union=int((pred | y).sum())
        dice=(2*intersection+1e-6)/(int(pred.sum()+y.sum())+1e-6)
        iou=(intersection+1e-6)/(union+1e-6)
        row=dict(dataset=r['dataset'],case_id=r['case_id'],frame=r['frame'],source=r['source'],dice=dice,iou=iou)
        metrics.append(row); groups[(r['dataset'],r['case_id'])].append(row)
    per_case=[dict(dataset=c[0],case_id=c[1],dice=float(np.mean([r['dice'] for r in rs])),
                   iou=float(np.mean([r['iou'] for r in rs])),frames=len(rs)) for c,rs in groups.items()]
    save_csv(out/'frames.csv',metrics); save_csv(out/'cases.csv',per_case)
    by_dataset={}
    for dataset in ('local','tg3k_unused'):
        subset=[r for r in metrics if r['dataset']==dataset]
        grouped=[r for r in per_case if r['dataset']==dataset]
        by_dataset[dataset]=dict(frames=len(subset),groups=len(grouped),
            frame_mean_dice=float(np.mean([r['dice'] for r in subset])),
            frame_mean_iou=float(np.mean([r['iou'] for r in subset])),
            group_macro_dice=float(np.mean([r['dice'] for r in grouped])),
            group_macro_iou=float(np.mean([r['iou'] for r in grouped])))
    save_json(out/'summary.json',dict(by_dataset=by_dataset,frames=len(rows),
        combined_frame_mean_dice=float(np.mean([r['dice'] for r in metrics])),
        combined_frame_mean_iou=float(np.mean([r['iou'] for r in metrics])),
        threshold=.5,checkpoint_sha256=digest(args.checkpoint),
        scope='Original local positive frames + unused TG3K pairs. TG3K groups are not patient IDs; former gap frames can be adjacent to pretraining frames.'))

if __name__=='__main__':
    main()
