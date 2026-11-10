"""Explicitly invoked training only. Stage 2 requires a TG3K checkpoint."""
import argparse
import random
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader,WeightedRandomSampler
from src.common import ROOT,config,read_csv,save_json,new_directory,digest,save_csv
from src.data import Frames
from src.model import MobileUNet,load_model,ARCHITECTURE

def loss_fn(logits,y,v,weights):
    logits=logits.float()
    bce=(F.binary_cross_entropy_with_logits(logits,y,reduction='none')*v).sum((1,2,3))/v.sum((1,2,3)).clamp_min(1)
    p=logits.sigmoid()*v
    target=y*v
    dice=(2*(p*target).sum((1,2,3))+1)/(p.sum((1,2,3))+target.sum((1,2,3))+1)
    return ((.5*bce+1-dice)*weights).sum()/weights.sum().clamp_min(1e-6)

def select_rows(rows,split,stage):
    # Both stages use original annotations only.
    return [r for r in rows if r['split']==split and r['source']=='provided']

@torch.inference_mode()
def validate(model,loader,device):
    model.eval()
    groups=defaultdict(list)
    for x,y,v,cids,_ in loader:
        x,y,v=x.to(device),y.to(device),v.to(device)
        pred=(model(x).sigmoid()>=.5)*v
        target=y*v
        scores=(2*(pred*target).sum((1,2,3))+1e-6)/(pred.sum((1,2,3))+target.sum((1,2,3))+1e-6)
        for cid,score in zip(cids,scores.tolist()):
            groups[cid].append(score)
    if not groups:
        raise ValueError('Empty validation set')
    return float(np.mean([np.mean(scores) for scores in groups.values()]))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['pretrain','finetune'],required=True)
    p.add_argument('--manifest',type=Path)
    p.add_argument('--init',type=Path)
    p.add_argument('--run',required=True)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--epochs',type=int)
    args=p.parse_args()
    cfg=config()
    if args.stage=='finetune' and (args.init is None or args.manifest is None):
        p.error('finetune requires --init TG3K_best.pt and --manifest data/original_pairs_v3/local_pairs.csv')
    if Path(args.run).name!=args.run:
        p.error('--run must be a simple directory name')
    seed=cfg['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.benchmark=False
    manifest=args.manifest or ROOT/'data/tg3k.csv'
    rows=read_csv(manifest)
    train=select_rows(rows,'train',args.stage)
    val=select_rows(rows,'val',args.stage)
    if not train or not val:
        raise ValueError('Train or validation set empty')
    if args.stage=='finetune':
        from src.policy import validate_rows,require_complete_manifest
        require_complete_manifest(manifest)
        validate_rows(rows)
    device=torch.device(args.device)
    if args.init:
        model,checkpoint=load_model(args.init,device)
        if args.stage=='finetune' and checkpoint['stage']!='pretrain':
            raise ValueError('Stage 2 must initialize from TG3K pretraining')
        if checkpoint['image_size']!=cfg['image_size']:
            raise ValueError('Preprocessing differs from checkpoint')
    else:
        model=MobileUNet(cfg['imagenet_initialization']).to(device)
    out=new_directory(ROOT/'runs'/args.run)
    save_json(out/'config.json',dict(config=cfg,stage=args.stage,manifest=str(manifest.resolve()),
        manifest_sha256=digest(manifest),init=str(args.init),init_sha256=digest(args.init) if args.init else None))
    ds=Frames(train,cfg['image_size'],True,cfg['pseudo_weight'])
    counts=Counter(r['case_id'] for r in train)
    sampler=WeightedRandomSampler([1/counts[r['case_id']] for r in train],len(train),replacement=True)
    loader=DataLoader(ds,batch_size=cfg['batch_size'],sampler=sampler,num_workers=cfg['num_workers'])
    vl=DataLoader(Frames(val,cfg['image_size']),batch_size=cfg['batch_size'],num_workers=cfg['num_workers'])
    epochs=args.epochs if args.epochs is not None else cfg['epochs']
    if epochs<1:
        raise ValueError('epochs must be positive')
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg[args.stage+'_lr'],weight_decay=cfg['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs)
    amp=device.type=='cuda'
    scaler=torch.amp.GradScaler('cuda',enabled=amp)
    best=-1.; stale=0; history=[]
    for epoch in range(1,epochs+1):
        model.train(); total=0.
        for x,y,v,_,weights in loader:
            x,y,v,weights=x.to(device),y.to(device),v.to(device),weights.to(device).float()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=amp):
                loss=loss_fn(model(x),y,v,weights)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            scaler.step(optimizer); scaler.update()
            total+=loss.item()*len(x)
        score=validate(model,vl,device)
        scheduler.step()
        history.append(dict(epoch=epoch,train_loss=total/len(ds),val_group_dice=score))
        save_csv(out/'history.csv',history)
        if score>best:
            best=score; stale=0
            torch.save(dict(model=model.state_dict(),architecture=ARCHITECTURE,stage=args.stage,
                image_size=cfg['image_size'],epoch=epoch,val_dice=score,
                preprocessing='gray_rgb_imagenet_letterbox_v1',manifest_sha256=digest(manifest)),out/'best.pt')
        else:
            stale+=1
        print(history[-1],flush=True)
        if stale>=cfg['patience']:
            break
    save_json(out/'complete.json',dict(best_val_dice=best,epochs_completed=epoch,test_used=False))

if __name__=='__main__':
    main()
