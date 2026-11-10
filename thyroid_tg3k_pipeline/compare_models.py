"""Compare TG3K-pretrained and locally fine-tuned models on original local frames.

Ground truth is read directly from the original gland NRRD, never predictions.
No optimizer, backward pass, source edits or training are performed.
"""
import argparse
from collections import defaultdict
from datetime import datetime
import html
from pathlib import Path
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import torch
from src.common import ROOT,read_json,save_json,save_csv,digest,new_directory
from src.nrrd_io import read_volume
from src.model import load_model
from src.predict import predict

def choose_cases(cases,split,count,seed,case_ids=None):
    eligible=sorted([r for r in cases if split=='all' or r['split']==split],key=lambda r:r['case_id'])
    if case_ids:
        wanted=set(case_ids)
        chosen=[r for r in eligible if r['case_id'] in wanted]
        if {r['case_id'] for r in chosen}!=wanted:
            raise ValueError('Requested case is not in the approved case list / requested split')
        return chosen
    if count<1 or count>len(eligible):
        raise ValueError(f'--cases must be between 1 and {len(eligible)}')
    return sorted(random.Random(seed).sample(eligible,count),key=lambda r:r['case_id'])

def choose_frames(volume,count,seed,cid,pool):
    candidates=np.flatnonzero(np.any(volume!=0,axis=(1,2))).tolist() if pool=='positive' else list(range(len(volume)))
    if count<1 or count>len(candidates):
        raise ValueError(f'Case {cid}: requested {count}, available {len(candidates)} frames')
    return sorted(random.Random(f'{seed}:{cid}').sample(candidates,count))

def metrics(pred,truth):
    intersection=int((pred & truth).sum())
    a,b=int(pred.sum()),int(truth.sum())
    union=a+b-intersection
    return dict(dice=2*intersection/(a+b) if a+b else 1.,
                iou=intersection/union if union else 1.,
                predicted_pixels=a,annotated_pixels=b)

def overlay(image,mask,color):
    rgb=np.asarray(image.convert('RGB')).copy()
    rgb[mask]=(rgb[mask]*.6+np.asarray(color)*.4).round().astype(np.uint8)
    binary=Image.fromarray(mask.astype(np.uint8)*255)
    eroded=np.asarray(binary.filter(ImageFilter.MinFilter(3)))>0
    rgb[mask & ~eroded]=color
    return Image.fromarray(rgb)

def font(size):
    for name in ('C:/Windows/Fonts/arial.ttf','DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(name,size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)

def comparison_image(image,truth,predictions,scores,title):
    w,h=image.size
    header=112;footer=40
    canvas=Image.new('RGB',(w*4,h+header+footer),(245,247,250))
    draw=ImageDraw.Draw(canvas)
    draw.text((16,8),title,font=font(24),fill=(20,30,45))
    labels=['Original ultrasound','Original NRRD annotation','TG3K pretrained','Local fine-tuned']
    if not truth.any():
        labels[1]='No annotation (status unknown)'
    panels=[image.convert('RGB'),overlay(image,truth,(40,210,100)),
            overlay(image,predictions[0],(255,160,35)),overlay(image,predictions[1],(30,170,255))]
    for i,(label,panel) in enumerate(zip(labels,panels)):
        draw.text((i*w+16,46),label,font=font(22),fill=(20,30,45))
        if i>=2:
            score=scores[i-2]
            score_text=f'Dice {score["dice"]:.4f} | IoU {score["iou"]:.4f}' if score is not None else 'Unlabeled: Dice / IoU not evaluated'
            draw.text((i*w+16,78),score_text,font=font(18),fill=(50,60,70))
        canvas.paste(panel,(i*w,header))
    draw.text((16,h+header+8),'Green: original annotation | Orange: TG3K prediction | Blue: local prediction | All panels use original coordinates',font=font(18),fill=(30,40,50))
    return canvas

def write_gallery(out,rows,summary):
    cards=[]
    for r in rows:
        title=f'WD{r["case_id"]:08d} / frame {r["frame"]} / {r["split"]}'
        cards.append(f'<article><h2>{html.escape(title)}</h2><a href="{r["figure"]}" target="_blank"><img loading="lazy" src="{r["figure"]}"></a></article>')
    body='''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>双模型腺体分割对比</title><style>body{margin:24px;font:16px sans-serif;background:#edf1f5;color:#152238}article{background:white;padding:16px;margin:20px 0;border-radius:10px}img{width:100%;height:auto}h2{font-size:18px}pre{white-space:pre-wrap}</style>
<h1>甲状腺分割：TG3K预训练 / 本地继续训练</h1>
<p>从左到右：原图、原始NRRD标注（绿）、TG3K预测（橙）、本地训练预测（蓝）。点击图像查看原尺寸。</p>
<p>同一批随机帧供两个模型共同推理；指标只反映本次抽样，不替代完整测试集结果。原标注为空也不代表已确认阴性。</p>'''
    import json
    body+='<pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre>'
    (out/'index.html').write_text(body+'\n'.join(cards)+'</html>',encoding='utf-8')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tg3k-checkpoint',type=Path,default=ROOT/'downloaded_weights/tg3k_best.pt')
    p.add_argument('--local-checkpoint',type=Path,default=ROOT/'downloaded_weights/local_best.pt')
    p.add_argument('--split',choices=['train','val','test','all'],default='test')
    p.add_argument('--cases',type=int,default=8)
    p.add_argument('--case-ids',type=int,nargs='+',help='Explicit approved WD numeric IDs; overrides --cases')
    p.add_argument('--frames-per-case',type=int,default=2)
    p.add_argument('--seed',type=int,default=20261017)
    p.add_argument('--frame-pool',choices=['positive','all'],default='positive')
    p.add_argument('--threshold',type=float,default=.5)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out',type=Path)
    args=p.parse_args()
    if not 0<args.threshold<1 or args.batch_size<1:
        p.error('Threshold must be in (0,1); batch size must be positive')
    chosen=choose_cases(read_json(ROOT/'configs/cases.json'),args.split,args.cases,args.seed,args.case_ids)
    device=torch.device(args.device)
    loaded=[];metadata=[]
    for name,path,stage in [('tg3k',args.tg3k_checkpoint,'pretrain'),('local',args.local_checkpoint,'finetune')]:
        model,checkpoint=load_model(path,device)
        if checkpoint['stage']!=stage:
            raise ValueError(f'{name}: expected stage {stage}, got {checkpoint["stage"]}')
        model.eval();loaded.append((model,checkpoint['image_size']))
        metadata.append(dict(name=name,path=str(path.resolve()),sha256=digest(path),
            stage=stage,epoch=checkpoint['epoch'],image_size=checkpoint['image_size']))
    out=new_directory(args.out or ROOT/'comparisons'/datetime.now().strftime('compare_%Y%m%d_%H%M%S'))
    rows=[];selection=[]
    save_json(out/'run.json',dict(models=metadata,seed=args.seed,split=args.split,
        frame_pool=args.frame_pool,threshold=args.threshold,device=str(device),cases=[r['case_id'] for r in chosen]))
    for case in chosen:
        cid=case['case_id']
        if digest(case['annotation'])!=case['annotation_sha256']:
            raise ValueError('Original NRRD changed: '+case['annotation'])
        volume,_,_=read_volume(case['annotation'])
        if len(volume)!=case['frames']:
            raise ValueError('Unexpected volume frame count')
        frames=choose_frames(volume,args.frames_per_case,args.seed,cid,args.frame_pool)
        for start in range(0,len(frames),args.batch_size):
            batch=frames[start:start+args.batch_size]
            images=[]
            for frame in batch:
                with Image.open(Path(case['image_directory'])/f'{frame}.bmp') as im:
                    if im.size!=(volume.shape[2],volume.shape[1]):
                        raise ValueError('Image / NRRD shape mismatch')
                    images.append(im.convert('RGB'))
            probs=[predict(model,images,side,device) for model,side in loaded]
            for i,(frame,image) in enumerate(zip(batch,images)):
                dest=out/f'WD{cid:08d}'/f'frame_{frame:05d}';dest.mkdir(parents=True)
                truth=volume[frame]!=0
                predictions=[prob[i]>=args.threshold for prob in probs]
                scores=[metrics(pred,truth) for pred in predictions]
                image.save(dest/'original.png')
                Image.fromarray(truth.astype(np.uint8)*255).save(dest/'annotation_mask.png')
                for name,pred,prob in zip(('tg3k','local'),predictions,probs):
                    Image.fromarray(pred.astype(np.uint8)*255).save(dest/f'{name}_mask.png')
                    np.save(dest/f'{name}_probability.npy',prob[i].astype(np.float32))
                title=f'WD{cid:08d} | frame {frame} | {case["split"]} | threshold {args.threshold:g}'
                figure=dest/'comparison.png'
                comparison_image(image,truth,predictions,scores,title).save(figure)
                row=dict(case_id=cid,frame=frame,split=case['split'],
                    image=str(Path(case['image_directory'])/f'{frame}.bmp'),annotation=case['annotation'],
                    annotated_pixels=int(truth.sum()),tg3k_dice=scores[0]['dice'],tg3k_iou=scores[0]['iou'],
                    local_dice=scores[1]['dice'],local_iou=scores[1]['iou'],
                    dice_change=scores[1]['dice']-scores[0]['dice'],figure=figure.relative_to(out).as_posix())
                rows.append(row)
                selection.append(dict(case_id=cid,frame=frame,image=row['image'],image_sha256=digest(row['image']),
                    annotation=case['annotation'],annotation_sha256=case['annotation_sha256']))
                print(f'WD{cid:08d}/{frame}: TG3K Dice={scores[0]["dice"]:.4f}, local Dice={scores[1]["dice"]:.4f}',flush=True)
        del volume
    summary=dict(cases=len(chosen),frames=len(rows),split=args.split,seed=args.seed,
        frame_pool=args.frame_pool,threshold=args.threshold,
        metrics_scope='Random sampled frames only; not full-test-set metrics')
    for name in ('tg3k','local'):
        grouped=defaultdict(list)
        for row in rows:
            grouped[row['case_id']].append(row)
        summary[name]={f'frame_mean_{m}':float(np.mean([r[f'{name}_{m}'] for r in rows])) for m in ('dice','iou')}
        summary[name].update({f'case_macro_{m}':float(np.mean([np.mean([r[f'{name}_{m}'] for r in rs]) for rs in grouped.values()])) for m in ('dice','iou')})
    save_csv(out/'metrics.csv',rows)
    save_json(out/'selection.json',selection)
    save_json(out/'summary.json',summary)
    write_gallery(out,rows,summary)
    save_json(out/'complete.json',dict(frames=len(rows),training_executed=False,original_files_modified=False))
    print(f'Completed. Open {out/"index.html"}',flush=True)

if __name__=='__main__':
    main()
