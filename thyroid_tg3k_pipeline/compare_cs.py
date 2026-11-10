"""Feature-selected CS examples: raw ultrasound, original CS NRRD, CMU-Net prediction."""
import argparse
from datetime import datetime
from pathlib import Path
import html
import numpy as np
from PIL import Image,ImageDraw
import torch
from src.common import ROOT,read_json,save_json,save_csv,digest,new_directory
from src.nrrd_io import read_volume
from src.cs_model import load_cs,predict_cs
from compare_models import overlay,font,metrics
from compare_diverse import extract_features,spaced,choose_diverse

def render_cs(image,truth,prediction,score,title):
    w,h=image.size
    result=Image.new('RGB',(w*3,h+152),(245,247,250));draw=ImageDraw.Draw(result)
    draw.text((16,8),title,font=font(24),fill=(20,30,45))
    panels=[image.convert('RGB'),overlay(image,truth,(40,210,100)),overlay(image,prediction,(30,170,255))]
    labels=['Original ultrasound','Original CS NRRD annotation' if truth.any() else 'No CS annotation (unknown)','CMU-Net CS prediction']
    for i,(panel,label) in enumerate(zip(panels,labels)):
        draw.text((i*w+16,46),label,font=font(22),fill=(20,30,45))
        result.paste(panel,(i*w,112))
    info=f'Dice {score["dice"]:.4f} | IoU {score["iou"]:.4f}' if score else 'Unlabeled: Dice / IoU not evaluated'
    draw.text((w*2+16,78),info,font=font(18),fill=(40,50,60))
    draw.text((16,h+120),'Green: original CS annotation | Blue: CS model prediction | Original image coordinates | Threshold 0.5',font=font(18),fill=(30,40,50))
    return result

def gallery(out,rows,summary):
    body='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>CS CMU-Net 分割对比</title>
<style>body{font:16px sans-serif;margin:24px;background:#edf1f5;color:#182334}article{background:white;padding:16px;margin:20px 0;border-radius:10px}article[data-kind=unknown]{border-left:6px solid #e5a32a}img{width:100%;height:auto}button{padding:10px 18px;margin-right:8px;cursor:pointer}h2{font-size:20px}</style>
<h1>CS 模型：原图、原始标注、预测分割</h1><p>左：原始超声图像；中：原始CS NRRD标注（绿色）；右：CMU-Net CS预测（蓝色）。点击图像可查看原尺寸。</p>
<p>CS是原项目的目标标签。模型只学习原CS标注定义，不把其预测解释为所有血管结构。输入是BMP超声图像，NRRD用作原始标注对照。</p>
<p>依据原图与标注的可观察特征选样，不按预测质量挑选。特征是相对数值，不是病理诊断。空标注帧实际是否含目标未知，不计算Dice/IoU、不改写标注。</p>'''
    body+=f'<p>{summary["cases"]} 个{summary["split"]}病例；{summary["provided"]} 张有CS标注，{summary["unknown"]} 张空CS标注。抽样分数不代表完整测试集。</p>'
    body+='''<button onclick="filterCards('all')">全部</button><button onclick="filterCards('provided')">有CS标注</button><button onclick="filterCards('unknown')">空CS标注</button>'''
    for r in rows:
        state='有原始CS标注' if r['annotation_status']=='provided' else '原始CS标注为空（未知）'
        body+=f'<article data-kind="{r["annotation_status"]}"><h2>WD{r["case_id"]:08d} · {r["frame"]}.bmp · {state}</h2><p>{html.escape(r["selection_reason"])}；特征候选分位 {r["feature_percentile"]:.1f}%</p><a target="_blank" href="{r["figure"]}"><img loading="lazy" src="{r["figure"]}"></a></article>'
    body+='''<script>function filterCards(kind){document.querySelectorAll('article').forEach(x=>x.hidden=kind!=='all'&&x.dataset.kind!==kind)}</script></html>'''
    (out/'index.html').write_text(body,encoding='utf-8')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'downloaded_weights/cmunet_cs_best.pt')
    p.add_argument('--split',choices=['train','val','test','all'],default='test')
    p.add_argument('--out',type=Path)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--positive-candidates-per-case',type=int,default=40)
    p.add_argument('--empty-candidates-per-region',type=int,default=10)
    p.add_argument('--min-gap',type=int,default=20)
    args=p.parse_args()
    if args.positive_candidates_per_case<1 or args.empty_candidates_per_region<1 or args.min_gap<0:
        p.error('Invalid selection parameters')
    cases=[r for r in read_json(ROOT/'configs/cs_cases.json') if args.split=='all' or r['split']==args.split]
    device=torch.device(args.device);torch.set_num_threads(4)
    model,checkpoint=load_cs(args.checkpoint,device)
    train=set(checkpoint['config']['train_cases']);val=set(checkpoint['config']['val_cases'])
    for r in cases:
        if (r['split']=='test' and r['case_id'] in train|val) or (r['split']=='train' and r['case_id'] not in train) or (r['split']=='val' and r['case_id'] not in val):
            raise ValueError('CS case split conflicts with checkpoint metadata')
        if not Path(r['annotation']).name.endswith('_cs.seg.nrrd'):
            raise ValueError('Expected CS annotation, not gland annotation')
    out=new_directory(args.out or ROOT/'comparisons'/datetime.now().strftime('cs_%Y%m%d_%H%M%S'))
    candidates=[]
    for case in cases:
        if digest(case['annotation'])!=case['annotation_sha256']:
            raise ValueError('Original CS NRRD changed')
        volume,_,_=read_volume(case['annotation'])
        if len(volume)!=case['frames']:
            raise ValueError('Volume frame count mismatch')
        present=np.any(volume!=0,axis=(1,2));positive=np.flatnonzero(present).tolist();empty=np.flatnonzero(~present).tolist()
        if not positive:
            raise ValueError('Previously accepted CS case now has no positive annotations')
        lo,hi=min(positive),max(positive)
        frames=spaced(positive,args.positive_candidates_per_case)
        for pool in ([f for f in empty if f<lo],[f for f in empty if lo<f<hi],[f for f in empty if f>hi]):
            frames+=spaced(pool,args.empty_candidates_per_region)
        for frame in sorted(frames):
            path=Path(case['image_directory'])/f'{frame}.bmp'
            with Image.open(path) as image:
                if image.size!=(volume.shape[2],volume.shape[1]):
                    raise ValueError('CS image / NRRD size mismatch')
                features=extract_features(image,volume[frame]!=0)
            if features is None:
                continue
            status='provided' if present[frame] else 'unknown'
            candidates.append(dict(case_id=case['case_id'],frame=frame,split=case['split'],image=str(path),annotation=case['annotation'],
                annotation_status=status,empty_region='' if status=='provided' else 'before' if frame<lo else 'after' if frame>hi else 'inside',**features))
        print(f'CS features: WD{case["case_id"]:08d}, {len(frames)} candidate frames',flush=True)
        del volume
    selected=choose_diverse(candidates,args.min_gap)
    for r in selected:
        r['selection_reason']=r['selection_reason'].replace('腺体','CS目标')
    save_json(out/'candidates.json',candidates);save_json(out/'selection.json',selected)
    save_json(out/'run.json',dict(checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=digest(args.checkpoint),
        epoch=checkpoint['epoch'],target='cs',architecture=checkpoint['config']['architecture'],
        preprocessing='Original RGB -> aspect-preserving 256 letterbox -> float32 /255; no ImageNet normalization',
        probability_restore='Remove padding then bilinear resize to original resolution',threshold=.5,
        min_gap=args.min_gap,split=args.split,device=str(device),selection_uses_predictions=False))
    results=[]
    for case in cases:
        chosen=[r for r in selected if r['case_id']==case['case_id']]
        if not chosen:
            continue
        volume,_,_=read_volume(case['annotation'])
        for r in chosen:
            with Image.open(r['image']) as im:
                image=im.convert('RGB')
            truth=volume[r['frame']]!=0
            probability=predict_cs(model,[image],device)[0]
            if not np.isfinite(probability).all():
                raise ValueError('Nonfinite CS probability')
            prediction=probability>.5
            score=metrics(prediction,truth) if truth.any() else None
            dest=out/f'WD{r["case_id"]:08d}'/f'frame_{r["frame"]:05d}';dest.mkdir(parents=True)
            image.save(dest/'original.png')
            Image.fromarray(truth.astype(np.uint8)*255).save(dest/'cs_annotation_mask.png')
            Image.fromarray(prediction.astype(np.uint8)*255).save(dest/'cs_prediction_mask.png')
            np.save(dest/'cs_probability.npy',probability.astype(np.float32))
            title=f'WD{r["case_id"]:08d} | frame {r["frame"]} | CS | {r["split"]} | {r["annotation_status"]}'
            render_cs(image,truth,prediction,score,title).save(dest/'comparison.png')
            results.append(dict(r,dice=score['dice'] if score else None,iou=score['iou'] if score else None,
                predicted_pixels=int(prediction.sum()),annotated_pixels=int(truth.sum()),
                image_sha256=digest(r['image']),annotation_sha256=case['annotation_sha256'],figure=(dest/'comparison.png').relative_to(out).as_posix()))
            text=f'{score["dice"]:.4f}' if score else 'N/A (unknown annotation)'
            print(f'CS WD{r["case_id"]:08d}/{r["frame"]} Dice={text}',flush=True)
        del volume
    lookup={(r['case_id'],r['frame']):r for r in results}
    results=[lookup[(r['case_id'],r['frame'])] for r in selected]
    scored=[r for r in results if r['dice'] is not None]
    summary=dict(target='cs',split=args.split,cases=len({r['case_id'] for r in results}),frames=len(results),
        provided=len(scored),unknown=len(results)-len(scored),candidate_frames=len(candidates),
        sampled_positive_mean_dice=float(np.mean([r['dice'] for r in scored])),
        sampled_positive_mean_iou=float(np.mean([r['iou'] for r in scored])),
        scope='Feature-selected CS examples at original resolution, not full test evaluation; unknowns excluded from metrics')
    save_json(out/'results.json',results);save_json(out/'summary.json',summary)
    save_csv(out/'frames.csv',results,list(dict.fromkeys(k for r in results for k in r)))
    gallery(out,results,summary)
    save_json(out/'complete.json',dict(frames=len(results),training_executed=False,source_modified=False))
    print(f'Completed: {out/"index.html"}',flush=True)

if __name__=='__main__':
    main()
