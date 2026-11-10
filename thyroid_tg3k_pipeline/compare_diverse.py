"""Feature-based selection of varied local frames, including unknown empty annotations.

Selection uses original image/annotation features only, never model performance.
These descriptive image features are not clinical diagnoses or lesion labels.
"""
import argparse
from collections import Counter
from datetime import datetime
import html
from pathlib import Path
import numpy as np
from PIL import Image,ImageFilter
import torch
from src.common import ROOT,read_json,save_json,save_csv,new_directory,digest
from src.nrrd_io import read_volume
from src.model import load_model
from src.predict import predict
from compare_models import comparison_image,metrics

FEATURES=[
 ('area_fraction',False,'腺体标注面积较小'),('area_fraction',True,'腺体标注面积较大'),
 ('aspect_ratio',False,'标注形状偏纵向'),('aspect_ratio',True,'标注形状偏横向'),
 ('center_y',False,'标注位于图像较上方'),('center_y',True,'标注位于图像较下方'),
 ('gland_brightness',False,'标注区域灰度较低'),('gland_brightness',True,'标注区域灰度较高'),
 ('contrast',False,'区域内外灰度差较小'),('contrast',True,'区域内外灰度差较大'),
 ('gland_texture',False,'标注区域灰度变化较少'),('gland_texture',True,'标注区域灰度变化较多'),
 ('boundary_gradient',False,'标注边缘梯度较低'),('boundary_gradient',True,'标注边缘梯度较高'),
 ('fill_ratio',False,'标注外接框填充率较低'),('fill_ratio',True,'标注外接框填充率较高'),
 ('center_x',False,'标注位于图像较左侧'),('center_x',True,'标注位于图像较右侧'),
 ('aspect_ratio',True,'横向形态补充'),('contrast',False,'弱区域对比补充')]

def spaced(values,count):
    values=list(values)
    return [values[i] for i in np.unique(np.linspace(0,len(values)-1,min(count,len(values))).astype(int))] if values else []

def extract_features(image,mask):
    gray=np.asarray(image.convert('L').resize((160,100),Image.Resampling.BILINEAR),dtype=np.float32)
    active=gray>8
    values=gray[active]
    if len(values)<100 or float(values.std())<2:
        return None
    result=dict(image_brightness=float(values.mean()),image_texture=float(values.std()))
    if not mask.any():
        return result
    ys,xs=np.nonzero(mask)
    w,h=image.size
    bw,bh=int(xs.max()-xs.min()+1),int(ys.max()-ys.min()+1)
    small=Image.fromarray(mask.astype(np.uint8)*255).resize((160,100),Image.Resampling.NEAREST)
    inside=np.asarray(small)>0
    if not inside.any():
        return None
    ring=(np.asarray(small.filter(ImageFilter.MaxFilter(11)))>0)&~inside&active
    boundary=inside & ~(np.asarray(small.filter(ImageFilter.MinFilter(3)))>0)
    dy,dx=np.gradient(gray)
    gradient=np.hypot(dx,dy)
    mean=float(gray[inside].mean())
    result.update(area_fraction=float(mask.mean()),aspect_ratio=bw/bh,
        center_x=float(xs.mean()/w),center_y=float(ys.mean()/h),fill_ratio=int(mask.sum())/(bw*bh),
        gland_brightness=mean,gland_texture=float(gray[inside].std()),
        contrast=abs(mean-float(gray[ring].mean())) if ring.any() else 0.,
        boundary_gradient=float(gradient[boundary].mean()) if boundary.any() else 0.)
    return result

def choose_diverse(candidates,min_gap=20):
    positive=[r for r in candidates if r['annotation_status']=='provided']
    unknown=[r for r in candidates if r['annotation_status']=='unknown']
    selected=[];seen=set();counts=Counter()
    def add(pool,feature,reverse,reason,prefer_new=False,limit=3):
        available=[r for r in pool if (r['case_id'],r['frame']) not in seen
            and counts[r['case_id']]<limit
            and all(r['case_id']!=s['case_id'] or abs(r['frame']-s['frame'])>=min_gap for s in selected)]
        if prefer_new:
            fresh=[r for r in available if counts[r['case_id']]==0]
            if fresh:
                available=fresh
        if not available:
            return
        row=dict(sorted(available,key=lambda r:(r[feature],r['case_id'],r['frame']),reverse=reverse)[0])
        vals=np.asarray([r[feature] for r in pool])
        row.update(selection_reason=reason,selection_feature=feature,selection_value=row[feature],
            feature_percentile=float(100*np.mean(vals<=row[feature])))
        selected.append(row);seen.add((row['case_id'],row['frame']));counts[row['case_id']]+=1
    for feature,reverse,reason in FEATURES:
        add(positive,feature,reverse,reason,prefer_new=True)
    # All original empty masks remain unknown, including gaps between positive frames.
    for region,name in [('inside','有标注区间内部的空标注帧'),('before','有标注区间之前的空标注帧'),('after','有标注区间之后的空标注帧')]:
        pool=[r for r in unknown if r['empty_region']==region]
        for reverse,tag in [(False,'灰度变化较少'),(True,'灰度变化较多')]:
            add(pool,'image_texture',reverse,name+'；'+tag,prefer_new=True,limit=4)
    for reverse,tag in [(False,'整体较暗'),(True,'整体较亮')]:
        add(unknown,'image_brightness',reverse,'空标注帧；'+tag,prefer_new=True,limit=4)
    return selected

def gallery(out,rows,summary):
    cards=[]
    for r in rows:
        label='已有原始腺体标注' if r['annotation_status']=='provided' else '原标注为空：实际是否含腺体未知'
        cards.append(f'''<article data-kind="{r['annotation_status']}"><h2>WD{r['case_id']:08d} · 第 {r['frame']} 帧 · {label}</h2>
<p>{html.escape(r['selection_reason'])}。特征值 {r['selection_value']:.3f}，候选池分位 {r['feature_percentile']:.1f}%。</p>
<a href="{r['figure']}" target="_blank"><img loading="lazy" src="{r['figure']}"></a></article>''')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>多种图像特征与空标注帧：双模型对比</title>
<style>body{margin:24px;background:#edf1f5;color:#182334;font:16px sans-serif}article{background:white;padding:16px;margin:18px 0;border-radius:10px}img{width:100%;height:auto}h2{font-size:20px}button{padding:10px 18px;margin-right:8px;cursor:pointer}article[data-kind=unknown]{border-left:6px solid #e5a32a}</style>
<h1>多种可观察特征与空标注图像帧</h1><p>四栏依次为：原图、原始NRRD标注（绿）、TG3K预训练预测（橙）、本地继续训练预测（蓝）。点击查看原尺寸。</p>
<p>依据图像灰度、纹理和原标注的形态/位置选样，未按模型表现挑选。特征为候选帧之间的相对量，不代表病理诊断；图像纵向坐标不等于有标定的物理深度。</p>
<p>原标注为空的帧可包含未标出的腺体，第二栏不会补造真值，Dice/IoU不计算。这些样本仅作观察，不修改训练数据集。</p>'''
    page+=f'<p>来源：{summary["split"]}；{summary["cases_selected"]} 个病例；{summary["provided"]} 张有标注帧，{summary["unknown"]} 张空标注帧。</p>'
    page+='''<button onclick="filterCards('all')">全部</button><button onclick="filterCards('provided')">已有腺体标注</button><button onclick="filterCards('unknown')">空标注帧</button>'''
    page+='\n'.join(cards)
    page+='''<script>function filterCards(kind){document.querySelectorAll('article').forEach(x=>x.hidden=kind!=='all'&&x.dataset.kind!==kind)}</script></html>'''
    (out/'index.html').write_text(page,encoding='utf-8')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split',choices=['train','val','test','all'],default='test')
    p.add_argument('--out',type=Path)
    p.add_argument('--positive-candidates-per-case',type=int,default=40)
    p.add_argument('--empty-candidates-per-region',type=int,default=10)
    p.add_argument('--min-gap',type=int,default=20)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args=p.parse_args()
    if args.positive_candidates_per_case<1 or args.empty_candidates_per_region<1 or args.min_gap<0:
        p.error('Invalid candidate counts or minimum frame gap')
    cases=[r for r in read_json(ROOT/'configs/cases.json') if args.split=='all' or r['split']==args.split]
    out=new_directory(args.out or ROOT/'comparisons'/datetime.now().strftime('diverse_%Y%m%d_%H%M%S'))
    candidates=[];case_map={r['case_id']:r for r in cases};rejected=0
    for case in cases:
        if digest(case['annotation'])!=case['annotation_sha256']:
            raise ValueError('Original annotation changed')
        volume,_,_=read_volume(case['annotation'])
        positives=np.flatnonzero(np.any(volume!=0,axis=(1,2))).tolist()
        if not positives or len(volume)!=case['frames']:
            raise ValueError('Unexpected annotation volume')
        lo,hi=min(positives),max(positives)
        empties=np.flatnonzero(~np.any(volume!=0,axis=(1,2))).tolist()
        chosen=spaced(positives,args.positive_candidates_per_case)
        for pool in ([f for f in empties if f<lo],[f for f in empties if lo<f<hi],[f for f in empties if f>hi]):
            chosen+=spaced(pool,args.empty_candidates_per_region)
        for frame in sorted(chosen):
            path=Path(case['image_directory'])/f'{frame}.bmp'
            with Image.open(path) as im:
                if im.size!=(volume.shape[2],volume.shape[1]):
                    raise ValueError('Image/annotation shape mismatch')
                features=extract_features(im,volume[frame]!=0)
            if features is None:
                rejected+=1;continue
            status='provided' if np.any(volume[frame]) else 'unknown'
            candidates.append(dict(case_id=case['case_id'],frame=frame,split=case['split'],image=str(path),
                annotation=case['annotation'],annotation_status=status,
                empty_region='' if status=='provided' else ('before' if frame<lo else 'after' if frame>hi else 'inside'),**features))
        print(f'Features: WD{case["case_id"]:08d}, {len(chosen)} candidate frames',flush=True)
        del volume
    save_json(out/'candidates.json',candidates)
    selected=choose_diverse(candidates,args.min_gap)
    save_json(out/'selection.json',selected)
    if not selected:
        raise ValueError('No eligible frames')
    device=torch.device(args.device);models=[];weights=[]
    for name,stage in [('tg3k','pretrain'),('local','finetune')]:
        path=ROOT/'downloaded_weights'/f'{name}_best.pt'
        model,ckpt=load_model(path,device)
        if ckpt['stage']!=stage:
            raise ValueError('Wrong model stage')
        model.eval();models.append((model,ckpt['image_size']))
        weights.append(dict(name=name,path=str(path),sha256=digest(path),epoch=ckpt['epoch']))
    save_json(out/'run.json',dict(models=weights,split=args.split,min_frame_gap=args.min_gap,
        candidate_sampling='Evenly spaced within positive and before/inside/after empty regions',
        selection='Feature ranking, prefer distinct cases; never use model predictions for selection'))
    results=[]
    for cid in sorted({r['case_id'] for r in selected}):
        case=case_map[cid];volume,_,_=read_volume(case['annotation'])
        for r in [r for r in selected if r['case_id']==cid]:
            with Image.open(r['image']) as im:
                image=im.convert('RGB')
            truth=volume[r['frame']]!=0
            probs=[predict(model,[image],side,device)[0] for model,side in models]
            preds=[p>=.5 for p in probs]
            scores=[metrics(m,truth) for m in preds] if truth.any() else [None,None]
            dest=out/f'WD{cid:08d}'/f'frame_{r["frame"]:05d}';dest.mkdir(parents=True)
            image.save(dest/'original.png')
            Image.fromarray(truth.astype(np.uint8)*255).save(dest/'annotation_mask.png')
            for name,mask,prob in zip(('tg3k','local'),preds,probs):
                Image.fromarray(mask.astype(np.uint8)*255).save(dest/f'{name}_mask.png')
                np.save(dest/f'{name}_probability.npy',prob.astype(np.float32))
            title=f'WD{cid:08d} | frame {r["frame"]} | {r["split"]} | {r["annotation_status"]} | threshold 0.5'
            comparison_image(image,truth,preds,scores,title).save(dest/'comparison.png')
            row=dict(r,figure=(dest/'comparison.png').relative_to(out).as_posix(),
                image_sha256=digest(r['image']),annotation_sha256=case['annotation_sha256'],
                tg3k_predicted_pixels=int(preds[0].sum()),local_predicted_pixels=int(preds[1].sum()))
            for i,name in enumerate(('tg3k','local')):
                for metric in ('dice','iou'):
                    row[f'{name}_{metric}']=scores[i][metric] if scores[i] else None
            results.append(row)
            print(f'Inference: WD{cid:08d}/{r["frame"]} {r["annotation_status"]}',flush=True)
        del volume
    # Keep selection-reason order rather than ranking by model performance.
    lookup={(r['case_id'],r['frame']):r for r in results}
    results=[lookup[(r['case_id'],r['frame'])] for r in selected]
    summary=dict(split=args.split,cases_selected=len({r['case_id'] for r in results}),
        candidate_frames=len(candidates),quality_rejected=rejected,
        provided=sum(r['annotation_status']=='provided' for r in results),
        unknown=sum(r['annotation_status']=='unknown' for r in results),
        unknown_metrics='Not computed; empty annotation is not confirmed background',
        scope='Feature-selected examples, not an unbiased performance evaluation')
    fields=list(dict.fromkeys(k for r in results for k in r))
    save_csv(out/'frames.csv',results,fields)
    save_json(out/'results.json',results);save_json(out/'summary.json',summary)
    gallery(out,results,summary)
    save_json(out/'complete.json',dict(frames=len(results),training_executed=False,source_files_modified=False))
    print(f'Completed: {out/"index.html"}',flush=True)

if __name__=='__main__':
    main()
