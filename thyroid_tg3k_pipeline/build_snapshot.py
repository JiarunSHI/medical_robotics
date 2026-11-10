"""One-time packaging utility; copies manifests only, never trains."""
from collections import Counter
from pathlib import Path
import shutil
from src.common import ROOT, read_json, save_json, digest

def main():
    previous=Path.home()/'thyroid_cmunet'
    selected=read_json(previous/'manifest.json')
    splits=read_json(previous/'case_split.json')
    audit=read_json(Path('E:/thyroid_negative_candidates_review/empty_frame_audit.json'))
    ids=sorted({r['case_id'] for r in selected if r['target']=='gland'})
    records=[]
    for cid in ids:
        rows=[r for r in selected if r['target']=='gland' and r['case_id']==cid]
        dirs={str(Path(r['image']).parent) for r in rows}
        if len(dirs)!=1 or {r['split'] for r in rows}!={splits[str(cid)]}:
            raise ValueError(cid)
        ar=next(r for r in audit['records'] if r['case_id']==cid and r['target']=='gland')
        folder=Path(dirs.pop())
        if folder!=Path(ar['image_directory']):
            raise ValueError(cid)
        annotation=Path(ar['annotation'])
        records.append(dict(case_id=cid,split=splits[str(cid)],image_directory=str(folder),
                            annotation=str(annotation),annotation_sha256=digest(annotation),
                            frames=ar['nrrd_frames']))
    save_json(ROOT/'configs/cases.json',records)
    files=[previous/n for n in ('manifest.json','case_split.json','excluded.json','selection_summary.json')]
    sources=[]
    for file in files:
        target=ROOT/'provenance'/file.name
        shutil.copy2(file,target)
        sources.append(dict(source=str(file),snapshot=str(target.relative_to(ROOT)),sha256=digest(file)))
    split_dir=ROOT.parent/'tg3k_split'
    for file in split_dir.iterdir():
        if file.is_file():
            dest=ROOT/'provenance/tg3k'/file.name
            dest.parent.mkdir(exist_ok=True)
            shutil.copy2(file,dest)
            sources.append(dict(source=str(file),snapshot=str(dest.relative_to(ROOT)),sha256=digest(file)))
    save_json(ROOT/'provenance/sources.json',sources)
    save_json(ROOT/'reports/snapshot.json',dict(case_count=len(records),
        splits=dict(Counter(r['split'] for r in records)),frames=sum(r['frames'] for r in records),
        policy='Exact gland case whitelist from previous manifest; all raw frames in these cases'))
    print(read_json(ROOT/'reports/snapshot.json'))

if __name__=='__main__':
    main()
