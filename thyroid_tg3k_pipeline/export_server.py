"""Build a self-contained Linux archive without training or remote access."""
import argparse
import csv
import hashlib
import io
import json
import tarfile
from pathlib import Path
from src.common import ROOT,read_csv,read_json,digest

def json_bytes(value):
    return (json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8')

def csv_bytes(rows):
    stream=io.StringIO(newline='')
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
    writer.writeheader();writer.writerows(rows)
    return stream.getvalue().encode('utf-8-sig')

def plan(source):
    files={};generated={}
    def add(name,path):
        path=Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        if name in files and files[name]!=path:
            raise ValueError('Archive name conflict: '+name)
        files[name]=path
    for path in source.glob('*.py'):
        add(path.name,path)
    for name in ('requirements.txt','README.md','run_server.sh'):
        add(name,source/name)
    for directory in ('src','configs','provenance','tests','reports'):
        for path in (source/directory).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix!='.pyc':
                add(path.relative_to(source).as_posix(),path)
    cfg=read_json(source/'configs/project.json')
    original_tg_root=Path(cfg['tg3k_root'])
    cfg['tg3k_root']='datasets/tg3k'
    generated['configs/project.json']=json_bytes(cfg)
    tg=read_csv(source/'data/tg3k.csv')
    for r in tg:
        for key in ('image','mask'):
            path=Path(r[key])
            name='datasets/tg3k/'+path.relative_to(original_tg_root).as_posix()
            add(name,path);r[key]=name
    generated['data/tg3k.csv']=csv_bytes(tg)
    base=source/'data/original_pairs_v3'
    complete=read_json(base/'complete.json')
    if complete['policy']!='original_local_positive_plus_tg3k_unused_test_v3':
        raise ValueError('Expected v3 prepared data')
    for filename in ('local_pairs.csv','final_test.csv'):
        if digest(base/filename)!=complete['manifest_sha256'][filename]:
            raise ValueError('Prepared manifest changed')
        rows=read_csv(base/filename)
        for r in rows:
            for key in ('image','mask'):
                path=Path(r[key])
                if r['dataset']=='local':
                    name=f'datasets/local/{key}/WD{int(r["case_id"]):08d}/{path.name}'
                else:
                    name='datasets/tg3k/'+path.relative_to(original_tg_root).as_posix()
                add(name,path);r[key]=name
        name='data/original_pairs_v3/'+filename
        generated[name]=csv_bytes(rows)
        complete['manifest_sha256'][filename]=hashlib.sha256(generated[name]).hexdigest()
    generated['data/original_pairs_v3/complete.json']=json_bytes(complete)
    add('data/original_pairs_v3/case_counts.csv',base/'case_counts.csv')
    generated['SERVER_README.txt']=('Run from the extracted project directory.\n'
        'Data is already prepared. Do not run prepare.py, prepare_pairs.py, build_snapshot.py or reconcile.py on Linux.\n'
        'The runtime CSV paths are portable; Windows paths in historical provenance/configs describe source records only.\n'
        'python verify_server.py\nbash run_server.sh\n'
        'No credentials are included in the archive.\n').encode()
    for name in generated:
        files.pop(name,None)
    return files,generated

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,default=ROOT)
    p.add_argument('--out',type=Path)
    p.add_argument('--check',action='store_true',help='Validate packaging plan without writing archive')
    args=p.parse_args()
    source=args.source.resolve()
    files,generated=plan(source)
    size=sum(p.stat().st_size for p in files.values())+sum(map(len,generated.values()))
    print(f'{len(files)+len(generated)} files; uncompressed {size/1024**3:.2f} GiB',flush=True)
    if args.check:
        print('Plan valid; no archive created; no training executed.');return
    out=args.out or source/'dist/thyroid_server_v3.tar.gz'
    out.parent.mkdir(parents=True,exist_ok=True)
    temporary=out.with_suffix(out.suffix+'.partial')
    if out.exists() or temporary.exists():
        raise FileExistsError('Archive/partial archive already exists; use a different --out path')
    hashes={}
    def put(tar,name,data):
        member=tarfile.TarInfo('thyroid_tg3k_pipeline/'+name)
        member.size=len(data);member.mode=0o644
        tar.addfile(member,io.BytesIO(data))
    with tarfile.open(temporary,'w:gz',compresslevel=1) as tar:
        for i,(name,path) in enumerate(sorted(files.items()),1):
            hashes[name]=digest(path)
            tar.add(path,arcname='thyroid_tg3k_pipeline/'+name,recursive=False)
            if i%1000==0:
                print(f'Packed {i}/{len(files)} source files',flush=True)
        for name,data in generated.items():
            hashes[name]=hashlib.sha256(data).hexdigest();put(tar,name,data)
        put(tar,'server_checksums.json',json_bytes(hashes))
    temporary.rename(out)
    checksum=digest(out)
    out.with_name(out.name+'.sha256').write_bytes((checksum+'  '+out.name+'\n').encode('ascii'))
    print(f'Created {out} ({out.stat().st_size/1024**3:.2f} GiB)\nSHA256 {checksum}')

if __name__=='__main__':
    main()
