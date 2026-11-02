"""Create and round-trip verify a source/data/results bundle from completed runs."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import tarfile
from audit_release import audit_release

ROOT=Path(__file__).resolve().parents[1]


def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):
            digest.update(chunk)
    return digest.hexdigest()


def bundle(run_dirs,output,allow_incomplete=False):
    runs=[(ROOT/run).resolve() for run in run_dirs]
    for folder in runs:
        if not folder.is_relative_to(ROOT):
            raise ValueError('Run must be inside the repository')
        status=json.loads((folder/'status.json').read_text())
        if status['state']!='complete':
            raise ValueError('Never package a run while it is still being written')
    if not allow_incomplete:
        audit=audit_release(runs)
        if not audit['passed']:
            raise ValueError('Release audit failed: '+'; '.join(audit['errors']))
    paths=set(ROOT.glob('*.py'))
    paths.update(ROOT/name for name in ('README.md','research-plan.md','requirements.txt','pytest.ini','download_models.sh'))
    for directory,patterns in {'scripts':['*.py'],'tests':['*.py'],'configs':['*.json'],
                                'docs':['*.md'], 'notebooks':['*.ipynb'],
                                'artifacts':['*.json','*.md','*.txt','*.csv','*.pdf','*.png'],
                                'paper':['*.tex','*.bib','*.bbl','*.pdf']}.items():
        for pattern in patterns:
            paths.update((ROOT/directory).rglob(pattern))
    for folder in runs:
        for filename in ('campaign.json','model-comparison.json'):
            path=folder.parent/filename
            if path.exists():
                paths.add(path)
        for pattern in ('*.json','*.jsonl','*.csv','vectors.pt','source/*.py','plots/*.pdf','plots/*.png'):
            paths.update(folder.glob(pattern))
    paths={p for p in paths if p.is_file()}
    if any(not p.resolve().is_relative_to(ROOT) or p.is_symlink() for p in paths):
        raise ValueError('Bundle inputs must be regular repository files, not symlinks')
    files={str(p.relative_to(ROOT)):{'sha256':sha256_file(p),'bytes':p.stat().st_size} for p in sorted(paths)}
    manifest={'schema_version':1,'created':datetime.now(timezone.utc).isoformat(),
              'status':'incomplete_draft' if allow_incomplete else 'release_candidate',
              'runs':[str(p.relative_to(ROOT)) for p in runs],
              'excludes':['downloaded model weights','virtual environment','local TeX binary','Git history'],
              'files':files}
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix(output.suffix+'.tmp')
    payload=(json.dumps(manifest,indent=2)+'\n').encode()
    with tarfile.open(temporary,'w:gz') as archive:
        info=tarfile.TarInfo('bundle-manifest.json'); info.size=len(payload); info.mtime=0
        archive.addfile(info,io.BytesIO(payload))
        for relative in files:
            archive.add(ROOT/relative,arcname=relative,recursive=False)
    verify_bundle(temporary)
    temporary.replace(output)
    report={'archive':str(output),'sha256':sha256_file(output),'files':len(files),
            'status':manifest['status'],'verified':True,'runs':manifest['runs']}
    output.with_suffix(output.suffix+'.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def verify_bundle(path):
    with tarfile.open(path,'r:gz') as archive:
        manifest=json.load(archive.extractfile('bundle-manifest.json'))
        names=archive.getnames()
        if len(names)!=len(set(names)) or set(names)!=set(manifest['files'])|{'bundle-manifest.json'}:
            raise ValueError('Archive membership differs from manifest')
        for name,expected in manifest['files'].items():
            if Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError('Unsafe member path')
            stream=archive.extractfile(name)
            digest=hashlib.sha256(); size=0
            for chunk in iter(lambda:stream.read(1<<20),b''):
                digest.update(chunk); size+=len(chunk)
            if digest.hexdigest()!=expected['sha256'] or size!=expected['bytes']:
                raise ValueError(f'Archive checksum mismatch: {name}')
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',nargs='+',required=True)
    parser.add_argument('--output',default='dist/reproduction.tar.gz')
    parser.add_argument('--allow-incomplete',action='store_true')
    args=parser.parse_args()
    print(json.dumps(bundle(args.runs,args.output,args.allow_incomplete),indent=2))
