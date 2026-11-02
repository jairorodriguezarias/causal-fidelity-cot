import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
import pytest

scripts=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(scripts))
from package_reproduction import verify_bundle
import package_reproduction


def archive(path,payload,expected):
    manifest={'files':{'source.py':{'sha256':hashlib.sha256(expected).hexdigest(),'bytes':len(expected)}}}
    with tarfile.open(path,'w:gz') as tar:
        data=json.dumps(manifest).encode()
        member=tarfile.TarInfo('bundle-manifest.json'); member.size=len(data)
        tar.addfile(member,io.BytesIO(data))
        member=tarfile.TarInfo('source.py'); member.size=len(payload)
        tar.addfile(member,io.BytesIO(payload))


def test_bundle_verifies_actual_member_bytes(tmp_path):
    target=tmp_path/'bundle.tar.gz'
    archive(target,b'original',b'original')
    assert 'source.py' in verify_bundle(target)['files']
    archive(target,b'changed!',b'original')
    with pytest.raises(ValueError,match='checksum mismatch'):
        verify_bundle(target)


def test_bundle_includes_notebook_and_interim_figure_dependencies(tmp_path, monkeypatch):
    monkeypatch.setattr(package_reproduction, 'ROOT', tmp_path)
    included = {
        'notebooks/tutorial.ipynb': '{}',
        'paper/interim.tex': r'\includegraphics{../artifacts/diagnostics/date/figures/plot.pdf}',
        'paper/interim.pdf': 'standalone report',
        'artifacts/diagnostics/date/figures/plot.pdf': 'figure',
        'artifacts/diagnostics/date/figures/table.csv': 'n,value\n1,2\n',
        'docs/README.md': 'documentation index',
        'runs/completed/status.json': '{"state": "complete"}',
    }
    excluded = {'.env': 'local setting', 'hf_cache/weights.safetensors': 'weights',
                'paper/interim.log': 'build output', 'runs/other/records.jsonl': '{}'}
    for name, content in {**included, **excluded}.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    target = tmp_path / 'dist/draft.tar.gz'
    package_reproduction.bundle(['runs/completed'], target, allow_incomplete=True)
    files = verify_bundle(target)['files']
    assert set(included) <= files.keys()
    assert not set(excluded) & files.keys()


def test_draft_bundle_rejects_generation_in_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(package_reproduction, 'ROOT', tmp_path)
    folder = tmp_path / 'runs/active'
    folder.mkdir(parents=True)
    status = folder / 'status.json'
    status.write_text('{"state": "running"}')
    before = status.read_bytes()
    target = tmp_path / 'dist/draft.tar.gz'
    with pytest.raises(ValueError, match='still being written'):
        package_reproduction.bundle(['runs/active'], target, allow_incomplete=True)
    assert status.read_bytes() == before
    assert not target.exists()
