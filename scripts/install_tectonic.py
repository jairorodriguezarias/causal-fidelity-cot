"""Install the pinned Linux Tectonic release into this repository's .tools directory."""
from __future__ import annotations
import hashlib
import io
from pathlib import Path
import platform
import subprocess
import tarfile
import requests

VERSION='0.17.0'
ASSETS={
    'aarch64':('aarch64-unknown-linux-musl','b10954a95404f3ab2328d2fa59a5ebab8e657f893fab096f98be8db7c0c979b8'),
    'x86_64':('x86_64-unknown-linux-musl','8533d07f9ccbd7a65824b9e0459041bca34af1eb33daba48f59215593753a3b7'),
}


def install():
    if platform.system()!='Linux' or platform.machine() not in ASSETS:
        raise SystemExit('This installer supports Linux aarch64/x86_64. Install Tectonic 0.17.0 manually into .tools/tectonic on other platforms.')
    target=Path(__file__).resolve().parents[1]/'.tools/tectonic'
    if target.exists():
        found=subprocess.check_output([str(target),'--version'],text=True).strip()
        if found==f'Tectonic {VERSION}':
            print(found+' is already installed.')
            return
        raise SystemExit(f'Existing binary has a different version: {found}')
    triple,expected=ASSETS[platform.machine()]
    url=f'https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%40{VERSION}/tectonic-{VERSION}-{triple}.tar.gz'
    response=requests.get(url,timeout=120)
    response.raise_for_status()
    if hashlib.sha256(response.content).hexdigest()!=expected:
        raise SystemExit('Release archive checksum mismatch')
    with tarfile.open(fileobj=io.BytesIO(response.content),mode='r:gz') as archive:
        members=[m for m in archive.getmembers() if m.isfile() and Path(m.name).name=='tectonic']
        if len(members)!=1:
            raise SystemExit('Unexpected release archive contents')
        binary=archive.extractfile(members[0]).read()
    target.parent.mkdir(exist_ok=True)
    temporary=target.with_suffix('.tmp')
    temporary.write_bytes(binary)
    temporary.chmod(0o755)
    temporary.replace(target)
    print(subprocess.check_output([str(target),'--version'],text=True).strip())


if __name__=='__main__':
    install()
