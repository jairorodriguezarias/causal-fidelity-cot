"""Record test results together with hashes of the tested source files."""
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import subprocess,sys
root=Path(__file__).resolve().parents[1]
paths=list(root.glob('*.py'))+list((root/'scripts').glob('*.py'))+list((root/'tests').glob('*.py'))
hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
result=subprocess.run([sys.executable,'-m','pytest','-q'],cwd=root,capture_output=True,text=True)
report={'timestamp':datetime.now(timezone.utc).isoformat(),'returncode':result.returncode,
        'stdout':result.stdout,'stderr':result.stderr,'source_sha256':hashes}
(root/'artifacts/test-results.json').write_text(json.dumps(report,indent=2)+'\n')
print(result.stdout,end=''); print(result.stderr,end='',file=sys.stderr)
raise SystemExit(result.returncode)
