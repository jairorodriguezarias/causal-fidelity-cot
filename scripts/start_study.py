"""Finish calibration, freeze the protocol, then execute the complete model campaign."""
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import subprocess,sys,time

ROOT=Path(__file__).resolve().parents[1]
PYTHON=ROOT/'env/bin/python'
CAL=ROOT/'runs/calibration-7b'
UNIT='causal-fidelity-calibration-7b.service'

while True:
    status_path=CAL/'status.json'
    status=json.loads(status_path.read_text()) if status_path.exists() else {}
    if status.get('state')=='complete':
        break
    state=subprocess.check_output(['systemctl','--user','show',UNIT,'--property=ActiveState','--value'],text=True).strip()
    pid=int(subprocess.check_output(['systemctl','--user','show',UNIT,'--property=MainPID','--value'],text=True).strip())
    if state!='active' or pid<=0 or not Path(f'/proc/{pid}').exists():
        raise SystemExit('Calibration is incomplete and its service is no longer running; inspect before resuming.')
    print(f'Waiting for calibration PID {pid}, {status.get("records",0)} records saved',flush=True)
    time.sleep(15)
subprocess.run([str(PYTHON),'scripts/audit_run.py',str(CAL)],cwd=ROOT,check=True)
subprocess.run([str(PYTHON),'analyze.py',str(CAL)],cwd=ROOT,check=True)
records=[json.loads(line) for line in (CAL/'records.jsonl').read_text().splitlines()]
calibration={}
for variant in ('entail','contra'):
    subset=[r for r in records if r['variant']==variant]
    calibration[variant]={'n':len(subset),'correct':sum(r['parsed']['label']==r['gold_label'] for r in subset),
                          'valid':sum(r['parsed']['label'] is not None for r in subset),
                          'unfinished':sum(r['parsed']['status']=='unfinished_thought' for r in subset)}
if calibration['entail']['correct']<10 or calibration['contra']['correct']<10:
    raise SystemExit('7B calibration is below 50% accuracy in a class; inspect task competence before the full study.')
config=json.loads((ROOT/'configs/study-draft.json').read_text())
config['batch_size']=16
config['generation']['max_new_tokens']=2048
frozen=ROOT/'configs/study.json'
if frozen.exists() and json.loads(frozen.read_text())!=config:
    raise SystemExit('A different frozen study configuration already exists.')
frozen.write_text(json.dumps(config,indent=2)+'\n')
freeze={'frozen_at':datetime.now(timezone.utc).isoformat(),'configuration':config,
        'config_sha256':hashlib.sha256(frozen.read_bytes()).hexdigest(),
        'calibration_run_identity':json.loads((CAL/'manifest.json').read_text())['identity'],
        'calibration_results':calibration,
        'decisions':['2048 generated-token ceiling to reduce truncation observed at 1024 on calibration.',
                     'Batch size 16 for throughput; identical in the primary 32B comparison.',
                     'Eight layers, four mean-difference intensities and both random controls on every test example.',
                     'No selection of test examples based on baseline results; report conditional metrics and missingness.',
                     'One seeded generation and one random direction per control/example; uncertainty is over examples.'],
        'registration':'Local protocol freeze before held-out execution; not an external preregistration.'}
freeze_path=ROOT/'artifacts/protocol-freeze.json'
if not freeze_path.exists():
    freeze_path.write_text(json.dumps(freeze,indent=2)+'\n')
print(json.dumps(freeze,indent=2),flush=True)
subprocess.run([str(PYTHON),'scripts/run_campaign.py','--config','configs/study.json','--name','study-20260925'],cwd=ROOT,check=True)
