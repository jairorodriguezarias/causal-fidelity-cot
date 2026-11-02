"""Run a frozen study sequentially, with durable per-stage logs and explicit failures."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from campaign_resources import prepare_stage

ROOT=Path(__file__).resolve().parents[1]
PYTHON=ROOT/'env/bin/python'


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n')
    temp.replace(path)


parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--config',required=True)
parser.add_argument('--name',required=True)
parser.add_argument('--models',nargs='+',default=['distill-7b','distill-32b','qwq-32b'])
parser.add_argument('--exclusive-vllm-container',
                    help='Stop this container for generation; start it and await readiness for judging')
args=parser.parse_args()
folder=ROOT/'runs'/args.name
folder.mkdir(parents=True,exist_ok=True)
config_path=ROOT/args.config
config_hash=hashlib.sha256(config_path.read_bytes()).hexdigest()
source_files=['experiment.py','dataset_builder.py','evaluation.py','llm_judge.py',
              'rejudge.py','analyze.py','compare.py','scripts/audit_run.py']
source_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_files}
state_path=folder/'campaign.json'
state={'state':'running','pid':os.getpid(),'started':now(),'models':args.models,
       'config':args.config,'config_sha256':config_hash,'source_sha256':source_hashes,'steps':[]}
if args.exclusive_vllm_container:
    state['resources']={'exclusive_vllm_container':args.exclusive_vllm_container}
if state_path.exists():
    previous=json.loads(state_path.read_text())
    if previous['config_sha256']!=config_hash or previous['models']!=args.models or previous.get('source_sha256')!=source_hashes:
        raise SystemExit('Campaign identity differs')
    # Do not start a duplicate while an earlier campaign process is alive.
    old_pid=previous.get('pid')
    if old_pid and old_pid!=os.getpid():
        try:
            cmdline=Path(f'/proc/{old_pid}/cmdline').read_bytes().decode().replace('\0',' ')
        except FileNotFoundError:
            cmdline=''
        if 'run_campaign.py' in cmdline and args.name in cmdline:
            raise SystemExit(f'Campaign is still running as PID {old_pid}')
    state['steps']=previous['steps']
atomic(state_path,state)


def stage(name,command):
    if any(s['name']==name and s.get('returncode')==0 for s in state['steps']):
        return
    if hashlib.sha256(config_path.read_bytes()).hexdigest()!=config_hash:
        raise RuntimeError('Frozen campaign configuration changed')
    for filename,digest in source_hashes.items():
        if hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()!=digest:
            raise RuntimeError(f'Frozen campaign source changed: {filename}')
    log=folder/(name+'.log')
    step={'name':name,'command':command,'started':now(),'log':str(log)}
    state['steps'].append(step)
    state['current_step']=name
    atomic(state_path,state)
    prepare_stage(name,args.exclusive_vllm_container)
    with log.open('a') as output:
        process=subprocess.Popen(command,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
        step['pid']=process.pid
        atomic(state_path,state)
        returncode=process.wait()
    step.update(returncode=returncode,finished=now())
    atomic(state_path,state)
    if returncode:
        raise RuntimeError(f'{name} failed with exit code {returncode}; see {log}')


try:
    stage('tests',[str(PYTHON),'scripts/record_tests.py'])
    stage('judge-validation',[str(PYTHON),'rejudge.py','--validate'])
    for model in args.models:
        stage(model+'-download',[str(PYTHON),'scripts/download_models.py',model])
        output=f'runs/{args.name}/{model}'
        stage(model+'-generate',[str(PYTHON),'experiment.py','--config',args.config,'--model',model,'--output',output])
        stage(model+'-audit',[str(PYTHON),'scripts/audit_run.py',output,'--require-test'])
        stage(model+'-analyze',[str(PYTHON),'analyze.py',output])
        stage(model+'-judge',[str(PYTHON),'rejudge.py','--run',output])
    if 'distill-32b' in args.models and 'qwq-32b' in args.models:
        stage('model-comparison',[str(PYTHON),'compare.py',f'runs/{args.name}/distill-32b',f'runs/{args.name}/qwq-32b',
                                  '--output',str(folder/'model-comparison.json')])
    state.update(state='complete',finished=now())
except Exception as error:
    state.update(state='failed',error=str(error),finished=now())
    atomic(state_path,state)
    raise
atomic(state_path,state)
