"""Publication-readiness gate based on executed evidence, not status claims alone."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
from audit_run import audit

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dataset_builder import load_contrast_dataset


def audit_release(run_dirs):
    errors=[]
    run_reports={}
    manifests={}
    load_contrast_dataset(ROOT/'artifacts/dataset.json')
    for run_dir in run_dirs:
        folder=Path(run_dir)
        report=audit(folder,require_test=True,output_name='release-run-audit.json')
        model=report['model']
        run_reports[model]=report
        if not report['passed']:
            errors.extend(f'{model}: {e}' for e in report['errors'])
        manifest=json.loads((folder/'manifest.json').read_text())
        manifests[model]=manifest
        cfg=manifest['config']
        if len(cfg['layer_fractions'])<8 or len(cfg['alphas'])<4:
            errors.append(f'{model}: incomplete layer/intensity sweep')
        if set(cfg['modes'])!={'steer','orthogonal','gaussian'}:
            errors.append(f'{model}: missing activation controls')
        if cfg['k_tokens']!=15:
            errors.append(f'{model}: intervention duration differs from protocol')
        if not cfg.get('text_interventions') or not cfg.get('swap_cot') or set(cfg['truncation_fractions'])!={0,.25,.5,.75,1}:
            errors.append(f'{model}: missing text interventions')
        for filename in ('summary.json','summary.csv','judge-summary.json','judge-manifest.json','judgments.jsonl','judge-retest.jsonl'):
            if not (folder/filename).exists():
                errors.append(f'{model}: missing {filename}')
        for name in ('layer_sweep_heatmap','alpha_sweep_curve','truncation_degradation'):
            if not (folder/'plots'/f'{name}.pdf').exists():
                errors.append(f'{model}: missing {name} figure')
        judge_path=folder/'judge-summary.json'
        if judge_path.exists():
            j=json.loads(judge_path.read_text())
            if j['assessed']!=j['eligible'] or j['status_counts'].get('error',0):
                errors.append(f'{model}: missing/failed judge assessments')
            if j['retest_n']<20:
                errors.append(f'{model}: insufficient judge retest')
            if j['agreement'] is None or j['agreement']<.8:
                errors.append(f'{model}: inspect weak judge retest agreement')
        summary_path=folder/'summary.json'
        if summary_path.exists():
            summary=json.loads(summary_path.read_text())
            if summary['identity']!=manifest['identity']:
                errors.append(f'{model}: stale summary')
    for model in ('distill-7b','distill-32b','qwq-32b'):
        if model not in manifests:
            errors.append(f'Missing required held-out model: {model}')
    if all(m in manifests for m in ('distill-32b','qwq-32b')):
        a,b=(manifests[m] for m in ('distill-32b','qwq-32b'))
        for key in ('config','dataset_sha256','eval_ids'):
            if a[key]!=b[key]:
                errors.append(f'32B comparison has unmatched {key}')
        if a['model']['quantization']!=b['model']['quantization']:
            errors.append('32B quantization is unmatched')
    authors_path=ROOT/'paper/authors.json'
    authors=json.loads(authors_path.read_text()) if authors_path.exists() else {}
    if not authors.get('authors') or any(not a.get('name','').strip() for a in authors.get('authors',[])):
        errors.append('Publication author metadata is missing')
    validation_path=ROOT/'artifacts/judge-validation-results.json'
    if not validation_path.exists():
        errors.append('Missing assessor validation')
    else:
        validation=json.loads(validation_path.read_text())
        if validation['correct']<validation['total']-1:
            errors.append('Assessor validation failed')
    tests_path=ROOT/'artifacts/test-results.json'
    if not tests_path.exists():
        errors.append('Missing recorded test evidence')
    else:
        tests=json.loads(tests_path.read_text())
        if tests.get('returncode')!=0:
            errors.append('Tests failed')
        for name,expected in tests.get('source_sha256',{}).items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
                errors.append(f'Test evidence is stale: {name}')
    return {'passed':not errors,'errors':errors,'runs':run_reports,
            'scope':'Technical and coverage gate; the manuscript claims and uncertainty still require substantive review.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',nargs='+')
    parser.add_argument('--output',default='artifacts/release-audit.json')
    args=parser.parse_args()
    report=audit_release(args.runs)
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if report['passed'] else 1)
