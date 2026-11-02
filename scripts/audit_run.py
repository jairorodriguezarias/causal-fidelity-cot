"""Independently check experiment coverage, provenance and parse validity."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dataset_builder import load_contrast_dataset
from evaluation import parse_output


def audit(folder, require_test=False, output_name='audit.json'):
    folder=Path(folder)
    manifest=json.loads((folder/'manifest.json').read_text())
    status=json.loads((folder/'status.json').read_text())
    config=manifest['config']
    records=[json.loads(line) for line in (folder/'records.jsonl').read_text().splitlines()]
    dataset=load_contrast_dataset(ROOT/'artifacts/dataset.json')
    errors=[]
    def check(condition,message):
        if not condition:
            errors.append(message)
    check(status['state']=='complete','Run is incomplete')
    check(manifest['dataset_sha256']==dataset['sha256'],'Dataset checksum differs')
    for name,digest in manifest['source_sha256'].items():
        path=folder/'source'/name
        check(path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==digest,f'Source snapshot mismatch: {name}')
    test_ids=[p['id'] for p in dataset['splits'][config['evaluation_split']][:config['eval_pairs']]]
    train_ids=[p['id'] for p in dataset['splits']['train'][:config['train_pairs']]]
    check(test_ids==manifest['eval_ids'],'Evaluation IDs differ')
    check(train_ids==manifest['train_ids'],'Construction IDs differ')
    check(not set(train_ids).intersection(test_ids),'Train/evaluation overlap')
    if require_test:
        check(config['evaluation_split']=='test','Not held-out test data')
        check(config['train_pairs']>=500,'Fewer than 500 construction pairs')
        check(config['eval_pairs']>=50,'Fewer than 50 evaluation examples')
    keys=[r['key'] for r in records]
    check(len(keys)==len(set(keys)),'Duplicate keys')
    indexed={r['key']:r for r in records}
    expected={f'baseline/{i}/entail' for i in test_ids}
    if config.get('text_interventions') or config.get('both_baseline_variants'):
        expected.update(f'baseline/{i}/contra' for i in test_ids)
    model=manifest['model']
    model_folder=ROOT/'hf_cache/hub'/('models--'+model['id'].replace('/','--'))/'snapshots'/model['revision']
    model_meta=json.loads((model_folder/'config.json').read_text())
    n_layers=model_meta['num_hidden_layers']
    layers=sorted(set(round(f*(n_layers-1)) for f in config['layer_fractions']))
    for layer in layers:
        for alpha in config['alphas']:
            for mode in config['modes']:
                for rep in range(config['control_replicates'] if mode!='steer' else 1):
                    expected.update(f'{mode}_L{layer}_a{alpha}_r{rep}/{i}/entail' for i in test_ids)
    if config.get('text_interventions'):
        valid=[i for i in test_ids if indexed.get(f'baseline/{i}/entail',{}).get('parsed',{}).get('status')=='ok']
        for fraction in config['truncation_fractions']:
            expected.update(f'truncate_{fraction}/{i}/entail' for i in valid)
        if config.get('swap_cot'):
            expected.update(f'shuffle_cot/{i}/entail' for i in valid)
            expected.update(f'swap_contradiction_cot/{i}/entail' for i in valid
                            if indexed.get(f'baseline/{i}/contra',{}).get('parsed',{}).get('status')=='ok')
    check(set(keys)==expected,f'Coverage mismatch: missing {len(expected-set(keys))}, extra {len(set(keys)-expected)}')
    check(status['records']==len(records),'Status record count differs')
    for record in records:
        check(record['example_id'] in test_ids,f"Unknown evaluation ID: {record['key']}")
        check(record['generated_tokens']==len(record['generated_token_ids']),f"Token count mismatch: {record['key']}")
        if record['condition']['kind']!='text':
            parsed=parse_output(record['generated_text']).asdict()
            check(parsed==record['parsed'],f"Parse mismatch: {record['key']}")
        if record['condition']['kind']=='activation':
            check(record['hook_steps']==min(config['k_tokens'],record['generated_tokens']),f"Incorrect hook count: {record['key']}")
    report={'passed':not errors,'errors':errors,'run':str(folder),'model':manifest['model_name'],
            'require_test':require_test,
            'records':len(records),'expected_records':len(expected),'conditions':dict(Counter(r['condition']['name'] for r in records)),
            'baseline_entailment_status':dict(Counter(indexed[k]['parsed']['status'] for k in indexed if k.startswith('baseline/') and k.endswith('/entail')))}
    (folder/output_name).write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run')
    parser.add_argument('--require-test',action='store_true')
    args=parser.parse_args()
    result=audit(args.run,args.require_test)
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['passed'] else 1)
