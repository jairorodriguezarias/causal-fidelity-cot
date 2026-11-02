"""Assess v4 records without overwriting generations; validate and retest the judge."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from transformers import AutoTokenizer
from dataset_builder import fingerprint
from evaluation import cohen_kappa
from llm_judge import Judge, SYSTEM_PROMPT, consistency_category


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()] if Path(path).exists() else []


def append(path, record):
    with Path(path).open('a') as output:
        output.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
        output.flush()


def validate(judge, output):
    fixtures=json.loads(Path('artifacts/judge-validation.json').read_text())
    results=[]
    for case in fixtures:
        assessment=judge.assess(case['cot'],case['context'])
        results.append({**case,'assessment':assessment,
                        'correct':assessment.get('verdict')==case['expected'] and assessment['status'] in ('ok','missing_reasoning')})
        print(case['id'],assessment['status'],assessment.get('verdict'),flush=True)
    report={'kind':'synthetic_unit_validation_not_human_annotation',
            'judge_model':judge.model,'judge_url':judge.url,'prompt_sha256':fingerprint(SYSTEM_PROMPT),
            'correct':sum(r['correct'] for r in results),'total':len(results),'cases':results}
    Path(output).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report


def evaluate(judge, run_dir, retest_size):
    folder=Path(run_dir)
    records=load_jsonl(folder/'records.jsonl')
    manifest=json.loads((folder/'manifest.json').read_text())
    judge_manifest={'model':judge.model,'url':judge.url,'backend':judge.backend,
                    'prompt_sha256':fingerprint(SYSTEM_PROMPT),'run_identity':manifest['identity'],
                    'code_sha256':hashlib.sha256(Path('llm_judge.py').read_bytes()).hexdigest()}
    meta_path=folder/'judge-manifest.json'
    if meta_path.exists() and json.loads(meta_path.read_text())!=judge_manifest:
        raise ValueError('Judge configuration changed; preserve previous assessments before rerunning')
    meta_path.write_text(json.dumps(judge_manifest,indent=2)+'\n')
    tokenizer=AutoTokenizer.from_pretrained(manifest['model']['id'],revision=manifest['model']['revision'],
                                           cache_dir='hf_cache/hub',local_files_only=True)
    baseline={r['example_id']:r for r in records if r['condition']['kind']=='baseline' and r['variant']=='entail'}
    output=folder/'judgments.jsonl'
    previous={r['key']:r for r in load_jsonl(output)}
    eligible=[r for r in records if r['variant']=='entail' and r['condition']['kind'] in ('baseline','activation')]
    for record in eligible:
        if record['key'] in previous:
            continue
        parsed=record['parsed']
        if parsed['status']!='ok':
            assessment={'status':'incomplete_generation','verdict':None,'attempts':[]}
        else:
            context=record['prompt'].split('Premise:',1)[-1].split('<｜Assistant｜>',1)[0].split('<|im_end|>',1)[0]
            assessment=judge.assess(parsed['cot'],context)
        quote=assessment.get('first_contradiction_quote','')
        drift=None
        if quote:
            char=parsed['cot'].index(quote)
            drift=len(tokenizer.encode(parsed['cot'][:char],add_special_tokens=False))
        result={'key':record['key'],'assessment':assessment,'first_contradiction_token':drift,
                'category':consistency_category(baseline[record['example_id']]['parsed']['label'],parsed['label'],assessment)}
        append(output,result)
        previous[record['key']]=result
        if len(previous)%20==0:
            print(f'Assessed {len(previous)}/{len(eligible)}',flush=True)
    candidates=[r for r in eligible if previous[r['key']]['assessment']['status']=='ok']
    random.Random(20260925).shuffle(candidates)
    chosen=candidates[:retest_size]
    retest_path=folder/'judge-retest.jsonl'
    retests={r['key']:r for r in load_jsonl(retest_path)}
    for record in chosen:
        if record['key'] not in retests:
            context=record['prompt'].split('Premise:',1)[-1].split('<｜Assistant｜>',1)[0].split('<|im_end|>',1)[0]
            result={'key':record['key'],'assessment':judge.assess(record['parsed']['cot'],context,seed=137)}
            append(retest_path,result)
            retests[record['key']]=result
    pairs=[(previous[r['key']]['assessment']['verdict'],retests[r['key']]['assessment']['verdict'])
           for r in chosen if retests[r['key']]['assessment']['status']=='ok']
    summary={'eligible':len(eligible),'assessed':len(previous),
             'status_counts':dict(Counter(r['assessment']['status'] for r in previous.values())),
             'category_counts':dict(Counter(r['category'] for r in previous.values())),
             'retest_n':len(pairs),'retest_requested':len(chosen),
             'kappa':cohen_kappa([p[0] for p in pairs],[p[1] for p in pairs]),
             'agreement':sum(a==b for a,b in pairs)/len(pairs) if pairs else None,
             'interpretation':'Retest consistency is not external validity; categories describe text/output agreement, not causal faithfulness.'}
    (folder/'judge-summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run')
    parser.add_argument('--url',default='http://localhost:8000')
    parser.add_argument('--model',default='nvidia/Qwen3.6-35B-A3B-NVFP4')
    parser.add_argument('--backend',choices=['openai_compatible','ollama'],default='openai_compatible')
    parser.add_argument('--validate',action='store_true')
    parser.add_argument('--validation-output',default='artifacts/judge-validation-results.json')
    parser.add_argument('--retest-size',type=int,default=50)
    args=parser.parse_args()
    judge=Judge(args.url,args.model,args.backend)
    if args.validate:
        report=validate(judge,args.validation_output)
        if report['correct']<report['total']-1:
            raise SystemExit('Synthetic judge validation failed; inspect before evaluating real outputs.')
    if args.run:
        evaluate(judge,args.run,args.retest_size)
