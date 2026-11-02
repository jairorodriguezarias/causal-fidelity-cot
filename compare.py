"""Compare matched 32B runs without conflating model identity with training causality."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from analyze import paired_difference


def load(folder):
    folder=Path(folder)
    manifest=json.loads((folder/'manifest.json').read_text())
    audit=json.loads((folder/'audit.json').read_text())
    if not audit['passed']:
        raise ValueError('Run audit has not passed')
    records=[json.loads(line) for line in (folder/'records.jsonl').read_text().splitlines()]
    return manifest,records


def compare(first,second,output):
    a,records_a=load(first)
    b,records_b=load(second)
    for key in ('config','dataset_sha256','eval_ids'):
        if a[key]!=b[key]:
            raise ValueError(f'Unmatched {key}')
    if a['model']['quantization']!=b['model']['quantization']:
        raise ValueError('Quantization differs')
    if a['config']['evaluation_split']!='test':
        raise ValueError('Comparison requires held-out data')
    baseline_a={r['example_id']:r for r in records_a if r['condition']['kind']=='baseline' and r['variant']=='entail'}
    baseline_b={r['example_id']:r for r in records_b if r['condition']['kind']=='baseline' and r['variant']=='entail'}
    common={i for i in baseline_a if baseline_a[i]['parsed']['label']==baseline_b[i]['parsed']['label']=='ENTAILMENT'}
    indexed_b={r['key']:r for r in records_b}
    groups=defaultdict(list)
    for record in records_a:
        if record['variant']=='entail' and record['key'] in indexed_b:
            groups[record['condition']['name']].append((record,indexed_b[record['key']]))
    result={'first':a['model_name'],'second':b['model_name'],'common_baseline_entailment_n':len(common),
            'interpretation':'Exploratory checkpoint contrast; training method is confounded with data and post-training choices.',
            'conditions':[]}
    for name,pairs in groups.items():
        eligible=[(x,y) for x,y in pairs if x['example_id'] in common]
        complete=[(x,y) for x,y in eligible if x['parsed']['label'] is not None and y['parsed']['label'] is not None]
        estimate=paired_difference([x['parsed']['label']=='CONTRADICTION' for x,y in complete],
                                   [y['parsed']['label']=='CONTRADICTION' for x,y in complete])
        result['conditions'].append({'name':name,'eligible':len(eligible),'missing_pairs':len(eligible)-len(complete),
                                     'contradiction_rate_difference_first_minus_second':estimate})
    Path(output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('first'); parser.add_argument('second')
    parser.add_argument('--output',default='artifacts/model-comparison.json')
    args=parser.parse_args()
    compare(args.first,args.second,args.output)
