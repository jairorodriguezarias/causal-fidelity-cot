"""Supplementary paired analyses with missingness bounds and multiplicity control.

This module never changes generation settings or selects evaluation examples.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from analyze import paired_difference
from evaluation import wilson_interval


def binary(label):
    if label is None:
        return None
    if label not in ('ENTAILMENT','CONTRADICTION'):
        raise ValueError(f'Unexpected outcome: {label}')
    return int(label=='CONTRADICTION')


def mcnemar_exact(first, second):
    """Two-sided conditional exact test on the discordant paired outcomes."""
    if len(first)!=len(second):
        raise ValueError('Unpaired outcomes')
    if not first:
        return None
    positive=sum(a==1 and b==0 for a,b in zip(first,second))
    negative=sum(a==0 and b==1 for a,b in zip(first,second))
    n=positive+negative
    if not n:
        return 1.0
    return min(1.0,2*sum(math.comb(n,k) for k in range(min(positive,negative)+1))/2**n)


def holm_adjust(p_values):
    """Keep unassessable planned contrasts in the family size; return null for them."""
    adjusted=[None]*len(p_values)
    order=sorted((i for i,p in enumerate(p_values) if p is not None),key=lambda i:p_values[i])
    previous=0.0
    for rank,index in enumerate(order):
        p=p_values[index]
        if not 0<=p<=1:
            raise ValueError('Invalid p value')
        previous=max(previous,min(1.0,(len(p_values)-rank)*p))
        adjusted[index]=previous
    return adjusted


def paired_binary(first,second):
    if len(first)!=len(second):
        raise ValueError('Unpaired outcomes')
    if any(x not in (0,1,None) for x in [*first,*second]):
        raise ValueError('Outcomes must be binary or missing')
    complete=[(a,b) for a,b in zip(first,second) if a is not None and b is not None]
    a=[x for x,y in complete]; b=[y for x,y in complete]
    low=high=0
    for x,y in zip(first,second):
        xlo,xhi=(0,1) if x is None else (x,x)
        ylo,yhi=(0,1) if y is None else (y,y)
        low+=xlo-yhi; high+=xhi-ylo
    n=len(first)
    return {'eligible_pairs':n,'complete_pairs':len(complete),'missing_pairs':n-len(complete),
            'complete_case_difference':paired_difference(a,b),
            'missingness_bounds':{'low':low/n if n else None,'high':high/n if n else None},
            'discordant_treatment_only':sum(x==1 and y==0 for x,y in complete),
            'discordant_control_only':sum(x==0 and y==1 for x,y in complete),
            'exact_p_complete_cases':mcnemar_exact(a,b)}


def clustered_baseline(baselines):
    by_id=defaultdict(dict)
    for record in baselines:
        by_id[record['example_id']][record['variant']]=record
    per_example=[]
    for example_id,pair in by_id.items():
        if set(pair)!={'entail','contra'}:
            continue
        per_example.append(sum(r['parsed']['label']==r['gold_label'] for r in pair.values())/2)
    if per_example:
        values=np.asarray(per_example)
        rng=np.random.default_rng(42)
        means=values[rng.integers(0,len(values),size=(5000,len(values)))].mean(axis=1)
        overall={'premises':len(values),'accuracy':float(values.mean()),
                 'low':float(np.quantile(means,.025)),'high':float(np.quantile(means,.975))}
    else:
        overall={'premises':0,'accuracy':None,'low':None,'high':None}
    by_class={}
    for variant in ('entail','contra'):
        rows=[r for r in baselines if r['variant']==variant]
        by_class[variant]={'accuracy':wilson_interval(sum(r['parsed']['label']==r['gold_label'] for r in rows),len(rows)),
                           'parse_status':dict(Counter(r['parsed']['status'] for r in rows)),
                           'finish_reason':dict(Counter(r['finish_reason'] for r in rows))}
    return {'by_class':by_class,'balanced_accuracy_clustered_by_premise':overall}


def analyze_robustness(folder):
    folder=Path(folder)
    manifest=json.loads((folder/'manifest.json').read_text())
    status=json.loads((folder/'status.json').read_text())
    if status['state']!='complete':
        raise ValueError('Supplementary inference requires a complete run')
    records_path=folder/'records.jsonl'
    records=[json.loads(line) for line in records_path.read_text().splitlines()]
    baseline=[r for r in records if r['condition']['kind']=='baseline']
    baseline_e={r['example_id']:r for r in baseline if r['variant']=='entail'}
    groups=defaultdict(dict)
    conditions={}
    for r in records:
        if r['variant']=='entail':
            name=r['condition']['name']
            groups[name][r['example_id']]=r
            conditions[name]=r['condition']
    def comparison(name,control,only_baseline_correct=True):
        ids=sorted(set(groups[name]).intersection(groups[control]))
        if only_baseline_correct:
            ids=[i for i in ids if baseline_e[i]['parsed']['label']=='ENTAILMENT']
        report=paired_binary([binary(groups[name][i]['parsed']['label']) for i in ids],
                             [binary(groups[control][i]['parsed']['label']) for i in ids])
        return {'treatment':name,'control':control,'example_ids':ids,
                'baseline_correct_only':only_baseline_correct,**report}
    activation=[]
    for name,condition in conditions.items():
        if condition.get('mode')=='steer':
            for mode in ('orthogonal','gaussian'):
                for rep in range(manifest['config']['control_replicates']):
                    control=f"{mode}_L{condition['layer']}_a{condition['alpha']}_r{rep}"
                    if control not in groups:
                        raise ValueError(f'Missing control {control}')
                    activation.append(comparison(name,control))
    for item,p in zip(activation,holm_adjust([r['exact_p_complete_cases'] for r in activation])):
        item['holm_adjusted_p_within_model_activation_family']=p
    text=[]
    text_control='truncate_1.0'
    if text_control in groups:
        for name,condition in conditions.items():
            if condition['kind']=='text' and name!=text_control:
                for baseline_correct in (False,True):
                    text.append(comparison(name,text_control,baseline_correct))
        # Full-prefix reconstruction may itself change the sampled answer.
        text.append(comparison(text_control,'baseline',False))
    result={'schema_version':1,'model':manifest['model_name'],'run_identity':manifest['identity'],
            'evaluation_split':manifest['config']['evaluation_split'],
            'records_sha256':hashlib.sha256(records_path.read_bytes()).hexdigest(),
            'analysis_source_sha256':{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                     for name in ('robustness.py','analyze.py','evaluation.py')},
            'baseline':clustered_baseline(baseline),'activation_contrasts':activation,
            'text_contrasts_against_full_prefix':text,
            'interpretation':['Every difference is the contradiction rate in treatment minus control.',
                              'Missingness bounds are identification bounds, not sampling confidence intervals.',
                              'Complete-case inference can be biased by treatment-dependent missingness.',
                              'The two baseline hypotheses sharing a premise are resampled as a cluster.',
                              'Holm adjustment covers all activation/control contrasts within a model, not cross-model or text families.',
                              'These supplemental analyses are exploratory; the generation protocol is unchanged.']}
    output=folder/'robustness.json'
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    plot_missingness(records,folder/'plots',manifest['model_name'])
    return result


def plot_missingness(records,output,model_name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[r for r in records if r['condition']['kind']=='activation' and r['condition']['replicate']==0]
    if not rows:
        return
    output.mkdir(exist_ok=True)
    layers=sorted({r['condition']['layer'] for r in rows})
    alphas=sorted({r['condition']['alpha'] for r in rows})
    groups=defaultdict(list)
    for record in rows:
        c=record['condition']
        groups[(c['mode'],c['layer'],c['alpha'])].append(record)
    fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
    for ax,mode in zip(axes,['steer','orthogonal','gaussian']):
        grid=np.full((len(layers),len(alphas)),np.nan)
        for i,layer in enumerate(layers):
            for j,alpha in enumerate(alphas):
                batch=groups[(mode,layer,alpha)]
                if batch:
                    grid[i,j]=sum(r['parsed']['label'] is None for r in batch)/len(batch)
        im=ax.imshow(np.ma.masked_invalid(grid),vmin=0,vmax=1,aspect='auto',cmap='magma')
        ax.set(xticks=range(len(alphas)),xticklabels=alphas,yticks=range(len(layers)),yticklabels=layers,
               xlabel='Intensity (mean-difference units)',ylabel='Layer',title=mode)
    fig.colorbar(im,ax=axes,label='Fraction without a valid final label')
    fig.suptitle(model_name+' — missing outcomes')
    fig.savefig(output/'missingness_heatmap.pdf')
    fig.savefig(output/'missingness_heatmap.png',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',nargs='+')
    args=parser.parse_args()
    for run in args.runs:
        result=analyze_robustness(run)
        print(json.dumps({'model':result['model'],'activation_contrasts':len(result['activation_contrasts']),
                          'text_contrasts':len(result['text_contrasts_against_full_prefix'])}))
