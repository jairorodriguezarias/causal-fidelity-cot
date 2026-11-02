"""Summarize complete runs, preserving missingness and pairing by example."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import numpy as np
from evaluation import wilson_interval


def paired_difference(first, second, seed=42, draws=5000):
    """Bootstrap paired example outcomes, not layer/cell counts."""
    if len(first)!=len(second):
        raise ValueError('Unpaired outcomes')
    if not first:
        return {'n':0,'difference':None,'low':None,'high':None}
    delta=np.asarray(first,dtype=float)-np.asarray(second,dtype=float)
    rng=np.random.default_rng(seed)
    means=delta[rng.integers(0,len(delta),size=(draws,len(delta)))].mean(axis=1)
    return {'n':len(delta),'difference':float(delta.mean()),
            'low':float(np.quantile(means,.025)),'high':float(np.quantile(means,.975))}


def summarize_condition(records, baseline):
    n=len(records)
    labels=[r['parsed']['label'] for r in records]
    eligible=[r for r in records if baseline[r['example_id']]['parsed']['label']=='ENTAILMENT']
    valid=[r for r in eligible if r['parsed']['label'] is not None]
    flips=sum(r['parsed']['label']=='CONTRADICTION' for r in valid)
    missing=len(eligible)-len(valid)
    return {'n':n,'valid_answers':sum(x is not None for x in labels),
            'status_counts':dict(Counter(r['parsed']['status'] for r in records)),
            'accuracy_all':wilson_interval(sum(r['parsed']['label']==r['gold_label'] for r in records),n),
            'contradiction_all':wilson_interval(labels.count('CONTRADICTION'),n),
            'baseline_entailment_n':len(eligible),'missing_among_baseline_entailment':missing,
            'flip_complete_cases':wilson_interval(flips,len(valid)),
            'flip_missingness_bounds':{'low':flips/len(eligible) if eligible else None,
                                      'high':(flips+missing)/len(eligible) if eligible else None},
            'mean_generated_tokens':float(np.mean([r['generated_tokens'] for r in records])) if records else None}


def analyze(run_dir, allow_incomplete=False):
    folder=Path(run_dir)
    status=json.loads((folder/'status.json').read_text())
    if status['state']!='complete' and not allow_incomplete:
        raise ValueError('Run is incomplete; use --allow-incomplete for diagnostics only')
    records=[json.loads(line) for line in (folder/'records.jsonl').read_text().splitlines()]
    manifest=json.loads((folder/'manifest.json').read_text())
    baseline={r['example_id']:r for r in records if r['condition']['kind']=='baseline' and r['variant']=='entail'}
    groups=defaultdict(list)
    for record in records:
        if record['variant']=='entail':
            groups[record['condition']['name']].append(record)
    rows=[]
    for name, group in groups.items():
        rows.append({'name':name,'condition':group[0]['condition'],**summarize_condition(group,baseline)})
    contrasts=[]
    for name, group in groups.items():
        condition=group[0]['condition']
        if condition.get('mode')!='steer':
            continue
        for mode in ('orthogonal','gaussian'):
            control_name=f"{mode}_L{condition['layer']}_a{condition['alpha']}_r0"
            controls={r['example_id']:r for r in groups.get(control_name,[])}
            matched=[(r,controls[r['example_id']]) for r in group
                     if r['example_id'] in controls and baseline[r['example_id']]['parsed']['label']=='ENTAILMENT']
            complete=[(a,b) for a,b in matched if a['parsed']['label'] is not None and b['parsed']['label'] is not None]
            contrast=paired_difference([a['parsed']['label']=='CONTRADICTION' for a,b in complete],
                                       [b['parsed']['label']=='CONTRADICTION' for a,b in complete])
            contrasts.append({'steer':name,'control':control_name,'eligible_n':len(matched),
                              'excluded_missing':len(matched)-len(complete),'complete_case_difference':contrast})
    result={'model':manifest['model_name'],'identity':manifest['identity'],
            'evaluation_split':manifest['config']['evaluation_split'],
            'run_state':status['state'],'records':len(records),'conditions':rows,'paired_controls':contrasts,
            'notes':['Missing labels are not counted as no effect.',
                     'Complete-case estimates may be biased by intervention-dependent missingness; inspect bounds.',
                     'Bootstrap sampling unit is the example; intervals are exploratory and not multiplicity adjusted.',
                     'CoT/answer agreement is descriptive, not a proof of causal mediation.']}
    (folder/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (folder/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['condition','n','valid_answers','baseline_entailment_n','flip_rate','flip_low','flip_high','missing','accuracy'])
        writer.writeheader()
        for row in rows:
            writer.writerow({'condition':row['name'],'n':row['n'],'valid_answers':row['valid_answers'],
                             'baseline_entailment_n':row['baseline_entailment_n'],
                             'flip_rate':row['flip_complete_cases']['rate'],'flip_low':row['flip_complete_cases']['low'],
                             'flip_high':row['flip_complete_cases']['high'],
                             'missing':row['missing_among_baseline_entailment'],'accuracy':row['accuracy_all']['rate']})
    plot(result,folder/'plots')
    print(json.dumps({'model':result['model'],'state':status['state'],'records':len(records),
                      'baseline':next((r for r in rows if r['name']=='baseline'),None)},indent=2))
    return result


def plot(result,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(exist_ok=True)
    activation=[r for r in result['conditions'] if r['condition']['kind']=='activation']
    if activation:
        layers=sorted({r['condition']['layer'] for r in activation})
        alphas=sorted({r['condition']['alpha'] for r in activation})
        fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
        for ax,mode in zip(axes,['steer','orthogonal','gaussian']):
            grid=np.full((len(layers),len(alphas)),np.nan)
            for r in activation:
                c=r['condition']
                if c['mode']==mode and c['replicate']==0:
                    value=r['flip_complete_cases']['rate']
                    if value is not None:
                        grid[layers.index(c['layer']),alphas.index(c['alpha'])]=value
            im=ax.imshow(np.ma.masked_invalid(grid),vmin=0,vmax=1,aspect='auto',cmap='viridis')
            ax.set(xticks=range(len(alphas)),xticklabels=alphas,yticks=range(len(layers)),yticklabels=layers,
                   xlabel='Intensity (mean-difference units)',ylabel='Layer',title=mode)
        fig.colorbar(im,ax=axes,label='Contradiction rate | baseline entailment, valid answers')
        fig.suptitle(f"{result['model']} — {result['evaluation_split']} ({result['run_state']})")
        fig.savefig(output/'layer_sweep_heatmap.pdf'); fig.savefig(output/'layer_sweep_heatmap.png',dpi=180)
        plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
        for layer in layers:
            values=sorted([r for r in activation if r['condition']['mode']=='steer' and r['condition']['layer']==layer],key=lambda r:r['condition']['alpha'])
            ax.plot([r['condition']['alpha'] for r in values],
                    [np.nan if r['flip_complete_cases']['rate'] is None else r['flip_complete_cases']['rate'] for r in values],
                    marker='o',label=f'Layer {layer}')
        ax.set(xlabel='Intensity (mean-difference units)',ylabel='Complete-case contradiction rate',ylim=(-.02,1.02))
        ax.legend(ncols=2); fig.savefig(output/'alpha_sweep_curve.pdf'); fig.savefig(output/'alpha_sweep_curve.png',dpi=180)
        plt.close(fig)
    trunc=sorted([r for r in result['conditions'] if r['name'].startswith('truncate_')],key=lambda r:r['condition']['fraction'])
    if trunc:
        fig,ax=plt.subplots(figsize=(6,4),constrained_layout=True)
        x=[r['condition']['fraction'] for r in trunc]
        y=[r['accuracy_all']['rate'] for r in trunc]
        error=np.array([[r['accuracy_all']['rate']-r['accuracy_all']['low'] for r in trunc],
                        [r['accuracy_all']['high']-r['accuracy_all']['rate'] for r in trunc]])
        ax.errorbar(x,y,yerr=error,marker='o',capsize=3)
        ax.set(xlabel='Fraction of original CoT retained',ylabel='Entailment accuracy (Wilson 95% interval)',ylim=(-.02,1.02))
        fig.savefig(output/'truncation_degradation.pdf'); fig.savefig(output/'truncation_degradation.png',dpi=180)
        plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir')
    parser.add_argument('--allow-incomplete',action='store_true')
    args=parser.parse_args()
    analyze(args.run_dir,args.allow_incomplete)
