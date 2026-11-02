#!/usr/bin/env python3
"""CPU-only figures from completed records, verified against existing reports.

Writes only to a separate output folder, never into a run or scientific source.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
from evaluation import wilson_interval
from robustness import binary, holm_adjust, paired_binary

NAMES = ['truncate_0.0', 'truncate_0.25', 'truncate_0.5', 'truncate_0.75',
         'swap_contradiction_cot', 'shuffle_cot']
LABELS = ['0% prefix retained', '25% prefix retained', '50% prefix retained', '75% prefix retained',
          'Contradiction trace swap', 'Shuffled tokens']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(fig, output, name):
    fig.savefig(output/f'{name}.png', dpi=190)
    fig.savefig(output/f'{name}.pdf')
    plt.close(fig)


def csv_table(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def flat_contrast(x):
    d, b = x['complete_case_difference'], x['missingness_bounds']
    fields = ['treatment', 'control', 'baseline_correct_only', 'eligible_pairs',
              'complete_pairs', 'missing_pairs', 'treatment_contradictions_all_eligible',
              'control_contradictions_all_eligible', 'discordant_treatment_only',
              'discordant_control_only', 'exact_p_complete_cases']
    return {**{k:x[k] for k in fields}, 'difference_complete_cases':d['difference'],
            'bootstrap_95_low':d['low'], 'bootstrap_95_high':d['high'],
            'missingness_bound_low':b['low'], 'missingness_bound_high':b['high'],
            'holm_activation_family':x.get('holm_adjusted_p_within_model_activation_family')}


def text_figure(rows, out, model):
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.8))
    fig.subplots_adjust(left=.21, right=.98, top=.79, bottom=.23, wspace=.21)
    for ax, correct in zip(axes, (False, True)):
        for y, name in enumerate(NAMES):
            row = next(r for r in rows if r['treatment']==name and r['baseline_correct_only']==correct)
            d, b = row['complete_case_difference'], row['missingness_bounds']
            ax.plot([d['low']*100,d['high']*100],[y,y],color='#276FBF',lw=2.2)
            ax.scatter([d['difference']*100],[y],color='#276FBF',s=34,zorder=3)
            ax.plot([b['low']*100,b['high']*100],[y+.17,y+.17],color='#B56800',lw=4,
                    solid_capstyle='butt',marker='|',markersize=6)
            ax.text(121,y+.06,f"{row['complete_pairs']}/{row['eligible_pairs']}",
                    ha='center',va='center',fontsize=9,color='#444444')
        ax.axvline(0,color='#888888',lw=1,linestyle='--',zorder=0)
        ax.set(xlim=(-24,132),ylim=(5.55,-.65),yticks=range(6),
               yticklabels=LABELS if not correct else [],
               xlabel='Change in contradiction rate (percentage points)')
        ax.set_title('Initially correct examples' if correct else 'All eligible examples',pad=25)
        ax.set_xticks([-20,0,20,40,60,80,100])
        ax.text(121,-.48,'Complete /\neligible pairs',ha='center',va='center',fontsize=8,color='#555555')
        ax.axvline(109,color='#EEEEEE',lw=1)
        ax.grid(axis='x',color='#EEEEEE')
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle(f'{model} · Interventions on written reasoning',fontsize=15,y=.96)
    fig.text(.5,.895,'Each intervention is compared with the full original prefix on the same examples.',
             ha='center',fontsize=10)
    handles=[Line2D([0],[0],color='#276FBF',marker='o',lw=2,label='Paired difference and bootstrap 95% CI (complete cases)'),
             Line2D([0],[0],color='#B56800',lw=4,label='Missing-outcome bounds (not confidence intervals)')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.55,.09),frameon=False,fontsize=9)
    fig.text(.04,.035,
             'Exploratory analysis; CIs are not multiplicity-adjusted. Rows reuse examples; eligibility differs across interventions.\n'
             '0% closes the thought block and permits up to 128 new tokens; it does not eliminate all computation or establish absence of reasoning.',
             fontsize=9,va='bottom')
    save(fig,out,'text_paired_contrasts')


def truncation_figure(rows,out,model):
    fig,ax=plt.subplots(figsize=(8.2,5.9))
    fig.subplots_adjust(left=.12,right=.96,top=.81,bottom=.24)
    x=np.array([r['fraction']*100 for r in rows])
    y=np.array([r['accuracy']['rate'] for r in rows])
    lo=np.array([r['accuracy']['low'] for r in rows])
    hi=np.array([r['accuracy']['high'] for r in rows])
    ax.errorbar(x,y,yerr=[y-lo,hi-y],fmt='o-',color='#276FBF',capsize=5)
    for xi,yi,r in zip(x,y,rows):
        ax.annotate(f"{r['correct']}/{r['n']}",(xi,yi),xytext=(0,-20),textcoords='offset points',ha='center',fontsize=10)
        ax.text(xi,.55,f"Missing label: {r['missing']}",ha='center',fontsize=8.5,color='#555555')
    n=rows[0]['n']
    ax.set(xlim=(-8,108),ylim=(.49,1.01),xticks=x,
           xlabel='Percentage of original reasoning prefix retained',ylabel=f'Correct answers / {n} examples')
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis='y',color='#EEEEEE'); ax.spines[['top','right']].set_visible(False)
    fig.suptitle(f'{model} · Prefix truncation',fontsize=15,y=.96)
    fig.text(.5,.895,f'Same {n} examples in all five conditions; bars: marginal Wilson 95% CIs.',ha='center',fontsize=10)
    fig.text(.06,.045,
             'Missing labels count as incorrect. Observations are paired: these marginal CIs do not establish\n'
             'differences or equivalence between conditions. At 0%, the thought block is closed and up to\n'
             '128 new tokens are allowed; this is not equivalent to removing all reasoning.',fontsize=9)
    save(fig,out,'truncation_accuracy')


def activation_figure(rows,conditions,out,model):
    layers=sorted({conditions[r['treatment']]['layer'] for r in rows})
    alphas=sorted({conditions[r['treatment']]['alpha'] for r in rows})
    fig,axes=plt.subplots(1,2,figsize=(10.5,7.6))
    fig.subplots_adjust(left=.09,right=.86,bottom=.19,top=.79,wspace=.25)
    for ax,mode in zip(axes,('orthogonal','gaussian')):
        values=np.full((len(layers),len(alphas)),np.nan); cells={}
        for row in rows:
            if conditions[row['control']]['mode']!=mode: continue
            c=conditions[row['treatment']]; i,j=layers.index(c['layer']),alphas.index(c['alpha'])
            values[i,j]=row['complete_case_difference']['difference']*100; cells[i,j]=row
        im=ax.imshow(values,vmin=-20,vmax=20,cmap='RdBu_r',aspect='auto')
        for (i,j),row in cells.items():
            ax.text(j,i,f"{values[i,j]:+.1f}\n{row['complete_pairs']}/{row['eligible_pairs']}",ha='center',va='center',
                    fontsize=9,color='white' if abs(values[i,j])>13 else '#111111')
        ax.set(xticks=range(len(alphas)),xticklabels=alphas,yticks=range(len(layers)),yticklabels=layers,
               xlabel='Intensity (mean-difference units)',ylabel='Layer',
               title='Steering − '+('orthogonal' if mode=='orthogonal' else 'Gaussian')+' control')
    cax=fig.add_axes([.89,.25,.022,.49])
    fig.colorbar(im,cax=cax,label='Contradiction-rate difference (percentage points)')
    fig.suptitle(f'{model} · Steering versus equal-norm controls',fontsize=14,y=.96)
    p=min(r['exact_p_complete_cases'] for r in rows)
    adjusted=min(r['holm_adjusted_p_within_model_activation_family'] for r in rows)
    fig.text(.5,.89,f'{len(rows)} paired contrasts · minimum exact p: {p:.4f} · minimum Holm-adjusted p: {adjusted:.2f}',ha='center',fontsize=10)
    fig.text(.07,.045,
             'Each cell shows the complete-case difference and complete pairs / eligible initially correct examples.\n'
             'Examples are reused: cells are not independent replications. Lack of statistical significance does\n'
             'not establish absence of an effect. See tables for CIs, missing-outcome bounds and exact p values.',fontsize=9)
    save(fig,out,'activation_control_contrasts')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'runs/study-20260925/distill-7b')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/diagnostics/20261102/figures')
    args=parser.parse_args()
    folder,out=args.run.resolve(),args.output.resolve()
    if out==folder or folder in out.parents:
        raise ValueError('Output must be outside the source run')
    paths={name:folder/name for name in ('records.jsonl','summary.json','robustness.json','manifest.json','status.json')}
    hashes={name:digest(path) for name,path in paths.items()}
    status=json.loads(paths['status.json'].read_text())
    if status['state']!='complete': raise ValueError('Figures require a complete run')
    records=[json.loads(line) for line in paths['records.jsonl'].read_text().splitlines()]
    assert len(records)==status['records'] and len({r['key'] for r in records})==len(records)
    report=json.loads(paths['robustness.json'].read_text())
    summary=json.loads(paths['summary.json'].read_text())
    assert report['records_sha256']==hashes['records.jsonl']
    groups=defaultdict(dict); conditions={}
    for r in records:
        if r['variant']=='entail':
            name=r['condition']['name']
            groups[name][r['example_id']]=r; conditions[name]=r['condition']
    for row in summary['conditions']:
        raw=list(groups[row['name']].values())
        assert row['n']==len(raw)
        assert row['accuracy_all']['successes']==sum(r['parsed']['label']==r['gold_label'] for r in raw)
        assert row['valid_answers']==sum(r['parsed']['label'] is not None for r in raw)
    baseline=groups['baseline']

    def verify(item):
        ids=sorted(set(groups[item['treatment']]) & set(groups[item['control']]))
        if item['baseline_correct_only']:
            ids=[i for i in ids if baseline[i]['parsed']['label']=='ENTAILMENT']
        assert item['example_ids']==ids
        a=[binary(groups[item['treatment']][i]['parsed']['label']) for i in ids]
        b=[binary(groups[item['control']][i]['parsed']['label']) for i in ids]
        computed=paired_binary(a,b)
        for name,value in computed.items():
            assert item[name]==value,(item['treatment'],name,item[name],value)
        return {**item,'treatment_contradictions_all_eligible':a.count(1),
                'control_contradictions_all_eligible':b.count(1)}

    text=[verify(x) for x in report['text_contrasts_against_full_prefix']]
    activation=[verify(x) for x in report['activation_contrasts']]
    assert holm_adjust([x['exact_p_complete_cases'] for x in activation])==[
        x['holm_adjusted_p_within_model_activation_family'] for x in activation]
    truncation=[]; ids=set(groups['truncate_1.0'])
    for fraction in (0.,.25,.5,.75,1.):
        group=groups[f'truncate_{fraction}']
        assert set(group)==ids,'Truncation conditions must use the same examples'
        correct=sum(r['parsed']['label']=='ENTAILMENT' for r in group.values())
        truncation.append({'fraction':fraction,'n':len(group),'correct':correct,
                           'missing':sum(r['parsed']['label'] is None for r in group.values()),
                           'accuracy':wilson_interval(correct,len(group))})
    out.mkdir(parents=True,exist_ok=True)
    tables={'model':report['model'],'run_state':'complete','records':len(records),
            'source_hashes':hashes,'builder_sha256':digest(Path(__file__)),
            'analysis_sources':{name:digest(ROOT/name) for name in ('robustness.py','analyze.py','evaluation.py')},
            'versions':{'python':sys.version,'numpy':np.__version__,'matplotlib':matplotlib.__version__},
            'paired_bootstrap':{'seed':42,'draws':5000,'unit':'matched example','interval':'percentile 95%'},
            'verification':'Raw counts, paired contrasts, missingness bounds, bootstrap intervals and Holm p values match existing reports.',
            'text_contrasts':text,'activation_contrasts':activation,'truncation':truncation,
            'notes':['Text contrasts and their p values/intervals are exploratory and unadjusted.',
                     'Activation Holm family contains all within-model activation/control contrasts.',
                     'Missingness bounds are identification bounds, not confidence intervals.',
                     'Every contrast uses matched examples; eligibility can differ between contrasts.',
                     'Conditions reuse examples and are not independent replication.',
                     'Substitution changes context and does not identify natural reasoning mediation.']}
    (out/'tables.json').write_text(json.dumps(tables,indent=2,ensure_ascii=False)+'\n')
    csv_table(out/'text_contrasts.csv',[flat_contrast(x) for x in text])
    csv_table(out/'activation_contrasts.csv',[flat_contrast(x) for x in activation])
    csv_table(out/'truncation.csv',[{k:v for k,v in r.items() if k!='accuracy'} |
              {'accuracy':r['accuracy']['rate'],'wilson_95_low':r['accuracy']['low'],
               'wilson_95_high':r['accuracy']['high']} for r in truncation])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
    text_figure(text,out,report['model'])
    truncation_figure(truncation,out,report['model'])
    activation_figure(activation,conditions,out,report['model'])
    assert hashes=={name:digest(path) for name,path in paths.items()},'Source data changed during figure generation'
    assert 'torch' not in sys.modules and 'transformers' not in sys.modules,'GPU framework unexpectedly imported'
    print(json.dumps({'output':str(out),'verified_records':len(records),'verified_text_contrasts':len(text),
                      'verified_activation_contrasts':len(activation),'figures':3,'source_files_unchanged':True}))


if __name__=='__main__':
    main()
