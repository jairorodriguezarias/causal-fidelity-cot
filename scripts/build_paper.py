"""Compile the manuscript; a submission build requires the complete evidence audit."""
import argparse
from datetime import date
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
from audit_release import audit_release

ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/'paper'


def tex(value):
    mapping={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}'}
    return ''.join(mapping.get(c,c) for c in str(value))


parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--runs',nargs='+',required=True)
parser.add_argument('--allow-incomplete',action='store_true')
args=parser.parse_args()
reports=[]
for run in args.runs:
    folder=ROOT/run
    manifest=json.loads((folder/'manifest.json').read_text())
    summary=json.loads((folder/'summary.json').read_text())
    status=json.loads((folder/'status.json').read_text())
    reports.append((folder,manifest,summary,status))
if not args.allow_incomplete:
    audit=audit_release([ROOT/r for r in args.runs])
    (ROOT/'artifacts/release-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    if not audit['passed']:
        raise SystemExit('Release audit failed:\n'+'\n'.join(audit['errors']))
    discussion=PAPER/'results-discussion.tex'
    if not discussion.exists() or 'PENDING' in discussion.read_text():
        raise SystemExit('The completed results still need a substantive results/discussion section.')
authors_path=PAPER/'authors.json'
authors=json.loads(authors_path.read_text()) if authors_path.exists() else None
status_text=r'\\[.5em]\large Working draft --- experiments incomplete' if args.allow_incomplete else ''
metadata='\\newcommand{\\StudyStatus}{'+status_text+'}\n'
if authors and authors.get('authors'):
    author_text=r' \and '.join(tex(a['name'])+r'\\'+tex(a.get('affiliation',''))+r'\\\texttt{'+tex(a.get('email',''))+'}' for a in authors['authors'])
else:
    author_text='Author details pending'
metadata+='\\newcommand{\\StudyAuthors}{'+author_text+'}\n'
metadata+='\\newcommand{\\StudyDate}{'+str(date.today())+'}\n'
(PAPER/'generated-metadata.tex').write_text(metadata)
if args.allow_incomplete:
    abstract='This working draft reports only completed implementation checks; the planned held-out checkpoint comparison remains incomplete.\n'
else:
    abstract='The study evaluates '+str(len(reports))+' checkpoints on 50 held-out premise pairs each; all missing generations and assessor errors are reported separately.\n'
(PAPER/'generated-abstract.tex').write_text(abstract)
lines=[]
if args.allow_incomplete:
    lines.append(r'\noindent\textbf{Draft status:} The held-out comparison is not yet complete. The table below is an execution audit, not evidence of a training-method effect.')
lines += [r'\begin{center}',r'\begin{tabular}{lrrrr}',r'\toprule',r'Run & Examples & Valid & Correct & Generations \\',r'\midrule']
for folder,manifest,summary,status in reports:
    row=next(r for r in summary['conditions'] if r['name']=='baseline')
    name=manifest['model_name']+' / '+manifest['config']['evaluation_split']
    lines.append(f"{tex(name)} & {row['n']} & {row['valid_answers']} & {row['accuracy_all']['successes']} & {summary['records']} \\\\")
lines += [r'\bottomrule',r'\end{tabular}',r'\end{center}',
          r'Valid denotes a parseable final answer. Correct uses the dataset label and counts missing answers as incorrect. Calibration observations are not held-out findings.']
validation=ROOT/'artifacts/judge-validation-results.json'
if validation.exists():
    v=json.loads(validation.read_text())
    lines.append(f"The assessor passed {v['correct']} of {v['total']} synthetic validation cases. These are engineered checks, not human judgments of generated traces.")
if not args.allow_incomplete:
    lines.append(r'\input{results-discussion}')
    figures=PAPER/'figures'
    figures.mkdir(exist_ok=True)
    for folder,manifest,summary,status in reports:
        for filename in ('layer_sweep_heatmap','truncation_degradation'):
            source=folder/'plots'/f'{filename}.pdf'
            name=f"{manifest['model_name']}-{filename}.pdf"
            shutil.copy2(source,figures/name)
            lines += [r'\begin{figure}[htbp]',r'\centering',
                      r'\includegraphics[width=\textwidth]{figures/'+name+'}',
                      r'\caption{'+tex(manifest['model_name']+' — '+filename.replace('_',' '))+r'. Complete-case rates must be interpreted alongside the reported missingness counts.}',
                      r'\end{figure}']
(PAPER/'generated-results.tex').write_text('\n'.join(lines)+'\n')
subprocess.run([str(ROOT/'.tools/tectonic'),'--keep-logs','--keep-intermediates','main.tex'],cwd=PAPER,check=True)
archive_name='arxiv-draft-source.tar.gz' if args.allow_incomplete else 'arxiv-source.tar.gz'
with tarfile.open(PAPER/archive_name,'w:gz') as archive:
    # Package the main manuscript only. Standalone interim reports reference
    # figures outside paper/ and belong in the full reproduction bundle.
    for filename in ('main.tex','references.bib','main.bbl','generated-metadata.tex',
                     'generated-abstract.tex','generated-results.tex','results-discussion.tex'):
        path=PAPER/filename
        if path.is_file():
            archive.add(path,arcname=path.name)
    if not args.allow_incomplete:
        for path in sorted((PAPER/'figures').glob('*.pdf')):
            archive.add(path,arcname='figures/'+path.name)
print(f'Built paper/main.pdf and paper/{archive_name}.')
