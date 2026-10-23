from pathlib import Path
import json,csv,statistics
import numpy as np
H=Path(__file__).resolve().parent;D=H/'directed_20261017';S=H.parent
r=json.loads((D/'review/results_snapshot.json').read_text());assert len(r)==126
idx={(x['review_split'],x['job']['dataset'],x['job']['seed'],x['job']['model']['name']):x for x in r};assert len(idx)==126
rows=[];groups=json.loads((D/'review/review_summary.json').read_text())['groups'];old=json.loads((D/'completed/complete_mechanism_analysis.json').read_text())['summary']
for g,o in zip(groups,old):
 assert (g['dataset'],g['direction'])==(o['dataset'],o['direction'])
 ds='WaRP-C' if g['dataset']=='warpc' else 'TrashNet';direction=r'$c\uparrow,d\downarrow$' if g['direction']=='c_up_d_down' else r'$c\downarrow,d\uparrow$'
 vals=[o['means']['gain_joint_pp'],g['metrics']['f1']['mean'],g['metrics']['accuracy']['mean']]
 rows.append(ds+' & '+direction+' & '+' & '.join(f'${v:+.2f}$' for v in vals)+f" & {g['metrics']['f1']['negative']}/9 "+chr(92)*2)
assert '\n'.join(rows)+'\n'==(S/'tables/directed_interaction_rows.tex').read_text()
for x in r:
 b=idx[x['review_split'],x['job']['dataset'],x['job']['seed'],'base']
 if x['job']['model']['name'] in ['base','joint_c_grow_d_shrink','joint_c_shrink_d_grow']:
  assert all(abs(x['cost'][k]/b['cost'][k]-1)<=.01 for k in ['params','macs'])
cp=list(csv.DictReader((D/'completed/macro_interaction_and_proxy_error.csv').open()));assert len(cp)==36
for z in json.loads((D/'joint_proxy_comparison.json').read_text()):
 a=[x for x in cp if (z['dataset']=='all' or x['dataset']==z['dataset']) and (z['direction']=='all' or x['direction']==z['direction'])]
 assert len(a)==z['n']
 assert abs(statistics.mean(abs(float(x['joint_proxy_error_pp'])) for x in a)-z['proxy_mae'])<1e-10
 assert abs(statistics.mean(abs(float(x['gain_joint_pp'])) for x in a)-z['zero_gain_mae'])<1e-10
print('PASS:126 unique cells,36 joint contrasts,matched Base/joint budgets,main interaction table,and all proxy MAE summaries.')
