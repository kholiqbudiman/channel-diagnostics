"""CPU-only verification of the distributed corrected ACRE analysis evidence.
Run from any directory: python3 research_artifacts/verify_corrected_evidence.py
Requires NumPy. Does not train models or claim checkpoint re-inference.
"""
from pathlib import Path
import json,hashlib,statistics,collections
import numpy as np
HERE=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source=HERE/'stability_sources';lock=read(source/'COHORT_LOCK.json')
for name,h in lock['sources'].items():assert sha(source/name)==h,name
for split,h in lock['split_locks'].items():
 folder=source/f'split_{split}';assert sha(folder/'protocol_lock.json')==h
 for name,v in read(folder/'protocol_lock.json')['sources'].items():assert sha(folder/name)==v,name
runs=read(HERE/'stability_audit/verified_results_snapshot.json');audit=read(HERE/'stability_audit/audit_results.json');assert len(runs)==72
seen=set();models=['base','base_no_residual','allocation_14_30_36_52','allocation_12_34_36_36'];labels=['Base','No residual','Allocation A','Allocation B'];rows=[]
for r in runs:
 j=r['job'];key=(r['split'],j['dataset'],j['model']['name'],j['seed']);assert key not in seen;seen.add(key)
 assert r['split'] in [4302,4303,4304] and j['seed'] in [2901,2902,2903]
 folder=source/f"split_{r['split']}";m=read(folder/f"data_{j['dataset']}.json");assert m['classes']==r['classes']
 cm=np.array(r['probe_metrics']['confusion_matrix']);support=cm.sum(1);tp=cm.diagonal();den=support+cm.sum(0)
 f1=np.divide(2*tp,den,out=np.zeros(len(tp)),where=den>0).mean()*100
 assert abs(f1-r['probe_metrics']['macro_f1_pct'])<1e-8
 expected=np.bincount([m['samples'][i][1] for i in m['split']['probe']],minlength=len(tp));assert np.array_equal(support,expected)
 assert abs(tp.sum()/cm.sum()*100-r['probe_metrics']['accuracy_pct'])<1e-8
 history=r['history'];vals=[x['selection_macro_f1_pct'] for x in history];best=int(np.argmax(vals))+1
 assert r['best_epoch']==best and vals[best-1]==r['selection_macro_f1_pct'] and len(vals)==min(best+20,100)
 for ds in [j['dataset']]:
  base=next(v for v in runs if v['split']==r['split'] and v['job']['dataset']==ds and v['job']['model']['name']=='base' and v['job']['seed']==j['seed'])
  assert all(abs(r['cost'][c]/base['cost'][c]-1)<=.01 for c in ['params','macs'])
for ds in ['warpc','trashnet']:
 for mi,model in enumerate(models):
  sub=[r for r in runs if r['job']['dataset']==ds and r['job']['model']['name']==model];assert len(sub)==9
  vals=[r['probe_metrics']['macro_f1_pct'] for r in sub];diff=[]
  for r in sub:
   b=next(v for v in runs if v['split']==r['split'] and v['job']['dataset']==ds and v['job']['seed']==r['job']['seed'] and v['job']['model']['name']=='base');diff.append(r['probe_metrics']['macro_f1_pct']-b['probe_metrics']['macro_f1_pct'])
  a=next(x for x in audit['summary'] if x['dataset']==ds and x['model']==model)
  assert abs(statistics.mean(vals)-a['mean_macro_f1_pct'])<1e-10 and abs(statistics.stdev(vals)-a['descriptive_sd'])<1e-10
  assert abs(statistics.mean(diff)-a['mean_delta_pp'])<1e-10
  delta='---' if model=='base' else f'${statistics.mean(diff):+.2f}$';pos='---' if model=='base' else f'{sum(x>0 for x in diff)}/9'
  rows.append(f"{'WaRP-C' if ds=='warpc' else 'TrashNet'} & {labels[mi]} & ${statistics.mean(vals):.2f}\\pm{statistics.stdev(vals):.2f}$ & {delta} & {pos} \\\\")
assert '\n'.join(rows)+'\n'==(HERE.parent/'tables/stability_rows.tex').read_text()
print('PASS: source locks,72 unique corrected runs, confusion metrics, selection history, joint budget band, and manuscript summary table.')
print('Checkpoint files, raw images, and saved-logit NPZ files are not embedded; their prior audit is documented, not rerun by this lightweight verifier.')
