"""Read-only reconciliation of retained empirical records and printed table values.
Run from any directory: python3 research_artifacts/verify_empirical_evidence.py
Needs Python 3 and NumPy. Does not re-run inference or authenticate remote binaries.
"""
from pathlib import Path
import json,csv,statistics,hashlib
import numpy as np
H=Path(__file__).resolve().parent
# During development the script lives in /tmp; installed copy is portable.
if H==Path('/tmp'):H=Path('/tmp/acre_empirical_20261017/source/research_artifacts')
S=H.parent;T=H/'proxy_transfer_20261017';C=T/'completed_20261017'
load=lambda p:json.loads(p.read_text())
rows=lambda p:list(csv.DictReader(p.open()))
pilot=load(H/'pilot_audit/verified_pilot_snapshot.json')['results'];stability=load(H/'stability_audit/verified_results_snapshot.json');directed=load(H/'directed_20261017/review/results_snapshot.json');dtd=load(C/'verified_results_snapshot.json')
assert list(map(len,[pilot,stability,directed,dtd]))==[36,72,126,63]
def key(r):
 return (r.get('review_split',r.get('split',r['job'].get('split_seed'))),r['job']['dataset'],r['job']['seed'],r['job']['model']['name'])
sidx={key(r):r for r in stability};didx={key(r):r for r in directed};nidx={key(r):r for r in dtd}
assert len(sidx)==72 and len(didx)==126 and len(nidx)==63
for k,r in didx.items():
 if k[-1]=='base':
  assert k in sidx and r['checkpoint_sha256']==sidx[k]['checkpoint_sha256']
  assert r['probe_metrics']==sidx[k]['probe_metrics']
unique={key(r) for cohort in [pilot,stability,directed,dtd] for r in cohort}
assert len(unique)==279
# Metrics and checkpoint selection histories.
for r in pilot+stability+directed+dtd:
 cm=np.array(r['probe_metrics']['confusion_matrix'],float);tp=np.diag(cm);den=cm.sum(0)+cm.sum(1)
 f1=float(np.divide(2*tp,den,out=np.zeros_like(tp),where=den!=0).mean()*100)
 accuracy=float(tp.sum()/cm.sum()*100)
 assert abs(f1-r['probe_metrics']['macro_f1_pct'])<1e-9
 assert abs(accuracy-r['probe_metrics']['accuracy_pct'])<1e-9
 hist=r['history'];best=max(hist,key=lambda h:h['selection_macro_f1_pct'])
 assert best['epoch']==r['best_epoch'] and abs(best['selection_macro_f1_pct']-r['selection_macro_f1_pct'])<1e-9
for idx in [didx,nidx]:
 for k,r in idx.items():
  if k[-1] in ['base','joint_c_grow_d_shrink','joint_c_shrink_d_grow']:
   b=idx[k[:-1]+('base',)]
   assert all(abs(r['cost'][q]/b['cost'][q]-1)<=.01 for q in ['params','macs'])
# Verify frozen source files and retained prospective release metadata locally.
lock=load(T/'COHORT_LOCK.json')
for name,digest in lock['sources'].items():
 assert hashlib.sha256((T/name).read_bytes()).hexdigest()==digest,name
barrier=C/'PREDICTION_BARRIER.json';release=load(C/'EVALUATION_RELEASE.json')
assert hashlib.sha256(barrier.read_bytes()).hexdigest()==release['barrier_sha256']
assert len(load(barrier)['predictions'])==9
receipt=load(C/'evaluation_audit.json')
assert (receipt['records'],receipt['checkpoint_hashes_verified'],receipt['initialization_hashes_verified'],receipt['prediction_commits_verified'])==(63,63,63,9)

# Prospective DTD grouping, role disjointness and exact-content boundary.
for sp in [6301,6302,6303]:
 z=load(T/f'data_{sp}.json');roles=list(z['split'].values());sets=[set(v) for v in roles]
 assert sum(map(len,sets))==3760 and len(set.union(*sets))==3760
 for i in range(4):
  for j in range(i):assert not sets[i]&sets[j]
 # Manifest SHA map is indexed by relative path; grouping holds even when byte-identical paths differ.
 hashes=z['image_sha256'];assert len(hashes)==3760
 hs=[{hashes[z['samples'][i][0]] for i in q} for q in sets]
 for i in range(4):
  for j in range(i):assert not hs[i]&hs[j]
# Every policy and reported gain/regret is reconciled to saved cell metrics.
all_dec=rows(T/'old_comparison_complete/old_comparison/all_decisions.csv')+rows(C/'dtd_comparison/all_decisions.csv')
idx={**didx,**nidx}
for x in all_dec:
 k=(int(x['split']),x['dataset'],int(x['seed']));b=idx[k+('base',)]['probe_metrics']['macro_f1_pct']
 gain=lambda n:idx[k+(n,)]['probe_metrics']['macro_f1_pct']-b
 assert abs(gain(x['selected'])-float(x['gain_pp']))<1e-9
 h=max(gain(n) for n in ['base','joint_c_grow_d_shrink','joint_c_shrink_d_grow'])
 assert abs(h-gain(x['selected'])-float(x['regret_pp']))<1e-9
 assert abs(h-float(x['base_regret_pp']))<1e-9
 assert abs(sum(gain(n) for n in ['base','joint_c_grow_d_shrink','joint_c_shrink_d_grow'])/3-float(x['uniform_random_expected_gain_pp']))<1e-9
summaries=load(T/'old_comparison_complete/old_comparison/summary.json')['methods']+load(C/'dtd_comparison/summary.json')['methods']
for z in summaries:
 a=[x for x in all_dec if (x['dataset'],x['method'])==(z['dataset'],z['method'])];assert len(a)==9
 assert abs(statistics.mean(float(x['gain_pp']) for x in a)-z['mean_selected_gain_pp'])<1e-9
 assert abs(statistics.mean(float(x['regret_pp']) for x in a)-z['mean_regret_pp'])<1e-9
# Interaction and error identity on all six task/direction groups.
ints=rows(T/'old_comparison_complete/old_comparison/all_interactions.csv')+rows(C/'dtd_comparison/all_interactions.csv')
for x in ints:
 k=(int(x['split']),x['dataset'],int(x['seed']));names=['joint_c_grow_d_shrink','c_grow','d_shrink','base'] if x['direction']=='c_up_d_down' else ['joint_c_shrink_d_grow','c_shrink','d_grow','base']
 for metric,col in [('macro_f1_pct','interaction_f1_pp'),('accuracy_pct','interaction_accuracy_pp')]:
  values=[idx[k+(n,)]['probe_metrics'][metric] for n in names];v=values[0]-values[1]-values[2]+values[3];assert abs(v-float(x[col]))<1e-9
 for m in ['mask_f1','architecture_ridge']:
  assert abs(float(x[m+'_joint_error'])-float(x['interaction_f1_pp'])-float(x[m+'_component_c_error'])-float(x[m+'_component_d_error']))<1e-9
# DTD headline endpoints, headroom and MAE.
ss=load(C/'dtd_comparison/summary.json');primary=ss['primary'];assert len(primary)==9
assert abs(statistics.mean(x['ce_policy_minus_f1_policy_pp'] for x in primary)-.11053045323651823)<1e-9
assert sum(abs(x['ce_policy_minus_f1_policy_pp'])>1e-12 for x in primary)==1
head=load(C/'candidate_headroom_review.json');assert sum(x['base_is_best'] for x in head)==7
assert sum(v>0 for x in head for v in x['joint_gains_pp'])==2
assert abs(statistics.mean(x['oracle_gain_pp'] for x in head)-.6118079267558644)<1e-9
fx=rows(C/'dtd_comparison/all_effects.csv');assert len(fx)==54
for m in ['mask_f1','architecture_ridge','choose_base']:
 mae=statistics.mean(abs(float(x['observed_f1_gain_pp'])-float(x[m])) for x in fx)
 assert abs(mae-next(z['mean_f1_gain_mae_pp'] for z in ss['methods'] if z['method']==m))<1e-9
# Exact printed policy table, ensuring rows track the preserved records.
methods=['mask_f1','mask_macro_ce','taylor_signed','taylor_squared','bn_gamma','architecture_ridge','choose_base']
labels=['F1 mask','Loss mask','Signed Taylor','Squared Taylor','BN scale','Architecture ridge','Retain Base']
lookup={(z['dataset'],z['method']):z for z in summaries};expected=[]
for m,label in zip(methods,labels):
 z=lookup['dtd',m];rho='---' if z['mean_spearman_f1'] is None else f"{z['mean_spearman_f1']:+.3f}"
 expected.append(label+' & '+' & '.join(f"${lookup[d,m]['mean_selected_gain_pp']:+.3f}$" for d in ['warpc','trashnet','dtd'])+f" & {z['mean_regret_pp']:.3f} & {rho} "+chr(92)*2)
assert '\n'.join(expected)+'\n'==(S/'tables/empirical_policy_rows.tex').read_text()
print(json.dumps({'status':'PASS','distinct_training_runs':279,'cohort_new_runs':[36,72,108,63],'reused_base_records':18,'metric_records_checked':297,'policy_decisions_checked':len(all_dec),'interaction_contexts_checked':len(ints),'dtd_all_effects_checked':54,'scope':'Retained records and printed summary reconciliation; no fresh inference or remote binary authentication.'},indent=2))
