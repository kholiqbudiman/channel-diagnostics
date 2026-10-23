"""Recompute post-outcome ACRE channel/retraining summaries, without inference."""
from pathlib import Path
import json,csv,hashlib
import numpy as np
H=Path(__file__).resolve().parent
if H==Path('/tmp'):H=Path('/tmp/acre_channel_retraining_20261017/source/research_artifacts/channel_retraining_20261017')
P=H.parent/'proxy_transfer_20261017';S=H.parent.parent
read=lambda p:list(csv.DictReader(p.open()))
load=lambda p:json.loads(p.read_text())
effects=read(P/'old_comparison_complete/old_comparison/all_effects.csv')+read(P/'completed_20261017/dtd_comparison/all_effects.csv');assert len(effects)==162
proj=read(H/'projection_contexts.csv');assert len(proj)==108
closures=read(H/'bidirectional_contexts.csv');assert len(closures)==108
local=read(H/'local_mask_response.csv');assert len(local)==135
names=['c_grow','d_shrink','joint_c_grow_d_shrink','c_shrink','d_grow','joint_c_shrink_d_grow'];contexts={(x['dataset'],int(x['split']),int(x['seed'])) for x in effects};assert len(contexts)==27
for r in proj:
 ds=r['dataset'];dd=6 if ds=='dtd' else 8;X=np.array([[np.log(c/40),np.log(d/64)] for c,d in [(42,64),(40,64-dd),(42,64-dd),(38,64),(40,64+dd),(38,64+dd)]])
 e={x['candidate']:x for x in effects if (x['dataset'],x['split'],x['seed'])==(r['dataset'],r['split'],r['seed'])}
 col='observed_f1_gain_pp' if r['target']=='f1' else 'observed_macro_ce_improvement';y=np.array([float(e[n][col]) for n in names]);q=np.array([float(e[n][r['method']]) for n in names])
 # Independent pseudoinverse implementation checks the saved least-squares solution.
 fitted=X@np.linalg.pinv(X)@y;minimum=float(np.sum((y-fitted)**2));actual=float(np.sum((y-q)**2));cal=float(np.sum((fitted-q)**2))
 assert abs(minimum-float(r['minimum_sse']))<1e-9
 assert abs(actual-float(r['actual_sse']))<1e-9
 assert abs(actual-minimum-cal)<1e-8
 assert minimum<=actual+1e-9
for r in closures:
 gp=float(r['growth_gain']);gm=float(r['shrink_gain']);assert (r['same_nonzero_sign']=='True')==(gp*gm>0)
 es={x['candidate']:x for x in effects if (x['dataset'],x['split'],x['seed'])==(r['dataset'],r['split'],r['seed'])}
 endpoint='observed_f1_gain_pp' if r['target']=='f1' else 'observed_macro_ce_improvement'
 plus='c_grow' if r['axis']=='c' else 'd_grow';minus='c_shrink' if r['axis']=='c' else 'd_shrink'
 assert abs(gp-float(es[plus][endpoint]))<1e-12 and abs(gm-float(es[minus][endpoint]))<1e-12
 center=40 if r['axis']=='c' else 64;step=2 if r['axis']=='c' else (6 if r['dataset']=='dtd' else 8)
 lp=np.log((center+step)/center);lm=np.log((center-step)/center)
 assert abs((lp*gm-lm*gp)/(lp-lm)-float(r['log_width_zero_intercept_closure']))<1e-12
 pr=next(x for x in proj if (x['dataset'],x['split'],x['seed'],x['target'])==(r['dataset'],r['split'],r['seed'],r['target']))
# Check every local-response row directly against the frozen diagnostic profiles.
profiles=load(P/'old_comparison_complete/old_diagnostics/all_profiles.json')
barrier=load(P/'completed_20261017/PREDICTION_BARRIER.json')['predictions']
for jid,sha in barrier.items():
 f=H/'dtd_diagnostic_profiles'/jid;cb=(f/'predictions_committed.json').read_bytes();assert hashlib.sha256(cb).hexdigest()==sha;c=json.loads(cb);db=(f/'diagnostic.json').read_bytes();assert hashlib.sha256(db).hexdigest()==c['diagnostic_sha256']
 parts=jid.split('_');sp=int(parts[1].replace('split',''));seed=int(parts[-1].lstrip('s'));profiles.append(dict(dataset='dtd',split=sp,seed=seed,profile=json.loads(db)))
assert len(profiles)==27
for r in local:
 z=next(x for x in profiles if (x['dataset'],str(x['split']),str(x['seed']))==(r['dataset'],r['split'],r['seed']));j=int(r['block']);p=z['profile'];m=[x for x in p['masks'] if x['stage']==f'block{j}'];a=np.mean([np.mean(x['class_loss_delta']) for x in m]);f=np.mean([x['removed_fraction'] for x in m]);q=f*p['taylor_signed_slope'][j-1]
 assert abs(a-float(r['observed_macro_ce_increase']))<1e-12 and abs(q-float(r['taylor_expected_macro_ce_increase']))<1e-12
summary=load(H/'analysis_summary.json');labels={'warpc':'WaRP-C','trashnet':'TrashNet','dtd':'DTD'}
expected=[]
for z in summary['local_taylor']:
 a=[r for r in local if r['dataset']==z['dataset']];assert len(a)==45
 assert abs(np.mean([float(r['absolute_error']) for r in a])-z['taylor_mae'])<1e-12
 b=[r for r in effects if r['dataset']==z['dataset']]
 assert abs(np.mean([abs(float(r['observed_macro_ce_improvement'])-float(r['taylor_signed'])) for r in b])-z['retrained_ce_mae'])<1e-12
 expected.append(labels[z['dataset']]+f" & {z['taylor_mae']:.3f} / {z['zero_mae']:.3f} & {z['retrained_ce_mae']:.3f} / {z['zero_retrained_ce_mae']:.3f} "+chr(92)*2)
assert '\n'.join(expected)+'\n'==(S/'tables/channel_local_rows.tex').read_text()
# Confirm original DTD primary analysis has not been redefined by this extension.
primary=load(P/'completed_20261017/dtd_comparison/summary.json')['primary'];assert abs(np.mean([x['ce_policy_minus_f1_policy_pp'] for x in primary])-.11053045323651823)<1e-10
assert hashlib.sha256((H/'ANALYSIS_PLAN.txt').read_bytes()).hexdigest()==load(H/'PLAN_RECEIPT.json')['plan_sha256']
print(json.dumps({'status':'PASS','training_runs_added':0,'inference_runs_added':0,'contexts':27,'retrained_effects':162,'local_block_responses':135,'projection_records':108,'bidirectional_records':108,'dtd_profile_commitments':9,'primary_dtd_contrast':'unchanged','scope':'Post-outcome observed-response analysis; no causal or population noise-free claim.'},indent=2))
