"""Post-outcome ACRE-only channel/width/retraining analysis; no training or inference.
All contexts retained. Oracles below are geometric diagnostics using known outcomes,
not predictive methods. Run with Python 3, NumPy, SciPy.
"""
from pathlib import Path
import json,csv,hashlib,math
import numpy as np
from scipy.stats import spearmanr
H=Path(__file__).resolve().parent
if H==Path('/tmp'):H=Path('/mnt/d/Coding/SERVER AI/KD/wasterecycle/acre_channel_retraining_analysis_20261017')
# Installed manuscript evidence has the same relative sibling structure.
P=H.parent/'acre_proxy_transfer_20261017'
if not P.exists():P=H.parent/'proxy_transfer_20261017'
load=lambda p:json.loads(p.read_text())
read=lambda p:list(csv.DictReader(p.open()))
effects=read(P/'old_comparison_complete/old_comparison/all_effects.csv')+read(P/'completed_20261017/dtd_comparison/all_effects.csv')
assert len(effects)==162
names=['c_grow','d_shrink','joint_c_grow_d_shrink','c_shrink','d_grow','joint_c_shrink_d_grow']
labels={'c_grow':'c growth','d_shrink':'d shrinkage','joint_c_grow_d_shrink':'c growth + d shrinkage','c_shrink':'c shrinkage','d_grow':'d growth','joint_c_shrink_d_grow':'c shrinkage + d growth'}
key=lambda r:(r['dataset'],int(r['split']),int(r['seed']))
contexts=sorted({key(r) for r in effects});assert len(contexts)==27
projection=[];closures=[];by_intervention=[];aggregated=[]
for k in contexts:
 a={r['candidate']:r for r in effects if key(r)==k};assert set(a)==set(names)
 ds,sp,seed=k;dd=6 if ds=='dtd' else 8
 widths={'c_grow':(42,64),'d_shrink':(40,64-dd),'joint_c_grow_d_shrink':(42,64-dd),'c_shrink':(38,64),'d_grow':(40,64+dd),'joint_c_shrink_d_grow':(38,64+dd)}
 X=np.array([[np.log(widths[n][0]/40),np.log(widths[n][1]/64)] for n in names]);assert np.linalg.matrix_rank(X)==2
 for target,predmethods in [('f1',['mask_f1','architecture_ridge']),('macro_ce',['mask_macro_ce','taylor_signed'])]:
  endpoint='observed_f1_gain_pp' if target=='f1' else 'observed_macro_ce_improvement'
  y=np.array([float(a[n][endpoint]) for n in names]);b=np.linalg.lstsq(X,y,rcond=None)[0];fitted=X@b;res=y-fitted
  assert np.linalg.norm(X.T@res)<1e-9
  ssefloor=float(res@res);ssezero=float(y@y)
  common=dict(dataset=ds,split=sp,seed=seed,target=target,oracle_c_slope=float(b[0]),oracle_d_slope=float(b[1]),minimum_sse=ssefloor,zero_sse=ssezero,minimum_rmse=float(np.sqrt(ssefloor/6)),zero_rmse=float(np.sqrt(ssezero/6)),scope='Post-outcome two-parameter zero-anchored log-width projection; not a prospective predictor or causal decomposition.')
  for m in predmethods:
   q=np.array([float(a[n][m]) for n in names]);qb=np.linalg.lstsq(X,q,rcond=None)[0]
   assert np.max(np.abs(q-X@qb))<1e-9
   err=y-q;cal=fitted-q;total=float(err@err);ssecal=float(cal@cal)
   assert abs(total-ssefloor-ssecal)<1e-8
   projection.append(dict(common,method=m,actual_sse=total,calibration_sse=ssecal,actual_rmse=float(np.sqrt(total/6)),minimum_fraction_of_actual_sse=ssefloor/total if total else None))
  for axis,plus,minus,j in [('c','c_grow','c_shrink',0),('d','d_grow','d_shrink',1)]:
   gp=float(a[plus][endpoint]);gm=float(a[minus][endpoint]);lp=X[names.index(plus),j];lm=X[names.index(minus),j]
   closure=(lp*gm-lm*gp)/(lp-lm)
   closures.append(dict(dataset=ds,split=sp,seed=seed,target=target,axis=axis,growth_gain=gp,shrink_gain=gm,both_positive=gp>0 and gm>0,both_negative=gp<0 and gm<0,same_nonzero_sign=gp*gm>0,log_width_zero_intercept_closure=float(closure)))
 for n in names:
  r=a[n];g=float(r['observed_f1_gain_pp']);q=float(r['mask_f1'])
  by_intervention.append(dict(dataset=ds,split=sp,seed=seed,candidate=n,observed_gain_pp=g,predicted_gain_pp=q,error_pp=g-q,abs_error_pp=abs(g-q),zero_abs_error_pp=abs(g),sign_agreement=(g*q>0 or g==q==0)))
# Descriptive seed/split averaging checks; no noise-free or population inference.
for ds in ['warpc','trashnet','dtd']:
 dd=6 if ds=='dtd' else 8
 X=np.array([[np.log(c/40),np.log(d/64)] for c,d in [(42,64),(40,64-dd),(42,64-dd),(38,64),(40,64+dd),(38,64+dd)]])
 for level,groups in [('task',[None]),('split',sorted({int(z['split']) for z in effects if z['dataset']==ds}))]:
  for sp in groups:
   a=[z for z in effects if z['dataset']==ds and (sp is None or int(z['split'])==sp)]
   for target,col in [('f1','observed_f1_gain_pp'),('macro_ce','observed_macro_ce_improvement')]:
    y=np.array([np.mean([float(z[col]) for z in a if z['candidate']==n]) for n in names]);b=np.linalg.lstsq(X,y,rcond=None)[0];res=y-X@b
    aggregated.append(dict(dataset=ds,level=level,split=sp,target=target,contexts_averaged=len(a)//6,minimum_rmse=float(np.sqrt(np.mean(res**2))),zero_rmse=float(np.sqrt(np.mean(y**2))),observed_mean_gains=y.tolist(),scope='Projection of averaged observed response; not population/causal inference.'))

# Local frozen-model response: Taylor class-balanced expected random removal vs observed random masks.
profiles=load(P/'old_comparison_complete/old_diagnostics/all_profiles.json')
base_records={r['job']['id']:r for r in load(P/'completed_20261017/verified_results_snapshot.json') if r['job']['model']['name']=='base'}
barrier=load(P/'completed_20261017/PREDICTION_BARRIER.json')['predictions']
for jid,b in base_records.items():
 f=H/'dtd_diagnostic_profiles'/jid;commitbytes=(f/'predictions_committed.json').read_bytes();c=json.loads(commitbytes)
 assert hashlib.sha256(commitbytes).hexdigest()==barrier[jid]
 db=(f/'diagnostic.json').read_bytes();assert hashlib.sha256(db).hexdigest()==c['diagnostic_sha256'];assert c['checkpoint_sha256']==b['checkpoint_sha256']
 profiles.append(dict(dataset='dtd',split=b['job']['split'],seed=b['job']['seed'],profile=json.loads(db)))
assert len(profiles)==27
local=[];localcontexts=[]
for z in profiles:
 pr=z['profile'];actual=[];pred=[];fractions=[]
 for j in range(5):
  masks=[m for m in pr['masks'] if m['stage']==f'block{j+1}'];assert len(masks)==3
  obs=float(np.mean([np.mean(m['class_loss_delta']) for m in masks]));frac=float(np.mean([m['removed_fraction'] for m in masks]));q=frac*pr['taylor_signed_slope'][j]
  actual.append(obs);pred.append(q);fractions.append(frac)
  local.append(dict(dataset=z['dataset'],split=z['split'],seed=z['seed'],block=j+1,mean_removed_fraction=frac,observed_macro_ce_increase=obs,taylor_expected_macro_ce_increase=q,absolute_error=abs(obs-q),zero_absolute_error=abs(obs)))
 actual=np.array(actual);pred=np.array(pred)
 localcontexts.append(dict(dataset=z['dataset'],split=z['split'],seed=z['seed'],taylor_mae=float(np.mean(abs(actual-pred))),zero_mae=float(np.mean(abs(actual))),taylor_rmse=float(np.sqrt(np.mean((actual-pred)**2))),spearman_five_blocks=float(spearmanr(actual,pred).statistic),target='Frozen-model class-balanced CE increase under three finite random masks; not retrained gain.'))
# Full descriptive summaries, no tests over overlapping contexts.
summary={'scope':'Post-outcome ACRE-only analysis. Existing prospective primary contrast unchanged. No new training/inference. All27 contexts retained.','projection':[],'averaged_response_projection':aggregated,'bidirectional':[],'intervention':[],'local_taylor':[]}
for ds in ['warpc','trashnet','dtd']:
 for target,methods in [('f1',['mask_f1','architecture_ridge']),('macro_ce',['mask_macro_ce','taylor_signed'])]:
  for m in methods:
   a=[r for r in projection if r['dataset']==ds and r['target']==target and r['method']==m];assert len(a)==9
   summary['projection'].append(dict(dataset=ds,target=target,method=m,contexts=9,mean_minimum_rmse=float(np.mean([r['minimum_rmse'] for r in a])),mean_actual_rmse=float(np.mean([r['actual_rmse'] for r in a])),mean_zero_rmse=float(np.mean([r['zero_rmse'] for r in a])),pooled_minimum_fraction_of_sse=sum(r['minimum_sse'] for r in a)/sum(r['actual_sse'] for r in a),mean_calibration_rmse=float(np.mean([np.sqrt(r['calibration_sse']/6) for r in a]))))
  for axis in ['c','d']:
   a=[r for r in closures if r['dataset']==ds and r['target']==target and r['axis']==axis]
   summary['bidirectional'].append(dict(dataset=ds,target=target,axis=axis,contexts=9,both_positive=sum(r['both_positive'] for r in a),both_negative=sum(r['both_negative'] for r in a),same_sign=sum(r['same_nonzero_sign'] for r in a),mean_closure=float(np.mean([r['log_width_zero_intercept_closure'] for r in a])),mean_absolute_closure=float(np.mean([abs(r['log_width_zero_intercept_closure']) for r in a]))))
 for n in names:
  a=[r for r in by_intervention if r['dataset']==ds and r['candidate']==n]
  summary['intervention'].append(dict(dataset=ds,candidate=n,contexts=9,mean_gain_pp=float(np.mean([r['observed_gain_pp'] for r in a])),proxy_mae=float(np.mean([r['abs_error_pp'] for r in a])),zero_mae=float(np.mean([r['zero_abs_error_pp'] for r in a])),positive=sum(r['observed_gain_pp']>0 for r in a),sign_agreement=sum(r['sign_agreement'] for r in a)))
 a=[r for r in localcontexts if r['dataset']==ds]
 b=[z for z in effects if z['dataset']==ds]
 summary['local_taylor'].append(dict(dataset=ds,contexts=9,blocks=45,retrained_ce_mae=float(np.mean([abs(float(z['observed_macro_ce_improvement'])-float(z['taylor_signed'])) for z in b])),zero_retrained_ce_mae=float(np.mean([abs(float(z['observed_macro_ce_improvement'])) for z in b])),taylor_mae=float(np.mean([r['taylor_mae'] for r in a])),zero_mae=float(np.mean([r['zero_mae'] for r in a])),mean_spearman=float(np.mean([r['spearman_five_blocks'] for r in a])),contexts_better_than_zero=sum(r['taylor_mae']<r['zero_mae'] for r in a)))
def writecsv(name,a):
 with (H/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(a[0]));w.writeheader();w.writerows(a)
for name,a in [('projection_contexts.csv',projection),('bidirectional_contexts.csv',closures),('intervention_effects.csv',by_intervention),('local_mask_response.csv',local),('local_taylor_contexts.csv',localcontexts)]:writecsv(name,a)
(H/'analysis_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
print(json.dumps(summary,indent=2,allow_nan=False))
