"""Post-cohort CPU development diagnostics; no scientific training or source edits."""
from pathlib import Path
import os,sys,json,hashlib,csv,statistics,copy
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
ARCHIVE=WORK/'acre_novelty_20261010/review_complete_pilot_20261016'
sys.path.insert(0,str(ARCHIVE))
from confusion_response import ConfusionResponse,macro_f1,pair_features,log_ratios,block_widths
from analyze_complete_pilot import reference_from_result,validate_surrogate
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def save(name,data):
 path=ROOT/name;assert not path.exists();path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def csvsave(name,data):
 with (ROOT/name).open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def rho(x,y):
 return None if np.std(x)<1e-12 or np.std(y)<1e-12 else float(spearmanr(x,y).statistic)
def smooth_cm(cm,alpha=.5):
 return cm.sum(1)[:,None]*(cm+alpha)/(cm.sum(1)[:,None]+len(cm)*alpha)
def simplex(v):
 u=np.sort(v)[::-1];css=np.cumsum(u)-1;ix=np.flatnonzero(u-css/(np.arange(len(v))+1)>0);theta=css[ix[-1]]/(ix[-1]+1)
 return np.maximum(v-theta,0)
def apply_response(ref,response,variant):
 cm=ref.cm;k=len(cm);off=np.zeros((k,k));t=0
 for i in range(k):
  for j in range(k):
   if i!=j:off[i,j]=response[t];t+=1
 if variant=='multiplicative_exact':
  # Zeros remain zero; this limitation is measured and reported, not hidden.
  probs=cm/cm.sum(1)[:,None];logp=np.full_like(probs,-np.inf);np.log(probs,out=logp,where=probs>0);logp+=off;logp-=logp.max(1,keepdims=True)
  p=np.exp(logp);p/=p.sum(1,keepdims=True)
 else:
  smooth=(cm+.5)/(cm.sum(1)[:,None]+k*.5);logp=np.log(smooth)+off;logp-=logp.max(1,keepdims=True);p=np.exp(logp);p/=p.sum(1,keepdims=True)
  if variant=='additive_projected':p=np.stack([simplex(row) for row in cm/cm.sum(1)[:,None]+p-smooth])
 return cm.sum(1)[:,None]*p
class Ablation(ConfusionResponse):
 def __init__(self,mode,variant):super().__init__(mode=mode,ridge=1.,alpha=.5);self.variant=variant
 def predict_cm(self,ref,channels):
  if self.variant=='original':return super().predict_cm(ref,channels)
  response=(pair_features(ref,channels,self.mode)/self.scale)@self.coef
  return apply_response(ref,response,self.variant)
lock=read(ROOT/'INPUT_LOCK.json');assert sha(ROOT/'DIAGNOSTIC_PLAN.txt')==lock['plan_sha256']
for path,h in lock['inputs'].items():assert sha(Path(path))==h
pilot=read(ARCHIVE/'verified_pilot_snapshot.json')['results'];stability=read(WORK/'acre_stability_20261016/audit_20261016/verified_results_snapshot.json')
def records_from(results,cohort):
 records=[]
 for r in results:
  j=r['job'];model=j['model'];ds=j['dataset'];seed=j['seed'];split=r.get('split',4301)
  if not(model['kind']=='allocation' and model['residual'] and model['name']!='base'):continue
  base=next(b for b in results if b.get('split',4301)==split and b['job']['dataset']==ds and b['job']['seed']==seed and b['job']['model']['name']=='base')
  records.append(dict(id=f'{cohort}_{split}_{j["id"]}',dataset=ds,seed=seed,split=split,channels=model['channels'],model=model['name'],reference=reference_from_result(base),target_cm=np.array(r['probe_metrics']['confusion_matrix']),cost=r['cost'],base_result=base,target_result=r))
 return records
calibration=records_from(pilot,'pilot');target=records_from(stability,'stability');assert len(calibration)==20 and len(target)==36
# Reproduce the originally reported grouped-holdout evaluation before adding ablations.
validation=validate_surrogate(calibration);reproduced=[]
for axis in ['allocation','dataset']:
 for mode in ['architecture_only','frequency_only','class_agnostic','class_pair']:
  preds=[p for v in validation if v['heldout_axis']==axis and v['mode']==mode for p in v['predictions']]
  reproduced.append(dict(axis=axis,mode=mode,n=len(preds),mae_pp=statistics.mean(abs(p['predicted_macro_f1_pct']-p['observed_macro_f1_pct']) for p in preds)))
expected={('allocation','class_pair'):4.137,('allocation','architecture_only'):4.094,('dataset','class_pair'):5.480,('dataset','architecture_only'):3.855}
for r in reproduced:
 if (r['axis'],r['mode']) in expected:assert abs(r['mae_pp']-expected[(r['axis'],r['mode'])])<.0006
save('original_pilot_reproduction.json',reproduced)
# Measure smoothing under exactly zero response, for every Base context.
anchors=[]
for cohort,allruns in [('pilot',pilot),('stability',stability)]:
 for r in allruns:
  if r['job']['model']['name']!='base':continue
  cm=np.array(r['probe_metrics']['confusion_matrix'],dtype=float);sm=smooth_cm(cm);delta=(macro_f1(sm)-macro_f1(cm))*100
  anchors.append(dict(cohort=cohort,dataset=r['job']['dataset'],split=r.get('split',4301),seed=r['job']['seed'],raw_f1_pct=macro_f1(cm)*100,zero_response_candidate_f1_pct=macro_f1(sm)*100,zero_response_bias_pp=delta,zero_diagonal_classes=[c for c,tp in zip(r['classes'],cm.diagonal()) if tp==0]))
save('zero_response_anchoring.json',anchors)
variants=['original','consistent_smoothed','multiplicative_exact','additive_projected'];modes=['architecture_only','frequency_only','class_agnostic','class_pair'];predrows=[];selection=[];invariants=[];models={}
for mode in modes:
 for variant in variants:
  model=Ablation(mode,variant).fit(calibration);models[(mode,variant)]=model
  check=[]
  for r in target:
   ref=r['reference'];pred=model.predict_cm(ref,r['channels']);baseline=model.predict_cm(ref,ref.channels)
   assert np.isfinite(pred).all() and (pred>=-1e-10).all() and np.allclose(pred.sum(1),ref.cm.sum(1),atol=1e-9)
   rawgain=(macro_f1(r['target_cm'])-macro_f1(ref.cm))*100;predgain=(macro_f1(pred)-macro_f1(baseline))*100
   predrows.append(dict(dataset=r['dataset'],split=r['split'],seed=r['seed'],candidate=r['model'],mode=mode,anchoring=variant,observed_gain_pp=rawgain,predicted_gain_pp=predgain,error_pp=predgain-rawgain))
   zero=apply_response(ref,np.zeros(len(ref.cm)*(len(ref.cm)-1)),variant)
   if variant in ['multiplicative_exact','additive_projected']:assert np.allclose(zero,ref.cm,atol=1e-9)
   check.append(float(np.abs(zero-ref.cm).max()))
  # Class-permutation equivariance on a real reference and candidate.
  rec=target[0];ref=rec['reference'];perm=np.arange(len(ref.cm))[::-1];rr=copy.deepcopy(ref);rr.cm=ref.cm[np.ix_(perm,perm)];rr.loss=ref.loss[perm];rr.sensitivity=ref.sensitivity[:,perm];rr.separation=ref.separation[:,perm][:,:,perm]
  assert np.allclose(model.predict_cm(rr,rec['channels']),model.predict_cm(ref,rec['channels'])[np.ix_(perm,perm)],atol=1e-9)
  invariants.append(dict(mode=mode,variant=variant,finite_support_nonnegative_checks=36,permutation='PASS',max_zero_response_cell_shift=max(check)))
  for ds,s,seed in sorted({(r['dataset'],r['split'],r['seed']) for r in target}):
   sub=[p for p in predrows if p['mode']==mode and p['anchoring']==variant and (p['dataset'],p['split'],p['seed'])==(ds,s,seed)]
   choices=[dict(candidate='base',predicted_gain_pp=0.,observed_gain_pp=0.)]+sub
   # Fixed common tie rule: no-change Base before A/B for exactly equal scores.
   choice=max(choices,key=lambda x:x['predicted_gain_pp']);oracle=max(x['observed_gain_pp'] for x in choices)
   selection.append(dict(dataset=ds,split=s,seed=seed,mode=mode,anchoring=variant,selected=choice['candidate'],gain_pp=choice['observed_gain_pp'],oracle_regret_pp=oracle-choice['observed_gain_pp'],base_regret_pp=oracle,uniform_random_expected_gain_pp=statistics.mean(x['observed_gain_pp'] for x in choices)))
csvsave('all_prediction_rows.csv',predrows);csvsave('all_selection_rows.csv',selection);save('mathematical_invariants.json',invariants)
summary=[]
for mode in modes:
 for variant in variants:
  for ds in ['all','warpc','trashnet']:
   sub=[r for r in predrows if r['mode']==mode and r['anchoring']==variant and (ds=='all' or r['dataset']==ds)];sel=[r for r in selection if r['mode']==mode and r['anchoring']==variant and (ds=='all' or r['dataset']==ds)]
   summary.append(dict(dataset=ds,mode=mode,anchoring=variant,n_effects=len(sub),n_decisions=len(sel),gain_mae_pp=statistics.mean(abs(r['error_pp']) for r in sub),zero_gain_mae_pp=statistics.mean(abs(r['observed_gain_pp']) for r in sub),spearman=rho([r['predicted_gain_pp'] for r in sub],[r['observed_gain_pp'] for r in sub]),mean_selection_gain_pp=statistics.mean(r['gain_pp'] for r in sel),mean_regret_pp=statistics.mean(r['oracle_regret_pp'] for r in sel),base_mean_regret_pp=statistics.mean(r['base_regret_pp'] for r in sel),uniform_random_expected_gain_pp=statistics.mean(r['uniform_random_expected_gain_pp'] for r in sel),nonbase_choices=sum(r['selected']!='base' for r in sel)))
save('predictor_diagnostic_summary.json',summary)
# Direct, fixed first-order mask-to-retraining proxies; no parameter fitting.
proxyrows=[];lossrows=[]
for r in target:
 b=r['base_result'];t=r['target_result'];delta=np.log(block_widths(r['channels'])/block_widths(r['reference'].channels));predloss=np.zeros(len(r['reference'].cm));predf1=0
 for s in range(5):
  masks=[m for m in b['channel_sensitivity'] if m['stage']==f'block{s+1}'];stage_loss=[];stage_f1=[]
  for m in masks:
   norm=-np.log1p(-m['removed_fraction']);stage_loss.append(np.array(m['class_loss_delta'])/norm);stage_f1.append(-m['macro_f1_delta_pp']/norm)
  predloss+=delta[s]*np.mean(stage_loss,axis=0);predf1+=delta[s]*statistics.mean(stage_f1)
 actualf1=(macro_f1(r['target_cm'])-macro_f1(r['reference'].cm))*100
 proxyrows.append(dict(dataset=r['dataset'],split=r['split'],seed=r['seed'],candidate=r['model'],predicted_f1_gain_pp=float(predf1),observed_f1_gain_pp=actualf1))
 actual_loss=np.array(b['probe_metrics']['class_loss'])-t['probe_metrics']['class_loss']
 for k,c in enumerate(b['classes']):lossrows.append(dict(dataset=r['dataset'],split=r['split'],seed=r['seed'],candidate=r['model'],class_name=c,predicted_loss_improvement=float(predloss[k]),observed_loss_improvement=float(actual_loss[k])))
csvsave('direct_mask_f1_proxy.csv',proxyrows);csvsave('direct_mask_class_loss_proxy.csv',lossrows)
proxy_summary=[]
for ds in ['all','warpc','trashnet']:
 for kind,data,xkey,ykey in [('macro_f1',proxyrows,'predicted_f1_gain_pp','observed_f1_gain_pp'),('class_loss',lossrows,'predicted_loss_improvement','observed_loss_improvement')]:
  sub=[r for r in data if ds=='all' or r['dataset']==ds];x=np.array([r[xkey] for r in sub]);y=np.array([r[ykey] for r in sub]);proxy_summary.append(dict(dataset=ds,endpoint=kind,rows=len(sub),mae=float(np.mean(abs(x-y))),nochange_mae=float(np.mean(abs(y))),spearman=rho(x,y),sign_agreement=float(np.mean(np.sign(x)==np.sign(y)))))
save('direct_proxy_summary.json',proxy_summary)
# Selection using direct F1 proxy, for all18 Base/A/B contexts.
proxysel=[]
for ds,s,seed in sorted({(r['dataset'],r['split'],r['seed']) for r in target}):
 sub=[r for r in proxyrows if (r['dataset'],r['split'],r['seed'])==(ds,s,seed)];choices=[dict(candidate='base',predicted_f1_gain_pp=0.,observed_f1_gain_pp=0.)]+sub;best=max(choices,key=lambda x:x['predicted_f1_gain_pp']);oracle=max(r['observed_f1_gain_pp'] for r in choices)
 proxysel.append(dict(dataset=ds,split=s,seed=seed,selected=best['candidate'],gain_pp=best['observed_f1_gain_pp'],regret_pp=oracle-best['observed_f1_gain_pp']))
save('direct_proxy_selection.json',proxysel)
widths=np.array([np.log(block_widths(r['channels'])/block_widths(r['reference'].channels)) for r in calibration]);feature_ranks=[]
for mode in modes:
 x=np.vstack([pair_features(r['reference'],r['channels'],mode) for r in calibration]);feature_ranks.append(dict(mode=mode,pair_rows=x.shape[0],features=x.shape[1],rank=int(np.linalg.matrix_rank(x)),zero_columns=int(np.sum(np.all(x==0,axis=0)))))
design=dict(calibration_runs=20,calibration_datasets=2,unique_width_transfers=len({tuple(r['channels']) for r in calibration}),five_stage_transfer_rank=int(np.linalg.matrix_rank(widths)),block2_block3_always_equal=bool(np.all(widths[:,1]==widths[:,2])),missing_grow_stages=[i+1 for i in range(5) if not np.any(widths[:,i]>0)],missing_shrink_stages=[i+1 for i in range(5) if not np.any(widths[:,i]<0)],feature_ranks=feature_ranks,stability_effects=36,stability_decision_contexts=18,no_residual_excluded=18)
save('design_identifiability.json',design)
for path,h in lock['inputs'].items():assert sha(Path(path))==h
save('diagnostic_receipt.json',dict(state='COMPLETE',source_sha256=sha(Path(__file__)),plan_sha256=sha(ROOT/'DIAGNOSTIC_PLAN.txt'),unchanged_inputs=True,no_gpu_training=True,target_outcomes_not_used_to_fit=True,limits=['Post-cohort development analysis; not independent confirmation.','Base probe-derived diagnostics and Base confusion are available at prediction time here; prospective deployment would need an appropriate separate diagnostic split.','Stability uses new splits/seeds on already seen dataset pools; target tasks are not unseen datasets.','Anchoring variants are diagnostic interventions, not validated new allocation algorithms.','Noiid p-values; class pairs and overlapping contexts are dependent.']))
print(json.dumps(dict(reproduced=reproduced,anchoring_stability=[r for r in anchors if r['cohort']=='stability'],all_dataset_summaries=[r for r in summary if r['dataset']=='all'],proxy=proxy_summary,design=design),indent=2))
