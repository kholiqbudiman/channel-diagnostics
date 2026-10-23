"""Locked full-cohort audit and finite width-intervention decomposition."""
from pathlib import Path
from datetime import datetime,timezone
import sys,os,json,hashlib,csv,statistics,time,fcntl,traceback,math
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,'/home/kholiq/KD/ACRE_method_development_20261010')
from analyze_complete_pilot import load_complete
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def save(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def f1(r):
 cm=np.array(r['probe_metrics']['confusion_matrix'],dtype=float);return 200*cm.diagonal()/(cm.sum(0)+cm.sum(1))
def proxy(base,widths):
 delta=np.log(np.array(widths,dtype=float)/[16,24,40,64])[[0,1,1,2,3]];pf=0.;pl=np.zeros(len(base['classes']))
 for s in range(5):
  masks=[m for m in base['channel_sensitivity'] if m['stage']==f'block{s+1}'];assert len(masks)==3
  pf+=delta[s]*statistics.mean(-m['macro_f1_delta_pp']/(-np.log1p(-m['removed_fraction'])) for m in masks)
  pl+=delta[s]*np.mean([np.array(m['class_loss_delta'])/(-np.log1p(-m['removed_fraction'])) for m in masks],axis=0)
 return float(pf),pl

def main():
 guard=(ROOT/'analysis.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 jobs=read(ROOT/'job_inventory.json');lock=read(ROOT/'COHORT_LOCK.json');decisions=read(ROOT/'EXPERIMENT_DECISIONS.json')
 for n,h in lock['sources'].items():assert sha(ROOT/n)==h,n
 for n,h in lock['analysis_dependencies'].items():assert sha(Path(n))==h,n
 while True:
  s=read(ROOT/'status.json');n=sum((Path(j['folder'])/'runs'/j['inner_id']/'result.json').exists() for j in jobs)
  save(ROOT/'analysis_status.json',dict(state='WAITING_FOR_COMPLETE_COHORT',completed=n,total=108))
  if s['state'] in ['FAILED','COMPLETE_WITH_FAILURES']:raise RuntimeError('Preserve failed cohort; no partial winner analysis')
  if n==108 and s['state']=='COMPLETE':break
  time.sleep(30)
 results={};audits={}
 for split in [4302,4303,4304]:
  folder=ROOT/f'split_{split}';assert sha(folder/'protocol_lock.json')==lock['split_locks'][str(split)]
  _,audits[str(split)]=load_complete(folder,folder/'runs');assert audits[str(split)]['completed']==42
  design=audits[str(split)]['design_identifiability']
  if not math.isfinite(design['condition_number']):
   assert design['four_width_transfer_rank']<4
   design['condition_number']=None;design['condition_number_note']='Not defined for full four-width design: a,b are held fixed intentionally; factorial c,d contrasts are evaluated directly, no four-stage regression.'
  for j in read(folder/'protocol.json')['jobs']:
   r=read(folder/'runs'/j['id']/'result.json');v=[h['selection_macro_f1_pct'] for h in r['history']];best=int(np.argmax(v))+1
   assert best==r['best_epoch'] and v[best-1]==r['selection_macro_f1_pct'] and len(v)==min(best+20,100)
   results[(split,j['dataset'],j['seed'],j['model']['name'])]=r
 for ref in read(ROOT/'REFERENCE_MANIFEST.json'):
  folder=ROOT/f"split_{ref['split']}"/'runs'/ref['job']
  for n,h in ref['files'].items():assert sha(folder/n)==h
 effects=[];classrows=[];lossrows=[];cellrows=[]
 for split in [4302,4303,4304]:
  for ds in ['warpc','trashnet']:
   for seed in [2901,2902,2903]:
    base=results[(split,ds,seed,'base')];bf=f1(base);bl=np.array(base['probe_metrics']['class_loss'])
    for c in decisions['configs']:
     r=results[(split,ds,seed,c['name'])];delta=f1(r)-bf
     cellrows.append(dict(split=split,dataset=ds,seed=seed,model=c['name'],role=c['role'],macro_f1_pct=float(f1(r).mean()),delta_vs_base_pp=float(delta.mean()),params=r['cost']['params'],macs=r['cost']['macs']))
    for direction in decisions['directions']:
     g=results[(split,ds,seed,direction['component_g'])];d=results[(split,ds,seed,direction['component_d'])];j=results[(split,ds,seed,direction['joint'])]
     gains=[f1(r)-bf for r in [g,d,j]];interaction=gains[2]-gains[0]-gains[1]
     pg,lg=proxy(base,g['job']['model']['channels']);pd,ld=proxy(base,d['job']['model']['channels']);pj,lj=proxy(base,j['job']['model']['channels'])
     assert abs(pj-pg-pd)<1e-9 and np.allclose(lj,lg+ld)
     eg=float(gains[0].mean())-pg;ed=float(gains[1].mean())-pd;ej=float(gains[2].mean())-pj;I=float(interaction.mean());assert abs(ej-I-eg-ed)<1e-8
     effects.append(dict(split=split,dataset=ds,seed=seed,direction=direction['name'],gain_component_g_pp=float(gains[0].mean()),gain_component_d_pp=float(gains[1].mean()),gain_joint_pp=float(gains[2].mean()),interaction_pp=I,predicted_joint_mask_gain_pp=pj,component_g_proxy_error_pp=eg,component_d_proxy_error_pp=ed,joint_proxy_error_pp=ej,identity_residual_pp=ej-I-eg-ed))
     loss_gains=[bl-np.array(r['probe_metrics']['class_loss']) for r in [g,d,j]];lossI=loss_gains[2]-loss_gains[0]-loss_gains[1]
     for k,c in enumerate(base['classes']):
      classrows.append(dict(split=split,dataset=ds,seed=seed,direction=direction['name'],class_name=c,component_g_f1_change_pp=float(gains[0][k]),component_d_f1_change_pp=float(gains[1][k]),joint_f1_change_pp=float(gains[2][k]),interaction_f1_pp=float(interaction[k])))
      ge=loss_gains[0][k]-lg[k];de=loss_gains[1][k]-ld[k];je=loss_gains[2][k]-lj[k];assert abs(je-lossI[k]-ge-de)<1e-8
      lossrows.append(dict(split=split,dataset=ds,seed=seed,direction=direction['name'],class_name=c,gain_joint_loss=float(loss_gains[2][k]),predicted_joint_loss_gain=float(lj[k]),interaction_loss=float(lossI[k]),component_g_error=float(ge),component_d_error=float(de),joint_error=float(je)))
 for name,rows in [('all_cell_metrics.csv',cellrows),('macro_interaction_and_proxy_error.csv',effects),('classwise_f1_interaction.csv',classrows),('classwise_loss_error_identity.csv',lossrows)]:
  with (ROOT/name).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 summary=[]
 for ds in ['warpc','trashnet']:
  for direction in decisions['directions']:
   sub=[r for r in effects if r['dataset']==ds and r['direction']==direction['name']]
   keys=['gain_component_g_pp','gain_component_d_pp','gain_joint_pp','interaction_pp','joint_proxy_error_pp']
   summary.append(dict(dataset=ds,direction=direction['name'],n_contexts=9,means={k:statistics.mean(r[k] for r in sub) for k in keys},mean_abs_interaction_pp=statistics.mean(abs(r['interaction_pp']) for r in sub),positive_interaction_contexts=sum(r['interaction_pp']>0 for r in sub),by_split={str(s):{k:statistics.mean(r[k] for r in sub if r['split']==s) for k in keys} for s in [4302,4303,4304]}))
 save(ROOT/'complete_mechanism_analysis.json',dict(audits=audits,summary=summary,scope='Development intervention analysis with reused Base references and overlapping split pools. Interaction is a finite difference of width-boundary interventions, not a universal stage mechanism. Error decomposition is an arithmetic identity, not a new estimator. No iid significance, automatic novelty, or untouched task claim.'))
 save(ROOT/'analysis_status.json',dict(state='COMPLETE',completed=108,total=108,reused_base=18))
if __name__=='__main__':
 try:main()
 except Exception:
  (ROOT/'analysis_failure.log').write_text(traceback.format_exc());save(ROOT/'analysis_status.json',dict(state='FAILED'));raise
