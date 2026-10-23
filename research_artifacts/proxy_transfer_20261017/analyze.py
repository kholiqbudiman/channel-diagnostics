"""All-method comparisons and complete-cohort DTD evaluation; no partial winner selection."""
from pathlib import Path
import os,json,csv,hashlib,time,argparse,traceback,statistics,warnings
import numpy as np,torch
from scipy.stats import spearmanr
import reference as e
from models import build_model
from scores import METHODS
from train import save,verify
ROOT=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def csvsave(path,rows):
 with path.open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def context(r):return(r['job']['dataset'],r.get('review_split',r['job'].get('split')),r['job']['seed'])
def ce(r):return float(np.mean(r['probe_metrics']['class_loss']))
def rho(x,y):
 if np.ptp(x)<1e-12 or np.ptp(y)<1e-12:return None
 return float(spearmanr(x,y).statistic)
def compare(records,profiles,out):
 out.mkdir(exist_ok=False);idx={(context(r),r['job']['model']['name']):r for r in records};effects=[];decisions=[];metrics=[];interactions=[]
 for p in profiles:
  key=(p['dataset'],p['split'],p['seed']);b=idx[key,'base'];bf=b['probe_metrics']['macro_f1_pct'];bc=ce(b);pred=p['predictions'];allowed=['base','joint_c_grow_d_shrink','joint_c_shrink_d_grow']
  gain={n:idx[key,n]['probe_metrics']['macro_f1_pct']-bf for n in allowed};oracle=max(gain.values());randomgain=statistics.mean(gain.values())
  for method in METHODS:
   selected=p['choices'][method];assert selected in allowed
   decisions.append(dict(dataset=key[0],split=key[1],seed=key[2],method=method,selected=selected,gain_pp=gain[selected],regret_pp=oracle-gain[selected],base_regret_pp=oracle,uniform_random_expected_gain_pp=randomgain,macro_ce_improvement=bc-ce(idx[key,selected])))
   ns=[n for n in pred if n!='base'];xs=[pred[n][method] for n in ns];ys=[idx[key,n]['probe_metrics']['macro_f1_pct']-bf for n in ns]
   z=dict(dataset=key[0],split=key[1],seed=key[2],method=method,spearman_f1=rho(xs,ys),effects=len(ns))
   if method in ['mask_f1','architecture_ridge','choose_base']:z['f1_gain_mae_pp']=float(np.mean(np.abs(np.array(xs)-ys)))
   if method in ['mask_macro_ce','taylor_signed']:z['macro_ce_gain_mae']=float(np.mean([abs(pred[n][method]-(bc-ce(idx[key,n]))) for n in ns]))
   metrics.append(z)
  for n,ss in pred.items():
   if n=='base':continue
   r=idx[key,n];row=dict(dataset=key[0],split=key[1],seed=key[2],candidate=n,observed_f1_gain_pp=r['probe_metrics']['macro_f1_pct']-bf,observed_macro_ce_improvement=bc-ce(r));row.update(ss);effects.append(row)
  for direction,c,d,j in [('c_up_d_down','c_grow','d_shrink','joint_c_grow_d_shrink'),('c_down_d_up','c_shrink','d_grow','joint_c_shrink_d_grow')]:
   rr=[idx[key,n] for n in ['base',c,d,j]];f=[v['probe_metrics']['macro_f1_pct'] for v in rr];a=[v['probe_metrics']['accuracy_pct'] for v in rr];loss=[-ce(v) for v in rr]
   I=f[3]-f[1]-f[2]+f[0];row=dict(dataset=key[0],split=key[1],seed=key[2],direction=direction,interaction_f1_pp=I,interaction_accuracy_pp=a[3]-a[1]-a[2]+a[0],interaction_negative_macro_ce=loss[3]-loss[1]-loss[2]+loss[0])
   for method in ['mask_f1','architecture_ridge']:
    ec=f[1]-f[0]-pred[c][method];ed=f[2]-f[0]-pred[d][method];ej=f[3]-f[0]-pred[j][method];assert abs(ej-I-ec-ed)<1e-8
    row[method+'_component_c_error']=ec;row[method+'_component_d_error']=ed;row[method+'_joint_error']=ej
   interactions.append(row)
 summary=[]
 for ds in sorted(set(p['dataset'] for p in profiles)):
  for method in METHODS:
   rr=[r for r in decisions if r['dataset']==ds and r['method']==method];mm=[r for r in metrics if r['dataset']==ds and r['method']==method];valid=[m['spearman_f1'] for m in mm if m['spearman_f1'] is not None]
   z=dict(dataset=ds,method=method,contexts=len(rr),mean_selected_gain_pp=statistics.mean(r['gain_pp'] for r in rr),descriptive_sd_gain_pp=statistics.stdev(r['gain_pp'] for r in rr),mean_regret_pp=statistics.mean(r['regret_pp'] for r in rr),mean_uniform_random_gain_pp=statistics.mean(r['uniform_random_expected_gain_pp'] for r in rr),mean_macro_ce_improvement=statistics.mean(r['macro_ce_improvement'] for r in rr),mean_spearman_f1=None if not valid else statistics.mean(valid),valid_rank_contexts=len(valid),nonbase_choices=sum(r['selected']!='base' for r in rr))
   if method in ['mask_f1','architecture_ridge','choose_base']:z['mean_f1_gain_mae_pp']=statistics.mean(m['f1_gain_mae_pp'] for m in mm)
   summary.append(z)
 primary=[]
 for p in profiles:
  key=(p['dataset'],p['split'],p['seed']);a=next(x for x in decisions if(x['dataset'],x['split'],x['seed'])==key and x['method']=='mask_macro_ce');b=next(x for x in decisions if(x['dataset'],x['split'],x['seed'])==key and x['method']=='mask_f1');primary.append(dict(dataset=key[0],split=key[1],seed=key[2],ce_policy_minus_f1_policy_pp=a['gain_pp']-b['gain_pp'],ce_policy_minus_base_pp=a['gain_pp']))
 for n,rows in [('all_effects.csv',effects),('all_decisions.csv',decisions),('all_context_metrics.csv',metrics),('all_interactions.csv',interactions),('primary_policy_contrast.csv',primary)]:csvsave(out/n,rows)
 save(out/'summary.json',dict(methods=summary,primary=primary,scope='All prespecified methods retained. Scores of ordinal diagnostics are not F1 units. Contexts overlap in image pools; descriptive comparisons, not iid p-values.'))
 return summary

def old_analysis():
 profiles=read(ROOT/'old_diagnostics/all_profiles.json');assert len(profiles)==18
 summary=compare(read(ROOT/'old_results.json'),profiles,ROOT/'old_comparison')
 txt=['PERBANDINGAN DIAGNOSTIK LAMA — RETROSPEKTIF','Hasil ini memakai probe WaRP-C/TrashNet yang telah terekspos, bukan konfirmasi baru.','Gain adalah hasil pilihan di antaraBase+2joint; semua metode dan hasil dipertahankan.','']
 for r in summary:txt.append(f"{r['dataset']} | {r['method']} | mean gain {r['mean_selected_gain_pp']:+.3f} pp | regret {r['mean_regret_pp']:.3f} pp | nonBase {r['nonbase_choices']}/{r['contexts']}")
 (ROOT/'HASIL_PERBANDINGAN_LAMA.txt').write_text('\n'.join(txt)+'\n')

def evaluate_complete():
 verify();status=read(ROOT/'status.json');assert status['state']=='COMPLETE' and status['completed']==63 and not status['failures']
 jobs=read(ROOT/'jobs.json');barrier=read(ROOT/'PREDICTION_BARRIER.json');assert len(barrier['predictions'])==9
 profiles=[]
 for jid,h in barrier['predictions'].items():
  f=ROOT/'runs'/jid;assert sha(f/'predictions_committed.json')==h;a=read(f/'predictions_committed.json');assert sha(f/'best.pt')==a['checkpoint_sha256'] and sha(f/'diagnostic.json')==a['diagnostic_sha256'];j=read(f/'training_complete.json')['job'];profiles.append(dict(dataset='dtd',split=j['split'],seed=j['seed'],predictions=a['scores'],choices=a['choices']))
 for j in jobs:
  f=ROOT/'runs'/j['id'];r=read(f/'training_complete.json');assert sha(f/'best.pt')==r['checkpoint_sha256'];assert sha(f/'initial_state.pt')==r['initialization_file_sha256'];assert sha(ROOT/f"data_{j['split']}.json")==r['manifest_sha256'];v=[x['selection_macro_f1_pct'] for x in r['history']];assert np.argmax(v)+1==r['best_epoch'] and len(v)==min(r['best_epoch']+20,100)
 m=read(ROOT/'data_6301.json')
 for rel,h in m['image_sha256'].items():assert sha(Path(m['root'])/rel)==h
 save(ROOT/'EVALUATION_RELEASE.json',dict(created_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),state='ALL63_TRAININGS_AND9_COMMITTED_PREDICTIONS_VERIFIED',barrier_sha256=sha(ROOT/'PREDICTION_BARRIER.json')))
 torch.set_num_threads(2);records=[];logits={};audit=[]
 for j in jobs:
  f=ROOT/'runs'/j['id'];r=read(f/'training_complete.json');m=read(ROOT/f"data_{j['split']}.json");k=len(m['classes']);model=build_model(j['model'],k);model.load_state_dict(torch.load(f/'best.pt',map_location='cpu',weights_only=True)['model'])
  ds=e.Images(Path(m['root']),[tuple(x) for x in m['samples']],m['split']['evaluation']);loader=e.make_loader(ds,False,j['split'],workers=0);l,y,_=e.evaluate(model,loader,torch.device('cpu'));metrics=e.metrics(l,y,k)
  np.savez_compressed(f/'evaluation_predictions.npz',labels=y.numpy(),logits=l.numpy());record=dict(r,classes=m['classes'],probe_metrics=metrics,review_split=j['split'],scope='Prospective DTD evaluation after whole-cohort prediction barrier; no official test access.');save(f/'result.json',record);records.append(record);logits[j['split'],j['seed'],j['model']['name']]=(l.numpy(),y.numpy());audit.append(dict(job=j['id'],evaluation_sha256=sha(f/'evaluation_predictions.npz'),result_sha256=sha(f/'result.json')));save(ROOT/'analysis_status.json',dict(state='EVALUATING_COMPLETE_COHORT',completed=len(records),total=63))
 save(ROOT/'verified_results_snapshot.json',records);summary=compare(records,profiles,ROOT/'dtd_comparison');boot=[]
 for p in profiles:
  split=p['split'];seed=p['seed'];names=[c['name'] for c in read(ROOT/'PROTOCOL.json')['configs']];yy=logits[split,seed,'base'][1];k=47;rng=np.random.default_rng(20261017+split+seed);inds=np.concatenate([rng.choice(np.flatnonzero(yy==c),(1000,int((yy==c).sum())),replace=True) for c in range(k)],axis=1);yb=yy[inds];values={}
  for n in names:
   l,y=logits[split,seed,n];assert np.array_equal(yy,y);pb=l.argmax(1)[inds];codes=yb*k+pb+np.arange(1000)[:,None]*k*k;cm=np.bincount(codes.ravel(),minlength=1000*k*k).reshape(1000,k,k);den=cm.sum(1)+cm.sum(2);values[n]=np.divide(200*np.diagonal(cm,axis1=1,axis2=2),den,out=np.zeros_like(den,dtype=float),where=den>0).mean(1)
  f1n=p['choices']['mask_f1'];cen=p['choices']['mask_macro_ce'];z=dict(split=split,seed=seed,ce_minus_f1_policy_interval_pp=np.quantile(values[cen]-values[f1n],[.025,.975]).tolist(),ce_minus_base_interval_pp=np.quantile(values[cen]-values['base'],[.025,.975]).tolist(),interaction_intervals={})
  for d,c,a,j in [('c_up_d_down','c_grow','d_shrink','joint_c_grow_d_shrink'),('c_down_d_up','c_shrink','d_grow','joint_c_shrink_d_grow')]:z['interaction_intervals'][d]=np.quantile(values[j]-values[c]-values[a]+values['base'],[.025,.975]).tolist()
  boot.append(z)
 save(ROOT/'conditional_bootstrap.json',dict(replicates=1000,rows=boot,scope='Image uncertainty conditional on fixed diagnostic choices, models and evaluation sets. Not diagnostic estimation/training/split/task uncertainty.'))
 save(ROOT/'evaluation_audit.json',dict(records=63,checkpoint_hashes_verified=63,initialization_hashes_verified=63,prediction_commits_verified=9,files=audit))
 txt=['HASIL TRANSFER DTD — SELURUH63 RUN SELESAI','Primary:gain kebijakanCE-mask minus kebijakanF1-mask, juga terhadapBase.','Konteks baru adalahDTD47; pooltrain1+val1 dibagi internal4bagian. Ini bukan protokol benchmark resmiDTD.','']
 for r in summary:txt.append(f"{r['method']}: mean gain {r['mean_selected_gain_pp']:+.3f} pp; mean regret {r['mean_regret_pp']:.3f} pp; mean rank correlation {r['mean_spearman_f1']}")
 txt+=['','Lihat primary_policy_contrast.csv untuk semua9konteks, termasuk hasilnegatif.','Tidak menyimpulkan semua estimator gagal atau mekanisme universal dari konteks tunggal.']
 (ROOT/'HASIL_TRANSFER_DTD.txt').write_text('\n'.join(txt)+'\n');save(ROOT/'analysis_status.json',dict(state='COMPLETE',completed=63,total=63,old_comparison_complete=(ROOT/'old_comparison/summary.json').exists()))

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');a=parser.parse_args();verify()
 if not a.watch:
  if (ROOT/'old_diagnostics/all_profiles.json').exists() and not(ROOT/'old_comparison').exists():old_analysis()
  return
 while True:
  if (ROOT/'old_diagnostic_status.json').exists() and read(ROOT/'old_diagnostic_status.json')['state']=='FAILED':raise RuntimeError('Old diagnostic failure needs inspection')
  if (ROOT/'old_diagnostic_status.json').exists() and read(ROOT/'old_diagnostic_status.json')['state']=='COMPLETE' and not(ROOT/'old_comparison').exists():old_analysis()
  if (ROOT/'status.json').exists():
   s=read(ROOT/'status.json')
   if s['state'] in ['FAILED','COMPLETE_WITH_FAILURES']:raise RuntimeError('Training failure retained; no partial-cohort analysis')
   if s['state']=='COMPLETE':evaluate_complete();return
  save(ROOT/'analysis_status.json',dict(state='WAITING_FOR_COMPLETE_COHORT',total=63));time.sleep(20)
if __name__=='__main__':
 try:main()
 except Exception:
  (ROOT/'analysis_failure.log').write_text(traceback.format_exc());save(ROOT/'analysis_status.json',dict(state='FAILED'));raise
