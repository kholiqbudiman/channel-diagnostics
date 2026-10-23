import json,csv,hashlib
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
raw=(P/'results_snapshot.json').read_bytes(); rs=json.loads(raw)
dec=json.loads((P.parent/'EXPERIMENT_DECISIONS.json').read_text())
idx={(r['review_split'],r['job']['dataset'],r['job']['seed'],r['job']['model']['name']):r for r in rs}
assert len(idx)==126
checks=[]
def cm(r): return np.array(r['probe_metrics']['confusion_matrix'],float)
def fv(c): return np.divide(200*np.diag(c),c.sum(0)+c.sum(1),out=np.zeros(len(c)),where=c.sum(0)+c.sum(1)>0)
def metrics(r):
 c=cm(r); l=np.array(r['probe_metrics']['class_loss']); n=c.sum(1)
 return dict(f1=fv(c).mean(),accuracy=100*np.trace(c)/c.sum(),balanced_accuracy=100*np.mean(np.diag(c)/n),negative_macro_loss=-l.mean(),negative_sample_loss=-np.dot(l,n)/n.sum())
for r in rs:
 c=cm(r); m=metrics(r); h=[x['selection_macro_f1_pct'] for x in r['history']]
 assert np.all(c>=0) and np.all(c.sum(1)>0)
 assert abs(m['f1']-r['probe_metrics']['macro_f1_pct'])<1e-8
 assert abs(m['accuracy']-r['probe_metrics']['accuracy_pct'])<1e-8
 assert np.argmax(h)+1==r['best_epoch']
 assert abs(max(h)-r['selection_macro_f1_pct'])<1e-8
 assert len(h)==min(r['best_epoch']+20,100)
rows=[]; classes=[]; curvature=[]; epochs=[]
for r in rs:
 epochs.append(dict(split=r['review_split'],dataset=r['job']['dataset'],seed=r['job']['seed'],model=r['job']['model']['name'],best_epoch=r['best_epoch'],trained_epochs=len(r['history']),selection_minus_probe_f1=r['selection_macro_f1_pct']-metrics(r)['f1']))
for s in (4302,4303,4304):
 for ds in ('warpc','trashnet'):
  for seed in (2901,2902,2903):
   b=idx[s,ds,seed,'base']
   for d in dec['directions']:
    g=idx[s,ds,seed,d['component_g']];v=idx[s,ds,seed,d['component_d']];j=idx[s,ds,seed,d['joint']]
    assert all(np.array_equal(cm(b).sum(1),cm(x).sum(1)) for x in (g,v,j))
    ms=list(map(metrics,(b,g,v,j)))
    row=dict(split=s,dataset=ds,seed=seed,direction=d['name'])
    for k in ms[0]: row[k+'_interaction']=ms[3][k]-ms[1][k]-ms[2][k]+ms[0][k]
    rows.append(row)
    cf=fv(cm(j))-fv(cm(g))-fv(cm(v))+fv(cm(b))
    for k,c in enumerate(b['classes']): classes.append(dict(split=s,dataset=ds,seed=seed,direction=d['name'],class_name=c,interaction=cf[k],leave_class_out_interaction=float(np.delete(cf,k).mean())))
    ca=cm(g)+cm(v)-cm(b); valid=bool((ca>=0).all())
    z=dict(split=s,dataset=ds,seed=seed,direction=d['name'],valid_additive_confusion=valid,negative_cells=int((ca<0).sum()))
    if valid:
     z.update(metric_curvature=float(fv(ca).mean()-ms[1]['f1']-ms[2]['f1']+ms[0]['f1']),output_departure=float(ms[3]['f1']-fv(ca).mean()))
    curvature.append(z)
def savecsv(n,rr):
 keys=list(dict.fromkeys(k for r in rr for k in r))
 with (P/n).open('w') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rr)
for n,rr in [('metric_interactions.csv',rows),('class_influence.csv',classes),('curvature_validity.csv',curvature),('checkpoint_epochs.csv',epochs)]:savecsv(n,rr)
out=dict(snapshot_sha256=hashlib.sha256(raw).hexdigest(),records=126,recomputed_cm_metrics=126,history_checks=126,scope='Post-outcome review; descriptive contexts, not independent population inference.',groups=[])
for ds in ('warpc','trashnet'):
 for d in dec['directions']:
  rr=[r for r in rows if r['dataset']==ds and r['direction']==d['name']]
  cc=[r for r in classes if r['dataset']==ds and r['direction']==d['name']]
  z=dict(dataset=ds,direction=d['name'],metrics={})
  for k in metrics(rs[0]):
   a=np.array([r[k+'_interaction'] for r in rr]);spl=[float(np.mean([r[k+'_interaction'] for r in rr if r['split']==s])) for s in (4302,4303,4304)]
   z['metrics'][k]=dict(mean=float(a.mean()),sd=float(a.std(ddof=1)),negative=int((a<0).sum()),split_means=spl,leave_one_split_mean_range=[float((sum(spl)-max(spl))/2),float((sum(spl)-min(spl))/2)])
  z['class_means']={c:float(np.mean([r['interaction'] for r in cc if r['class_name']==c])) for c in sorted(set(r['class_name'] for r in cc))}
  z['leave_each_class_out_means']={c:float(np.mean([r['leave_class_out_interaction'] for r in cc if r['class_name']==c])) for c in sorted(set(r['class_name'] for r in cc))}
  z['valid_additive_confusion_contexts']=sum(r['valid_additive_confusion'] for r in curvature if r['dataset']==ds and r['direction']==d['name'])
  out['groups'].append(z)
(P/'review_summary.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
