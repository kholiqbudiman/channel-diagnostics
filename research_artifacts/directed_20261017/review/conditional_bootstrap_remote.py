import pathlib,json,hashlib
import numpy as np
P=pathlib.Path('/home/kholiq/KD/ACRE_transfer_mechanism_20261016')
dec=json.loads((P/'EXPERIMENT_DECISIONS.json').read_text()); out=[];checked=0
for split in (4302,4303,4304):
 records={}
 for f in sorted((P/f'split_{split}'/'runs').glob('*/result.json')):
  r=json.loads(f.read_text());a=np.load(f.parent/'probe_features.npz');y=a['labels'];pred=a['logits'].argmax(1);k=len(r['classes']);cm=np.bincount(y*k+pred,minlength=k*k).reshape(k,k)
  assert np.array_equal(cm,r['probe_metrics']['confusion_matrix']);records[r['job']['dataset'],r['job']['seed'],r['job']['model']['name']]=(y,pred,r)
  assert hashlib.sha256((f.parent/'best.pt').read_bytes()).hexdigest()==r['checkpoint_sha256']
  assert hashlib.sha256((f.parent/'initial_state.pt').read_bytes()).hexdigest()==r['initialization_file_sha256']
  checked+=1
 for ds in ('warpc','trashnet'):
  for seed in (2901,2902,2903):
   for direction in dec['directions']:
    names=['base',direction['component_g'],direction['component_d'],direction['joint']]
    aa=[records[ds,seed,n] for n in names];y=aa[0][0];k=len(aa[0][2]['classes']);assert all(np.array_equal(y,a[0]) for a in aa)
    rng=np.random.default_rng(20261017+split+seed);groups=[np.flatnonzero(y==c) for c in range(k)]
    ind=np.concatenate([rng.choice(g,(1000,len(g)),replace=True) for g in groups],axis=1);yb=y[ind];vals=[];acs=[]
    for _,pred,_ in aa:
     pb=pred[ind];codes=yb*k+pb+np.arange(1000)[:,None]*k*k;c=np.bincount(codes.ravel(),minlength=1000*k*k).reshape(1000,k,k)
     den=c.sum(1)+c.sum(2);f=np.divide(200*np.diagonal(c,axis1=1,axis2=2),den,out=np.zeros_like(den,dtype=float),where=den>0).mean(1);vals.append(f);acs.append(100*(yb==pb).mean(1))
    I=vals[3]-vals[1]-vals[2]+vals[0];A=acs[3]-acs[1]-acs[2]+acs[0]
    out.append(dict(split=split,dataset=ds,seed=seed,direction=direction['name'],conditional_f1_interval=np.quantile(I,[.025,.975]).tolist(),conditional_accuracy_interval=np.quantile(A,[.025,.975]).tolist()))
print(json.dumps(dict(prediction_recomputations=checked,checkpoint_hashes=checked,initialization_hashes=checked,bootstrap_replicates=1000,scope='Paired class-stratified image bootstrap conditional on fitted checkpoints and fixed probe; not training, split, source-object or task uncertainty. No pooled iid p-values.',rows=out)))
