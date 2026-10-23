"""Label-free preparation, fixed old-data control, numerical preflight; no CUDA training."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys,random,collections
import numpy as np,torch
from torch import nn
from models import build_model,count_model
from budget_design import allocation_cost
import reference as e
from scores import predictions,choices,delta
ROOT=Path(__file__).resolve().parent
RAW=Path('/home/kholiq/KD/datasets_paper1/_raw/dtd/dtd')
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 assert not(ROOT/'COHORT_LOCK.json').exists();assert not(ROOT/'runs').exists()
 torch.set_num_threads(2);assert torch.__version__=='2.10.0+cu128',torch.__version__
 labels=[RAW/'labels/train1.txt',RAW/'labels/val1.txt'];paths=sorted(set(sum([p.read_text().splitlines() for p in labels],[])))
 assert len(paths)==3760;assert all(len(Path(n).parts)==2 and '..' not in Path(n).parts for n in paths)
 classes=sorted(set(Path(n).parts[0] for n in paths));assert len(classes)==47
 root=RAW/'images';samples=[(n,classes.index(Path(n).parts[0])) for n in paths]
 assert set(collections.Counter(y for _,y in samples).values())=={80}
 hashes={n:sha(root/n) for n in paths};groups=collections.defaultdict(list)
 for n,y in samples:groups[hashes[n]].append((n,y))
 conflicts={h for h,ss in groups.items() if len({y for _,y in ss})>1};exclusions=[dict(hash=h,paths=groups[h],reason='Conflicting-label exact content: exclude every copy') for h in sorted(conflicts)]
 samples=[(n,y) for n,y in samples if hashes[n] not in conflicts];save(ROOT/'DATA_EXCLUSIONS.json',exclusions)
 manifest_checks={}
 for split in [6301,6302,6303]:
  rng=np.random.default_rng(split);parts={n:[] for n in ['fit','selection','diagnostic','evaluation']};byhash=collections.defaultdict(list)
  for i,(n,y) in enumerate(samples):byhash[hashes[n]].append(i)
  for k in range(47):
   gs=[v for v in byhash.values() if samples[v[0]][1]==k];gs.sort(key=lambda v:samples[v[0]][0]);rng.shuffle(gs)
   N=sum(map(len,gs));assert N>=40;target=max(2,int(N*.1))
   for part in ['selection','diagnostic','evaluation']:
    while len([i for i in parts[part] if samples[i][1]==k])<target:parts[part].extend(gs.pop())
   parts['fit'].extend(i for g in gs for i in g)
  parts={k:sorted(v) for k,v in parts.items()};assert sorted(sum(parts.values(),[]))==list(range(len(samples)))
  sets={k:{hashes[samples[i][0]] for i in ids} for k,ids in parts.items()}
  assert all(not sets[a]&sets[b] for a in sets for b in sets if a<b)
  assert all(len(set(samples[i][1] for i in ids))==47 for ids in parts.values())
  manifest=dict(root=str(root),classes=classes,samples=samples,split=parts,image_sha256=hashes,official_list_sha256={str(p):sha(p) for p in labels},excluded=exclusions,scope='Only official train1+val1 file whitelist. No test images read. Four-way exact-content grouped internal splits; previously used by other research does not mean an untouched global benchmark.')
  save(ROOT/f'data_{split}.json',manifest);manifest_checks[str(split)]={k:len(v) for k,v in parts.items()}
 base=allocation_cost([16,24,40,64],classes=47);steps=[]
 for dd in range(2,34,2):
  chs=[[16,24,42,64-dd],[16,24,38,64+dd]]
  if any(ch!=sorted(ch) for ch in chs):continue
  mismatch=max(abs(allocation_cost(ch,classes=47)[v]/base[v]-1) for ch in chs for v in ['params','macs'])
  if mismatch<=.01:steps.append((mismatch,dd))
 score,dd=min(steps);assert dd==6
 def configs(dd):
  return [dict(name=n,kind='allocation',channels=ch,expansions=[2]*5,residual=True) for n,ch in [('base',[16,24,40,64]),('c_grow',[16,24,42,64]),('d_shrink',[16,24,40,64-dd]),('joint_c_grow_d_shrink',[16,24,42,64-dd]),('c_shrink',[16,24,38,64]),('d_grow',[16,24,40,64+dd]),('joint_c_shrink_d_grow',[16,24,38,64+dd])]]
 cfg=configs(dd);oldcfg=configs(8)
 old=read(ROOT/'old_results.json');assert len(old)==126;idx={(r['review_split'],r['job']['dataset'],r['job']['seed'],r['job']['model']['name']):r for r in old}
 X=[];Y=[]
 for r in old:
  j=r['job']
  if j['model']['name']=='base':continue
  b=idx[r['review_split'],j['dataset'],j['seed'],'base'];X.append(np.log(np.array(j['model']['channels'])[2:]/[40,64]));Y.append(r['probe_metrics']['macro_f1_pct']-b['probe_metrics']['macro_f1_pct'])
 X=np.array(X);Y=np.array(Y);assert X.shape==(108,2);rms=np.sqrt((X*X).mean(0));Z=X/rms;coef=np.linalg.solve(Z.T@Z+np.eye(2),Z.T@Y)
 ridge=dict(features=['log(c/40)','log(d/64)'],rms=rms.tolist(),coefficients=coef.tolist(),lambda_l2=1.,intercept=0.,fit_rows=108,source_sha256=sha(ROOT/'old_results.json'),scope='Fixed architecture-only comparator calibrated on old exposed outcomes, never DTD. No hyperparameter search.')
 save(ROOT/'ridge.json',ridge)
 costs={};checks=[]
 m=read(ROOT/'data_6301.json');ds=e.Images(root,[tuple(s) for s in samples],m['split']['fit'][:2]);x=torch.stack([ds[i][0] for i in range(2)]);y=torch.tensor([ds[i][1] for i in range(2)])
 for c in cfg:
  torch.manual_seed(817);model=build_model(c,47);cost=count_model(model);formula=allocation_cost(c['channels'],classes=47)
  assert all(cost[k]==formula[k] for k in ['params','macs'])
  if c['name'] in ['base','joint_c_grow_d_shrink','joint_c_shrink_d_grow']:assert all(abs(cost[k]/base[k]-1)<=.01 for k in ['params','macs'])
  model.train();loss=nn.functional.cross_entropy(model(x),y,label_smoothing=.1);loss.backward();assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
  costs[c['name']]=cost;checks.append(c['name'])
 # Verify gate derivatives against symmetric finite differences on a scratch model/fit images.
 model=build_model(cfg[0],47).double().eval();xx=x.double();check=[]
 for name in ['block4','block5']:
  mod=getattr(model,name);gate=torch.tensor(1.,dtype=torch.double,requires_grad=True)
  h=mod.register_forward_hook(lambda _m,_i,out:out*gate)
  l=nn.functional.cross_entropy(model(xx),y);g=float(torch.autograd.grad(l,gate)[0]);h.remove()
  vals=[];eps=1e-4
  for t in [eps,-eps]:
   h=mod.register_forward_hook(lambda _m,_i,out,t=t:out*np.exp(t));vals.append(float(nn.functional.cross_entropy(model(xx),y).detach()));h.remove()
  finite=(vals[0]-vals[1])/(2*eps);assert abs(g-finite)<1e-6+abs(g)*1e-3,(g,finite)
  check.append(dict(stage=name,autograd=g,finite_difference=finite))
 mock={k:[1.,2.,3.,4.,5.] for k in ['mask_f1_slope','mask_ce_slope','taylor_signed_slope','taylor_squared_importance','bn_gamma']};p=predictions(mock,cfg,ridge);assert choices({n:{k:0 for k in p[n]} for n in p})=={k:'base' for k in p['base']}
 for key in p['base']:
  assert abs(p['joint_c_grow_d_shrink'][key]-p['c_grow'][key]-p['d_shrink'][key])<1e-9
  assert abs(p['joint_c_shrink_d_grow'][key]-p['c_shrink'][key]-p['d_grow'][key])<1e-9
 jobs=[dict(id=f'dtd_split{s}_{c["name"]}_s{seed}',split=s,seed=seed,dataset='dtd',model=c) for s in [6301,6302,6303] for seed in [3901,3902,3903] for c in cfg]
 random.Random(20261017).shuffle(jobs);jobs.sort(key=lambda j:j['model']['name']!='base');assert len(jobs)==63
 protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),context='DTD47, official train1+val1, internal four-way split; ACRE transfer study, not official DTD benchmark protocol',dataset_root=str(root),configs=cfg,old_configs=oldcfg,split_seeds=[6301,6302,6303],model_seeds=[3901,3902,3903],new_runs=63,reused_runs=0,label_free_delta_d=dd,joint_minimax_mismatch=score,primary_policy_contrast=['mask_macro_ce','mask_f1'],primary_endpoint='Paired evaluation macro-F1 gain of CE-mask choice minus F1-mask choice, also versus Base; all policies retained.',decision_candidates=['base','joint_c_grow_d_shrink','joint_c_shrink_d_grow'],evaluation_gate='All63 training completions and9 checkpoint-bound predictions committed before nonBase training. No evaluation inference until complete cohort.',training='Same100epoch/patience20 AdamW smoothedCE recipe; source-level custom runner only separates diagnostic/evaluation stages.',uncertainty='Descriptive split/seed contrasts; 1000 class-stratified paired image bootstraps conditional on fixed checkpoints; no iid tests over overlapping splits.',failure='No auto retry or exclusion; stop new launches and retain active jobs.',no_tuning='All five diagnostic mappings and architecture comparator fixed before DTD outcomes; report failures, floor performance and mixed directions without replacing context.')
 save(ROOT/'PROTOCOL.json',protocol);save(ROOT/'jobs.json',jobs);save(ROOT/'preflight.json',dict(state='PASS',torch=torch.__version__,costs=costs,split_sizes=manifest_checks,raw_image_count=len(paths),used_image_count=len(samples),excluded_content_groups=len(conflicts),forward_backward=checks,gate_derivative_checks=check,zero_and_additivity_checks='PASS'))
 names=['PLAN_SEBELUM_EKSEKUSI.txt','prepare.py','scores.py','diagnostics.py','train.py','old_profile.py','analyze.py','scheduler.py','admission.py','reference.py','models.py','legacy_issue3.py','legacy_table3.py','budget_design.py','old_results.json','ridge.json','PROTOCOL.json','jobs.json','preflight.json','DATA_EXCLUSIONS.json']+[f'data_{s}.json' for s in [6301,6302,6303]]
 save(ROOT/'COHORT_LOCK.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),sources={n:sha(ROOT/n) for n in names},scope='Frozen prospective DTD protocol and retrospective old-data diagnostics. Before GO/new training.'))
 print(json.dumps(read(ROOT/'preflight.json'),indent=2))
if __name__=='__main__':main()
