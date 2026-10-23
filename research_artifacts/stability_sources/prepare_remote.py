"""Prospective pilot-informed stability cohort. No GPU training in preparation."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys,shutil,random
import numpy as np
import torch
OLD=Path('/home/kholiq/KD/ACRE_novelty_20261010')
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(OLD))
import experiment as e
from models import build_model,count_model,stage_modules
from budget_design import allocation_cost

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def grouped_split(samples, hashes, seed):
 # Group byte-identical images before splitting; never resample by outcome.
 rng=np.random.default_rng(seed); groups={}
 for i,(name,label) in enumerate(samples):groups.setdefault(hashes[name],[]).append((i,label))
 groups={h:g for h,g in groups.items() if len({label for _,label in g})==1} # Exclude every conflicting-label content group, without choosing a preferred label.
 parts={'fit':[],'selection':[],'probe':[]}
 for label in sorted({v[1] for v in samples}):
  gs=[sorted(i for i,_ in g) for g in groups.values() if g[0][1]==label]
  gs.sort(key=lambda g:g[0]);rng.shuffle(gs)
  target=max(2,int(sum(map(len,gs))*.1));selection=[];probe=[];fit=[]
  for g in gs:
   if len(selection)<target:selection.extend(g)
   elif len(probe)<target:probe.extend(g)
   else:fit.extend(g)
  assert selection and probe and fit
  parts['fit'].extend(fit);parts['selection'].extend(selection);parts['probe'].extend(probe)
 return {k:sorted(v) for k,v in parts.items()}

def main():
 assert not (HERE/'COHORT_LOCK.json').exists(),'Existing lock must be preserved'
 torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
 oldlock=e.verify_lock();old=json.loads((OLD/'protocol.json').read_text())
 names=['base','base_no_residual','allocation_14_30_36_52','allocation_12_34_36_36']
 configs=[next(c for c in old['models'] if c['name']==n) for n in names]
 seeds=[2901,2902,2903];splits=[4302,4303,4304]; exclusions={}
 manifests={ds:json.loads((OLD/f'data_{ds}.json').read_text()) for ds in old['datasets']}
 for ds,m in manifests.items():
  groups={}
  for i,(name,label) in enumerate(m['samples']):groups.setdefault(m['image_sha256'][name],[]).append((i,name,label))
  exclusions[ds]=[{'sha256':h,'members':g} for h,g in groups.items() if len({v[2] for v in g})>1]
 save(HERE/'DATA_EXCLUSIONS.json',exclusions)
 for ds,root in old['datasets'].items():
  samples,classes=e.discover(Path(root),ds);m=manifests[ds]
  assert samples==[tuple(v) for v in m['samples']] and classes==m['classes']
  for name,expected in m['image_sha256'].items():assert sha(Path(root)/name)==expected,(ds,name)
 alljobs=[];report={'datasets_verified':list(manifests),'checks':[]}
 files=['experiment.py','models.py','legacy_issue3.py','legacy_table3.py','budget_design.py']
 for splitseed in splits:
  folder=HERE/f'split_{splitseed}';folder.mkdir()
  for name in files:shutil.copy2(OLD/name,folder/name)
  protocol=dict(old,version='pilot_informed_stability_v1',models=configs,split_seed=splitseed,max_workers=4,scope='Prospective stability replication, pilot-informed candidates; reused training pools; no independent new-task confirmation')
  jobs=[{'id':f'{ds}_{c["name"]}_s{seed}','dataset':ds,'model':c,'seed':seed} for ds in old['datasets'] for seed in seeds for c in configs]
  protocol['jobs']=jobs;save(folder/'protocol.json',protocol)
  preflight={'model_costs':{},'datasets':{},'checks':[]}
  for ds,root in old['datasets'].items():
   m=manifests[ds];samples=[tuple(v) for v in m['samples']];split=grouped_split(samples,m['image_sha256'],splitseed)
   parts=[set(split[k]) for k in ['fit','selection','probe']]
   assert not(parts[0]&parts[1] or parts[0]&parts[2] or parts[1]&parts[2]);excluded={v[0] for g in exclusions[ds] for v in g['members']};assert set.union(*parts)==set(range(len(samples)))-excluded
   hs=[{m['image_sha256'][samples[i][0]] for i in ids} for ids in parts]
   assert not(hs[0]&hs[1] or hs[0]&hs[2] or hs[1]&hs[2])
   new=dict(m,split=split,split_seed=splitseed,excluded_conflicting_content_groups=exclusions[ds],split_method='Class-stratified exact-content groups',scope='New subdivision of existing training pool. No official test access; previously exposed pilot data, not a new untouched dataset.')
   save(folder/f'data_{ds}.json',new)
   k=len(m['classes']);costs={};ref=allocation_cost([16,24,40,64],classes=k)
   diagnostic=[next(i for i in split['probe'] if samples[i][1]==c) for c in range(k)]
   dataset=e.Images(Path(root),samples,diagnostic)
   x=torch.stack([dataset[i][0] for i in range(k)]);y=torch.arange(k)
   for c in configs:
    torch.manual_seed(812);model=build_model(c,k);cost=count_model(model);formula=allocation_cost(c['channels'],c['expansions'],k)
    assert all(cost[n]==formula[n] for n in ['params','macs'])
    assert all(abs(cost[n]/ref[n]-1)<=.01 for n in ['params','macs'])
    model.train();loss=torch.nn.functional.cross_entropy(model(x),y,label_smoothing=.1);loss.backward()
    assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    costs[c['name']]=cost
   for seed in seeds:
    torch.manual_seed(seed);a=build_model(configs[0],k).state_dict()
    torch.manual_seed(seed);b=build_model(configs[1],k).state_dict()
    assert all(torch.equal(a[n],b[n]) for n in a)
   preflight['model_costs'][ds]=costs
   preflight['datasets'][ds]={'classes':m['classes'],'n_enumerated':len(samples),'n_used':sum(map(len,parts)),'excluded_conflicting_content_groups':exclusions[ds],'class_counts':{part:np.bincount([samples[i][1] for i in ids],minlength=k).tolist() for part,ids in split.items()},'exact_cross_split_duplicates':0}
  preflight['checks']=['All 8 task/model forward/backward checks finite','Actual and symbolic costs match; all joint budget bands within 1%','Base/no-residual initial weights equal for all three model seeds','Every class present, split IDs disjoint, no exact content hash crossing splits']
  save(folder/'preflight.json',preflight)
  locked=files+['protocol.json','preflight.json']+[f'data_{ds}.json' for ds in old['datasets']]
  save(folder/'protocol_lock.json',{'scope':'Prospective pilot-informed stability cohort, no powered confirmatory or independent-dataset claim','sources':{n:sha(folder/n) for n in locked}})
  for j in jobs:alljobs.append({'id':f'split{splitseed}_{j["id"]}','split':splitseed,'folder':str(folder),'inner_id':j['id'],'dataset':j['dataset'],'seed':j['seed'],'model':j['model']['name']})
  report['checks'].append({'split':splitseed,'forward_backward':8,'initial_weight_pairs':6,'lock_sha256':sha(folder/'protocol_lock.json')})
 random.Random(7016).shuffle(alljobs)
 save(HERE/'job_inventory.json',alljobs);save(HERE/'preflight_report.json',report)
 decision={'created_utc':datetime.now(timezone.utc).isoformat(),'kind':'Pilot-informed prospective stability replication','jobs':72,'model_seeds':seeds,'split_seeds':splits,'data_exclusion':'Exclude all copies in every exact-content group assigned conflicting labels, consistently before all splits/configs. TrashNet glass176.jpg and plastic152.jpg share identical bytes but different labels; retain raw files and old pilot. No outcome-based exclusion.','split_method':'Class-stratified grouping by exact image SHA256; all byte-identical copies stay together. Approximate 80/10/10; no source-object/near-duplicate guarantee.','models':names,'primary_development_contrasts':['allocation_14_30_36_52 minus base','base_no_residual minus base'],'secondary_control':'allocation_12_34_36_36 minus base','endpoint':'probe macro-F1 at selection-selected checkpoint','classwise':'all classes, canister explicitly retained even if zero','sample_size_basis':'Finite crossed design of three split levels and three model-seed levels to inspect sign/rank and optimization/split sensitivity. Resource-bounded development replication, not a formal power claim. No optional stopping or selective exclusions.','inference_scope':'Splits overlap in input images; nine runs are not nine independent datasets. Report paired differences by split and seed; no naive iid p-values or winner-only reporting.','training_recipe':'Byte-identical experiment.py and model sources to corrected pilot, including AdamW, clipping1, smoothing.1, 100epochs/patience20; only split/config/seed inventory differs.','selection_bias':'Configs informed by completed pilot; new retraining/splits improve stability assessment but do not create an untouched test set or establish algorithm novelty.','resource_policy':'At most two workers per GPU, four total, first-epoch and GPU admission checks; wait for DKD post-primary measurements AND aggregation completion; stop new jobs on failures; no automatic retry.','source_pilot_lock_sha256':sha(OLD/'protocol_lock.json')}
 save(HERE/'EXPERIMENT_DECISIONS.json',decision)
 locked=['prepare_remote.py','scheduler.py','resource_policy.py','job_inventory.json','preflight_report.json','EXPERIMENT_DECISIONS.json','DATA_EXCLUSIONS.json']
 lock={'created_utc':decision['created_utc'],'sources':{n:sha(HERE/n) for n in locked},'split_locks':{str(s):sha(HERE/f'split_{s}'/'protocol_lock.json') for s in splits}}
 save(HERE/'COHORT_LOCK.json',lock)
 print(json.dumps(report,indent=2));print('LOCKED72')
if __name__=='__main__':main()
