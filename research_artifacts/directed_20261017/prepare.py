"""Prepare a finite two-factor ACRE intervention; no GPU training or outcome search."""
from pathlib import Path
from datetime import datetime,timezone
import sys,json,hashlib,shutil,itertools,random
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent;OLD=Path('/home/kholiq/KD/ACRE_stability_20261016')
sys.path.insert(0,str(OLD/'split_4302'))
import experiment as e
from models import build_model,count_model
from budget_design import allocation_cost
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 assert not(ROOT/'COHORT_LOCK.json').exists()
 torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
 assert torch.__version__=='2.10.0+cu128',torch.__version__
 locked=read(OLD/'COHORT_LOCK.json')
 for n,h in locked['sources'].items():assert sha(OLD/n)==h,n
 refs={k:allocation_cost([16,24,40,64],classes=k) for k in [5,6]};candidates=[]
 # The smallest even c step is2. Select a symmetric d step by minimax joint
 # budget mismatch across both directions and class counts, without outcomes.
 for dd in range(2,34,2):
  pair=[[16,24,42,64-dd],[16,24,38,64+dd]]
  if any(ch!=sorted(ch) for ch in pair):continue
  deviations=[abs(allocation_cost(ch,classes=k)[v]/refs[k][v]-1) for ch in pair for k in [5,6] for v in ['params','macs']]
  if max(deviations)<=.01:candidates.append((max(deviations),dd))
 assert candidates;score,dd=min(candidates);assert dd==8
 settings=[('base',[16,24,40,64],'reference'),('c_grow',[16,24,42,64],'component_control'),('d_shrink',[16,24,40,64-dd],'component_control'),('joint_c_grow_d_shrink',[16,24,42,64-dd],'joint_budget_transfer'),('c_shrink',[16,24,38,64],'component_control'),('d_grow',[16,24,40,64+dd],'component_control'),('joint_c_shrink_d_grow',[16,24,38,64+dd],'joint_budget_transfer')]
 configs=[dict(name=n,kind='allocation',channels=ch,expansions=[2]*5,residual=True,cost_k5=allocation_cost(ch)) for n,ch,role in settings]
 jobs=[];references=[];checks=[]
 for split in [4302,4303,4304]:
  src=OLD/f'split_{split}';folder=ROOT/f'split_{split}';folder.mkdir()
  assert sha(src/'protocol_lock.json')==locked['split_locks'][str(split)]
  for n,h in read(src/'protocol_lock.json')['sources'].items():assert sha(src/n)==h,n
  for n in ['experiment.py','models.py','legacy_issue3.py','legacy_table3.py','budget_design.py','data_warpc.json','data_trashnet.json']:shutil.copy2(src/n,folder/n)
  protocol=read(src/'protocol.json');oldbase=next(m for m in protocol['models'] if m['name']=='base');assert oldbase==configs[0]
  protocol.update(version='directed_late_width_factorial_v1',models=configs,scope='Development intervention on reused grouped split pools;18 archived Base references and108 new training runs',max_workers=2)
  protocol['jobs']=[dict(id=f'{ds}_{c["name"]}_s{seed}',dataset=ds,model=c,seed=seed) for ds in protocol['datasets'] for seed in [2901,2902,2903] for c in configs]
  save(folder/'protocol.json',protocol);(folder/'runs').mkdir();pre=dict(model_costs={},datasets=read(src/'preflight.json')['datasets'],checks=[])
  for ds,path in protocol['datasets'].items():
   manifest=read(folder/f'data_{ds}.json');k=len(manifest['classes']);parts=manifest['split'];samples=manifest['samples']
   for name,h in manifest['image_sha256'].items():assert sha(Path(path)/name)==h
   indices=[next(i for i in parts['probe'] if samples[i][1]==c) for c in range(k)];data=e.Images(Path(path),samples,indices);x=torch.stack([data[i][0] for i in range(k)]);y=torch.arange(k)
   costs={}
   for c in configs:
    torch.manual_seed(812);model=build_model(c,k);cost=count_model(model);formula=allocation_cost(c['channels'],classes=k);assert all(cost[v]==formula[v] for v in ['params','macs'])
    role=next(role for name,ch,role in settings if name==c['name'])
    if role in ['reference','joint_budget_transfer']:assert all(abs(cost[v]/refs[k][v]-1)<=.01 for v in ['params','macs'])
    model.train();loss=torch.nn.functional.cross_entropy(model(x),y,label_smoothing=.1);loss.backward();assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    costs[c['name']]=cost
   pre['model_costs'][ds]=costs
   for seed in [2901,2902,2903]:
    jid=f'{ds}_base_s{seed}';oldfolder=src/'runs'/jid;r=read(oldfolder/'result.json')
    for name,key in [('best.pt','checkpoint_sha256'),('initial_state.pt','initialization_file_sha256')]:assert sha(oldfolder/name)==r[key]
    assert r['split_manifest_sha256']==sha(folder/f'data_{ds}.json')
    torch.manual_seed(seed);base=build_model(configs[0],k).state_dict();recorded=torch.load(oldfolder/'initial_state.pt',map_location='cpu',weights_only=True)['model'];assert base.keys()==recorded.keys() and all(torch.equal(base[n],recorded[n]) for n in base)
    for c in configs[1:]:
     torch.manual_seed(seed);state=build_model(c,k).state_dict()
     early=[n for n in base if n.startswith(('stem.','block1.','block2.','block3.'))];assert all(torch.equal(base[n],state[n]) for n in early)
    shutil.copytree(oldfolder,folder/'runs'/jid)
    references.append(dict(split=split,dataset=ds,seed=seed,job=jid,original_folder=str(oldfolder),files={n:sha(oldfolder/n) for n in ['result.json','best.pt','initial_state.pt','probe_features.npz']}))
  save(folder/'preflight.json',pre)
  names=['experiment.py','models.py','legacy_issue3.py','legacy_table3.py','budget_design.py','data_warpc.json','data_trashnet.json','protocol.json','preflight.json'];save(folder/'protocol_lock.json',dict(sources={n:sha(folder/n) for n in names},scope=protocol['scope']))
  for j in protocol['jobs']:
   if j['model']['name']!='base':jobs.append(dict(id=f'split{split}_{j["id"]}',inner_id=j['id'],folder=str(folder),split=split,dataset=j['dataset'],seed=j['seed'],model=j['model']['name']))
  checks.append(dict(split=split,finite_forward_backward=14,base_initializations_exact=6,early_initial_tensor_comparisons=36))
 assert len(jobs)==108 and len(references)==18;random.Random(7017).shuffle(jobs)
 save(ROOT/'job_inventory.json',jobs);save(ROOT/'REFERENCE_MANIFEST.json',references)
 decision=dict(created_utc=datetime.now(timezone.utc).isoformat(),kind='Directed two-factor mechanistic development, not independent confirmation',source_stability_lock_sha256=sha(OLD/'COHORT_LOCK.json'),new_training_jobs=108,reused_base_runs=18,contexts_per_dataset=9,datasets=['warpc','trashnet'],splits=[4302,4303,4304],model_seeds=[2901,2902,2903],configs=[dict(name=n,channels=ch,role=role) for n,ch,role in settings],selection_rule=dict(delta_c=2,symmetric_delta_d=dd,label_free_minimax_budget_mismatch=score),research_question='Does masking-to-retraining proxy failure arise from errors of the individual width interventions, non-additivity of their joint effect, or both?',directions=[dict(name='c_up_d_down',component_g='c_grow',component_d='d_shrink',joint='joint_c_grow_d_shrink'),dict(name='c_down_d_up',component_g='d_grow',component_d='c_shrink',joint='joint_c_shrink_d_grow')],analysis='After all108 new results only. I=Fjoint-Fg-Fd+Fbase. For fixed mask predictor, errorjoint=I+errorg+errord. Report all classes, split/seed effects, signs and arithmetic precision/recall attribution. No iid significance across overlapping data splits. No after-outcome estimator fit or winning-ablation selection.',hypothesis_scope='Additivity and proxy fidelity can fail or hold. Do not assume non-additivity causes the previously observed failure before these controls exist. This tests width-boundary interventions, not independent biological/semantic roles of blocks.',joint_constraints='Base and joint transfer cells within1% params andConv/LinearMACs forK5/K6. Single-factor cells intentionally outside this band to identify finite component effects; never call them resource-matched superiority controls.',training='Byte-identical locked stability runner, model sources, data manifests, checkpoint-selection and100epoch/patience20 AdamW recipe. RuntimeTorch2.10.0+cu128. No official test access. Base references reused, not retrained or counted as new observations.',size_basis='Six nonreference cells are the minimal union of two opposing2x2 factorials sharing one Base. Reuse all existing3splits x3modelseeds x2datasets to preserve pairing and avoid choosing favorable contexts. No formal population power claim.',operational='At most one ACRE worker perGPU. DKD priority: admit alongside scientific DKD only when that GPU already has4 live DKD workers with epoch>=1, telemetry fresh, no unknown CUDA processes, util<75%, freeVRAM>=8GiB and CPUload<80%. When DKD ends, wait until all15 isolated cost profiles complete before further ACRE admission. Never interrupt existing jobs, no automatic retry on failure.',predictive_scope='Base diagnostics are development-probe-derived. Calibration/tasks already exposed; this does not create an untouched final test or validate a new allocation algorithm.')
 save(ROOT/'EXPERIMENT_DECISIONS.json',decision);save(ROOT/'preflight_report.json',dict(state='PASS',torch=torch.__version__,python=sys.version,checks=checks,raw_hashes_verified_per_split=10842))
 analysis=Path('/home/kholiq/KD/ACRE_method_development_20261010');names=['prepare.py','scheduler.py','analyze.py','job_inventory.json','REFERENCE_MANIFEST.json','EXPERIMENT_DECISIONS.json','preflight_report.json'];save(ROOT/'COHORT_LOCK.json',dict(sources={n:sha(ROOT/n) for n in names},split_locks={str(s):sha(ROOT/f'split_{s}'/'protocol_lock.json') for s in [4302,4303,4304]},analysis_dependencies={str(analysis/n):sha(analysis/n) for n in ['analyze_complete_pilot.py','confusion_response.py']}))
 print(json.dumps(dict(state='PREFLIGHT_PASS',jobs=108,reused=18,delta_d=dd,checks=checks),indent=2))
if __name__=='__main__':main()
