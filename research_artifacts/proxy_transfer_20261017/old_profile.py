"""Retrospective old-cohort diagnostic comparison; CPU only, no retraining."""
from pathlib import Path
import os,json,hashlib,sys,time
import numpy as np,torch
import reference as e
from models import build_model
from diagnostics import profile_model
from scores import predictions,choices
from train import save,verify
ROOT=Path(__file__).resolve().parent
OLD=Path('/home/kholiq/KD/ACRE_transfer_mechanism_20261016')
def main():
 verify();torch.set_num_threads(2);out=ROOT/'old_diagnostics';out.mkdir(exist_ok=True);jobs=json.loads((ROOT/'old_results.json').read_text());rr=[]
 for r in sorted([x for x in jobs if x['job']['model']['name']=='base'],key=lambda x:(x['review_split'],x['job']['dataset'],x['job']['seed'])):
  split=r['review_split'];j=r['job'];f=OLD/f'split_{split}'/'runs'/j['id'];target=out/f'{split}_{j["id"]}.json';assert not target.exists()
  sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert sha(f/'best.pt')==r['checkpoint_sha256']
  m=json.loads((OLD/f'split_{split}'/f'data_{j["dataset"]}.json').read_text());assert sha(OLD/f'split_{split}'/f'data_{j["dataset"]}.json')==r['split_manifest_sha256']
  root=Path(json.loads((OLD/f'split_{split}'/'protocol.json').read_text())['datasets'][j['dataset']]);samples=[tuple(x) for x in m['samples']]
  model=build_model(j['model'],len(r['classes']));model.load_state_dict(torch.load(f/'best.pt',map_location='cpu',weights_only=True)['model'])
  ds=e.Images(root,samples,m['split']['probe']);loader=e.make_loader(ds,False,split,workers=0)
  profile,l,y=profile_model(model,loader,r['classes'],torch.device('cpu'));a=np.load(f/'probe_features.npz');assert np.array_equal(y.numpy(),a['labels']);assert np.isfinite(l.numpy()).all();maxerr=float(np.max(np.abs(l.numpy()-a['logits'])));argmax_disagreement=int((l.argmax(1).numpy()!=a['logits'].argmax(1)).sum())
  cfg=json.loads((ROOT/'PROTOCOL.json').read_text())['old_configs'];pred=predictions(profile,cfg,json.loads((ROOT/'ridge.json').read_text()))
  v=dict(split=split,dataset=j['dataset'],seed=j['seed'],profile=profile,predictions=pred,choices=choices(pred),checkpoint_sha256=r['checkpoint_sha256'],max_cpu_saved_logit_difference=maxerr,cpu_saved_argmax_disagreement=argmax_disagreement,original_saved_channel_masks=r['channel_sensitivity'],all_diagnostics_runtime='CPU; all mask effects recomputed on same runtime as Taylor/BN; archived GPU masks retained separately',scope='Retrospective: original development probe, exposed outcomes.')
  save(target,v);rr.append(v);save(ROOT/'old_diagnostic_status.json',dict(state='RUNNING',completed=len(rr),total=18));print('PROFILED',target.name,flush=True)
 save(out/'all_profiles.json',rr);save(ROOT/'old_diagnostic_status.json',dict(state='COMPLETE',completed=18,total=18))
if __name__=='__main__':
 try:main()
 except Exception:
  import traceback
  (ROOT/'old_diagnostic_failure.log').write_text(traceback.format_exc());save(ROOT/'old_diagnostic_status.json',dict(state='FAILED',total=18));raise
