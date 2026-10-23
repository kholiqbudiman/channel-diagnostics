"""DTD training; final evaluation is withheld for the complete-cohort analysis."""
from pathlib import Path
import os,sys,json,hashlib,argparse,random,math,time
import numpy as np,torch
from torch import nn
import reference as e
from models import build_model,count_model
from diagnostics import profile_model
from scores import predictions,choices
ROOT=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
 p.parent.mkdir(exist_ok=True,parents=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def verify():
 lock=read(ROOT/'COHORT_LOCK.json')
 for n,h in lock['sources'].items():assert sha(ROOT/n)==h,n
 assert read(ROOT/'GO.json')['state']=='GO'
def run(j,gpu=True):
 verify();folder=ROOT/'runs'/j['id'];assert not(folder/'training_complete.json').exists();folder.mkdir(exist_ok=True,parents=True)
 m=read(ROOT/f"data_{j['split']}.json");samples=[tuple(x) for x in m['samples']];classes=m['classes'];root=Path(m['root']);K=len(classes)
 for rel,h in m['image_sha256'].items():assert sha(root/rel)==h,rel
 torch.set_num_threads(2);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 torch.manual_seed(j['seed']);model=build_model(j['model'],K);cost=count_model(model);assert cost==read(ROOT/'preflight.json')['costs'][j['model']['name']]
 torch.save(dict(model=model.state_dict(),job=j),folder/'initial_state.pt')
 seed=j['seed']+100000;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 assert torch.cuda.is_available();device=torch.device('cuda:0');model.to(device)
 fit=e.make_loader(e.Images(root,samples,m['split']['fit'],True),True,seed)
 select=e.make_loader(e.Images(root,samples,m['split']['selection']),False,j['split'])
 opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001);sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda ep:(ep+1)/5 if ep<5 else .5*(1+math.cos(math.pi*(ep-5)/95)))
 best=-math.inf;epochbest=0;bad=0;history=[];start=time.time()
 for epoch in range(100):
  model.train()
  for x,y in fit:
   opt.zero_grad(set_to_none=True);loss=nn.functional.cross_entropy(model(x.to(device)),y.to(device),label_smoothing=.1);assert torch.isfinite(loss)
   loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
  sched.step();l,y,_=e.evaluate(model,select,device);f=e.metrics(l,y,K)['macro_f1_pct'];history.append(dict(epoch=epoch+1,selection_macro_f1_pct=f))
  if f>best:
   best=f;epochbest=epoch+1;bad=0;torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},job=j,classes=classes,cost=cost,epoch=epochbest,selection_macro_f1_pct=best),folder/'best.pt')
  else:bad+=1
  save(folder/'progress.json',dict(epoch=epoch+1,elapsed_s=time.time()-start,state='TRAINING',selection_macro_f1_pct=f))
  if bad>=20:break
 model.load_state_dict(torch.load(folder/'best.pt',map_location='cpu',weights_only=True)['model'])
 if j['model']['name']=='base':
  diagnostic=e.make_loader(e.Images(root,samples,m['split']['diagnostic']),False,j['split'])
  profile,l,y=profile_model(model,diagnostic,classes,device)
  save(folder/'diagnostic.json',profile);np.savez_compressed(folder/'diagnostic_predictions.npz',labels=y.numpy(),logits=l.numpy())
  pred=predictions(profile,read(ROOT/'PROTOCOL.json')['configs'],read(ROOT/'ridge.json'))
  save(folder/'predictions_committed.json',dict(scores=pred,choices=choices(pred),checkpoint_sha256=sha(folder/'best.pt'),diagnostic_sha256=sha(folder/'diagnostic.json'),manifest_sha256=sha(ROOT/f"data_{j['split']}.json"),scope='Committed before nonBase training; uses diagnostic subset, never evaluation outcomes.'))
 save(folder/'training_complete.json',dict(job=j,cost=cost,history=history,best_epoch=epochbest,selection_macro_f1_pct=best,checkpoint_sha256=sha(folder/'best.pt'),initialization_file_sha256=sha(folder/'initial_state.pt'),manifest_sha256=sha(ROOT/f"data_{j['split']}.json"),elapsed_s=time.time()-start,scope='No final evaluation performed.'))
 save(folder/'progress.json',dict(epoch=len(history),state='TRAINING_COMPLETE_EVALUATION_WITHHELD'))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--job-id',required=True);a=p.parse_args();run(next(x for x in read(ROOT/'jobs.json') if x['id']==a.job_id))
