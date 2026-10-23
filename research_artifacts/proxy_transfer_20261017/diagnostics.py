"""Frozen-model diagnostics. Taylor/BN adaptations, not published pruning pipelines."""
import hashlib,numpy as np,torch
from torch import nn
import reference as e
from models import stage_modules

def profile_model(model,loader,classes,device,stored_masks=None):
 model.eval();K=len(classes);logits,y,_=e.evaluate(model,loader,device)
 base=e.metrics(logits,y,K);masks=[] if stored_masks is None else stored_masks
 if stored_masks is None:
  for name,module in stage_modules(model):
   width=module.conv[-1].num_features
   for seed in [5101,5102,5103]:
    ids=np.random.default_rng(seed).choice(width,max(1,round(width*.1)),replace=False)
    def hook(_,inputs,out):
     mask=torch.ones(out.shape[1],device=out.device);mask[torch.as_tensor(ids,device=out.device)]=0
     return out*mask[None,:,None,None]
    h=module.register_forward_hook(hook)
    try:l,yy,_=e.evaluate(model,loader,device)
    finally:h.remove()
    assert torch.equal(y,yy);z=e.metrics(l,yy,K)
    masks.append(dict(stage=name,mask_seed=seed,removed_channels=ids.tolist(),removed_fraction=len(ids)/width,macro_f1_delta_pp=z['macro_f1_pct']-base['macro_f1_pct'],class_loss_delta=(np.array(z['class_loss'])-base['class_loss']).tolist()))
 f1=[];ce=[];gamma=[];signed=[];squared=[];contrib={};handles=[]
 for name,module in stage_modules(model):
  a=[m for m in masks if m['stage']==name];assert len(a)==3
  f1.append(float(np.mean([-m['macro_f1_delta_pp']/(-np.log1p(-m['removed_fraction'])) for m in a])))
  ce.append(float(np.mean([np.mean(m['class_loss_delta'])/(-np.log1p(-m['removed_fraction'])) for m in a])))
  bn=module.conv[-1];assert isinstance(bn,nn.BatchNorm2d);gamma.append(float(bn.weight.detach().abs().mean().cpu()))
  contrib[name]=dict(signed=np.zeros(K),squared=np.zeros(K),count=np.zeros(K,dtype=int))
 retained={}
 def capture(name):
  def hook(_,inputs,out):out.retain_grad();retained[name]=out
  return hook
 for name,module in stage_modules(model):handles.append(module.register_forward_hook(capture(name)))
 try:
  for x,labels in loader:
   model.zero_grad(set_to_none=True);z=model(x.to(device));nn.functional.cross_entropy(z,labels.to(device),reduction='sum').backward()
   yy=labels.numpy()
   for name,h in retained.items():
    t=(h.detach()*h.grad.detach()).sum((2,3));sg=t.sum(1).cpu().numpy();sq=t.square().mean(1).cpu().numpy();a=contrib[name]
    for k in range(K):
     a['signed'][k]+=sg[yy==k].sum();a['squared'][k]+=sq[yy==k].sum();a['count'][k]+=int((yy==k).sum())
   retained.clear()
 finally:
  for h in handles:h.remove()
  model.zero_grad(set_to_none=True)
 for name,_ in stage_modules(model):
  a=contrib[name];assert (a['count']>0).all();signed.append(float(-(a['signed']/a['count']).mean()));squared.append(float((a['squared']/a['count']).mean()))
 out=dict(mask_f1_slope=f1,mask_ce_slope=ce,taylor_signed_slope=signed,taylor_squared_importance=squared,bn_gamma=gamma,baseline_metrics=base,masks=masks,sample_count=len(y),classes=classes,scope='Diagnostic subset only; smooth-loss signed slopes and ordinal importance are adaptations to width decisions.')
 assert all(np.isfinite(out[k]).all() for k in ['mask_f1_slope','mask_ce_slope','taylor_signed_slope','taylor_squared_importance','bn_gamma'])
 return out,logits,y
