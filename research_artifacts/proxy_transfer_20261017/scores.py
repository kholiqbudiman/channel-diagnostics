"""Fixed diagnostic-to-width mappings. Ordinal scores are not F1 predictions."""
import numpy as np
METHODS=['mask_f1','mask_macro_ce','taylor_signed','taylor_squared','bn_gamma','architecture_ridge','choose_base']
def delta(ch): return np.log(np.array(ch,float)/[16,24,40,64])[[0,1,1,2,3]]
def predictions(profile,configs,ridge):
 out={}
 for c in configs:
  d=delta(c['channels']);x=np.log(np.array(c['channels'],float)[2:]/[40,64])/np.array(ridge['rms'])
  out[c['name']]={
   'mask_f1':float(d@np.array(profile['mask_f1_slope'])),
   'mask_macro_ce':float(d@np.array(profile['mask_ce_slope'])),
   'taylor_signed':float(d@np.array(profile['taylor_signed_slope'])),
   'taylor_squared':float(d@np.array(profile['taylor_squared_importance'])),
   'bn_gamma':float(d@np.array(profile['bn_gamma'])),
   'architecture_ridge':float(x@np.array(ridge['coefficients'])),
   'choose_base':0.}
 assert all(abs(v)<1e-12 for v in out['base'].values())
 return out
def choices(pred):
 names=['base','joint_c_grow_d_shrink','joint_c_shrink_d_grow']
 return {m:('base' if m=='choose_base' else max(names,key=lambda n:pred[n][m])) for m in METHODS}
