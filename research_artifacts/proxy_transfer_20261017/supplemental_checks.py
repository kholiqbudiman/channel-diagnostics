"""Pre-GO numerical checks on scratch weights and fit images only."""
from pathlib import Path
import json,hashlib
import numpy as np,torch
from torch.utils.data import DataLoader
from models import build_model
from diagnostics import profile_model
from scores import predictions,choices
import reference as e
ROOT=Path(__file__).resolve().parent
m=json.loads((ROOT/'data_6301.json').read_text());cfg=json.loads((ROOT/'PROTOCOL.json').read_text())['configs'];ridge=json.loads((ROOT/'ridge.json').read_text());samples=[tuple(x) for x in m['samples']]
torch.set_num_threads(2);ids=[next(i for i in m['split']['fit'] if samples[i][1]==k) for k in range(47)];ds=e.Images(Path(m['root']),samples,ids)
torch.manual_seed(19817);model=build_model(cfg[0],47);before={k:v.clone() for k,v in model.state_dict().items()};fake=[]
for j,w in enumerate([16,24,24,40,64]):
 for seed in [5101,5102,5103]:fake.append(dict(stage=f'block{j+1}',removed_fraction=max(1,round(.1*w))/w,macro_f1_delta_pp=0.,class_loss_delta=[0.]*47))
a,_,_=profile_model(model,DataLoader(ds,batch_size=47,shuffle=False),m['classes'],torch.device('cpu'),stored_masks=fake)
b,_,_=profile_model(model,DataLoader(ds,batch_size=11,shuffle=False),m['classes'],torch.device('cpu'),stored_masks=fake)
assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
for k in ['taylor_signed_slope','taylor_squared_importance','bn_gamma']:assert np.allclose(a[k],b[k],atol=1e-7,rtol=2e-3),(k,a[k],b[k])
for seed in [3901,3902,3903]:
 torch.manual_seed(seed);base=build_model(cfg[0],47).state_dict()
 for c in cfg[1:]:
  torch.manual_seed(seed);other=build_model(c,47).state_dict();assert all(torch.equal(v,other[n]) for n,v in base.items() if n.startswith(('stem.','block1.','block2.','block3.')))
assert not(ROOT/'runs').exists() and not(ROOT/'GO.json').exists()
assert all(x['split']['evaluation']!=x['split']['diagnostic'] for x in [json.loads((ROOT/f'data_{s}.json').read_text()) for s in [6301,6302,6303]])
receipt=dict(state='PASS',profile_parameters_and_BN_buffers_unchanged=True,gradient_scores_batch_size_invariant=True,early_tensor_pairs_checked=18,finite_difference_checks='See preflight.json',test_data_access='Fit images only; no evaluation inference',training_outcomes_available=0,check_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(ROOT/'SUPPLEMENTAL_CHECKS.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
