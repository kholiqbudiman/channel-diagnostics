"""Synthetic resource checks, never scientific outcomes or training."""
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import importlib.util,time,json,copy
root=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('admission',root/'scheduler.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
active=[dict(gpu=g,pid=100+g*4+i,epoch=2) for g in [0,1] for i in range(4)]
base=dict(state='RUNNING',total=75,completed=0,active=active,failures=[])
rows='0, GPU-a, 3000, 81920, 50\n1, GPU-b, 3000, 81920, 65\n'
procs=''.join(f"GPU-{'a' if r['gpu']==0 else 'b'}, {r['pid']}\n" for r in active)
def run(status=None,cost=None,devices=rows,processes=procs,residents=[],stale=False):
 with patch.object(m,'read',side_effect=[status or copy.deepcopy(base),cost or dict(state='WAITING',completed=6)]),patch.object(m.os,'getloadavg',return_value=(0.,0.,0.)),patch.object(m.os,'cpu_count',return_value=80),patch.object(Path,'stat',return_value=SimpleNamespace(st_mtime=time.time()-(120 if stale else 0))),patch.object(Path,'exists',return_value=True),patch.object(Path,'read_text',return_value='MemAvailable: 67108864 kB\n'),patch.object(m.subprocess,'check_output',side_effect=[devices,processes]):return m.available(residents)[0]
checks=[]
def check(name,ok):assert ok,name;checks.append(name)
check('Spare capacity with four DKD workers perGPU admitted',run()==0)
check('Isolated profiling blocks ACRE',run(cost=dict(state='RUNNING')) is None)
check('Cost failure blocks ACRE',run(cost=dict(state='FAILED')) is None)
check('Stale DKD telemetry blocks ACRE',run(stale=True) is None)
b=copy.deepcopy(base);b['active']=[r for r in b['active'] if r['pid'] not in [103,107]]
check('DKD refill priority blocks ACRE',run(status=b) is None)
b=copy.deepcopy(base);b['failures']=['job']
check('DKD failure blocks ACRE',run(status=b) is None)
check('Unknown CUDA processes block ACRE',run(processes=procs+'GPU-a, 999\nGPU-b, 998\n') is None)
check('One ACRE worker perGPU enforced',run(residents=[dict(gpu=0),dict(gpu=1)]) is None)
check('High GPU utilization blocks admission',run(devices=rows.replace(', 50',', 90').replace(', 65',', 90')) is None)
check('Low VRAM blocks admission',run(devices=rows.replace('81920','4096')) is None)
b=dict(state='COMPLETE',total=75,completed=75,active=[],failures=[])
check('Completed DKD waits for isolated profiles',run(status=b) is None)
check('Completed DKD and15profiles permits emptyGPU',run(status=b,cost=dict(state='COMPLETE',completed=15),processes='')==0)
(root/'ADMISSION_CHECKS.json').write_text(json.dumps(dict(state='PASS',checks=checks),indent=2)+'\n')
print('PASS',len(checks),'admission checks')
