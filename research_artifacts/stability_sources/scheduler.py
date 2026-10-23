"""Finite stability queue. Never overlaps the isolated DKD post-primary pipeline."""
from pathlib import Path
from datetime import datetime,timezone
import fcntl,hashlib,json,os,subprocess,sys,time,traceback
import resource_policy as policy
HERE=Path(__file__).resolve().parent
POST=Path('/home/kholiq/KD/DKD11_benchmark/post_primary_20261016')
def save(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def verify():
 lock=json.loads((HERE/'COHORT_LOCK.json').read_text())
 for name,expected in lock['sources'].items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==expected,name
 for s,expected in lock['split_locks'].items():
  folder=HERE/f'split_{s}';f=folder/'protocol_lock.json';assert hashlib.sha256(f.read_bytes()).hexdigest()==expected
  for name,h in json.loads(f.read_text())['sources'].items():assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==h,name

def gate():
 try:
  a=json.loads((POST/'status.json').read_text());b=json.loads((POST/'aggregation_status.json').read_text())
  if a.get('state')!='COMPLETED' or b.get('state')!='COMPLETED':return False,'Wait for DKD post-primary measurements and aggregation to complete'
  if b.get('metric_records')!=250 or b.get('profiles')!=5:return False,'DKD aggregate inventory is not complete'
  return True,'DKD post-primary complete'
 except (OSError,ValueError):return False,'DKD post-primary telemetry unavailable'
def main():
 lock=(HERE/'scheduler.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify()
 jobs=json.loads((HERE/'job_inventory.json').read_text());assert len(jobs)==len({j['id'] for j in jobs})==72
 claims=HERE/'claims';claims.mkdir(exist_ok=True);assert not list(claims.iterdir()),'Existing claims need reconciliation'
 active={};done=[];failed=[];queue=list(jobs);reason='Starting';handles={}
 def publish(state=None):
  view=[]
  for jid,a in active.items():
   progress=Path(a['jobdef']['folder'])/'runs'/a['jobdef']['inner_id']/'progress.json'
   a['epoch']=json.loads(progress.read_text())['epoch'] if progress.exists() else 0
   view.append({k:a[k] for k in ['job','gpu','pid','epoch']})
  save(HERE/'status.json',{'state':state or ('RUNNING' if active else 'WAITING'),'reason':reason,'pid':os.getpid(),'jobs_total':72,'jobs_completed':len(done),'queued':len(queue),'active':view,'failures':failed,'timestamp_utc':datetime.now(timezone.utc).isoformat(),'per_gpu_limit':2,'parallel_limit':4})
 try:
  while queue or active:
   for jid,a in list(active.items()):
    code=a['process'].poll()
    if code is None:continue
    handles.pop(jid).close();j=a['jobdef'];folder=Path(j['folder'])/'runs'/j['inner_id']
    if code or not (folder/'result.json').exists():failed.append(jid);save(folder/'failure.json',{'exit_code':code,'policy':'No automatic retries; preserve run'})
    else:done.append(jid)
    save(HERE/'claims'/f'{jid}_completion.json',{'exit_code':code,'utc':datetime.now(timezone.utc).isoformat()});del active[jid]
   publish()
   if failed:
    reason='Failure preserved; no new launches'
    if not active:break
   elif queue:
    ready,reason=gate()
    if ready:
     gpu,reason=policy.available(list(active.values()))
     if gpu is not None:
      verify();ready,_=gate();again,_=policy.available(list(active.values()))
      if ready and again==gpu:
       j=queue[0];folder=Path(j['folder'])/'runs'/j['inner_id'];assert not folder.exists()
       with (claims/f'{j["id"]}.json').open('x') as f:json.dump({'job':j,'gpu':gpu},f)
       folder.mkdir(parents=True);out=(folder/'stdout.log').open('x')
       env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
       proc=subprocess.Popen([sys.executable,'-B',str(Path(j['folder'])/'experiment.py'),'--job-id',j['inner_id']],cwd=j['folder'],env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT)
       active[j['id']]={'job':j['id'],'gpu':gpu,'pid':proc.pid,'epoch':0,'process':proc,'jobdef':j};handles[j['id']]=out;queue.pop(0)
       print('RUNNING',j['id'],'GPU',gpu,flush=True)
   publish();time.sleep(5)
  reason='Finite stability cohort finished' if not failed else reason;publish('COMPLETE' if not failed else 'COMPLETE_WITH_FAILURES')
 except Exception:
  reason='Scheduler error: preserve active workers; no automatic retry';(HERE/'scheduler_error.log').write_text(traceback.format_exc());publish('FAILED');raise
if __name__=='__main__':main()
