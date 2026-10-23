"""Two-stage, outcome-withheld63-run cohort; DKD/cost-priority admission."""
from pathlib import Path
from datetime import datetime,timezone
import os,sys,json,time,subprocess,hashlib,fcntl,traceback
from admission import available
from train import verify,save
ROOT=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 guard=(ROOT/'scheduler.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);verify()
 jobs=read(ROOT/'jobs.json');assert len(jobs)==63;assert not(ROOT/'runs').exists();(ROOT/'runs').mkdir();(ROOT/'claims').mkdir()
 done=[];active={};failed=[];handles={};queue=list(jobs);reason='Starting';barrier=False
 def publish(state=None):
  view=[]
  for jid,r in active.items():
   p=ROOT/'runs'/jid/'progress.json';ep=read(p).get('epoch',0) if p.exists() else 0
   view.append(dict(job=jid,pid=r['pid'],gpu=r['gpu'],epoch=ep))
  save(ROOT/'status.json',dict(state=state or ('RUNNING' if active else 'WAITING'),reason=reason,total=63,completed=len(done),queued=len(queue),active=view,failures=failed,stage='CANDIDATES' if barrier else 'BASE_AND_DIAGNOSTICS',prediction_barrier=barrier,utc=datetime.now(timezone.utc).isoformat(),per_gpu_cap=1))
 try:
  while queue or active:
   for jid,r in list(active.items()):
    code=r['process'].poll()
    if code is None:continue
    handles.pop(jid).close();folder=ROOT/'runs'/jid
    if code or not(folder/'training_complete.json').exists():failed.append(jid);save(folder/'failure.json',dict(exit_code=code,policy='No auto retry'))
    else:done.append(jid)
    del active[jid]
   if (ROOT/'old_diagnostic_status.json').exists() and read(ROOT/'old_diagnostic_status.json')['state']=='FAILED' and 'OLD_DIAGNOSTICS' not in failed:failed.append('OLD_DIAGNOSTICS')
   basejobs=[j for j in jobs if j['model']['name']=='base']
   if not barrier and all(j['id'] in done for j in basejobs):
    assert not any((ROOT/'runs'/j['id']).exists() for j in jobs if j['model']['name']!='base')
    pred={}
    for j in basejobs:
     f=ROOT/'runs'/j['id'];a=read(f/'predictions_committed.json');assert a['checkpoint_sha256']==sha(f/'best.pt');assert a['diagnostic_sha256']==sha(f/'diagnostic.json');pred[j['id']]=sha(f/'predictions_committed.json')
    save(ROOT/'PREDICTION_BARRIER.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),predictions=pred,state='ALL9_BASE_PREDICTIONS_COMMITTED_BEFORE_NONBASE'));barrier=True
   publish()
   if failed:
    reason='Failure retained; hold all new launches'
    if not active:break
   elif queue:
    eligible=next((j for j in queue if barrier or j['model']['name']=='base'),None)
    if eligible is None:reason='Waiting for all9 Base diagnostics/prediction commits'
    else:
     gpu,reason=available(list(active.values()))
     if gpu is not None:
      verify();confirm,_=available(list(active.values()))
      if confirm==gpu:
       j=eligible;queue.remove(j);folder=ROOT/'runs'/j['id'];assert not folder.exists();folder.mkdir();out=(folder/'stdout.log').open('x')
       env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
       save(ROOT/'claims'/(j['id']+'.json'),dict(job=j,gpu=gpu,created_utc=datetime.now(timezone.utc).isoformat(),prediction_barrier_sha256=sha(ROOT/'PREDICTION_BARRIER.json') if barrier else None))
       proc=subprocess.Popen([sys.executable,'-B',str(ROOT/'train.py'),'--job-id',j['id']],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT)
       active[j['id']]=dict(pid=proc.pid,gpu=gpu,process=proc);handles[j['id']]=out;print('START',j['id'],'GPU',gpu,flush=True)
   publish();time.sleep(10)
  reason='All63 training completed; evaluation released to complete-cohort analysis' if not failed else reason;publish('COMPLETE' if not failed else 'COMPLETE_WITH_FAILURES')
 except Exception:
  reason='Scheduler exception; preserve active jobs';(ROOT/'scheduler_failure.log').write_text(traceback.format_exc());publish('FAILED');raise
if __name__=='__main__':main()
