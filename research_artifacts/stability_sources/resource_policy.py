"""Operational parallel queue for the frozen ACRE pilot.
Two workers per DKD-free GPU, first-epoch admission, no selective reruns.
"""
from pathlib import Path
from datetime import datetime, timezone
import fcntl, hashlib, json, os, subprocess, sys, time, traceback
HERE=Path(__file__).resolve().parent
DKD_STATUS=Path('/home/kholiq/KD/DKD11_benchmark/phase3_primary_n50_20261009/operations/status.json')
OPS=HERE/'parallel_queue_20261016'
PER_GPU=2


def choose_gpu(status,devices,processes,active):
    if status.get('state') not in {'RUNNING','COMPLETED'} or status.get('queued')!=0 or status.get('failure_count')!=0:
        return None,'DKD queue, state or failure requires priority'
    if status.get('final_grid_total')!=250 or not isinstance(status.get('active'),list):
        return None,'Unverified DKD inventory'
    if any(not isinstance(x,dict) or not isinstance(x.get('gpu'),int) for x in status['active']):
        return None,'Invalid DKD GPU assignments'
    if status['state']=='COMPLETED' and (status['active'] or status.get('scientific_grid_completed')!=250):
        return None,'Inconsistent completed DKD inventory'
    reserved={x['gpu'] for x in status['active']}
    options=[]
    for g in devices:
        gpu=g['index']
        if gpu in reserved: continue
        residents=[x for x in active if x['gpu']==gpu]
        own={x['pid'] for x in residents}
        cuda={x['pid'] for x in processes if x['uuid']==g['uuid']}
        if cuda-own: continue
        if len(residents)>=PER_GPU or g['total_mib']-g['memory_mib']<4096: continue
        if not residents:
            if cuda or g['memory_mib']>=1000 or g['util_percent']>=10: continue
        else:
            if any(x.get('epoch',0)<1 for x in residents): continue
            if g['util_percent']>=85: continue
        options.append((len(residents),g['memory_mib'],gpu))
    if options: return min(options)[2],'GPU free of DKD/other CUDA users; ACRE admission passed'
    return None,'Wait for a DKD-free GPU, first epoch, or available ACRE capacity'


def gpu_snapshot():
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    devices=[]
    for line in rows.splitlines():
        if not line.strip(): continue
        index,uuid,memory,total,util=[x.strip() for x in line.split(',')]
        devices.append(dict(index=int(index),uuid=uuid,memory_mib=float(memory),total_mib=float(total),util_percent=float(util)))
    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    processes=[]
    for line in rows.splitlines():
        if not line.strip(): continue
        uuid,pid=[x.strip() for x in line.split(',')]
        processes.append(dict(uuid=uuid,pid=int(pid)))
    return devices,processes


def available(active):
    try:
        status=json.loads(DKD_STATUS.read_text())
        if status.get('state')=='RUNNING' and time.time()-DKD_STATUS.stat().st_mtime>90:
            return None,'DKD telemetry stale'
        if os.getloadavg()[0]>.8*(os.cpu_count() or 1):
            return None,'CPU load threshold; defer a new worker'
        devices,processes=gpu_snapshot()
        return choose_gpu(status,devices,processes,active)
    except (OSError,ValueError,KeyError,TypeError,subprocess.SubprocessError):
        return None,'Telemetry unavailable; hold new ACRE launches'


def main():
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(HERE))
    from experiment import atomic_json,verify_lock
    OPS.mkdir(exist_ok=True)
    guard=(HERE/'scheduler.lock').open('a')
    fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify_lock()
    protocol=json.loads((HERE/'protocol.json').read_text())
    jobs=protocol['jobs']
    if len(jobs)!=36 or len({j['id'] for j in jobs})!=36 or protocol['max_workers']!=1:
        raise RuntimeError('Frozen inventory changed; operational amendment expects original max_workers=1')
    claims=OPS/'claims';claims.mkdir(exist_ok=True)
    if list(claims.glob('*.json')): raise RuntimeError('Existing claims require explicit reconciliation; no automatic retries')
    active={};completed=[];failures=[];queue=[];handles={}
    for j in jobs:
        out=HERE/'runs'/j['id']
        if (out/'result.json').exists(): completed.append(j['id'])
        elif (out/'failure.json').exists(): failures.append(j['id'])
        elif out.exists() and any(out.iterdir()): raise RuntimeError('Partial existing run requires reconciliation: '+j['id'])
        else: queue.append(j)
    reason='Starting verified parallel queue'
    def publish(state=None):
        view=[]
        for jid,a in active.items():
            epoch=0;p=HERE/'runs'/jid/'progress.json'
            if p.exists(): epoch=int(json.loads(p.read_text()).get('epoch',0))
            a['epoch']=epoch
            view.append({k:a[k] for k in ['job','gpu','pid','epoch','started_utc']})
        payload={'state':state or ('RUNNING' if active else 'WAITING_FOR_IDLE_GPU'),'reason':reason,
            'pid':os.getpid(),'jobs_total':len(jobs),'jobs_completed':len(completed),'active':view,
            'queued':len(queue),'next_job':queue[0]['id'] if queue else None,
            'failures':failures,'failure_count':len(failures),'per_gpu_limit':PER_GPU,'parallel_limit':2*PER_GPU,
            'timestamp_unix':time.time(),'operational_amendment':'parallel_queue_20261016/OPERATIONS_AMENDMENT.json'}
        atomic_json(HERE/'status.json',payload)
        atomic_json(OPS/'status.json',payload)
        return payload
    try:
        publish()
        while queue or active:
            for jid,a in list(active.items()):
                proc=a['process'];code=proc.poll()
                if code is None: continue
                handles.pop(jid).close()
                out=HERE/'runs'/jid
                if code!=0 or not (out/'result.json').exists():
                    failures.append(jid)
                    atomic_json(out/'failure.json',{'exit_code':code,'policy':'No automatic retry; preserve run and inspect cause'})
                else: completed.append(jid)
                atomic_json(OPS/(jid+'_completion.json'),{'job':jid,'gpu':a['gpu'],'pid':a['pid'],'exit_code':code,'completed_utc':datetime.now(timezone.utc).isoformat()})
                del active[jid]
            publish()
            if failures:
                reason='Failure preserved; hold new launches and let active workers finish'
                if not active: break
            elif queue:
                gpu,reason=available(list(active.values()))
                if gpu is not None:
                    # Recheck immediately before the unique claim/launch operation.
                    confirm,_=available(list(active.values()))
                    if confirm==gpu:
                        job=queue[0];jid=job['id'];out=HERE/'runs'/jid
                        claim=claims/(jid+'.json')
                        with claim.open('x') as f:
                            json.dump({'job':jid,'gpu':gpu,'claimed_utc':datetime.now(timezone.utc).isoformat()},f)
                        out.mkdir(parents=True,exist_ok=True)
                        handle=(out/'stdout.log').open('a')
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
                        try:
                            proc=subprocess.Popen([sys.executable,'-B',str(HERE/'experiment.py'),'--job-id',jid],cwd=HERE,env=env,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT)
                        except Exception:
                            handle.close();raise
                        queue.pop(0)
                        a={'job':jid,'gpu':gpu,'pid':proc.pid,'process':proc,'epoch':0,'started_utc':datetime.now(timezone.utc).isoformat()}
                        active[jid]=a;handles[jid]=handle
                        atomic_json(claim,{k:a[k] for k in ['job','gpu','pid','started_utc']})
                        print('RUNNING',jid,'GPU',gpu,flush=True)
            publish()
            time.sleep(5)
        reason='Finite pilot finished' if not failures else reason
        publish('COMPLETE_WITH_FAILURES' if failures else 'COMPLETE')
    except Exception:
        reason='Scheduler error; active workers preserved, no automatic retries'
        (OPS/'scheduler_error.log').write_text(traceback.format_exc())
        publish('FAILED')
        raise


if __name__=='__main__': main()
