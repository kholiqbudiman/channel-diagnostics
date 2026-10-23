from pathlib import Path
import os,time,json,subprocess
SECONDARY=Path("/home/kholiq/KD/DKD11_benchmark/secondary_pairs_20261016/status.json")
COST=Path("/home/kholiq/KD/DKD11_benchmark/cost_recheck_20261016/status.json")
read=lambda p:json.loads(p.read_text())
def available(active):
 try:
  dkd=read(SECONDARY);cost=read(COST)
  if dkd['state'] not in ['RUNNING','COMPLETE']:return None,'DKD status/failure priority'
  if dkd.get('failures'):return None,'DKD failures require inspection'
  if dkd.get('total')!=75 or (dkd['state']=='COMPLETE' and dkd.get('completed')!=75):return None,'DKD inventory mismatch'
  if dkd['state']=='RUNNING' and time.time()-SECONDARY.stat().st_mtime>60:return None,'DKD status stale'
  if cost['state']=='RUNNING':return None,'Isolated cost profile priority'
  if cost['state'] not in ['WAITING','COMPLETE']:return None,'Cost pipeline failure/unknown state requires inspection'
  if dkd['state']=='COMPLETE' and (cost['state']!='COMPLETE' or cost.get('completed')!=15):return None,'Wait for15 isolated profiles before using released GPUs'
  if os.getloadavg()[0]>.8*(os.cpu_count() or 1):return None,'CPU capacity hold'
  mem={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')}
  if mem['MemAvailable:'.rstrip(':')]<32*1024*1024:return None,'Host memory hold'
  devices=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)
  procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True);cuda={}
  for line in procs.splitlines():
   if line.strip():u,p=[v.strip() for v in line.split(',')];cuda.setdefault(u,set()).add(int(p))
  options=[]
  for line in devices.splitlines():
   g,u,used,total,util=[v.strip() for v in line.split(',')];g=int(g)
   if any(a['gpu']==g for a in active):continue
   known=[a for a in dkd.get('active',[]) if a['gpu']==g]
   if dkd['state']=='RUNNING' and (len(known)!=4 or any(a['epoch']<1 or not Path(f"/proc/{a['pid']}").exists() for a in known)):continue
   if cuda.get(u,set())!={a['pid'] for a in known}:continue
   if float(total)-float(used)<8192 or float(util)>=75:continue
   options.append((float(util),float(used),g))
  return (min(options)[2],'Cooperative admission; DKD slots already full') if options else (None,'Wait for full DKD slots and spare capacity')
 except (OSError,ValueError,KeyError,subprocess.SubprocessError):return None,'Telemetry unavailable; hold'
