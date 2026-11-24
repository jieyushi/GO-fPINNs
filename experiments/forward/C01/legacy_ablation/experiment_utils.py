import json,time,platform,random,csv
from pathlib import Path
import numpy as np
import torch

def seed_all(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
def synchronize(device):
 if device.type=='cuda':torch.cuda.synchronize(device)
def metrics(pred,target):
 p=pred.detach().double();y=target.detach().double();err=p-y
 pc=torch.complex(p[:,0],p[:,1]);yc=torch.complex(y[:,0],y[:,1])
 amp=yc.abs();mask=amp>max(float(amp.max())*1e-3,1e-8)
 return {'relative_l2':float(torch.linalg.vector_norm(err)/torch.linalg.vector_norm(y).clamp_min(1e-30)),
 'real_relative_l2':float(torch.linalg.vector_norm(err[:,0])/torch.linalg.vector_norm(y[:,0]).clamp_min(1e-30)),
 'imag_relative_l2':float(torch.linalg.vector_norm(err[:,1])/torch.linalg.vector_norm(y[:,1]).clamp_min(1e-30)),
 'amplitude_relative_l2':float(torch.linalg.vector_norm(pc.abs()-amp)/torch.linalg.vector_norm(amp).clamp_min(1e-30)),
 'phase_mae_radians':float(torch.angle(pc[mask]*yc[mask].conj()).abs().mean()) if mask.any() else None,
 'complex_rmse':float((err.square().sum(1).mean()).sqrt()),'max_abs_error':float((pc-yc).abs().max())}
def save_json(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2))
def save_csv(path,rows):
 if not rows:return
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with Path(path).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def prepare_out(path):
 p=Path(path);p.mkdir(parents=True,exist_ok=True)
 if (p/'config.json').exists():raise FileExistsError(f'Run already exists: {p}; choose a new output directory')
 return p
def environment(device):
 return {'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'device':str(device),
 'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None}
def timed_inference(func,device):
 with torch.no_grad():
  for _ in range(3):func()
  synchronize(device);t=time.perf_counter()
  for _ in range(10):func()
  synchronize(device)
 return (time.perf_counter()-t)/10
