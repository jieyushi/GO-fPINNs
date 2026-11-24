import contextlib
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import numpy as np
import torch


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def cpu_tree(v):
    if isinstance(v,torch.Tensor):return v.detach().cpu().clone()
    if isinstance(v,dict):return {k:cpu_tree(x) for k,x in v.items()}
    if isinstance(v,list):return [cpu_tree(x) for x in v]
    if isinstance(v,tuple):return tuple(cpu_tree(x) for x in v)
    return v


def capture_rng():
    s=np.random.get_state()
    return dict(python=random.getstate(),numpy=[s[0],s[1].tolist(),s[2],s[3],s[4]],
                torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(r):
    random.setstate(r['python']);s=r['numpy']
    np.random.set_state((s[0],np.asarray(s[1],dtype=np.uint32),s[2],s[3],s[4]))
    torch.set_rng_state(r['torch'].cpu())
    if r['cuda'] is not None:
        if not torch.cuda.is_available():raise ValueError('CUDA checkpoint needs CUDA for exact continuation.')
        torch.cuda.set_rng_state_all([x.cpu() for x in r['cuda']])


def digest(v):
    h=hashlib.sha256()
    def walk(x):
        if isinstance(x,torch.Tensor):
            h.update(str(x.dtype).encode());h.update(str(tuple(x.shape)).encode())
            h.update(x.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str):h.update(str(k).encode());walk(x[k])
        elif isinstance(x,(tuple,list)):
            for y in x:walk(y)
        else:h.update(repr(x).encode())
    walk(v);return h.hexdigest()


def clean(v):
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [clean(x) for x in v]
    if isinstance(v,np.generic):return clean(v.item())
    if isinstance(v,float) and not math.isfinite(v):return None
    return v


def atomic_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(clean(data),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(tmp,path)


def atomic_torch(path,data):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    torch.save(data,tmp);os.replace(tmp,path)


def write_csv(path,rows):
    if not rows:return
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');keys=list(dict.fromkeys(k for row in rows for k in row))
    with tmp.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows([clean(r) for r in rows])
    os.replace(tmp,path)


@contextlib.contextmanager
def output_lock(path):
    import fcntl
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    with (path/'.run.lock').open('a+') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another process owns '+str(path))
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)


def independent_points(seed,count):
    from protocol import XL,XR
    rng=np.random.default_rng(seed)
    return np.column_stack([rng.uniform(0,1,count),rng.uniform(XL,XR,count)]).astype(np.float32)


def metrics(pred,true):
    pred=np.asarray(pred,dtype=np.complex128);true=np.asarray(true,dtype=np.complex128)
    error=np.abs(pred-true);den=max(np.linalg.norm(true),1e-30)
    phase=np.abs(np.angle(pred*np.conj(true)))
    mask=np.abs(true)>max(1e-3*np.abs(true).max(),1e-8)
    return dict(relative_l2=float(np.linalg.norm(error)/den),
                amplitude_relative_l2=float(np.linalg.norm(np.abs(pred)-np.abs(true))/den),
                phase_mae_radians=float(phase.mean()),
                phase_mae_masked_radians=float(phase[mask].mean()) if mask.any() else None,
                mean_absolute_complex_error=float(error.mean()),max_absolute_complex_error=float(error.max()))
