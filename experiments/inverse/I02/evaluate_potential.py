"Collect one-dimensional potential errors and export checkpoint predictions."
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from common import atomic_json,digest,write_csv
from train_potential import FORMAT,load_for_prediction,predict_numpy,potential_numpy


def collect(root):
    root=Path(root);rows=[];groups={}
    for path in sorted(root.rglob('summary.json')):
        s=json.loads(path.read_text())
        if s.get('format')!=FORMAT:continue
        c=json.loads((path.parent/'config.json').read_text());a=c['args'];compare=dict(a)
        for key in ['output','resume','seed']:compare.pop(key,None)
        key=digest(compare)
        row=dict(problem=a['problem'],method=a['method'],supervised_points=a['supervised_points'],seed=a['seed'],
                 configuration=key,smoke=a['smoke'],noise_level=a['noise_level'],**s['metrics'],
                 total_updates=s['total_updates'],optimization_seconds=s['resources']['optimization_seconds'])
        rows.append(row);groups.setdefault((a['problem'],a['method'],a['supervised_points'],key),[]).append(row)
    if not rows:raise FileNotFoundError('No completed I02 runs.')
    write_csv(root/'potential_results.csv',rows);aggregates=[]
    for (problem,method,count,key),values in groups.items():
        row=dict(problem=problem,method=method,supervised_points=count,configuration=key,seeds=[v['seed'] for v in values])
        for metric in ['eta_relative_l2','eta_mae','solution_relative_l2','optimization_seconds']:
            array=np.asarray([v[metric] for v in values]);row[metric+'_mean']=float(array.mean())
            row[metric+'_sample_std']=float(array.std(ddof=1)) if len(array)>1 else None
        aggregates.append(row)
    atomic_json(root/'potential_results.json',dict(runs=rows,aggregates=aggregates,
                note='Numerical/analytic field references and eta truth are diagnostic; no error-based selection.'))
    print('Collected %d completed runs.'%len(rows),flush=True);return rows


def export_run(run,device='cpu'):
    run=Path(run);state,s=load_for_prediction(run/'final.pt',device)
    x=np.linspace(0,np.pi,101);t=np.linspace(0,1,101);tt,xx=np.meshgrid(t,x,indexing='ij')
    uv=predict_numpy(state,np.column_stack([tt.ravel(),xx.ravel()]))
    out=run/'evaluated';out.mkdir(exist_ok=True)
    np.savez_compressed(out/'predictions.npz',x=x,t=t,field=(uv[:,0]+1j*uv[:,1]).reshape(101,101),potential=potential_numpy(state,x))
    print('Saved '+str(out/'predictions.npz'));return out


def main(argv=None):
    q=argparse.ArgumentParser(description=__doc__);q.add_argument('--root',default='runs/I02_gradient_v2');q.add_argument('--run');q.add_argument('--device',default='cpu')
    a=q.parse_args(argv);torch.set_num_threads(1)
    return export_run(a.run,a.device) if a.run else collect(a.root)


if __name__=='__main__':main()
