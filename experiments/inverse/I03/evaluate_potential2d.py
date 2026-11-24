"Collect two-dimensional potential errors and export checkpoint predictions."
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from common import atomic_json,digest,metrics,write_csv
from problem import eta_truth,exact_numpy
from train_potential2d import FORMAT,load_for_prediction,potential_numpy,predict_numpy


def collect(root):
    root=Path(root);rows=[];groups={}
    for path in sorted(root.rglob('summary.json')):
        s=json.loads(path.read_text())
        if s.get('format')!=FORMAT:continue
        a=json.loads((path.parent/'config.json').read_text())['args'];compare=dict(a)
        for key in ['output','resume','seed']:compare.pop(key,None)
        key=digest(compare)
        row=dict(method=a['method'],supervised_points=a['supervised_points'],seed=a['seed'],configuration=key,
                 smoke=a['smoke'],noise_level=a['noise_level'],**s['metrics'],total_updates=s['total_updates'],
                 optimization_seconds=s['resources']['optimization_seconds'],training_wall_seconds=s['wall_seconds'],
                 peak_training_allocated_mib=(None if s['resources']['peak_training_allocated_bytes'] is None else s['resources']['peak_training_allocated_bytes']/2**20))
        rows.append(row);groups.setdefault((a['method'],a['supervised_points'],key),[]).append(row)
    if not rows:raise FileNotFoundError('No completed I03 runs.')
    write_csv(root/'potential2d_results.csv',rows);aggregates=[]
    for (method,count,key),values in groups.items():
        row=dict(method=method,supervised_points=count,configuration=key,seeds=[v['seed'] for v in values])
        for metric in ['eta_relative_l2','eta_mae','solution_relative_l2','optimization_seconds','training_wall_seconds']:
            array=np.asarray([v[metric] for v in values]);row[metric+'_mean']=float(array.mean())
            row[metric+'_sample_std']=float(array.std(ddof=1)) if len(array)>1 else None
        aggregates.append(row)
    atomic_json(root/'potential2d_results.json',dict(runs=rows,aggregates=aggregates,
                note='eta evaluated on a fixed full Cartesian grid; errors never select checkpoints or stop training.'))
    print('Collected %d completed runs.'%len(rows),flush=True);return rows


def export_run(run,device='cpu',nodes=41):
    run=Path(run);state,s=load_for_prediction(run/'final.pt',device)
    axis=np.linspace(0,1,nodes);tt,xx,yy=np.meshgrid(axis,axis,axis,indexing='ij')
    tx=np.column_stack([tt.ravel(),xx.ravel(),yy.ravel()]);uv=predict_numpy(state,tx)
    pred=uv[:,0]+1j*uv[:,1];ref=exact_numpy(tx)
    out=run/'evaluated';out.mkdir(exist_ok=True)
    np.savez_compressed(out/'field_predictions.npz',t=axis,x=axis,y=axis,prediction=pred.reshape((nodes,)*3),reference=ref.reshape((nodes,)*3))
    x,y=np.meshgrid(axis,axis,indexing='ij');xy=np.column_stack([x.ravel(),y.ravel()]);eta=potential_numpy(state,xy);truth=eta_truth(xy)
    np.savez_compressed(out/'potential_prediction.npz',x=axis,y=axis,prediction=eta.reshape(nodes,nodes),reference=truth.reshape(nodes,nodes))
    atomic_json(out/'metrics.json',dict(checkpoint='final.pt',field_grid_nodes=nodes,solution_metrics=metrics(pred,ref),
                eta_relative_l2=float(np.linalg.norm(eta-truth)/np.linalg.norm(truth))))
    print('Saved '+str(out));return out


def main(argv=None):
    q=argparse.ArgumentParser(description=__doc__);q.add_argument('--root',default='runs/I03_gradient_v2');q.add_argument('--run');q.add_argument('--device',default='cpu');q.add_argument('--nodes',type=int,default=41)
    a=q.parse_args(argv);torch.set_num_threads(1)
    if a.nodes<2:q.error('At least two nodes required.')
    return export_run(a.run,a.device,a.nodes) if a.run else collect(a.root)


if __name__=='__main__':main()
