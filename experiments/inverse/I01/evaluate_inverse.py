"Collect inverse-run summaries and optionally export checkpoint field predictions."
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from common import atomic_json,digest,metrics,write_csv
from problem import exact_numpy
from train_inverse import FORMAT,load_for_prediction,predict_numpy,errors


def evaluate_run(run,device='cpu',nodes=101,checkpoint='final.pt'):
    run=Path(run);state,saved=load_for_prediction(run/checkpoint,device)
    x=np.linspace(0,1,nodes);t=np.linspace(0,1,nodes)
    xx,tt=np.meshgrid(x,t,indexing='ij');tx=np.column_stack([tt.ravel(),xx.ravel()])
    uv=predict_numpy(state,tx);pred=uv[:,0]+1j*uv[:,1];ref=exact_numpy(tx)
    out=run/'evaluated';out.mkdir(exist_ok=True)
    np.savez_compressed(out/'fields.npz',x=x,t=t,prediction=pred.reshape(nodes,nodes),reference=ref.reshape(nodes,nodes))
    result=dict(checkpoint=checkpoint,parameters=saved['parameters'],
                parameter_errors=errors(saved['parameters'],saved['args']['unknown']),
                field_metrics=metrics(pred,ref),nodes=[nodes,nodes])
    atomic_json(out/'metrics.json',result);return result


def collect(root,fields=False,device='cpu'):
    root=Path(root);rows=[];groups={}
    for path in sorted(root.rglob('summary.json')):
        summary=json.loads(path.read_text())
        if summary.get('format')!=FORMAT:continue
        run=path.parent;config=json.loads((run/'config.json').read_text());a=config['args']
        case=run.parent.parent.name
        comparable=dict(a)
        for key in ['output','resume','seed']:comparable.pop(key,None)
        key_hash=digest(comparable)
        row=dict(case=case,method=a['method'],unknown=a['unknown'],initial=a['initial'],seed=a['seed'],
                 configuration=key_hash,smoke=a['smoke'],noise_level=a['noise_level'],
                 **summary['parameters'],**summary['parameter_errors'],total_updates=summary['total_updates'],
                 selected_stage=summary['selected_stage'],selected_epoch=summary['selected_epoch'],
                 solution_relative_l2=summary['reference_solution_metrics']['relative_l2'],
                 optimization_seconds=summary['resources']['optimization_seconds'])
        rows.append(row);groups.setdefault((case,a['method'],key_hash),[]).append(row)
        if fields:evaluate_run(run,device)
    if not rows:raise FileNotFoundError('No completed inverse runs under '+str(root))
    write_csv(root/'inverse_results.csv',rows);aggregates=[]
    for (case,method,key_hash),values in groups.items():
        row=dict(case=case,method=method,configuration=key_hash,seeds=[r['seed'] for r in values],count=len(values))
        for name in ['alpha','beta','gamma','unknown_relative_l2','solution_relative_l2','optimization_seconds']:
            a=np.array([r[name] for r in values]);row[name+'_mean']=float(a.mean())
            row[name+'_sample_std']=float(a.std(ddof=1)) if len(a)>1 else None
        aggregates.append(row)
    atomic_json(root/'inverse_results.json',dict(runs=rows,aggregates=aggregates,
                note='Parameter errors are diagnostics, never training/early-stop/model-selection objectives.'))
    print('Collected %d runs under %s'%(len(rows),root),flush=True);return rows


def main(argv=None):
    from protocol import RUN_ROOT
    q=argparse.ArgumentParser(description=__doc__);q.add_argument('--root',default=RUN_ROOT)
    q.add_argument('--run');q.add_argument('--fields',action='store_true');q.add_argument('--device',default='cpu')
    q.add_argument('--nodes',type=int,default=101);q.add_argument('--checkpoint',default='final.pt')
    a=q.parse_args(argv);torch.set_num_threads(1)
    if a.nodes<2:q.error('At least two grid nodes required.')
    return evaluate_run(a.run,a.device,a.nodes,a.checkpoint) if a.run else collect(a.root,a.fields,a.device)


if __name__=='__main__':main()
