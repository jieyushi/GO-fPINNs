"""Post-training errors against the analytic manufactured field or a matching numerical reference."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
import backend as b
from common import atomic_json,metrics,write_csv
from problem import reference_field,SOURCE_MODEL
from protocol import LABELS
from numerical_reference import Reference
from train_largescale import FORMAT,load_for_prediction


def infer(states,domains,tx,device,chunk=8192):
    dtype=states[0]['dtype'];values=[]
    with torch.no_grad():
        for start in range(0,len(tx),chunk):
            loc=torch.tensor(tx[start:start+chunk],device=device,dtype=dtype)
            uv=b.predict(states,domains,loc).cpu().double().numpy()
            values.append(uv[:,0]+1j*uv[:,1])
    return np.concatenate(values)


def main(argv=None):
    q=argparse.ArgumentParser(description=__doc__)
    q.add_argument('--root',type=Path,default=Path('runs/C02_gradient_v2'))
    q.add_argument('--reference',type=Path)
    q.add_argument('--exact-series','--nominal-series',dest='exact_series',action='store_true',help='Use the analytic manufactured reference (the default). --nominal-series is a compatibility alias.')
    q.add_argument('--methods',default='go,fpinn,fl1')
    q.add_argument('--seeds',default='1234,2024,3407')
    q.add_argument('--plot-seed',type=int)
    q.add_argument('--device',default='cpu')
    q.add_argument('--space-points',type=int,default=401)
    q.add_argument('--time-points',type=int,default=201)
    q.add_argument('--allow-smoke',action='store_true')
    a=q.parse_args(argv);methods=a.methods.split(',');seeds=[int(v) for v in a.seeds.split(',')]
    if a.reference and a.exact_series:q.error('Choose an analytic or a numerical reference.')
    if not seeds or len(set(seeds))!=len(seeds) or min(a.space_points,a.time_points)<2:q.error('Invalid seed/grid settings.')
    plot_seed=a.plot_seed if a.plot_seed is not None else seeds[0]
    if plot_seed not in seeds:q.error('Plot seed must be included.')
    if a.reference:
        reference=Reference(a.reference);role='NUMERICAL_MANUFACTURED_SOURCE_REFERENCE'
    else:
        reference=reference_field;role='ANALYTIC_MANUFACTURED_SERIES'
    torch.set_num_threads(1)
    curves=[];results=[];stages=[];resources=[];reference_coords=None;all_smoke=[]
    from protocol import XL,XR
    x=np.linspace(XL,XR,a.space_points,dtype=np.float64);t=np.linspace(0,1,a.time_points,dtype=np.float64)
    xx,tt=np.meshgrid(x,t,indexing='ij');tx=np.column_stack([tt.ravel(),xx.ravel()])
    out=a.root/'evaluated';out.mkdir(parents=True,exist_ok=True)
    fields={'x':x,'t':t,'reference':reference(tx).reshape(xx.shape)}
    for method in methods:
        for seed in seeds:
            run=a.root/method/('seed_'+str(seed))
            summary=json.loads((run/'summary.json').read_text());smoke=summary['status']=='SMOKE_ONLY';all_smoke.append(smoke)
            if summary.get('format')!=FORMAT or summary.get('source_model')!=SOURCE_MODEL:
                raise ValueError('Run used an older or different source model; use newly trained manufactured-source results.')
            if smoke and not a.allow_smoke:raise ValueError('Smoke output needs --allow-smoke.')
            with np.load(run/'evaluation_coordinates.npz') as data:
                val=data['validation'].copy();test=data['test'].copy()
            coords=(val,test)
            if reference_coords is not None and any(not np.array_equal(u,v) for u,v in zip(reference_coords,coords)):
                raise ValueError('Evaluation coordinates differ between methods/seeds.')
            reference_coords=coords
            target_val=reference(val);target_test=reference(test)
            states,saved=load_for_prediction(run/'final.pt',a.device)
            pred=infer(states,saved['domains'],test,a.device)
            results.append(dict(method=method,seed=seed,reference_role=role,**metrics(pred,target_test)))
            resources.append(dict(method=method,seed=seed,updates=summary['total_updates'],
                                  optimization_seconds=summary['resources']['optimization_seconds'],
                                  total_wall_seconds=summary['wall_seconds'],
                                  residual_points_processed=summary['residual_points_processed'],
                                  peak_cuda_allocated_bytes=summary['resources']['peak_training_allocated_bytes']))
            with (run/'training.csv').open(newline='') as f:log=list(csv.DictReader(f))
            for row in log:
                path=run/'monitor_predictions'/('s%d_e%d.npz'%(int(row['stage']),int(row['epoch'])))
                with np.load(path) as data:
                    if int(data['global_step'])!=int(row['global_step']):raise ValueError('Monitor prediction/step mismatch.')
                    uv=data['prediction'];metric=metrics(uv[:,0]+1j*uv[:,1],target_val)
                curves.append(dict(method=method,seed=seed,stage=int(row['stage']),epoch=int(row['epoch']),
                                   global_step=int(row['global_step']),training_loss=float(row['loss']),
                                   validation_relative_l2=metric['relative_l2'],
                                   optimization_seconds=float(row['optimization_seconds']),
                                   interface_loss=float(row['loss_interface'])))
            for stage in summary['stages']:
                checkpoint=run/('stage_%02d_last.pt'%stage['stage'])
                ss,record=load_for_prediction(checkpoint,a.device)
                stage_metric=metrics(infer(ss,record['domains'],test,a.device),target_test)
                stages.append(dict(method=method,seed=seed,stage=stage['stage'],phase=stage['phase'],
                                   parts=stage['parts'],M=stage['M'],N=stage['N'],epochs=stage['epochs'],
                                   reference_role=role,**stage_metric))
            if seed==plot_seed:fields['prediction_'+method]=infer(states,saved['domains'],tx,a.device).reshape(xx.shape)
    if any(all_smoke) and not all(all_smoke):raise ValueError('Do not combine smoke/main results.')
    write_csv(out/'per_seed_metrics.csv',results);write_csv(out/'training_curves.csv',curves)
    write_csv(out/'stage_metrics.csv',stages);write_csv(out/'resources.csv',resources)
    statistics=[]
    for method in methods:
        for key in ['relative_l2','amplitude_relative_l2','phase_mae_radians','mean_absolute_complex_error']:
            values=[row[key] for row in results if row['method']==method]
            statistics.append(dict(method=method,metric=key,seeds=len(values),mean=float(np.mean(values)),
                                   sample_sd=float(np.std(values,ddof=1)) if len(values)>1 else None))
    write_csv(out/'mean_std.csv',statistics)
    np.savez_compressed(out/'plot_fields.npz',**fields)
    atomic_json(out/'evaluation.json',dict(reference_role=role,reference_is_analytic=a.reference is None,source_model=SOURCE_MODEL,
                reference_file=str(a.reference) if a.reference else None,
                reference_mesh_convergence_certified=reference.metadata['mesh_convergence_certified'] if a.reference else None,
                methods=methods,seeds=seeds,plot_seed=plot_seed,smoke_only=all(all_smoke),
                model_selection_uses_test_errors=False,baseline_implementations='C01-defined controlled L1/Fast-L1 coordinate networks'))
    print('Evaluated actual saved checkpoints against '+role)


if __name__=='__main__':main()
