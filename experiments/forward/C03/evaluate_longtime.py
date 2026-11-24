"Export full-interval and stage-wise errors from a trained long-time model."
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from common import atomic_json,metrics,write_csv
from problem import exact_numpy
from protocol import T
from train_longtime import load_for_prediction,prediction_numpy
from physics import continuity_report


def evaluate(run,device='cpu',nx=201,nt=801,checkpoint='final.pt'):
    run=Path(run);out=run/'evaluated';out.mkdir(exist_ok=True)
    config=json.loads((run/'config.json').read_text());summary=json.loads((run/'summary.json').read_text())
    coords=np.load(run/'evaluation_coordinates.npz',allow_pickle=False)['test']
    target=exact_numpy(coords);stage_rows=[]
    for item in summary['stages']:
        sid=item['stage'];states,saved=load_for_prediction(run/('stage_%02d_last.pt'%sid),device)
        uv=prediction_numpy(states,saved['windows'],coords)
        row=dict(stage=sid,windows=len(states),global_step=item['global_step'],
                 cumulative_optimization_seconds=item['cumulative_optimization_seconds'],
                 **metrics(uv[:,0]+1j*uv[:,1],target))
        stage_rows.append(row)
    states,saved=load_for_prediction(run/checkpoint,device)
    test_uv=prediction_numpy(states,saved['windows'],coords)
    x=np.linspace(0,1,nx);t=np.linspace(0,T,nt)
    xx,tt=np.meshgrid(x,t,indexing='ij');tx=np.column_stack([tt.ravel(),xx.ravel()])
    uv=prediction_numpy(states,saved['windows'],tx);true=exact_numpy(tx)
    field=uv[:,0]+1j*uv[:,1]
    np.savez_compressed(out/'full_interval.npz',x=x,t=t,prediction=field.reshape(nx,nt),reference=true.reshape(nx,nt))
    report=dict(format=saved['format'],checkpoint=checkpoint,selected_stage=saved['stage'],selected_window=saved['window'],
                selected_windows=saved['windows'],smoke=config['args']['smoke'],seed=config['args']['seed'],
                test_metrics=metrics(test_uv[:,0]+1j*test_uv[:,1],target),grid_metrics=metrics(field,true),
                grid_nodes=[nx,nt],continuity=continuity_report(states,saved['windows'],201),
                measured_optimization_seconds=summary['resources']['optimization_seconds'],
                baseline_included=False,raw_run_signature=config['signature'],
                selection='independent validation only; stages below are stage-LAST models')
    report['target_relative_l2']=1e-4
    report['target_reached_on_independent_test']=report['test_metrics']['relative_l2']<=1e-4
    report['target_reached_on_output_grid']=report['grid_metrics']['relative_l2']<=1e-4
    atomic_json(out/'metrics.json',report);write_csv(out/'stage_errors.csv',stage_rows)
    print('Selected model: %s | independent test L2=%.6e | grid L2=%.6e' %
          (checkpoint,report['test_metrics']['relative_l2'],report['grid_metrics']['relative_l2']),flush=True)
    return report


def main(argv=None):
    q=argparse.ArgumentParser(description=__doc__)
    q.add_argument('--run',default='runs/C03/seed_1234');q.add_argument('--device',default='cpu')
    q.add_argument('--nx',type=int,default=201);q.add_argument('--nt',type=int,default=801)
    q.add_argument('--checkpoint',default='final.pt',help='final.pt is best validation; stage_08_last.pt is terminal 16-window model.')
    a=q.parse_args(argv)
    if min(a.nx,a.nt)<2:q.error('At least two nodes per axis required.')
    torch.set_num_threads(1)
    return evaluate(a.run,a.device,a.nx,a.nt,a.checkpoint)


if __name__=='__main__':main()
