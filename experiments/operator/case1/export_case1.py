"""Export clean t=0.5 fields using the median held-out value, before seeing errors."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import single_support as p
import training_support as s
from train_single_operator import load_for_prediction


def main(argv=None):
    q=argparse.ArgumentParser(description=__doc__)
    q.add_argument('--root',type=Path,required=True)
    q.add_argument('--operators',default='alpha,beta,gamma')
    q.add_argument('--seed',type=int,default=1234)
    q.add_argument('--device',default='cpu')
    q.add_argument('--time',type=float,default=.5)
    q.add_argument('--space-points',type=int,default=101)
    a=q.parse_args(argv)
    if not 0<=a.time<=1 or a.space_points<2:q.error('Time in [0,1] and >=2 spatial nodes required.')
    torch.set_num_threads(1)
    out=a.root/'paper_data'/('seed_'+str(a.seed));out.mkdir(parents=True,exist_ok=True)
    axis=np.linspace(0,1,a.space_points,dtype=np.float32)
    xx,yy=np.meshgrid(axis,axis,indexing='ij')
    tx=np.column_stack([np.full(xx.size,a.time,dtype=np.float32),xx.ravel(),yy.ravel()])
    exported=[]
    for operator in a.operators.split(','):
        run=a.root/'operators'/operator/'go'/('seed_'+str(a.seed))
        split=json.loads((run/'parameter_split.json').read_text())
        summary=json.loads((run/'summary.json').read_text())
        candidates=sorted(split['test'],key=lambda r:r['active_value'])
        parameter=candidates[len(candidates)//2]
        # Deterministic PARAMETER selection, no comparison of test errors.
        state,saved=load_for_prediction(run/'final.pt',a.device)
        loc=torch.tensor(tx,device=a.device)
        with torch.no_grad():
            u,v=state['model'](loc,loc.new_tensor([parameter['mu']]))
            true=p.exact_real(loc,parameter['mu']).cpu().numpy().reshape(xx.shape)
        prediction=torch.cat([u,v],1).cpu().numpy().reshape(xx.shape+(2,))
        meta={'operator':operator,'seed':a.seed,'time':a.time,'parameter':parameter,
              'representative_rule':'upper median active value of the predetermined held-out set; no use of test errors',
              'checkpoint':'validation-selected final.pt','selected_stage':saved['stage'],
              'selected_epoch_zero_based':saved['epoch'],'design_sha256':split['design_sha256'],
              'status':summary['status'],'coordinate_order':['x','y'],'noise_level':0.}
        np.savez_compressed(out/('fields_'+operator+'.npz'),x=axis,y=axis,prediction=prediction,
                            exact=np.stack([true,true],-1),mu=np.asarray(parameter['mu']))
        s.atomic_json(out/('fields_'+operator+'.json'),meta);exported.append(meta)
    s.atomic_json(out/'export_manifest.json',{'seed':a.seed,'fields':exported,'references':'manufactured analytical family'})
    print('Exported analytical references and predictions to '+str(out))


if __name__=='__main__':main()
