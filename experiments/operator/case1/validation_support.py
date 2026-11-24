"""Loss-only stopping on held-out active-parameter values; no training changes.

Validation L2 selects checkpoints; learning rates are fixed within each stage.
Stopping uses LF + LBC + LIC, without stochastic weights, GE or consistency.
"""
import math

import numpy as np
import torch

import operator_alpha_core as core


class ValidationLossStop:
    """Ten consecutive checks without a STRICT new best; no relative tolerance."""
    def __init__(self, patience=10, best=float('inf'), count=0):
        self.patience=int(patience)
        self.best=float(best)
        self.count=int(count)
        if self.patience<1 or self.count<0 or math.isnan(self.best):
            raise ValueError('Invalid early-stopping state.')

    def update(self, current):
        current=float(current)
        if not math.isfinite(current):
            raise FloatingPointError('Nonfinite validation physical loss.')
        improved=current<self.best
        if improved:
            self.best=current
            self.count=0
        else:
            self.count+=1
        return improved,self.count>=self.patience

    def state_dict(self):
        return {'patience':self.patience,'best':self.best,'count':self.count}


def build_validation_datasets(design, grid, boundary_points):
    """Fixed final-size tensor grid for ALL stages, held-out parameter values.

    Separate local BC/IC RNG stream. This routine never changes the NumPy,
    Python or Torch training RNG. Random 4096-point validation L2 coordinates
    remain unchanged in single_support; a tensor grid is needed for history.
    """
    old_seed=core.SEED
    core.SEED=int(design['split_seed'])+700001
    result=[]
    try:
        for parameter in design['validation']:
            data=core.build_dataset(*parameter['mu'],M1=grid,M2=grid,N=grid,
                                    boundnum=boundary_points)
            data['parameter_id']=parameter['parameter_id']
            result.append(data)
    finally:
        core.SEED=old_seed
    return result


def unweighted_physics_terms(model, data):
    """Uploaded fast L2-1sigma PDE, BC and IC; no noise or parameter backward."""
    with torch.enable_grad():
        tx=torch.cat([data[k] for k in
                      ['pt_t_collocation','pt_x1_collocation','pt_x2_collocation']],1)
        sig=torch.cat([data[k] for k in ['pt_t_sigma','pt_x1_sigma','pt_x2_sigma']],1)
        full_u,full_v=model(tx,data['mu'])
        u,v=model(sig,data['mu'])
        g=data['N'];m=data['M1'];m2=data['M2']
        u_mat=full_u.view(m+1,g+1,m2+1)
        v_mat=full_v.view(m+1,g+1,m2+1)
        alpha=float(data['alpha'].detach().cpu())
        dtu=core.fast_caputo(u_mat[1:m,:,1:m2].permute(0,2,1),alpha,data['tau']) \
            .permute(0,2,1).reshape(-1,1).to(u)
        dtv=core.fast_caputo(v_mat[1:m,:,1:m2].permute(0,2,1),alpha,data['tau']) \
            .permute(0,2,1).reshape(-1,1).to(v)
        laplacians=[]
        for field in [u,v]:
            grad=torch.autograd.grad(field,sig,torch.ones_like(field),create_graph=True)[0]
            lap=0
            for dim in [1,2]:
                first=grad[:,dim:dim+1]
                second=torch.autograd.grad(first,sig,torch.ones_like(first),create_graph=True)[0][:,dim:dim+1]
                lap=lap+second
            laplacians.append(lap)
        amp2=u*u+v*v
        real=-dtv+data['D']*laplacians[0]+data['beta']*amp2*u
        imag=dtu+data['D']*laplacians[1]+data['beta']*amp2*v
        lf=((real-data['pt_f_real_collocation'])**2).mean() \
           +((imag-data['pt_f_image_collocation'])**2).mean()
        bc=torch.cat([data[k] for k in
                      ['pt_tbc_collocation','pt_x1bc_collocation','pt_x2bc_collocation']],1)
        ic=torch.cat([data[k] for k in
                      ['pt_t0_collocation','pt_x1t0_collocation','pt_x2t0_collocation']],1)
        ub,vb=model(bc,data['mu']);ui,vi=model(ic,data['mu'])
        lbc=((ub-data['pt_ubc_collocation'])**2).mean()+((vb-data['pt_vbc_collocation'])**2).mean()
        lic=((ui-data['pt_ut0_collocation'])**2).mean()+((vi-data['pt_vt0_collocation'])**2).mean()
        return {'validation_loss':lf+lbc+lic,'validation_loss_f':lf,
                'validation_loss_bc':lbc,'validation_loss_ic':lic}


def validate_physics(model, data_list):
    was_training=model.training
    model.eval()
    rows=[]
    try:
        for data in data_list:
            terms=unweighted_physics_terms(model,data)
            values={key:float(value.detach().cpu()) for key,value in terms.items()}
            rows.append({'parameter_id':data['parameter_id'],**values})
            del terms
    finally:
        model.train(was_training)
    macro={key:float(np.mean([row[key] for row in rows]))
           for key in ['validation_loss','validation_loss_f','validation_loss_bc','validation_loss_ic']}
    if not all(math.isfinite(value) for value in macro.values()):
        raise FloatingPointError('Nonfinite validation loss.')
    return macro,rows
