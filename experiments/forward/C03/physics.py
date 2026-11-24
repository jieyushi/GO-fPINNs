"Build the equation residuals and the configured gradient-enhancement terms."
import numpy as np
import torch
import history
import models
from protocol import ALPHA,BETA,GAMMA,T
from problem import boundary_target,source,residual_scale


def mesh(space,times):
    xx,tt=np.meshgrid(space,times,indexing='ij')
    return np.column_stack([tt.ravel(),xx.ravel()])


def build_data(states,item,window,args,device,previous_item=None,validation=False):
    s=states[window];dtype=s['dtype'];M,N=item['M'],item['N']
    start=window*N;lo,hi=item['windows'][window];tau=item['tau'];sigma=1-ALPHA/2
    axis=np.linspace(0,1,M+1);inner=axis[1:-1]
    if validation:inner=(np.arange(1,M)-.5)/M
    tensor=lambda a:torch.tensor(a,device=device,dtype=dtype)
    nodal=tensor(mesh(inner,lo+np.arange(N+1)*tau)).view(M-1,N+1,2)
    shifted=tensor(mesh(inner,lo+(np.arange(N)+sigma)*tau)).view(M-1,N,2)
    prefix=None
    if window:
        
        past_tx=tensor(mesh(inner,np.arange(start+1)*tau))
        with torch.no_grad():
            prefix=models.predict(states[:window],item['windows'][:window],past_tx).view(M-1,start+1,2).detach()
        assert not prefix.requires_grad
    rng=np.random.default_rng(args.validation_seed+window)
    xic=rng.uniform(0,1,args.interface_points) if validation else np.linspace(0,1,args.interface_points)
    ic_tx=tensor(np.column_stack([np.full(len(xic),lo),xic]))
    with torch.no_grad():
        target=(models.predict_one(states[window-1],ic_tx) if window else boundary_target(ic_tx)).detach()
    bt=rng.uniform(lo,hi,args.boundary_points) if validation else np.linspace(lo,hi,args.boundary_points)
    left=tensor(np.column_stack([bt,np.zeros(len(bt))]))
    right=tensor(np.column_stack([bt,np.ones(len(bt))]))
    teacher=None
    if previous_item is not None:
        inherited=(np.arange(previous_item['total_N'])+sigma)*previous_item['tau']
        inherited=inherited[(inherited>=lo)&(inherited<=hi)]
        anchor=tensor(mesh(inner,inherited))
        with torch.no_grad():teacher=dict(tx=anchor,uv=models.predict_one(s,anchor).detach())
    return dict(nodal=nodal,shifted=shifted,prefix=prefix,start=start,M=M,N=N,tau=tau,
                window=window,lo=lo,hi=hi,kernel=history.build(tau,item['total_N'],device,args.soe_eps),
                ic_tx=ic_tx,ic_target=target,bc_left=left,bc_right=right,
                bc_left_target=boundary_target(left),bc_right_target=boundary_target(right)),teacher


def chunks(data,args):
    count=data['M']-1;size=args.spatial_batch or count
    return [slice(i,min(i+size,count)) for i in range(0,count,size)]


def physics_chunk(predictor,data,args,rows,with_gradient=True):
    nodal=data['nodal'][rows];shifted=data['shifted'][rows]
    ns=nodal.shape[0];N=data['N'];tx=shifted.reshape(-1,2).detach().requires_grad_(True)
    current=predictor(nodal.reshape(-1,2)).view(ns,N+1,2)
    values=current if data['prefix'] is None else torch.cat([data['prefix'][rows],current[:,1:]],1)
    
    deriv=history.apply(values.permute(0,2,1),data['kernel'],args.history_mode)
    deriv=deriv[:,:,data['start']:data['start']+N].permute(0,2,1).reshape(-1,2)
    field=predictor(tx);local=[]
    for component in [0,1]:
        value=field[:,component:component+1]
        grad=torch.autograd.grad(value,tx,torch.ones_like(value),create_graph=True)[0]
        dx=grad[:,1:2]
        dxx=torch.autograd.grad(dx,tx,torch.ones_like(dx),create_graph=True)[0][:,1:2]
        local.append(BETA*dxx+(GAMMA*field.square().sum(1,keepdim=True)+args.potential_sign*tx[:,1:2])*value)
    local=torch.cat(local,1);forcing,fx,ft=source(tx,args.potential_sign)
    residual=torch.stack([-deriv[:,1],deriv[:,0]],1)+local-forcing
    scale=residual_scale(tx)
    lf=(residual/scale).square().sum(1).mean()
    physical_lf=residual.square().sum(1).mean()
    lg=lf.new_zeros(())
    if with_gradient and args.w_grad>0:
        for component in [0,1]:
            dl=torch.autograd.grad(local[:,component:component+1],tx,
                                   torch.ones_like(local[:,component:component+1]),create_graph=True)[0]
            
            lg=lg+((dl[:,1:2]-fx[:,component:component+1])/scale).square().mean()
            lg=lg+((dl[:,0:1]-ft[:,component:component+1])/scale).square().mean()
    return lf,lg,physical_lf


def conditions(state,data):
    ic=(models.predict_one(state,data['ic_tx'])-data['ic_target']).square().sum(1).mean()/T**2
    bc=ic.new_zeros(())
    for side in ['left','right']:
        bc=bc+(models.predict_one(state,data['bc_'+side])-data['bc_'+side+'_target']).square().sum(1).mean()/T**2
    return bc,ic


def previous_loss(state,teacher):
    if teacher is None:return state['a1'].new_zeros(())
    return (models.predict_one(state,teacher['tx'])-teacher['uv']).square().sum(1).mean()/T**2


def refresh_teacher(state,teacher):
    if teacher is not None:
        with torch.no_grad():teacher['uv']=models.predict_one(state,teacher['tx']).detach()


def objective(state,data,args,epoch,teacher,lf,lg):
    bc,ic=conditions(state,data)
    wf,wb,wi=state['hypernet'](torch.stack([lf,bc,ic]))
    wg=args.w_grad*torch.exp(-3*lf.detach())
    wp=1/(epoch+1) if teacher is not None else 0.
    if wp<1e-2:wp=0.
    lp=previous_loss(state,teacher)
    factor=args.continuity_weight if data['window'] else 1.
    total=wf*lf+wb*bc+wi*factor*ic+wg*lg+wp*lp
    return total,dict(loss_f=float(lf.detach()),loss_grad=float(lg.detach()),
                      loss_bc=float(bc.detach()),loss_ic=float(ic.detach()) if not data['window'] else 0.,
                      loss_cont=float(ic.detach()) if data['window'] else 0.,
                      loss_cont_physical=float(ic.detach())*T**2 if data['window'] else 0.,
                      loss_previous=float(lp.detach()),w_f=float(wf.detach()),w_bc=float(wb.detach()),
                      w_ic=float(wi.detach()),w_grad=float(wg.detach()),w_previous=wp)


def gradients(state,data,args,epoch,teacher):
    "Gradients."
    totals=[0.,0.,0.];parts=chunks(data,args)
    for rows in parts:
        frac=(rows.stop-rows.start)/(data['M']-1)
        terms=physics_chunk(lambda tx:models.predict_one(state,tx),data,args,rows)
        for k,value in enumerate(terms):totals[k]+=frac*float(value.detach())
        del terms
    leaves=torch.tensor(totals[:2],device=state['device'],dtype=state['dtype'],requires_grad=True)
    total,report=objective(state,data,args,epoch,teacher,leaves[0],leaves[1])
    if not bool(torch.isfinite(total)):raise FloatingPointError('Nonfinite objective.')
    total.backward();coeff=leaves.grad.detach().clone()
    for rows in parts:
        frac=(rows.stop-rows.start)/(data['M']-1)
        lf,lg,_=physics_chunk(lambda tx:models.predict_one(state,tx),data,args,rows)
        (frac*(coeff[0]*lf+coeff[1]*lg)).backward()
    return dict(report,loss=float(total.detach()),loss_f_physical=totals[2])


def continuity_report(states,windows,points):
    result=[];s=states[0]
    x=torch.linspace(0,1,points,device=s['device'],dtype=s['dtype'])
    with torch.no_grad():
        for j in range(len(states)-1):
            tx=torch.stack([x.new_full(x.shape,windows[j][1]),x],1)
            error=models.predict_one(states[j+1],tx)-models.predict_one(states[j],tx)
            result.append(dict(left=j,right=j+1,time=windows[j][1],
                               complex_jump_mse=float(error.square().sum(1).mean()),
                               max_complex_jump=float(error.square().sum(1).sqrt().max())))
    return result


def validation_loss(state,data,args):
    score=0.
    for rows in chunks(data,args):
        lf,_,_=physics_chunk(lambda tx:models.predict_one(state,tx),data,args,rows,with_gradient=False)
        score+=(rows.stop-rows.start)/(data['M']-1)*float(lf.detach())
    score+=sum(float(v.detach()) for v in conditions(state,data))
    if not np.isfinite(score):raise FloatingPointError('Nonfinite validation loss.')
    return score
