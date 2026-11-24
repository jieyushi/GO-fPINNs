"Build the equation residuals and the configured gradient-enhancement terms."
import torch
import models
import fractional
from problem import initial_values,forcing


def pieces(data,args):
    size=args.spatial_batch or data['M']-1
    return [slice(s,min(s+size,data['M']-1)) for s in range(0,data['M']-1,size)]


def time_difference(values,dt):
    """sigma-grid time difference: centered interior, one-sided first/last."""
    if values.shape[1]<2:raise ValueError('Time GE requires at least two residual times.')
    return torch.cat([(values[:,1:2]-values[:,:1])/dt,
                      (values[:,2:]-values[:,:-2])/(2*dt),
                      (values[:,-1:]-values[:,-2:-1])/dt],dim=1)


def residual_fields(field_predict,potential_predict,data,args,rows):
    """Construct residual and GE fields without changing the fractional operator."""
    active=args.method=='go_multi' and bool(args.w_grad)
    full_ge=active
    # Each spatial site keeps every nodal time, including t=0. History x derivatives
    # must be taken on this leaf, not on the independent sigma-grid leaf below.
    q=data['nodal'][rows].detach().requires_grad_(full_ge);ns=q.shape[0];N=data['N']
    values=field_predict(q.reshape(-1,2)).view(ns,N+1,2)
    derivative=fractional.apply(values.permute(0,2,1),data['kernel'],args.history_evaluation).permute(0,2,1).reshape(-1,2)
    tx=data['shifted'][rows].reshape(-1,2).detach().requires_grad_(True)
    field=field_predict(tx);potential=potential_predict(tx[:,1:2]);source=forcing(tx,args.problem)
    local=[]
    for component in [0,1]:
        value=field[:,component:component+1]
        dx=torch.autograd.grad(value,tx,torch.ones_like(value),create_graph=True)[0][:,1:2]
        dxx=torch.autograd.grad(dx,tx,torch.ones_like(dx),create_graph=True)[0][:,1:2]
        local.append(.5*dxx-field.square().sum(1,keepdim=True)*value+potential*value)
    local=torch.cat(local,1);ordinary=local-source
    residual=(torch.stack([-derivative[:,1],derivative[:,0]],1)+ordinary).view(ns,N,2)
    result=dict(residual=residual,local=local,shifted=tx)
    if full_ge:
        nodal_x=[];local_x=[]
        for component in [0,1]:
            value=values[:,:,component]
            nodal_x.append(torch.autograd.grad(value,q,torch.ones_like(value),create_graph=True)[0][:,:,1])
            value=ordinary[:,component:component+1]
            # Includes eta_x * field, eta * field_x, and the fixed source derivative -f_x.
            local_x.append(torch.autograd.grad(value,tx,torch.ones_like(value),create_graph=True)[0][:,1])
        history_x=fractional.apply(torch.stack(nodal_x,1),data['kernel'],args.history_evaluation).permute(0,2,1)
        result['space_gradient']=torch.stack([-history_x[:,:,1],history_x[:,:,0]],2)+torch.stack(local_x,1).view(ns,N,2)
        result['time_gradient']=time_difference(residual,1/N)
    return result


def residual_parts(field_predict,potential_predict,data,args,rows):
    fields=residual_fields(field_predict,potential_predict,data,args,rows)
    lf=fields['residual'].square().sum(-1).mean();lg=lf.new_zeros(())
    for key in ['space_gradient','time_gradient']:
        if key in fields:lg=lg+fields[key].square().sum(-1).mean()
    return lf,lg


def residual_terms(state,data,args,rows):
    return residual_parts(lambda q:models.predict(state,q),lambda x:models.eta(state,x),data,args,rows)


def conditions(state,data):
    ic=(models.predict(state,data['ic_tx'])-initial_values(data['ic_tx'])).square().sum(1).mean()
    bc=sum(models.predict(state,data[k]).square().sum(1).mean() for k in ['left_tx','right_tx'])
    obs=(models.predict(state,data['obs_tx'])-data['obs_target']).square().sum(1).mean()
    eta=(models.eta(state,data['eta_x'])-data['eta_target']).square().mean()
    return bc,ic,obs,eta


def make_teacher(state,previous,args):
    if previous is None or args.consistency_mode=='none':return None
    tx=previous['shifted'].reshape(-1,2).detach()
    with torch.no_grad():uv=models.predict(state,tx).detach()
    return dict(tx=tx,uv=uv)


def refresh_teacher(state,teacher,args):
    if teacher is not None and args.consistency_mode=='rolling':
        with torch.no_grad():teacher['uv']=models.predict(state,teacher['tx']).detach()


def objective(state,data,args,epoch,teacher,lf,lg):
    bc,ic,obs,eta=conditions(state,data)
    if args.method=='fl1':weights=lf.new_ones(3)
    else:weights=state['hypernet'](torch.stack([lf,bc,ic]).to(state['dtype'])).to(lf)
    wf,wb,wi=weights
    previous=lf.new_zeros(()) if teacher is None else (models.predict(state,teacher['tx'])-teacher['uv']).square().sum(1).mean()
    wp=0. if teacher is None else 1/(epoch+1)
    if wp<1e-2:wp=0.
    wg=args.w_grad*torch.exp(-3*lf.detach()) if args.method=='go_multi' else 0.
    total=wf*lf+wb*bc+wi*ic+args.observation_weight*obs+args.eta_weight*eta+wg*lg+wp*previous
    report=dict(loss=float(total.detach()),loss_f=float(lf.detach()),loss_grad=float(lg.detach()),
                loss_bc=float(bc.detach()),loss_ic=float(ic.detach()),loss_observation=float(obs.detach()),
                loss_eta_supervised=float(eta.detach()),loss_previous=float(previous.detach()),
                w_f=float(wf.detach()),w_bc=float(wb.detach()),w_ic=float(wi.detach()),w_previous=wp,
                w_grad=float(wg) if isinstance(wg,float) else float(wg.detach()))
    return total,report


def backward(state,data,args,epoch,teacher):
    if not args.spatial_batch:
        lf,lg=residual_terms(state,data,args,slice(0,data['M']-1))
        total,report=objective(state,data,args,epoch,teacher,lf,lg)
        if not bool(torch.isfinite(total)):raise FloatingPointError('Nonfinite objective.')
        total.backward();return report
    totals=[0.,0.]
    for rows in pieces(data,args):
        weight=(rows.stop-rows.start)/(data['M']-1)
        terms=residual_terms(state,data,args,rows)
        totals=[s+weight*float(t.detach()) for s,t in zip(totals,terms)]
        del terms
    leaves=torch.tensor(totals,dtype=torch.float64,device=state['device'],requires_grad=True)
    total,report=objective(state,data,args,epoch,teacher,leaves[0],leaves[1])
    if not bool(torch.isfinite(total)):raise FloatingPointError('Nonfinite objective.')
    total.backward();coefficient=leaves.grad.detach()
    for rows in pieces(data,args):
        lf,lg=residual_terms(state,data,args,rows)
        weight=(rows.stop-rows.start)/(data['M']-1)
        (weight*(coefficient[0]*lf+coefficient[1]*lg)).backward()
    return report


def validation_loss(state,data,args):
    score=0.
    for rows in pieces(data,args):
        lf,_=residual_terms(state,data,args,rows)
        score+=(rows.stop-rows.start)/(data['M']-1)*float(lf.detach())
    score+=sum(float(term.detach()) for term in conditions(state,data))
    if not torch.isfinite(torch.tensor(score)):raise FloatingPointError('Nonfinite validation loss.')
    return score
