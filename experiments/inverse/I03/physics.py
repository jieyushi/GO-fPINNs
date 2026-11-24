"Build the equation residuals and the configured gradient-enhancement terms."
import torch
import models
import fractional
from problem import known_values,forcing
from protocol import BETA,GAMMA


def pieces(data,args):
    size=args.spatial_batch or data['spatial_sites']
    return [slice(s,min(s+size,data['spatial_sites'])) for s in range(0,data['spatial_sites'],size)]


def local_gradient_mismatch(local,true_local,tx):
    """Return component x coordinate derivatives, with coordinate order t,x,y."""
    error=local-true_local
    return torch.stack([torch.autograd.grad(error[:,c:c+1],tx,
                        torch.ones_like(error[:,c:c+1]),create_graph=True)[0]
                        for c in [0,1]],1)


def residual_parts(field_predict,potential_predict,data,args,rows):
    q=data['nodal'][rows];ns=q.shape[0];N=data['N']
    values=field_predict(q.reshape(-1,3)).view(ns,N+1,2)
    derivative=fractional.apply(values.permute(0,2,1),data['kernel'],args.history_evaluation).permute(0,2,1).reshape(-1,2)
    tx=data['shifted'][rows].reshape(-1,3).detach().requires_grad_(True)
    field=field_predict(tx);potential=potential_predict(tx[:,1:3]);source,true_local=forcing(tx)
    local=[]
    for component in [0,1]:
        value=field[:,component:component+1]
        first=torch.autograd.grad(value,tx,torch.ones_like(value),create_graph=True)[0]
        lap=0.
        for direction in [1,2]:
            part=first[:,direction:direction+1]
            lap=lap+torch.autograd.grad(part,tx,torch.ones_like(part),create_graph=True)[0][:,direction:direction+1]
        local.append(BETA*lap+GAMMA*field.square().sum(1,keepdim=True)*value+potential*value)
    local=torch.cat(local,1)
    residual=torch.stack([-derivative[:,1],derivative[:,0]],1)+local-source
    lf=residual.square().sum(1).mean();lg=lf.new_zeros(())
    if args.method=='go_multi' and args.w_grad:
        # Fixed nonzero manufactured source: match all t/x/y local derivatives.
        # The trainable potential stays in local, retaining eta_x and eta_y.
        lg=local_gradient_mismatch(local,true_local,tx).square().sum((1,2)).mean()
    return lf,lg


def residual_terms(state,data,args,rows):
    return residual_parts(lambda q:models.predict(state,q),lambda xy:models.eta(state,xy),data,args,rows)


def conditions(state,data):
    ic=(models.predict(state,data['ic_tx'])-known_values(data['ic_tx'])).square().sum(1).mean()
    
    bc=(models.predict(state,data['bc_tx'])-known_values(data['bc_tx'])).square().sum(1).mean()
    obs=(models.predict(state,data['obs_tx'])-data['obs_target']).square().sum(1).mean()
    eta=(models.eta(state,data['eta_xy'])-data['eta_target']).square().mean()
    return bc,ic,obs,eta


def make_teacher(state,previous,args):
    if previous is None or args.consistency_mode=='none':return None
    tx=previous['shifted'].reshape(-1,3).detach()
    with torch.no_grad():uv=models.predict(state,tx).detach()
    return dict(tx=tx,uv=uv)


def refresh_teacher(state,teacher,args):
    if teacher is not None and args.consistency_mode=='rolling':
        with torch.no_grad():teacher['uv']=models.predict(state,teacher['tx']).detach()


def objective(state,data,args,epoch,teacher,lf,lg):
    bc,ic,obs,eta=conditions(state,data)
    if args.method=='fl1':weights=lf.new_ones(4)
    else:weights=state['hypernet'](torch.stack([lf,bc,ic,obs]).to(state['dtype'])).to(lf)
    wf,wb,wi,wo=weights
    previous=lf.new_zeros(()) if teacher is None else (models.predict(state,teacher['tx'])-teacher['uv']).square().sum(1).mean()
    wp=0. if teacher is None else 1/(epoch+1)
    if wp<1e-2:wp=0.
    wg=args.w_grad*torch.exp(-3*lf.detach()) if args.method=='go_multi' else 0.
    total=wf*lf+wb*bc+wi*ic+wo*obs+args.eta_weight*eta+wg*lg+wp*previous
    report=dict(loss=float(total.detach()),loss_f=float(lf.detach()),loss_grad=float(lg.detach()),loss_bc=float(bc.detach()),
                loss_ic=float(ic.detach()),loss_observation=float(obs.detach()),loss_eta_supervised=float(eta.detach()),
                loss_previous=float(previous.detach()),w_f=float(wf.detach()),w_bc=float(wb.detach()),
                w_ic=float(wi.detach()),w_observation=float(wo.detach()),w_previous=wp,
                w_grad=float(wg) if isinstance(wg,float) else float(wg.detach()))
    return total,report


def backward(state,data,args,epoch,teacher):
    if not args.spatial_batch:
        lf,lg=residual_terms(state,data,args,slice(0,data['spatial_sites']))
        total,report=objective(state,data,args,epoch,teacher,lf,lg)
        if not bool(torch.isfinite(total)):raise FloatingPointError('Nonfinite objective.')
        total.backward();return report
    totals=[0.,0.]
    for rows in pieces(data,args):
        weight=(rows.stop-rows.start)/data['spatial_sites']
        terms=residual_terms(state,data,args,rows)
        totals=[s+weight*float(t.detach()) for s,t in zip(totals,terms)];del terms
    leaves=torch.tensor(totals,dtype=torch.float64,device=state['device'],requires_grad=True)
    total,report=objective(state,data,args,epoch,teacher,leaves[0],leaves[1])
    if not bool(torch.isfinite(total)):raise FloatingPointError('Nonfinite objective.')
    total.backward();coefficient=leaves.grad.detach()
    for rows in pieces(data,args):
        lf,lg=residual_terms(state,data,args,rows);weight=(rows.stop-rows.start)/data['spatial_sites']
        (weight*(coefficient[0]*lf+coefficient[1]*lg)).backward()
    return report


def validation_loss(state,data,args):
    score=0.
    for rows in pieces(data,args):
        lf,_=residual_terms(state,data,args,rows)
        score+=(rows.stop-rows.start)/(data['spatial_sites'])*float(lf.detach())
    score+=sum(float(term.detach()) for term in conditions(state,data))
    if not torch.isfinite(torch.tensor(score)):raise FloatingPointError('Nonfinite validation loss.')
    return score
