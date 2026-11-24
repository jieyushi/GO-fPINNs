"Build the equation residuals and the configured gradient-enhancement terms."
import torch
import fractional
import models
from data import shifted
from problem import known_values,forcing
from protocol import gradient_axes


def pieces(data,args):
    size=args.spatial_batch or data['M']-1
    return [slice(j,min(j+size,data['M']-1)) for j in range(0,data['M']-1,size)]


def residual_terms(predictor,physical,data,args,rows):
    a,b,g=[physical[k] for k in ['alpha','beta','gamma']]
    nodes=data['nodal'][rows];ns=nodes.shape[0];N=data['N']
    vals=predictor(nodes.reshape(-1,2)).view(ns,N+1,2)
    kernel=fractional.build(a,data['basis'],N,data['scheme'],args.history_rule=='direct')
    d=fractional.apply(vals.permute(0,2,1),kernel,args.history_evaluation).permute(0,2,1).reshape(-1,2)
    tx=shifted(data,a,rows);field=predictor(tx);local=[]
    for component in [0,1]:
        value=field[:,component:component+1]
        dx=torch.autograd.grad(value,tx,torch.ones_like(value),create_graph=True)[0][:,1:2]
        dxx=torch.autograd.grad(dx,tx,torch.ones_like(dx),create_graph=True)[0][:,1:2]
        local.append(b*dxx+g*field.square().sum(1,keepdim=True)*value)
    local=torch.cat(local,1);f,ft,fx=forcing(tx)
    residual=torch.stack([-d[:,1],d[:,0]],1)+local-f
    lf=residual.square().sum(1).mean();lg=lf.new_zeros(())
    axes=gradient_axes(args.unknown,args.gradient_mode) if args.method!='fl1' and args.w_grad else ()
    if axes:
        for component in [0,1]:
            grad=torch.autograd.grad(local[:,component:component+1],tx,
                                     torch.ones_like(local[:,component:component+1]),create_graph=True)[0]
            for axis in axes:
                target=ft if axis==0 else fx
                lg=lg+(grad[:,axis:axis+1]-target[:,component:component+1]).square().mean()
    return lf,lg


def conditions(state,data):
    zero=models.predict(state,data['ic_tx'])-known_values(data['ic_tx'])
    ic=zero.square().sum(1).mean();bc=ic.new_zeros(())
    for side in ['left','right']:
        error=models.predict(state,data[side+'_tx'])-known_values(data[side+'_tx'])
        bc=bc+error.square().sum(1).mean()
    obs=(models.predict(state,data['obs_tx'])-data['obs_target']).square().sum(1).mean()
    return bc,ic,obs


def teacher_loss(state,teacher):
    if teacher is None:return state['physical']['alpha'].new_zeros(())
    return (models.predict(state,teacher['tx'])-teacher['uv']).square().sum(1).mean()


def make_teacher(state,previous_data,args):
    if previous_data is None or args.consistency_mode=='none':return None
    tx=shifted(previous_data,state['physical']['alpha'],slice(0,previous_data['M']-1)).detach()
    with torch.no_grad():target=models.predict(state,tx).detach()
    return dict(tx=tx,uv=target)


def refresh_teacher(state,teacher,args):
    if teacher is not None and args.consistency_mode=='rolling':
        with torch.no_grad():teacher['uv']=models.predict(state,teacher['tx']).detach()


def objective(state,data,args,epoch,teacher,lf,lg):
    bc,ic,obs=conditions(state,data)
    if args.method=='fl1':weights=lf.new_ones(4)
    else:weights=state['hypernet'](torch.stack([lf,bc,ic,obs]).to(state['dtype'])).to(lf)
    wf,wb,wi,wo=weights
    wg=args.w_grad*torch.exp(-3*lf.detach()) if args.method!='fl1' else 0.
    wp=1/(epoch+1) if teacher is not None else 0.
    if wp<1e-2:wp=0.
    lp=teacher_loss(state,teacher)
    total=wf*lf+wb*bc+wi*ic+wo*obs+wg*lg+wp*lp
    report=dict(loss=float(total.detach()),loss_f=float(lf.detach()),loss_grad=float(lg.detach()),
                loss_bc=float(bc.detach()),loss_ic=float(ic.detach()),loss_observation=float(obs.detach()),
                loss_previous=float(lp.detach()),w_f=float(wf.detach()),w_bc=float(wb.detach()),
                w_ic=float(wi.detach()),w_observation=float(wo.detach()),w_grad=float(wg) if isinstance(wg,float) else float(wg.detach()),w_previous=wp)
    return total,report


def backward(state,data,args,epoch,teacher):
    if not args.spatial_batch:
        lf,lg=residual_terms(lambda tx:models.predict(state,tx),state['physical'],data,args,slice(0,data['M']-1))
        total,report=objective(state,data,args,epoch,teacher,lf,lg)
        if not bool(torch.isfinite(total)):raise FloatingPointError('Nonfinite objective.')
        total.backward();return report
    
    totals=[0.,0.]
    for rows in pieces(data,args):
        fraction=(rows.stop-rows.start)/(data['M']-1)
        terms=residual_terms(lambda tx:models.predict(state,tx),state['physical'],data,args,rows)
        totals=[old+fraction*float(value.detach()) for old,value in zip(totals,terms)]
        del terms
    leaves=torch.tensor(totals,dtype=torch.float64,device=state['device'],requires_grad=True)
    total,report=objective(state,data,args,epoch,teacher,leaves[0],leaves[1])
    if not bool(torch.isfinite(total)):raise FloatingPointError('Nonfinite objective.')
    total.backward();coeff=leaves.grad.detach().clone()
    for rows in pieces(data,args):
        fraction=(rows.stop-rows.start)/(data['M']-1)
        lf,lg=residual_terms(lambda tx:models.predict(state,tx),state['physical'],data,args,rows)
        (fraction*(coeff[0]*lf+coeff[1]*lg)).backward()
    return report


def validation_loss(state,data,args):
    score=0.
    for rows in pieces(data,args):
        lf,_=residual_terms(lambda tx:models.predict(state,tx),state['physical'],data,args,rows)
        score+=(rows.stop-rows.start)/(data['M']-1)*float(lf.detach())
    score+=sum(float(term.detach()) for term in conditions(state,data))
    if not torch.isfinite(torch.tensor(score)):raise FloatingPointError('Nonfinite validation loss.')
    return score
