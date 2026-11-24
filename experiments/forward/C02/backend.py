"""C01 coordinate models, inherited Adam states and spatial C1 interfaces.

Only global physical endpoints receive prescribed data. Subdomain interfaces
are trained by differences between predictions, never by exact interface labels.
Absolute x,t inputs are retained when cloning a parent: no hidden remapping.
"""
import types
import numpy as np
import torch
from torch import nn
from c01_source import baseline_original as c01
from c01_source.weight_schemes import install_weights,set_stage
from c01_source.direct_reference import direct_caputo
import fractional_schemes as schemes
from common import cpu_tree,digest
from problem import reference_field,forcing_numpy
from protocol import XL,XR,BETA,GAMMA


def _ordinary_tanh(self,tx):
    return torch.tanh(self.linear(tx))


def create_state(args,device):
    state=c01.create_training_objects(args,device)
    if args.method=='go':
        install_weights(state,'softmax_noise',args.noise_amplitude,args.weight_lower_bound)
    else:
        for name in ['realnet','imagenet']:
            for block in state[name].layers:block.forward=types.MethodType(_ordinary_tanh,block)
        # Baselines use uniform weights. Their unused a1/a2 are not optimized.
        for key in ['a1','a2']:state[key].requires_grad_(False)
        state['optimizers']=[torch.optim.Adam(state['realnet'].parameters(),lr=1e-3),
                             torch.optim.Adam(state['imagenet'].parameters(),lr=1e-3)]
    # C01 fn is a fixed scale (never in an optimizer).
    state['fn'].requires_grad_(False)
    return state


def snapshot(state):
    return cpu_tree(dict(realnet=state['realnet'].state_dict(),imagenet=state['imagenet'].state_dict(),
                         hypernet=state['hypernet'].state_dict(),a1=state['a1'],a2=state['a2'],fn=state['fn'],
                         optimizers=[o.state_dict() for o in state['optimizers']]))


def restore(state,saved,optimizer=True):
    for key in ['realnet','imagenet','hypernet']:state[key].load_state_dict(saved[key])
    with torch.no_grad():
        for key in ['a1','a2','fn']:state[key].copy_(saved[key].to(state[key]))
    if optimizer:
        if len(state['optimizers'])!=len(saved['optimizers']):raise ValueError('Optimizer ownership differs.')
        for opt,data in zip(state['optimizers'],saved['optimizers']):opt.load_state_dict(data)


def weight_hash(saved):
    return digest({k:saved[k] for k in ['realnet','imagenet','hypernet','a1','a2','fn']})


def moment_hash(saved):
    return digest([v['state'] for v in saved['optimizers']])


def clone_parent(parent,args,device):
    child=create_state(args,device);saved=snapshot(parent);restore(child,saved)
    assert weight_hash(snapshot(child))==weight_hash(saved)
    assert moment_hash(snapshot(child))==moment_hash(saved)
    for key in ['a1','a2','fn']:
        assert child[key].data_ptr()!=parent[key].data_ptr()
    return child


def set_learning_rate(states,lr,stage):
    for state in states:
        set_stage(state,stage)
        for optimizer in state['optimizers']:
            for group in optimizer.param_groups:group['lr']=lr


def predict_region(state,tx):
    return torch.cat([state['realnet'](tx),state['imagenet'](tx)],1)


def predict(states,domains,tx):
    if len(states)==1:return predict_region(states[0],tx)
    bounds=tx.new_tensor([r[1] for r in domains[:-1]])
    indices=torch.bucketize(tx[:,1].detach().contiguous(),bounds,right=True)
    result=tx.new_zeros((len(tx),2))
    for i,state in enumerate(states):
        select=torch.where(indices==i)[0]
        if select.numel():result=result.index_copy(0,select,predict_region(state,tx.index_select(0,select)))
    return result


def tensor(a,device,dtype,grad=False):
    return torch.tensor(a,dtype=dtype,device=device,requires_grad=grad)


def mesh_tx(space,times):
    xx,tt=np.meshgrid(space,times,indexing='ij')
    return np.column_stack([tt.ravel(),xx.ravel()])


def build_data(domain,M,N,args,device,dtype,validation=False,space_sites=None):
    lo,hi=domain;scheme=args.scheme
    tau=1./N;times=np.linspace(0,1,N+1)
    sigma=schemes.sigma(args.alpha,scheme)
    shifted=(np.arange(N)+sigma)*tau
    if space_sites is None:axis=np.linspace(lo,hi,M+1)
    else:axis=np.concatenate(([lo],np.sort(space_sites),[hi]));M=len(axis)-1
    # Same resampling order as C01: random times, then random initial x.
    if validation:
        rng=np.random.default_rng(args.validation_seed+2201)
        bt=rng.uniform(0,1,(128,1));ix=rng.uniform(lo,hi,(128,1))
    else:
        bt=np.random.uniform(0,1,(M,1));ix=np.random.uniform(lo,hi,(M,1))
    bc=[]
    if abs(lo-XL)<1e-12:bc.append(np.column_stack([bt.ravel(),np.full(len(bt),XL)]))
    if abs(hi-XR)<1e-12:bc.append(np.column_stack([bt.ravel(),np.full(len(bt),XR)]))
    bc=np.concatenate(bc) if bc else np.empty((0,2))
    ic=np.column_stack([np.zeros(len(ix)),ix.ravel()])
    target_bc=reference_field(bc,args.alpha) if len(bc) else np.empty(0,dtype=complex)
    initial=np.exp(1j*ic[:,1])
    kernel=(c01.build_fastL21sigma_kernel(args.alpha,tau,device,args.kernel_mode)
            if scheme=='fast_l21' else schemes.build_kernel(args.alpha,tau,N,device,scheme,c01))
    source=forcing_numpy(mesh_tx(axis[1:-1],shifted),args.alpha)
    return dict(domain=list(domain),M=M,N=N,tau=tau,
                source=tensor(np.column_stack([source.real,source.imag]),device,dtype),
                full_tx=tensor(mesh_tx(axis,times),device,dtype),
                sigma_tx=tensor(mesh_tx(axis[1:-1],shifted),device,dtype,True),
                ic_tx=tensor(ic,device,dtype),bc_tx=tensor(bc,device,dtype),
                ic_target=tensor(np.column_stack([initial.real,initial.imag]),device,dtype),
                bc_target=tensor(np.column_stack([target_bc.real,target_bc.imag]),device,dtype),
                kernel=kernel)


def history(values,data,args):
    kernel=data['kernel'];N=data['N']
    if args.scheme!='fast_l21':
        return schemes.apply_history(values,args.alpha,data['tau'],kernel,direct_caputo)
    kd=kernel['ws'].dtype;values=values.to(kd)
    memory=values.new_zeros((values.shape[0],kernel['nexp']))
    alpha=torch.tensor([args.alpha],device=values.device,dtype=kd)
    gi=1/c01.gamma_torch(1-alpha)
    local=c01.a_hat(1-alpha/2,alpha,kernel['tau'])/c01.gamma_torch(1-alpha)
    result=[]
    for n in range(1,N+1):
        current=values[:,n]-values[:,n-1]
        if n>1:
            previous=values[:,n-1]-values[:,n-2]
            memory=memory*kernel['exp_xs_tau']+previous[:,None]@kernel['A_l']+current[:,None]@kernel['B_l']
        result.append(gi*(memory@kernel['ws']).squeeze(-1)+local*current)
    return torch.stack(result,1)


def residual_time_difference(residual,tau):
    """Differentiate [space, sigma-time, component] on its actual time axis.

    Sigma times are uniformly spaced by tau. Interior values use centered
    differences; the first and last values use forward/backward differences.
    This acts on the COMPLETE discrete residual, including its history term.
    """
    if residual.shape[1]<2:
        raise ValueError('Residual time differences require at least two sigma times.')
    return torch.cat([(residual[:,1:2]-residual[:,:1])/tau,
                      (residual[:,2:]-residual[:,:-2])/(2*tau),
                      (residual[:,-1:]-residual[:,-2:-1])/tau],dim=1)


def discrete_residual(predictor,data,args,gradient=True):
    """Manufactured-source PDE residual and gradients of the same discrete operator.

    Independent nodal/shifted leaves make spatial differentiation mean a common
    displacement of every history node and its sigma collocation point. The
    history rule itself remains the selected, original fractional discretization.
    """
    full_tx=data['full_tx'].detach().clone().requires_grad_(gradient)
    tx=data['sigma_tx'].detach().clone().requires_grad_(True)
    full_flat=predictor(full_tx)
    full=full_flat.view(data['M']+1,data['N']+1,2)
    field=predictor(tx)
    du=history(full[1:-1,:,0],data,args).reshape(-1,1).to(field)
    dv=history(full[1:-1,:,1],data,args).reshape(-1,1).to(field)
    local=[]
    for dim in [0,1]:
        u=field[:,dim:dim+1]
        grad=torch.autograd.grad(u,tx,torch.ones_like(u),create_graph=True)[0]
        ux=grad[:,1:2]
        uxx=torch.autograd.grad(ux,tx,torch.ones_like(ux),create_graph=True)[0][:,1:2]
        local.append(BETA*uxx+GAMMA*field.square().sum(1,keepdim=True)*u)
    source=data['source'].to(field)
    residual=(torch.cat([-dv+local[0],du+local[1]],dim=1)-source).view(data['M']-1,data['N'],2)
    space=time=None
    if gradient:
        history_dx=[];local_dx=[]
        for component,value in enumerate(local):
            nodal=full_flat[:,component:component+1]
            dx=torch.autograd.grad(nodal,full_tx,torch.ones_like(nodal),create_graph=True)[0][:,1]
            dx=dx.view(data['M']+1,data['N']+1)
            history_dx.append(history(dx[1:-1],data,args).reshape(-1,1).to(field))
            local_dx.append(torch.autograd.grad(value,tx,torch.ones_like(value),create_graph=True)[0][:,1:2])
        # The fixed source satisfies f_x=i*f; its real/imaginary derivatives
        # are (-Im(f), Re(f)). History and local derivatives are unchanged.
        source_dx=torch.stack([-source[:,1],source[:,0]],dim=1)
        space=(torch.cat([-history_dx[1]+local_dx[0],history_dx[0]+local_dx[1]],dim=1)-source_dx).view_as(residual)
        time=residual_time_difference(residual,data['tau'])
    return dict(residual=residual,spatial_gradient=space,temporal_gradient=time,
                full_tx=full_tx,sigma_tx=tx)


def physics(predictor,data,args,gradient=True):
    operator=discrete_residual(predictor,data,args,gradient)
    residual=operator['residual']
    lf=residual[...,0].square().mean()+residual[...,1].square().mean()
    lg=lf.new_zeros(())
    if gradient:
        lg=(operator['spatial_gradient'].square().sum(-1).mean()
            +operator['temporal_gradient'].square().sum(-1).mean())
    ic=predictor(data['ic_tx'])-data['ic_target']
    lic=ic.square().sum(1).mean()
    lbc=(predictor(data['bc_tx'])-data['bc_target']).square().sum(1).mean() if len(data['bc_tx']) else lf.new_zeros(())
    # C01 sums the two physical faces, rather than averaging the faces.
    if len(data['bc_tx']) and abs(data['domain'][0]-XL)<1e-12 and abs(data['domain'][1]-XR)<1e-12:lbc=2*lbc
    return dict(loss_f=lf,loss_ic=lic,loss_bc=lbc,loss_grad=lg)


def interface_loss(states,domains,times,details=False):
    """Exactly sum_pairs mean_points(|psi_p-psi_q|²+|psi_x,p-psi_x,q|²)."""
    device=states[0]['device'];dtype=states[0]['dtype']
    total=torch.zeros((),device=device,dtype=dtype);records=[]
    for pair in range(len(states)-1):
        x=domains[pair][1]
        tx=torch.column_stack([times,times.new_full(times.shape,x)]).detach().requires_grad_(True)
        left=predict_region(states[pair],tx);right=predict_region(states[pair+1],tx)
        derivatives=[]
        for field in [left,right]:
            dx=[]
            for component in [0,1]:
                value=field[:,component:component+1]
                dx.append(torch.autograd.grad(value,tx,torch.ones_like(value),create_graph=True)[0][:,1:2])
            derivatives.append(torch.cat(dx,1))
        lv=(left-right).square().sum(1).mean()
        ld=(derivatives[0]-derivatives[1]).square().sum(1).mean()
        total=total+lv+ld
        if details:records.append(dict(pair=pair,interface_x=x,points=len(times),
                                       value_jump_mse=float(lv.detach()),spatial_derivative_jump_mse=float(ld.detach())))
    return (total,records) if details else total


def teacher_loss(states,domains,teacher):
    if teacher is None:return states[0]['a1'].new_zeros(())
    tx=teacher['tx'].to(states[0]['device']);target=teacher['uv'].to(tx)
    return (predict(states,domains,tx)-target).square().sum(1).mean()


def refresh_teacher(states,domains,datasets):
    tx=torch.cat([d['sigma_tx'].detach() for d in datasets])
    with torch.no_grad():uv=predict(states,domains,tx)
    return dict(tx=tx.detach(),uv=uv.detach())


def objective(states,domains,datasets,args,epoch,teacher):
    values={k:0. for k in ['loss_f','loss_bc','loss_ic','loss_grad','w_f','w_bc','w_ic']}
    total=states[0]['a1'].new_zeros(())
    for state,domain,data in zip(states,domains,datasets):
        terms=physics(lambda tx:predict_region(state,tx),data,args,gradient=args.method=='go' and args.w_grad>0)
        if args.method=='go':wf,wb,wi=state['hypernet'](torch.stack([terms['loss_f'],terms['loss_bc'],terms['loss_ic']]))
        else:wf=wb=wi=total.new_ones(())
        volume=(domain[1]-domain[0])/(XR-XL)
        wg=args.w_grad*torch.exp(-3*terms['loss_f'].detach()) if args.method=='go' else 0.
        total=total+volume*(wf*terms['loss_f']+wi*terms['loss_ic']+wg*terms['loss_grad'])+wb*terms['loss_bc']
        for key in terms:values[key]+=float(terms[key].detach())*(1 if key=='loss_bc' else volume)
        for key,value in [('w_f',wf),('w_bc',wb),('w_ic',wi)]:values[key]+=float(value.detach())*volume
    times=torch.linspace(0,1,args.interface_points,device=states[0]['device'],dtype=states[0]['dtype'])
    lif=interface_loss(states,domains,times) if len(states)>1 else total.new_zeros(())
    lp=teacher_loss(states,domains,teacher) if args.method=='go' else total.new_zeros(())
    wp=1./(epoch+1) if teacher is not None and args.method=='go' else 0.
    if wp<1e-2:wp=0.
    total=total+args.interface_weight*lif+wp*lp
    return total,dict(values,loss=float(total.detach()),loss_interface=float(lif.detach()),
                      loss_previous=float(lp.detach()),w_previous=wp)


def physical_validation(states,domains,args,device):
    """Common physical grid across ALL stages; no solution interior targets."""
    rng=np.random.default_rng(args.validation_seed+1101)
    sites=rng.uniform(XL,XR,args.validation_spatial_sites)
    data=build_data([XL,XR],len(sites)+1,args.validation_time_intervals,args,device,states[0]['dtype'],True,sites)
    terms=physics(lambda tx:predict(states,domains,tx),data,args,gradient=False)
    times=torch.linspace(0,1,args.interface_points,device=device,dtype=states[0]['dtype'])
    lif,interfaces=interface_loss(states,domains,times,True)
    score=terms['loss_f']+terms['loss_bc']+terms['loss_ic']+args.interface_weight*lif
    return dict(score=float(score.detach()),**{k:float(v.detach()) for k,v in terms.items()},
                loss_interface=float(lif.detach()),interfaces=interfaces)
