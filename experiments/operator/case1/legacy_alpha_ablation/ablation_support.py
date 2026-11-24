"""V6 switches/budgets unchanged. Local GE now includes x/y/time."""
import math
import torch
from torch import nn
import operator_alpha_core as core
import training_support as s

_SOE_CAPUTO = core.fast_caputo

VARIANTS = {
    "full": {},
    "no_enrichment": {"grid_growth": False, "parameter_growth": False, "consistency": False},
    "direct_operator": {"grid_growth": False, "parameter_growth": False,
                        "consistency": False, "best_transfer": False},
    "no_grid_enrichment": {"grid_growth": False},
    "no_parameter_enrichment": {"parameter_growth": False},
    "no_consistency": {"consistency": False},
    "no_gradient": {"gradient": False},
    "no_soe": {"soe": False},
    "no_adaptive_weights": {"adaptive_weights": False},
    "no_dual_tanh": {"dual_tanh": False},
    "no_fourier": {"fourier": False},
    "plain_operator": {"grid_growth": False, "parameter_growth": False,
        "consistency": False, "gradient": False, "adaptive_weights": False,
        "dual_tanh": False, "fourier": False, "best_transfer": False},
}
PRIMARY = ["full", "no_enrichment", "direct_operator", "no_grid_enrichment",
           "no_parameter_enrichment", "no_consistency"]
COMPONENTS = ["full", "no_dual_tanh", "no_gradient",
              "no_adaptive_weights", "no_fourier", "no_soe"]

def flags(variant):
    if variant not in VARIANTS: raise ValueError("Unknown variant: "+variant)
    return dict(dict(grid_growth=True, parameter_growth=True, consistency=True,
                     gradient=True, adaptive_weights=True, dual_tanh=True, fourier=True, soe=True,
                     best_transfer=True),
                **VARIANTS[variant])

def point_cost(stage):
    return len(stage["alphas"])*(stage["grid"]-1)**2*stage["grid"]

def make_plan(a):
    f=flags(a.variant);nested=s.nested_sets(a.alpha_order)
    a.reference_stage_epochs=list(a.stage_epochs)
    reference=[dict(stage=i,grid=a.grids[i],boundary_points_per_face=a.boundary_points[i],
                    alphas=nested[i],update_cap=a.reference_stage_epochs[i]) for i in range(3)]
    single=not f['grid_growth'] and not f['parameter_growth']
    if single:
        plan=[dict(stage=0,grid=a.grids[-1],boundary_points_per_face=a.boundary_points[-1],
                   alphas=nested[-1],update_cap=sum(a.reference_stage_epochs))]
        if a.phase_seconds is not None:a.phase_seconds=[sum(a.phase_seconds)]
    else:
        plan=[dict(stage=i,grid=a.grids[i] if f['grid_growth'] else a.grids[-1],
                   boundary_points_per_face=a.boundary_points[i] if f['grid_growth'] else a.boundary_points[-1],
                   alphas=nested[i] if f['parameter_growth'] else nested[-1],
                   update_cap=a.reference_stage_epochs[i]) for i in range(3)]
    for i,p in enumerate(plan):
        reference_work=(sum(r['update_cap']*point_cost(r) for r in reference) if single
                        else reference[i]['update_cap']*point_cost(reference[i]))
        if a.budget_mode=='matched_point_work':
            p['update_cap']=reference_work//point_cost(p)
            if p['update_cap']<1:raise ValueError('Update budget is too small.')
        elif a.budget_mode=='matched_time':p['update_cap']=1000000000
        p['residual_points_per_update']=point_cost(p)
        p['reference_residual_point_budget']=reference_work
        p['planned_residual_point_budget']=None if a.budget_mode=='matched_time' else p['update_cap']*point_cost(p)
        p['reference_updates']=sum(a.reference_stage_epochs) if single else a.reference_stage_epochs[i]
    a.stage_epochs=[p['update_cap'] for p in plan]
    return plan,reference

class OrdinaryTanhLayer(core.DualTanhLayer):
    def forward(self,x):return torch.tanh(self.linear(x))

class UniformStageWeights(s.StagewiseHyperNet):
    @property
    def weight_mode(self):return "fixed_uniform"
    def forward(self,losses):return tuple(losses.new_ones(3).unbind())

class RawParameterBranch(nn.Module):
    """No periodic/replicated/padded features; real 3-input parameter MLP."""
    def __init__(self,latent_dim,width):
        super().__init__()
        self.register_buffer("mu_min",torch.tensor([.6,.5,1.]))
        self.register_buffer("mu_max",torch.tensor([.9,1.5,2.]))
        self.net=nn.Sequential(nn.Linear(3,width),nn.Tanh(),
            nn.Linear(width,width),nn.Tanh(),nn.Linear(width,latent_dim))
    def forward(self,mu):
        if mu.dim()==1:mu=mu.unsqueeze(0)
        normalized=2*(mu-self.mu_min)/(self.mu_max-self.mu_min)-1
        return self.net(normalized)

def configure_state(a,device,original_factory):
    f=flags(a.variant)
    core.fast_caputo=_SOE_CAPUTO if f['soe'] else direct_caputo
    state=original_factory(a,device)
    if a.variant=="full":return state
    if not f["fourier"]:
        target=sum(p.numel() for p in state["model"].branch.parameters())
        count=lambda width:width*width+(3+a.latent_dim+2)*width+a.latent_dim
        width=64 if a.variant=='no_fourier' else min(range(1,1025),key=lambda x:(abs(count(x)-target),x))
        state["model"].branch=RawParameterBranch(a.latent_dim,width).to(device)
    if not f["dual_tanh"]:
        for trunk in [state["model"].real_trunk,state["model"].imag_trunk]:
            for layer in trunk.hidden:layer.__class__=OrdinaryTanhLayer
        state["a1"].requires_grad_(False);state["a2"].requires_grad_(False)
    if not f["adaptive_weights"]:
        state["hyper"].__class__=UniformStageWeights
        for p in state["hyper"].parameters():p.requires_grad_(False)
    core.hypernet=state["hyper"]
    state["predictor"]=[p for p in list(state["model"].parameters())+[state["a1"],state["a2"]] if p.requires_grad]
    # Preserve V6 parameter ownership/order, excluding actually disabled parameters.
    state["trainables"]=[p for p in list(state["model"].parameters())+
        list(state["hyper"].parameters())+[state["a1"],state["a2"]] if p.requires_grad]
    state["optimizer"]=torch.optim.Adam(state["trainables"],lr=a.initial_lr)
    return state

def architecture(state,a):
    f=flags(a.variant);m=state["model"]
    return {
        "branch_dimensions":[m.branch.net[0].in_features,m.branch.net[0].out_features,
                             m.branch.net[2].out_features,a.latent_dim],
        "parameter_features":"normalized parameters + sin(pi*mu) + cos(pi*mu)" if f["fourier"] else "normalized raw 3 parameters, no Fourier lift",
        "trunk_input_dimensions":3+a.latent_dim,"trunk_hidden_layers":a.hidden_layers,
        "trunk_width":a.width,"trunk_count":2,
        "trunk_activation":"original adaptive dual-tanh" if f["dual_tanh"] else "ordinary tanh",
        "shared_activation_initial":[.2,.2] if f["dual_tanh"] else None,
        "fixed_fn":5. if f["dual_tanh"] else None,
        "prediction_parameters":sum(p.numel() for p in state["predictor"]),
        "registered_predictor_parameters":sum(p.numel() for p in m.parameters()),
        "loss_weight_dimensions":[3,16,3] if f["adaptive_weights"] else None,
        "loss_weight_parameters":sum(p.numel() for p in state["hyper"].parameters() if p.requires_grad),
        "total_optimizer_parameters":sum(p.numel() for p in state["trainables"]),
        "frozen_unused_hypernet_tensors_saved_for_checkpoint_compatibility":not f["adaptive_weights"],
        "model_structure":str(m),"loss_weight_structure":str(state["hyper"]),
    }

def fpde_without_gradient(tx,tx_sigma,real,imag,data):
    """Literal V6 PDE arithmetic through second derivatives, omit local third derivatives."""
    m=data["M1"];m2=data["M2"];n=data["N"]
    u=real(tx);v=imag(tx);us=real(tx_sigma);vs=imag(tx_sigma)
    um=u.view(m+1,n+1,m2+1);vm=v.view(m+1,n+1,m2+1)
    alpha=float(data["alpha"].detach().cpu().item())
    du=core.fast_caputo(um[1:m,:,1:m2].permute(0,2,1),alpha,data["tau"]).permute(0,2,1).reshape(-1,1).to(us)
    dv=core.fast_caputo(vm[1:m,:,1:m2].permute(0,2,1),alpha,data["tau"]).permute(0,2,1).reshape(-1,1).to(vs)
    gu=torch.autograd.grad(us,tx_sigma,torch.ones_like(us),create_graph=True)[0]
    gv=torch.autograd.grad(vs,tx_sigma,torch.ones_like(vs),create_graph=True)[0]
    ux,uy=gu[:,1:2],gu[:,2:3];vx,vy=gv[:,1:2],gv[:,2:3]
    uxx=torch.autograd.grad(ux,tx_sigma,torch.ones_like(ux),create_graph=True)[0][:,1:2]
    uyy=torch.autograd.grad(uy,tx_sigma,torch.ones_like(uy),create_graph=True)[0][:,2:3]
    vxx=torch.autograd.grad(vx,tx_sigma,torch.ones_like(vx),create_graph=True)[0][:,1:2]
    vyy=torch.autograd.grad(vy,tx_sigma,torch.ones_like(vy),create_graph=True)[0][:,2:3]
    amplitude=us*us+vs*vs;uuu=amplitude*us;vvv=amplitude*vs
    return -dv+data["D"]*(uxx+uyy)+data["beta"]*uuu,du+data["D"]*(vxx+vyy)+data["beta"]*vvv

def physics_terms(model,data,w_grad,variant):
    if flags(variant)["gradient"]:return s.physics_terms(model,data,w_grad)
    real=core.ComponentWrapper(model,data["mu"],"real")
    imag=core.ComponentWrapper(model,data["mu"],"imag")
    txbc=torch.cat([data[k] for k in ["pt_tbc_collocation","pt_x1bc_collocation","pt_x2bc_collocation"]],1)
    tx0=torch.cat([data[k] for k in ["pt_t0_collocation","pt_x1t0_collocation","pt_x2t0_collocation"]],1)
    ub,vb=real(txbc),imag(txbc);u0,v0=real(tx0),imag(tx0)
    mse=nn.MSELoss()
    lbc=mse(data["pt_ubc_collocation"],ub)+mse(data["pt_vbc_collocation"],vb)
    lic=mse(data["pt_ut0_collocation"],u0)+mse(data["pt_vt0_collocation"],v0)
    tx=torch.cat([data[k] for k in ["pt_t_collocation","pt_x1_collocation","pt_x2_collocation"]],1)
    sig=torch.cat([data[k] for k in ["pt_t_sigma","pt_x1_sigma","pt_x2_sigma"]],1)
    fr,fi=fpde_without_gradient(tx,sig,real,imag,data)
    lf=mse(fr,data["pt_f_real_collocation"])+mse(fi,data["pt_f_image_collocation"])
    wf,wb,wi=core.hypernet(torch.stack([lf,lbc,lic],dim=-1))
    return dict(loss=wf*lf+wi*lic+wb*lbc,loss_f=lf,loss_bc=lbc,loss_ic=lic,
                loss_grad=lf.new_zeros(()),loss_grad_space=lf.new_zeros(()),
                loss_grad_time=lf.new_zeros(()),w_g=lf.new_zeros(()),w_f=wf,w_bc=wb,w_ic=wi)


def direct_caputo(values,alpha,tau):
    y=values.to(torch.float64)
    a=torch.as_tensor(alpha,dtype=y.dtype,device=y.device).reshape(())
    delta=y[...,1:]-y[...,:-1];sigma=1-a/2
    scale=tau**(-a)*torch.exp(-torch.lgamma(1-a));output=[]
    for n in range(1,delta.shape[-1]+1):
        value=sigma**(1-a)/(1-a)*delta[...,n-1]
        if n>1:
            j=torch.arange(1,n,dtype=y.dtype,device=y.device);shift=n-1+sigma
            hi=shift-j+1;lo=shift-j
            w0=(hi**(1-a)-lo**(1-a))/(1-a)
            w1=(shift-j+.5)*w0-(hi**(2-a)-lo**(2-a))/(2-a)
            value=value+(delta[...,:n-1]*w0).sum(-1)
            value=value+((delta[...,1:n]-delta[...,:n-1])*w1).sum(-1)
        output.append(value*scale)
    return torch.stack(output,-1)
