"""C01 matched comparisons built around the uploaded original, kept pristine."""
import argparse,copy,inspect,types,time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import baseline_original as b
from operator_architecture import OperatorBranch
from direct_reference import direct_caputo
from experiment_utils import *
BASE_FPDE_SOURCE=inspect.getsource(b.fpde_fast)
VARIANTS=['full_original','coordinate_matched','no_gradient','no_attention','no_dual_tanh','no_consistency','no_grid_enrichment','no_soe','fixed_operator_branch']
class FixedWeights(nn.Module):
 def __init__(self,dtype,device):
  super().__init__();self.anchor=nn.Parameter(torch.zeros((),dtype=dtype,device=device),requires_grad=False)
 def forward(self,losses):return torch.ones_like(losses)
class ConditionedNet(nn.Module):
 def __init__(self,trunk,branch,mu):
  super().__init__();self.trunk=trunk;object.__setattr__(self,'branch',branch);self.register_buffer('mu',mu)
 def forward(self,tx):
  z=self.branch(self.mu).expand(tx.shape[0],-1)
  return self.trunk(torch.cat([tx,z],1))
def configure(args,device):
 state=b.create_training_objects(args,device)
 if args.variant in ['no_dual_tanh','coordinate_matched']:
  for net in [state['realnet'],state['imagenet']]:
   for block in net.layers:
    block.forward=types.MethodType(lambda self,x:torch.tanh(self.linear(x)),block)
 if args.variant in ['no_attention','coordinate_matched']:
  state['hypernet']=FixedWeights(state['dtype'],device);state['optimizers']=state['optimizers'][:2]
 if args.variant=='fixed_operator_branch':
  branch=OperatorBranch().to(device);mu=torch.tensor([[args.alpha,1.,1.]],device=device)
  # Branch included in one Adam only; a1/a2 retain original double updates.
  for key in ['realnet','imagenet']:
   trunk=b.FastNetwork(args.extra_hidden_layers,18,args.width,torch.tanh,state['a1'],state['a2'],state['fn']).to(device)
   state[key]=ConditionedNet(trunk,branch,mu)
  ro=torch.optim.Adam(state['realnet'].parameters(),lr=1e-3);io=torch.optim.Adam(state['imagenet'].parameters(),lr=1e-3)
  ro.add_param_group({'params':list(branch.parameters())})
  for opt in [ro,io]:opt.add_param_group({'params':[state['a1'],state['a2']]})
  state['optimizers']=[ro,io,state['optimizers'][2]];state['branch']=branch
 # Isolated fpde variants, not an edit of baseline_original.py.
 src=BASE_FPDE_SOURCE
 if args.variant=='no_soe':
  start=src.index('    # ---- kernel setup ----');end=src.index('    # ---- autodifferentiation terms ----')
  src=src[:start]+'''    Dtu_sigma = direct_caputo(u[1:M],alpha,tau).reshape(-1,1)
    Dtv_sigma = direct_caputo(v[1:M],alpha,tau).reshape(-1,1)
\n'''+src[end:]
 if args.variant in ['no_gradient','coordinate_matched']:
  start=src.index('    # gradient residuals:')
  src=src[:start]+'''    zero=torch.zeros_like(u_sigma)
    return f_real,f_imag,zero,zero,zero,zero
'''
 scope=dict(b.__dict__);scope['direct_caputo']=direct_caputo;exec(src,scope);b.fpde_fast=scope['fpde_fast']
 return state
def snapshot(state,args,sid,epoch,M,N,values,prev):
 c=b.make_checkpoint(state,args,sid,epoch,M,N,values,prev)
 c['experiment']='R2_C01';c['variant']=args.variant
 if 'branch' in state:c['branch']=copy.deepcopy(state['branch'].state_dict())
 return c
def restore(state,c):
 b.apply_checkpoint(state,c)
 if 'branch' in state:state['branch'].load_state_dict(c['branch'])
def main():
 p=argparse.ArgumentParser();p.add_argument('--variant',choices=VARIANTS,default='full_original');p.add_argument('--seed',type=int,default=1234)
 p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');p.add_argument('--output',required=True)
 p.add_argument('--budget-mode',choices=['matched_steps','original_stop'],default='matched_steps')
 p.add_argument('--epochs-per-stage',type=int,default=1000);p.add_argument('--eval-every',type=int,default=100);p.add_argument('--width',type=int,default=60)
 p.add_argument('--extra-hidden-layers',type=int,default=2);p.add_argument('--kernel-mode',choices=['original','stable64'],default='original')
 p.add_argument('--alpha',type=float,default=.9);p.add_argument('--eval-points',type=int,default=4096);p.add_argument('--threads',type=int,default=1);p.add_argument('--smoke',action='store_true')
 a=p.parse_args();args=b.parse_args([])
 for k,v in vars(a).items():setattr(args,k,v)
 if min(a.epochs_per_stage,a.eval_every,a.width,a.eval_points)<1:p.error('Positive counts required')
 if a.smoke:args.grids=[(4,4),(6,6)];args.width=8;args.extra_hidden_layers=1;args.epochs_per_stage=3;args.eval_every=1;args.eval_points=64
 if a.variant=='no_grid_enrichment':
  args.epochs_per_stage*=len(args.grids);args.grids=[args.grids[-1]]
 torch.set_num_threads(a.threads);device=torch.device(a.device);seed_all(a.seed);out=prepare_out(a.output);state=configure(args,device)
 val_tx,val_y=b.independent_points(98101,args.eval_points,device,state['dtype']);test_tx,test_y=b.independent_points(98102,args.eval_points,device,state['dtype'])
 unique={id(x):x for o in state['optimizers'] for g in o.param_groups for x in g['params'] if x.requires_grad}
 config={'suite':'C01','variant':a.variant,'args':vars(args),'environment':environment(device),'trainable_parameter_count':sum(x.numel() for x in unique.values()),
 'selection':'independent validation only','test_used_for_training':False,'same_update_budget':a.budget_mode=='matched_steps',
 'grid_phases':args.grids,'fixed_branch_is_single_parameter_control':a.variant=='fixed_operator_branch',
 'g0_original_and_coordinate_reduced_architecture_identical':True}
 save_json(out/'config.json',config)
 if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
 records=[];stages=[];prev=(None,None,None);best_val=float('inf');best_ck=None;step=0;work=0;synchronize(device);start=time.perf_counter()
 for sid,(M,N) in enumerate(args.grids):
  for opt in state['optimizers']:
   for g in opt.param_groups:g['lr']=b.get_stage_lr(sid)
  data=b.build_stage_data(M,N,args.alpha,device,state['dtype'],args.kernel_mode);best_loss=float('inf');bad=0
  for epoch in range(args.epochs_per_stage if a.budget_mode=='matched_steps' else 10**9):
   active=(None,None,None) if sid==0 or a.variant in ['no_consistency','no_grid_enrichment','coordinate_matched'] else prev
   # Coordinate comparator uses same multi-stage grid/rolling transfer as original.
   if a.variant=='coordinate_matched' and sid>0:active=prev
   wg=0. if a.variant in ['no_gradient','coordinate_matched'] else args.w_grad
   terms=b.training_step(state,data,M,N,args.alpha,wg,active,epoch);step+=1;work+=(M-1)*N
   row={'stage':sid,'epoch':epoch,'global_step':step,**{k:float(v.detach().cpu()) for k,v in terms.items()}}
   evaluate=epoch%args.eval_every==0 or (a.budget_mode=='matched_steps' and epoch+1==args.epochs_per_stage)
   if evaluate:
    l2,tx,pred=b.uploaded_sigma_l2(state,data);vl=b.relative_l2(b.prediction(state,val_tx),val_y)
    prev=(tx.detach(),pred[:,0:1].detach(),pred[:,1:2].detach());row.update(l2=l2,validation_l2=vl)
    ck=snapshot(state,args,sid,epoch,M,N,{'validation_l2':vl,'sigma_l2':l2},prev)
    if vl<best_val:best_val=vl;best_ck=ck;torch.save(ck,out/'best_validation.pt')
    print(f'{a.variant} [Stage {sid}] Epoch {epoch:6d} | Loss {row["loss"]:.3e} | L2 {l2:.3e} | val {vl:.3e}',flush=True)
    if row['loss']<best_loss-args.tol:best_loss=row['loss'];bad=0
    else:bad+=1
   records.append(row)
   if a.budget_mode=='original_stop' and evaluate and bad>=args.patience_evals:break
  torch.save(snapshot(state,args,sid,epoch,M,N,{},prev),out/f'stage_{sid:02d}_last.pt');stages.append({'stage':sid,'epochs':epoch+1,'M':M,'N':N})
 synchronize(device);seconds=time.perf_counter()-start
 torch.save(snapshot(state,args,sid,epoch,M,N,{},prev),out/'last.pt')
 with torch.no_grad():last=metrics(b.prediction(state,test_tx),test_y)
 restore(state,best_ck)
 selected=metrics(b.prediction(state,test_tx),test_y)
 np.savez(out/'test_predictions.npz',tx=test_tx.cpu().numpy(),exact=test_y.cpu().numpy(),prediction=b.prediction(state,test_tx).cpu().numpy())
 result={'suite':'C01','variant':a.variant,'seed':a.seed,'status':'SMOKE_ONLY' if a.smoke else 'TRAINING_COMPLETED',
 'validation_best':best_val,'test':selected,'test_last':last,'steps':step,'residual_points_processed':work,'training_seconds':seconds,
 'inference_seconds_per_eval_set':timed_inference(lambda:b.prediction(state,test_tx),device),'peak_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
 'parameter_count':config['trainable_parameter_count'],'stages':stages}
 save_csv(out/'training.csv',records);save_json(out/'summary.json',result)
if __name__=='__main__':main()
