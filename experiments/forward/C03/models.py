"Construct solution and inverse-parameter models and transfer inherited optimizer states."
import torch
from c01_source import baseline_original as c01
from c01_source.weight_schemes import install_weights,set_stage
from common import cpu_tree,digest
from protocol import T


def create(args,device):
    s=c01.create_training_objects(args,device)
    install_weights(s,'softmax_noise',args.noise_amplitude,args.weight_lower_bound)
    s['fn'].requires_grad_(False)
    return s


def predict_one(state,tx):
    
    
    q=torch.cat([2*tx[:,0:1]/T-1,2*tx[:,1:2]-1],1)
    return T*torch.cat([state['realnet'](q),state['imagenet'](q)],1)


def predict(states,windows,tx):
    if len(states)==1:return predict_one(states[0],tx)
    
    bounds=tx.new_tensor([w[1] for w in windows[:-1]])
    ids=torch.bucketize(tx[:,0].detach().contiguous(),bounds,right=False)
    result=tx.new_zeros((len(tx),2))
    for j,state in enumerate(states):
        rows=torch.where(ids==j)[0]
        if len(rows):result=result.index_copy(0,rows,predict_one(state,tx.index_select(0,rows)))
    return result


def snapshot(state):
    return cpu_tree(dict(realnet=state['realnet'].state_dict(),imagenet=state['imagenet'].state_dict(),
                         hypernet=state['hypernet'].state_dict(),a1=state['a1'],a2=state['a2'],fn=state['fn'],
                         optimizers=[o.state_dict() for o in state['optimizers']]))


def restore(state,saved,optimizers=True):
    for key in ['realnet','imagenet','hypernet']:state[key].load_state_dict(saved[key])
    with torch.no_grad():
        for key in ['a1','a2','fn']:state[key].copy_(saved[key].to(state[key]))
    if optimizers:
        if len(state['optimizers'])!=len(saved['optimizers']):raise ValueError('Optimizer layout changed.')
        for opt,row in zip(state['optimizers'],saved['optimizers']):opt.load_state_dict(row)


def hashes(state):
    saved=snapshot(state)
    return dict(weights=digest({k:v for k,v in saved.items() if k!='optimizers'}),
                moments=digest([o['state'] for o in saved['optimizers']]))


def clone(parent,args,device):
    s=create(args,device);restore(s,snapshot(parent))
    assert hashes(s)==hashes(parent)
    assert s['a1'].data_ptr()!=parent['a1'].data_ptr()
    return s


def set_lr(states,lr,stage):
    for s in states:
        set_stage(s,stage)
        for opt in s['optimizers']:
            for g in opt.param_groups:g['lr']=lr


def parameter_count(states):
    return sum(p.numel() for p in {id(p):p for s in states for o in s['optimizers']
                                  for g in o.param_groups for p in g['params']}.values())
