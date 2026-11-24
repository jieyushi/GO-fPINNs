"""Complete forward Case 2 workflow; first refinement, then spatial decomposition."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from protocol import plan,LABELS

ROOT=Path(__file__).resolve().parent


def main(argv=None):
    q=argparse.ArgumentParser(description=__doc__)
    q.add_argument('command',choices=['plan','train','reference','convergence','evaluate','plot','all','smoke'])
    q.add_argument('--methods',default='go,fpinn,fl1')
    q.add_argument('--seeds',default='1234,2024,3407')
    q.add_argument('--plot-seed',type=int)
    q.add_argument('--device',default='auto')
    q.add_argument('--output',type=Path,default=Path('runs/C02_gradient_v2'))
    q.add_argument('--max-epochs-per-stage',type=int,default=0)
    q.add_argument('--eval-every',type=int,default=100)
    q.add_argument('--patience-evals',type=int,default=10)
    q.add_argument('--kernel-mode',choices=['original','stable64'],default='original')
    q.add_argument('--w-grad',type=float,default=.001,
                   help='GO full discrete residual spatial/time GE; weight=w_grad*exp(-3*LF.detach()); baselines disable GE.')
    q.add_argument('--reference-space-intervals',type=int,default=800)
    q.add_argument('--reference-time-intervals',type=int,default=2048)
    q.add_argument('--font-dir',type=Path)
    q.add_argument('--font-family',default='Times New Roman')
    q.add_argument('--dpi',type=int,default=300)
    a=q.parse_args(argv);methods=a.methods.split(',')
    try:seeds=[int(v) for v in a.seeds.split(',')]
    except ValueError:q.error('Comma-separated integer seeds required.')
    if not methods or not set(methods)<=set(LABELS) or len(set(methods))!=len(methods):q.error('Choose a subset of go,fpinn,fl1.')
    if not seeds or min(seeds)<0 or len(set(seeds))!=len(seeds):q.error('Distinct nonnegative seeds required.')
    if a.max_epochs_per_stage<0 or min(a.eval_every,a.patience_evals)<1:q.error('Invalid cap/stopping settings.')
    seed=a.plot_seed if a.plot_seed is not None else seeds[0]
    if seed not in seeds:q.error('Plot seed must be included.')
    out=a.output.resolve() if a.output.is_absolute() else ROOT/a.output
    if a.command=='plan':
        result={m:plan(m) for m in methods}
        print(json.dumps(result,indent=2));return result
    smoke=a.command=='smoke'
    if smoke and a.output==Path('runs/C02_gradient_v2'):out=ROOT/'scratch_runs/C02_gradient_v2_smoke'
    if a.command in ['plot','all','smoke']:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        from plot_style import configure_typography
        configure_typography(plt,font_manager,a.font_dir,a.font_family)
    if a.device=='auto':
        import torch
        a.device='cuda:0' if torch.cuda.is_available() else 'cpu'
    def run(script,args):
        subprocess.run([sys.executable,'-u',str(ROOT/script)]+list(map(str,args)),cwd=ROOT,check=True)
    if a.command in ['reference','all','smoke']:
        args=['--output',out/'reference/reference.npz','--space-intervals',a.reference_space_intervals,
              '--time-intervals',a.reference_time_intervals]
        run('numerical_reference.py',args+(['--smoke'] if smoke else []))
    if a.command=='convergence':run('convergence_check.py',['--output',out/'reference/convergence'])
    if a.command in ['train','all','smoke']:
        for method in methods:
            for training_seed in seeds:
                args=['--method',method,'--seed',training_seed,'--output',out/method/('seed_'+str(training_seed)),
                      '--device',a.device,'--max-epochs-per-stage',a.max_epochs_per_stage,
                      '--eval-every',a.eval_every,'--patience-evals',a.patience_evals,
                      '--kernel-mode',a.kernel_mode,'--w-grad',a.w_grad,'--resume']
                run('train_largescale.py',args+(['--smoke'] if smoke else []))
    if a.command in ['evaluate','all','smoke']:
        args=['--root',out,'--methods',a.methods,'--seeds',a.seeds,'--plot-seed',seed,'--device',a.device]
        if smoke:args+=['--allow-smoke','--space-points',61,'--time-points',31]
        run('evaluate_largescale.py',args)
    if a.command in ['plot','all','smoke']:
        args=['--root',out,'--seed',seed,'--font-family',a.font_family,'--dpi',a.dpi]
        if a.font_dir:args+=['--font-dir',a.font_dir]
        run('plot_largescale.py',args)


if __name__=='__main__':main()
