"Run the one-dimensional potential reconstruction under the two supervision settings."
import argparse
import json
from pathlib import Path
import subprocess
import sys
from protocol import METHODS,PROBLEMS,plan,supervised_x


def main(argv=None):
    q=argparse.ArgumentParser(description=__doc__)
    q.add_argument('action',choices=['plan','train','evaluate','plot','all'])
    q.add_argument('--root',default='runs/I02_gradient_v2');q.add_argument('--problem',choices=PROBLEMS,default='manufactured_series')
    q.add_argument('--supervision',default='3,2');q.add_argument('--methods',default=','.join(METHODS));q.add_argument('--seeds',default='1234')
    q.add_argument('--device');q.add_argument('--font-dir');q.add_argument('--font-family')
    q.add_argument('--reference');q.add_argument('--reference-nx',type=int,default=256);q.add_argument('--reference-nt',type=int,default=1024)
    a,extra=q.parse_known_args(argv)
    if a.action=='plan':print(json.dumps(dict(problem=a.problem,methods={m:plan(m) for m in METHODS},supervised={n:supervised_x(n) for n in [2,3]},max_epochs_per_stage=10000),indent=2));return
    try:counts=[int(v) for v in a.supervision.split(',')];seeds=[int(v) for v in a.seeds.split(',')]
    except ValueError:q.error('Invalid counts/seeds.')
    methods=a.methods.split(',')
    if not counts or not all(n in [2,3] for n in counts) or not all(m in METHODS for m in methods) or min(seeds)<0:q.error('Invalid selection.')
    if any(len(set(v))!=len(v) for v in [counts,seeds,methods]):q.error('Use distinct settings.')
    reserved={'--method','--supervised-points','--seed','--output','--resume'}
    if any(v.split('=')[0] in reserved for v in extra):q.error('Use --methods/--supervision/--seeds here, or train_potential.py for a custom run.')
    if extra and a.action not in ['train','all']:q.error('Extra options apply to training only.')
    if a.action=='all':
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        from plot_style import configure_typography
        configure_typography(plt,font_manager,a.font_dir,a.font_family)
    root=Path(a.root);package=Path(__file__).resolve().parent
    reference=a.reference or str(Path('runs/I02')/('reference_zero_source_%dx%d.npz'%(a.reference_nx,a.reference_nt)))
    if a.action in ['train','all']:
        for count in counts:
            for method in methods:
                for seed in seeds:
                    run=root/a.problem/('eta_'+str(count))/method/('seed_'+str(seed));logs=root/'logs';logs.mkdir(parents=True,exist_ok=True)
                    command=[sys.executable,'-u',str(package/'train_potential.py'),'--method',method,'--problem',a.problem,
                             '--supervised-points',str(count),'--seed',str(seed),'--output',str(run),'--resume',
                             '--reference',reference,'--reference-nx',str(a.reference_nx),'--reference-nt',str(a.reference_nt)]
                    if a.device:command+=['--device',a.device]
                    command+=extra
                    with (logs/('%s_eta%d_%s_seed%d.log'%(a.problem,count,method,seed))).open('a',encoding='utf-8') as log:
                        child=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                        try:
                            for line in child.stdout:print(line,end='',flush=True);log.write(line);log.flush()
                            code=child.wait()
                        except KeyboardInterrupt:
                            import signal
                            child.send_signal(signal.SIGINT);child.wait();raise
                        if code:raise SystemExit(code)
    if a.action in ['evaluate','all']:
        from evaluate_potential import collect
        collect(root)
    if a.action in ['plot','all']:
        from plot_potential import plot
        for seed in seeds:plot(root,a.problem,counts,seed,a.font_dir,a.font_family)


if __name__=='__main__':main()
