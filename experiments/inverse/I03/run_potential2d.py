"Run the two-dimensional potential reconstruction under the two supervision settings."
import argparse
import json
from pathlib import Path
import subprocess
import sys
from protocol import METHODS,plan,supervised_xy


def main(argv=None):
    q=argparse.ArgumentParser(description=__doc__);q.add_argument('action',choices=['plan','train','evaluate','plot','all'])
    q.add_argument('--root',default='runs/I03_gradient_v2');q.add_argument('--supervision',default='9,4')
    q.add_argument('--methods',default=','.join(METHODS));q.add_argument('--seeds',default='1234')
    q.add_argument('--device');q.add_argument('--font-dir');q.add_argument('--font-family')
    a,extra=q.parse_known_args(argv)
    if a.action=='plan':print(json.dumps(dict(methods={m:plan(m) for m in METHODS},supervision={n:supervised_xy(n).tolist() for n in [9,4]},max_epochs_per_stage=10000),indent=2));return
    try:counts=[int(v) for v in a.supervision.split(',')];seeds=[int(v) for v in a.seeds.split(',')]
    except ValueError:q.error('Invalid counts/seeds.')
    methods=a.methods.split(',')
    if not counts or not all(n in [4,9] for n in counts) or not all(m in METHODS for m in methods) or min(seeds)<0:q.error('Invalid selection.')
    if any(len(set(v))!=len(v) for v in [counts,seeds,methods]):q.error('Use distinct settings.')
    reserved={'--method','--supervised-points','--seed','--output','--resume'}
    if any(v.split('=')[0] in reserved for v in extra):q.error('Use --methods/--supervision/--seeds here, or train_potential2d.py for one custom run.')
    if extra and a.action not in ['train','all']:q.error('Extra options apply to training only.')
    if a.action=='all':
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        from plot_style import configure_typography
        configure_typography(plt,font_manager,a.font_dir,a.font_family)
    root=Path(a.root);package=Path(__file__).resolve().parent
    if a.action in ['train','all']:
        for count in counts:
            for method in methods:
                for seed in seeds:
                    run=root/('eta_'+str(count))/method/('seed_'+str(seed));logs=root/'logs';logs.mkdir(parents=True,exist_ok=True)
                    command=[sys.executable,'-u',str(package/'train_potential2d.py'),'--method',method,'--supervised-points',str(count),
                             '--seed',str(seed),'--output',str(run),'--resume']
                    if a.device:command+=['--device',a.device]
                    command+=extra
                    with (logs/('eta%d_%s_seed%d.log'%(count,method,seed))).open('a',encoding='utf-8') as log:
                        child=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                        try:
                            for line in child.stdout:print(line,end='',flush=True);log.write(line);log.flush()
                            code=child.wait()
                        except KeyboardInterrupt:
                            import signal
                            child.send_signal(signal.SIGINT);child.wait();raise
                        if code:raise SystemExit(code)
    if a.action in ['evaluate','all']:
        from evaluate_potential2d import collect
        collect(root)
    if a.action in ['plot','all']:
        from plot_potential2d import plot
        for seed in seeds:plot(root,counts,seed,a.font_dir,a.font_family)


if __name__=='__main__':main()
