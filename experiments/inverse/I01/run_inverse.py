"Run the coefficient-identification cases, methods, and training seeds."
import argparse
import json
from pathlib import Path
import subprocess
import sys
from protocol import CASES,FIGURE_CASES,METHODS,RUN_ROOT,plan,gradient_contract


def main(argv=None):
    q=argparse.ArgumentParser(description=__doc__)
    q.add_argument('action',choices=['plan','train','evaluate','plot','all'])
    q.add_argument('--root',default=RUN_ROOT);q.add_argument('--cases',default=','.join(CASES))
    q.add_argument('--methods',default=','.join(METHODS));q.add_argument('--seeds',default='1234')
    q.add_argument('--device');q.add_argument('--font-dir');q.add_argument('--font-family')
    a,extra=q.parse_known_args(argv)
    if a.action=='plan':print(json.dumps(dict(cases=CASES,methods={m:plan(m) for m in METHODS},max_epochs_per_stage=10000,
                                             default_run_root=RUN_ROOT,
                                             default_gradient_enhancement={m:gradient_contract('alpha',m) for m in METHODS}),indent=2));return
    cases=a.cases.split(',');methods=a.methods.split(',');seeds=[int(v) for v in a.seeds.split(',')]
    if not all(v in CASES for v in cases) or not all(v in METHODS for v in methods):q.error('Unknown case or method.')
    if len(set(cases))!=len(cases) or len(set(methods))!=len(methods) or len(set(seeds))!=len(seeds) or min(seeds)<0:q.error('Use distinct cases/methods and nonnegative seeds.')
    reserved={'--unknown','--initial','--seed','--output','--method','--resume'}
    if any(v.split('=')[0] in reserved for v in extra):q.error('Use --cases/--methods/--seeds here, or train_inverse.py for an individual custom run.')
    if extra and a.action not in ['train','all']:q.error('Extra options apply to training only.')
    if a.action=='all':
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        from plot_style import configure_typography
        configure_typography(plt,font_manager,a.font_dir,a.font_family)
    package=Path(__file__).resolve().parent
    if a.action in ['train','all']:
        for case in cases:
            spec=CASES[case]
            for method in methods:
                for seed in seeds:
                    run=Path(a.root)/case/method/('seed_'+str(seed));logs=Path(a.root)/'logs';logs.mkdir(parents=True,exist_ok=True)
                    command=[sys.executable,'-u',str(package/'train_inverse.py'),'--unknown',spec['unknown'],
                             '--initial',','.join(str(v) for v in spec['initial']),'--method',method,
                             '--seed',str(seed),'--output',str(run),'--resume']
                    if a.device:command+=['--device',a.device]
                    command+=extra
                    with (logs/('%s_%s_seed_%d.log'%(case,method,seed))).open('a',encoding='utf-8') as log:
                        child=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                        try:
                            for line in child.stdout:print(line,end='',flush=True);log.write(line);log.flush()
                            code=child.wait()
                        except KeyboardInterrupt:
                            import signal
                            child.send_signal(signal.SIGINT);child.wait();raise
                        if code:raise SystemExit(code)
    if a.action in ['evaluate','all']:
        from evaluate_inverse import collect
        collect(a.root)
    if a.action in ['plot','all']:
        from plot_inverse import plot
        for seed in seeds:plot(a.root,seed,[c for c in cases if c in FIGURE_CASES],a.font_dir,a.font_family)


if __name__=='__main__':main()
