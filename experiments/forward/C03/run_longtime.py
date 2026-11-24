"Run the long-time forward experiment and its evaluation and plotting workflow."
import argparse
import json
from pathlib import Path
import subprocess
import sys
from protocol import plan


def main(argv=None):
    q=argparse.ArgumentParser(description=__doc__)
    q.add_argument('action',choices=['plan','train','evaluate','plot','all'])
    q.add_argument('--root',default='runs/C03');q.add_argument('--seeds',default='1234')
    q.add_argument('--device',default=None);q.add_argument('--font-dir');q.add_argument('--font-family')
    a,extra=q.parse_known_args(argv)
    if a.action=='plan':print(json.dumps(plan(),indent=2));return
    seeds=[int(v) for v in a.seeds.split(',')]
    if not seeds or len(set(seeds))!=len(seeds) or min(seeds)<0:q.error('Use distinct nonnegative seeds.')
    if extra and a.action not in ['train','all']:q.error('Extra arguments are training options; use the individual evaluation/plot script for other options.')
    if a.action=='all':
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        from plot_style import configure_typography
        configure_typography(plt,font_manager,a.font_dir,a.font_family)
    package=Path(__file__).resolve().parent
    for seed in seeds:
        run=Path(a.root)/('seed_'+str(seed))
        if a.action in ['train','all']:
            logs=Path(a.root)/'logs';logs.mkdir(parents=True,exist_ok=True)
            command=[sys.executable,'-u',str(package/'train_longtime.py'),'--seed',str(seed),'--output',str(run),'--resume']
            if a.device:command+=['--device',a.device]
            command+=extra
            with (logs/('seed_%d_train.log'%seed)).open('a',encoding='utf-8') as log:
                child=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                try:
                    for line in child.stdout:print(line,end='',flush=True);log.write(line);log.flush()
                    code=child.wait()
                except KeyboardInterrupt:
                    import signal
                    child.send_signal(signal.SIGINT);child.wait();raise
                if code:raise SystemExit(code)
        if a.action in ['evaluate','all']:
            from evaluate_longtime import evaluate
            evaluate(run,a.device or 'cpu')
        if a.action in ['plot','all']:
            from plot_longtime import plot
            plot(run,a.font_dir,a.font_family)


if __name__=='__main__':main()
