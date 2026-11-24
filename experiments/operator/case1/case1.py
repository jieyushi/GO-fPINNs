"""Case 1 workflow: plan, train, collect, export, plot, all, smoke.

Default main experiment: three SEPARATE operators and three training seeds.
The historical fixed-budget alpha ablations live in legacy_alpha_ablation/.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent


def main(argv=None):
    q=argparse.ArgumentParser(description=__doc__)
    q.add_argument('command',choices=['plan','train','collect','export','plot','all','smoke'])
    q.add_argument('--operators',default='alpha,beta,gamma')
    q.add_argument('--seeds',default='1234,1235,1236')
    q.add_argument('--plot-seed',type=int)
    q.add_argument('--device',default='auto')
    q.add_argument('--output',type=Path,default=Path('runs/Case1'))
    q.add_argument('--max-epochs',type=int,default=5000,help='Maximum PER STAGE, not fixed epochs.')
    q.add_argument('--eval-every',type=int,default=100)
    q.add_argument('--patience-evals',type=int,default=10)
    q.add_argument('--include-l1-baseline',action='store_true',
                   help='Also run optional L1-DeepONet RECONSTRUCTION; not the original baseline source.')
    q.add_argument('--font-family',default='Times New Roman')
    q.add_argument('--font-dir',type=Path)
    q.add_argument('--dpi',type=int,default=300)
    a=q.parse_args(argv)
    try:seeds=[int(v) for v in a.seeds.split(',')]
    except ValueError:q.error('Seeds must be comma-separated integers.')
    operators=a.operators.split(',')
    if not seeds or min(seeds)<0 or len(seeds)!=len(set(seeds)):
        q.error('Distinct nonnegative seeds required.')
    if not operators or len(operators)!=len(set(operators)) or not set(operators)<=set(['alpha','beta','gamma']):
        q.error('Operators must be a non-repeated subset of alpha,beta,gamma.')
    if not 1<=a.max_epochs<=5000 or min(a.eval_every,a.patience_evals,a.dpi)<1:
        q.error('Cap must be <=5000; checks, patience and dpi must be positive.')
    plot_seed=a.plot_seed if a.plot_seed is not None else seeds[0]
    if plot_seed not in seeds:q.error('--plot-seed must belong to --seeds.')
    out=a.output.resolve() if a.output.is_absolute() else ROOT/a.output
    plan={'case':1,'operators':operators,'three_separately_trained_models':True,
          'inputs':{'alpha':['alpha',1,1],'beta':[.8,'beta',1],'gamma':[.8,1,'gamma']},
          'open_ranges':{'alpha':[.6,.9],'beta':[.5,1.5],'gamma':[1,2]},
          'train_counts':[4,5,6],'validation_count':6,'test_count':30,
          'grid_nodes_per_axis':[6,8,10],'grid_intervals_per_axis':[5,7,9],
          'evaluation_points_per_parameter':4096,'stage_learning_rates':[.001,.0005,.00025],
          'max_epochs_per_stage':a.max_epochs,
          'early_stop':{'check_every_updates':a.eval_every,'patience_checks':a.patience_evals,'min_delta':0},
          'stop_monitor':'unweighted fixed clean validation LF+LBC+LIC',
          'checkpoint_selection':'validation macro relative L2, never test',
          'training_seeds':seeds,'split_seed':20261017,'data_noise_level':0.,
          'optional_baseline':'L1-DeepONet reconstruction; original baseline source unavailable',
          'controlled_ablation':'separate legacy entry, fixed 5000x3, 20 test orders'}
    if a.command=='plan':print(json.dumps(plan,indent=2));return plan
    smoke=a.command=='smoke'
    if smoke and a.output==Path('runs/Case1'):out=ROOT/'scratch_runs/smoke'
    if a.command in ['all','plot','smoke']:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        from plot_style import configure_typography
        # Fail BEFORE training if the requested paper font is unavailable.
        configure_typography(plt,font_manager,a.font_dir,a.font_family)
    if a.device=='auto':
        import torch
        a.device='cuda:0' if torch.cuda.is_available() else 'cpu'
    def run(script,args):
        subprocess.run([sys.executable,'-u',str(ROOT/script)]+list(map(str,args)),cwd=ROOT,check=True)
    methods=['go']+(['l1_reimplementation'] if a.include_l1_baseline else [])
    if a.command in ['train','all','smoke']:
        for operator in operators:
            for method in methods:
                for seed in seeds:
                    script='train_single_operator.py' if method=='go' else 'train_l1_deeponet.py'
                    args=['--operator',operator,'--seed',seed,'--device',a.device,
                          '--output',out/'operators'/operator/method/('seed_'+str(seed)),
                          '--epochs-per-stage',a.max_epochs,'--eval-every',a.eval_every,
                          '--patience-evals',a.patience_evals,'--resume']
                    if smoke:args+=['--smoke']
                    run(script,args)
    common=['--root',out,'--operators',a.operators,'--seeds',a.seeds]
    if a.include_l1_baseline:common+=['--include-l1-baseline']
    if a.command in ['collect','all','smoke']:
        run('collect_case1.py',common+(['--allow-smoke'] if smoke else []))
    if a.command in ['export','all','smoke']:
        run('export_case1.py',['--root',out,'--operators',a.operators,'--seed',plot_seed,'--device',a.device])
    if a.command in ['plot','all','smoke']:
        args=['--root',out,'--operators',a.operators,'--seed',plot_seed,
              '--font-family',a.font_family,'--dpi',a.dpi]
        if a.font_dir:args+=['--font-dir',a.font_dir]
        run('plot_case1.py',args)


if __name__=='__main__':main()
