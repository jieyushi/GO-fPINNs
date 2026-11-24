"""Count actual unique optimizer-registered trainable tensors on CPU; no training."""
import argparse
import csv
import json
from pathlib import Path
import sys

from experiment_plan import SUITES, make_plan, grids_string
from run_case1 import csv_list


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suites', default='all')
    p.add_argument('--output', type=Path, default=Path('parameter_counts'))
    a = p.parse_args()
    import torch
    import train_case1 as trainer
    from experiment_utils import seed_all
    selected = list(SUITES) if a.suites == 'all' else csv_list(a.suites)
    jobs = make_plan(selected, [1234])
    rows = []
    for job in jobs:
        seed_all(1234)
        if job['kind'] == 'fixed_ablation':
            import train_c01_weights as legacy
            args = legacy.parse_args(['--output', 'unused', '--variant', job['variant'], '--device', 'cpu'])
            state = legacy.configure(args, torch.device('cpu'))
        else:
            argv = ['--output', 'unused', '--device', 'cpu']
            for key, value in job['settings'].items():
                argv += ['--'+key.replace('_','-'), grids_string(value) if key == 'grids' else str(value)]
            args = trainer.parse_args(argv)
            state = trainer.configure(args, torch.device('cpu'))
        unique = {id(p):p for opt in state['optimizers'] for group in opt.param_groups
                  for p in group['params'] if p.requires_grad}
        weight_ids = {id(p) for p in state['hypernet'].parameters() if p.requires_grad}
        shared_ids = {id(state[k]) for k in ('a1', 'a2')}
        row = dict(experiment_id=job['experiment_id'],
                   optimizer_registered_unique_parameters=sum(p.numel() for p in unique.values()),
                   weighting_MLP_parameters=sum(p.numel() for i,p in unique.items() if i in weight_ids),
                   shared_activation_parameters=sum(p.numel() for i,p in unique.items() if i in shared_ids),
                   hidden_blocks=args.extra_hidden_layers+1, width=args.width,
                   repeated_optimizer_entries_deduplicated=True,
                   fn_optimizer_registered=id(state['fn']) in unique)
        rows.append(row)
        print('%-48s %8d' % (row['experiment_id'], row['optimizer_registered_unique_parameters']))
    a.output.mkdir(parents=True, exist_ok=True)
    with (a.output/'parameter_counts.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (a.output/'parameter_counts.json').write_text(json.dumps(rows, indent=2))
    print('Counts are architecture-dependent, not seed-dependent. No optimizer step was executed.')


if __name__ == '__main__':
    main()
