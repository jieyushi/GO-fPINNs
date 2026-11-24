"""Run the experiment workflows, numerical figures, and result-table exports.

Commands are executed with the appropriate case working directory. Use
--dry-run to inspect the complete command sequence without running it.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = json.loads((ROOT / 'configs/experiments.json').read_text())


def commands(case, stage, args):
    item = REGISTRY.get(case, {'path':'experiments/operator/case1/legacy_alpha_ablation'})
    cwd = ROOT / item['path']
    result = args.results_root / case
    seeds = args.seeds or ('1234,2024,3407' if case in {'C01', 'C02', 'OPA'}
                           else '1234,1235,1236')
    first = seeds.split(',')[0]
    jobs = []

    def add(script, values=(), directory=None):
        jobs.append((directory or cwd, [sys.executable, '-u', script] + list(map(str, values))))

    def native(action, extra=(), device=False, font=False, seed=None):
        values = [action]
        if case == 'C02':
            values += ['--output', result, '--seeds', seeds, '--plot-seed', first]
        elif case in {'C03', 'I01', 'I02', 'I03'}:
            values += ['--root', result, '--seeds', seeds]
        elif case in {'OP1', 'OP2'}:
            values += ['--output', result, '--seeds', seeds, '--plot-seed', first]
            if case == 'OP1' and args.include_l1_baseline:
                values += ['--include-l1-baseline']
        elif case == 'OP3':
            values += ['--root', result, '--seeds', seeds]
        elif case == 'OP4':
            values += ['--runs-root', result/'runs', '--paper-root', result/'paper',
                       '--cache', result/'references/cache',
                       '--adi-root', result/'references/imported_adi',
                       '--generated-root', result/'references/generated_adi',
                       '--seeds', seeds, '--seed', seed or first]
        if device:
            values += ['--device', args.device]
        if case == 'I02':
            values += ['--problem', 'manufactured_series']
        if font:
            values += ['--font-family', args.font_family]
        add(item['entrypoint'], values + list(extra))

    if case == 'OPA':
        cwd = ROOT / 'experiments/operator/case1/legacy_alpha_ablation'
        if stage == 'plan':
            return jobs
        if stage == 'train':
            add('run_suite.py', ['--preset', 'all', '--seeds', seeds,
                '--device', args.device, '--epochs-per-stage', 5000,
                '--budget-mode', 'matched_steps', '--output', result, '--resume'])
        elif stage in {'evaluate', 'tables'}:
            add('collect_results.py', ['--root', result, '--expected-seeds', seeds])
        return jobs

    if stage == 'plan':
        if case == 'C01':
            add(item['entrypoint'], ['--suites', 'all', '--seeds', seeds,
                                    '--output', result, '--plan-only'])
        else:
            native('plan')
    elif stage == 'reference':
        if case == 'C02': native('reference', device=True)
        elif case in {'OP3', 'OP4'}: native('references')
    elif stage == 'train':
        if case == 'C01':
            add(item['entrypoint'], ['--suites', 'all', '--seeds', seeds,
                '--output', result, '--device', args.device, '--resume'])
        else:
            native('train', device=True)
    elif stage == 'evaluate':
        if case == 'C01': add('collect_case1.py', ['--root', result])
        elif case in {'C02', 'C03'}: native('evaluate', device=True)
        elif case in {'I01', 'I02', 'I03'}: native('evaluate')
        elif case in {'OP1', 'OP2'}:
            native('collect')
            native('export', device=True)
        elif case == 'OP3': native('export', device=True)
        elif case == 'OP4':
            for seed in seeds.split(','):
                native('evaluate', device=True, seed=seed)
            native('collect')
    elif stage == 'figures':
        if case == 'C01': add('plot_case1.py', ['--root', result, '--seed', first])
        elif case in {'C02', 'C03', 'I01', 'I02', 'I03', 'OP1', 'OP2'}:
            native('plot', font=True)
        elif case == 'OP3':
            data = result/'reports/case3_lhs/data'
            for script, name in [('plot_stage_errors.py', 'Figure_add3'),
                                 ('plot_solution_comparison.py', 'Figure_add2')]:
                add(script, ['--data', data, '--seed', first,
                             '--output', data.parent/'figures'/name])
        elif case == 'OP4': native('figures', device=True, font=True)
    elif stage == 'convergence':
        if case == 'C02': native('convergence', device=True)
        elif case == 'OP3': native('converge')
        elif case == 'OP4':
            for mode in ['spatial', 'temporal']:
                add('check_reference_convergence.py', ['--mode', mode,
                    '--alphas', '.5', '--nt', 512, '--nx', 1024,
                    '--output', result/'convergence', '--resume'])
    elif stage == 'benchmark' and case == 'OP4':
        add('benchmark_bright_hump_matched_grid.py', ['--project-root', cwd,
            '--run', result/'runs'/('seed_'+first), '--cache', result/'references/cache',
            '--adi-root', result/'references/generated_adi', '--alphas', '.5',
            '--device', args.device, '--query-nx', 1024, '--query-nt', 512,
            '--threads', 1, '--repeats', 10, '--output', result/'timing'])
        add('summarize_timing.py', ['--run', result/'runs'/('seed_'+first),
                                   '--timing', result/'timing'])
    return jobs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cases', nargs='*', help='Case IDs; omitted means all ten main cases.')
    parser.add_argument('--stage', default='plan', choices=[
        'plan', 'reference', 'train', 'evaluate', 'figures', 'tables',
        'convergence', 'benchmark', 'diagrams', 'all'])
    parser.add_argument('--results-root', type=Path, default=ROOT/'runs')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seeds', help='Comma-separated distinct training seeds; defaults are case-specific.')
    parser.add_argument('--font-family', default='DejaVu Serif')
    parser.add_argument('--include-l1-baseline', action='store_true',
                        help='Include the explicitly reconstructed OP1 L1 baseline.')
    parser.add_argument('--include-operator-ablations', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args(argv)
    args.results_root = args.results_root.expanduser().resolve()
    if args.seeds:
        try:
            numbers = [int(v) for v in args.seeds.split(',')]
            if min(numbers) < 0 or len(set(numbers)) != len(numbers): raise ValueError
        except ValueError:
            parser.error('Use distinct nonnegative integer seeds.')
    if args.list:
        for key, value in REGISTRY.items(): print(f'{key:4s} {value["title"]}')
        print('OPA  Fixed-budget alpha-operator component ablations')
        return 0
    cases = args.cases or list(REGISTRY)
    if args.include_operator_ablations and 'OPA' not in cases: cases.append('OPA')
    if len(set(cases)) != len(cases) or set(cases) - (set(REGISTRY) | {'OPA'}):
        parser.error('Choose distinct case IDs from --list.')
    stages = ['reference', 'train', 'evaluate', 'figures', 'convergence', 'benchmark'] \
        if args.stage == 'all' else [args.stage]
    jobs = []
    for case in cases:
        for stage in stages: jobs.extend(commands(case, stage, args))
    if args.stage in {'all', 'tables'}:
        jobs.append((ROOT, [sys.executable, '-u', 'scripts/export_tables.py',
            '--results-root', str(args.results_root), '--output', str(args.results_root/'tables')]))
    if args.stage in {'all', 'diagrams'}:
        jobs.append((ROOT, [sys.executable, '-u', 'scripts/plot_method_diagrams.py',
            '--output', str(args.results_root/'method_diagrams')]))
    if args.stage in {'all', 'figures', 'diagrams'}:
        jobs.append((ROOT, [sys.executable, '-u', 'scripts/collect_figures.py',
            '--results-root', str(args.results_root), '--seed', (args.seeds or '1234').split(',')[0],
            '--output', str(args.results_root/'manuscript_figures')]))
    if not jobs:
        print('No command is defined for the selected case/stage combination.')
        return 0
    for cwd, command in jobs:
        print('Directory: ' + str(cwd), flush=True)
        print('Command: ' + shlex.join(command), flush=True)
        if not args.dry_run: subprocess.run(command, cwd=cwd, check=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
