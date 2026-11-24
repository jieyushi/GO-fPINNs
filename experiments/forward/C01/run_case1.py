"""Run any subset of the Case-1 experiments with separate resumable outputs."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

from experiment_plan import ABLATIONS, SUITES, grids_string, make_plan

ROOT = Path(__file__).resolve().parent


def csv_list(value, cast=str):
    values = [cast(s.strip()) for s in value.split(',') if s.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError('Expected a nonempty list without duplicates: '+value)
    return values


def arguments(argv=None, default_suites=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suites', default=default_suites or 'all')
    p.add_argument('--seeds', default='1234,2024,3407')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--output', type=Path, default=Path('runs/C01_case1_complete'))
    p.add_argument('--ablation-root', type=Path, default=None,
                   help='Existing original ablation root; unchanged core sources/configurations can resume or skip there.')
    p.add_argument('--ablation-variants', default=','.join(ABLATIONS))
    p.add_argument('--ablation-epochs-per-stage', type=int, default=2000)
    p.add_argument('--depths', default='2,3,4,5', help='Total hidden-layer depths for the activation sweep; the scalar output layer is excluded.')
    p.add_argument('--widths', default='20,40,60,80')
    p.add_argument('--width', type=int, default=60)
    p.add_argument('--extra-hidden-layers', type=int, default=2,
                   help='Additional hidden blocks after the first activated affine mapping; 2 gives 3 hidden layers in total.')
    p.add_argument('--eval-every', type=int, default=100)
    p.add_argument('--patience-evals', type=int, default=10)
    p.add_argument('--loss-min-delta', type=float, default=0.0)
    p.add_argument('--max-epochs-per-stage', type=int, default=0,
                   help='Optional explicit safety cap for non-ablation runs; 0 means loss stopping only.')
    p.add_argument('--eval-points', type=int, default=4096)
    p.add_argument('--validation-seed', type=int, default=98101)
    p.add_argument('--test-seed', type=int, default=98102)
    p.add_argument('--kernel-mode', choices=['original', 'stable64'], default='original')
    p.add_argument('--network-dtype', choices=['float32', 'float64'], default='float32')
    p.add_argument('--shared-update', choices=['legacy_double', 'once'], default='legacy_double')
    p.add_argument('--noise-amplitude', type=float, default=0.05)
    p.add_argument('--weight-lower-bound', type=float, default=1e-6)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--defer-shared-full', action='store_true',
                   help='Assign the shared Full to the other terminal; collect it later without a second launch.')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--plan-only', action='store_true')
    a = p.parse_args(argv)
    a.selected_suites = list(SUITES) if a.suites == 'all' else csv_list(a.suites)
    if set(a.selected_suites)-set(SUITES):
        p.error('Unknown suite; choices: '+','.join(SUITES))
    if min(a.width, a.eval_every, a.patience_evals, a.eval_points, a.ablation_epochs_per_stage, a.threads) < 1:
        p.error('Counts must be positive.')
    if a.max_epochs_per_stage < 0 or a.loss_min_delta < 0 or a.extra_hidden_layers < 0:
        p.error('Invalid negative value.')
    return a


def jobs_for(a):
    jobs = make_plan(a.selected_suites, csv_list(a.seeds, int), csv_list(a.depths, int),
                     csv_list(a.widths, int), a.width, a.extra_hidden_layers,
                     csv_list(a.ablation_variants))
    for job in jobs:
        if job['kind'] == 'fixed_ablation':
            job['output'] = str((a.ablation_root or a.output/'ablation')/'softmax_noise'/job['variant']/('seed_%d' % job['seed']))
        else:
            job['output'] = str(a.output/job['experiment_id']/('seed_%d' % job['seed']))
        job['output'] = str(Path(job['output']).resolve())
    return jobs


def command_for(job, a):
    legacy = job['kind'] == 'fixed_ablation'
    script = ROOT/'legacy_ablation/train_c01_weights.py' if legacy else ROOT/'train_case1.py'
    argv = [sys.executable, '-u', str(script), '--device', a.device, '--seed', str(job['seed']),
            '--output', str(Path(job['output']).resolve()), '--weight-scheme', 'softmax_noise',
            '--noise-amplitude', str(a.noise_amplitude), '--weight-lower-bound', str(a.weight_lower_bound),
            '--kernel-mode', a.kernel_mode, '--network-dtype', a.network_dtype,
            '--shared-update', a.shared_update, '--eval-points', str(a.eval_points),
            '--validation-seed', str(a.validation_seed), '--test-seed', str(a.test_seed),
            '--threads', str(a.threads)]
    if legacy:
        argv += ['--variant', job['variant'], '--budget-mode', 'matched_steps',
                 '--epochs-per-stage', str(a.ablation_epochs_per_stage),
                 '--width', str(a.width), '--extra-hidden-layers', str(a.extra_hidden_layers),
                 '--grids', '10x10,20x20,30x30,40x40,50x50',
                 '--eval-every', '100', '--patience-evals', '10', '--tol', '1e-8']
    else:
        settings = job['settings']
        argv += ['--experiment-id', job['experiment_id'], '--budget-mode', 'original_stop',
                 '--eval-every', str(a.eval_every), '--patience-evals', str(a.patience_evals),
                 '--tol', str(a.loss_min_delta), '--max-epochs-per-stage', str(a.max_epochs_per_stage)]
        for key, value in settings.items():
            argv += ['--'+key.replace('_', '-'), grids_string(value) if key == 'grids' else str(value)]
    if a.resume:
        argv.append('--resume')
    if a.smoke:
        argv.append('--smoke')
    return argv


def main(argv=None, default_suites=None):
    a = arguments(argv, default_suites)
    jobs = jobs_for(a)
    if a.plan_only:
        print(json.dumps({'jobs': jobs, 'number_of_jobs': len(jobs),
                          'non_ablation_stop': {'every': a.eval_every, 'patience': a.patience_evals,
                                               'absolute_min_delta': a.loss_min_delta,
                                               'safety_cap': a.max_epochs_per_stage},
                          'ablation_updates_per_stage': a.ablation_epochs_per_stage}, indent=2))
        return 0
    a.output.mkdir(parents=True, exist_ok=True)
    # A suite-specific plan prevents two disjoint terminal queues from overwriting each other's manifests.
    plan_path = a.output/('plan_'+('_'.join(a.selected_suites))+'.json')
    plan_path.write_text(json.dumps({'jobs': jobs, 'smoke': a.smoke}, indent=2), encoding='utf-8')
    for job in jobs:
        if a.defer_shared_full and job['experiment_id'] == 'full/GO_fPINNs':
            print('Shared Full is assigned to the other queue: '+job['output'], flush=True)
            continue
        command = command_for(job, a)
        print('Running %s / seed=%d / %s -> %s' %
              (job['experiment_id'], job['seed'], job['kind'], job['output']), flush=True)
        log_path = a.output/'logs'/job['experiment_id']/('seed_%d.log' % job['seed'])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open('a', encoding='utf-8') as log:
            log.write('\nCOMMAND '+json.dumps(command)+'\n')
            log.flush()
            process = subprocess.Popen(command, cwd=str(ROOT), stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            try:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
            except KeyboardInterrupt:
                # The child handles SIGINT and saves its optimizer/RNG/data progress.
                if process.poll() is None:
                    import signal
                    process.send_signal(signal.SIGINT)
                    process.wait()
                return 130
        if code:
            raise RuntimeError('Run exited %d; inspect %s, then use the same command with --resume.' % (code, log_path))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
