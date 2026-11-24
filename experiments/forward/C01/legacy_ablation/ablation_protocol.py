"""Fixed C01 Softmax+noise protocol; numerical source files stay unchanged."""
import argparse
import json
from pathlib import Path

import numpy as np

import train_c01_weights as trainer


VARIANTS = tuple(trainer.original.VARIANTS)
LABELS = {
    'full_original': 'Full',
    'no_dual_tanh': 'w/o Dual-tanh',
    'no_gradient': 'w/o Gradient',
    'no_attention': 'w/o Attention',
    'no_soe': 'w/o SOE',
    'no_consistency': 'w/o Consistency',
    'no_grid_enrichment': 'Fixed final grid',
    'coordinate_matched': 'Coordinate PINN',
    'fixed_operator_branch': 'Fixed-parameter operator branch',
}
ORDER = (
    'full_original', 'no_dual_tanh', 'no_gradient', 'no_attention', 'no_soe',
    'no_consistency', 'no_grid_enrichment', 'coordinate_matched',
    'fixed_operator_branch',
)
PRESETS = {
    'all': ORDER,
    'components': ('no_dual_tanh', 'no_gradient', 'no_attention', 'no_soe'),
    'controls': ('no_consistency', 'no_grid_enrichment', 'coordinate_matched',
                 'fixed_operator_branch'),
}
TRAINING_KEYS = (
    'device', 'budget_mode', 'epochs_per_stage', 'grids', 'width',
    'extra_hidden_layers', 'alpha', 'w_grad', 'eval_every', 'eval_points',
    'validation_seed', 'test_seed', 'kernel_mode', 'network_dtype',
    'shared_update', 'noise_amplitude', 'weight_lower_bound', 'weight_floor',
    'patience_evals', 'tol', 'safety_stage_cap', 'checkpoint_every', 'threads',
)


def parser():
    p = argparse.ArgumentParser(description='C01 ablations with fixed Softmax+noise weights.')
    p.add_argument('--variants', default=None)
    p.add_argument('--preset', choices=tuple(PRESETS), default='all')
    p.add_argument('--seeds', default='1234,2024,3407')
    p.add_argument('--output', default='runs/C01_softmax_ablation_budget2000')
    p.add_argument('--reuse-full-from', type=Path, default=None,
                   help='Existing weighting-comparison result root, not just its CSV files.')
    p.add_argument('--train-full', action='store_true',
                   help='Explicitly permit a fresh Full when there is no matching completed run.')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--budget-mode', choices=['matched_steps'], default='matched_steps')
    p.add_argument('--epochs-per-stage', type=int, default=2000)
    p.add_argument('--grids', default='10x10,20x20,30x30,40x40,50x50')
    p.add_argument('--width', type=int, default=60)
    p.add_argument('--extra-hidden-layers', type=int, default=2)
    p.add_argument('--alpha', type=float, default=0.9)
    p.add_argument('--w-grad', type=float, default=0.001)
    p.add_argument('--eval-every', type=int, default=100)
    p.add_argument('--eval-points', type=int, default=4096)
    p.add_argument('--validation-seed', type=int, default=98101)
    p.add_argument('--test-seed', type=int, default=98102)
    p.add_argument('--kernel-mode', choices=['original', 'stable64'], default='original')
    p.add_argument('--network-dtype', choices=['float32', 'float64'], default='float32')
    p.add_argument('--shared-update', choices=['legacy_double', 'once'], default='legacy_double')
    p.add_argument('--noise-amplitude', type=float, default=0.05)
    p.add_argument('--weight-lower-bound', type=float, default=1e-6)
    # Included in the old signature even though Softmax+noise does not use it.
    p.add_argument('--weight-floor', type=float, default=0.2)
    p.add_argument('--patience-evals', type=int, default=10)
    p.add_argument('--tol', type=float, default=1e-8)
    p.add_argument('--safety-stage-cap', type=int, default=0)
    p.add_argument('--checkpoint-every', type=int, default=100)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--plan-only', action='store_true')
    return p


def parse_list(value, cast=str):
    values = [cast(x.strip()) for x in value.split(',') if x.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError('List must be nonempty and contain no duplicates: ' + value)
    return values


def training_argv(a, variant, seed, destination):
    result = ['--weight-scheme', 'softmax_noise', '--variant', variant,
              '--seed', str(seed), '--output', str(destination)]
    for key in TRAINING_KEYS:
        result.extend(['--' + key.replace('_', '-'), str(getattr(a, key))])
    if a.resume:
        result.append('--resume')
    if a.smoke:
        result.append('--smoke')
    return result


def destination(root, variant, seed):
    return Path(root) / 'softmax_noise' / variant / ('seed_%d' % seed)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def expected_budget(args):
    return (len(args.grids) * args.epochs_per_stage,
            sum((M - 1) * N for M, N in args.grids) * args.epochs_per_stage)


def validate_completed(directory, args):
    """Reject stale weights, old budgets and mismatched sources before reuse."""
    directory = Path(directory)
    required = ('config.json', 'summary.json', 'final.pt',
                'best_validation.pt', 'test_predictions.npz', 'evaluation_points.npz')
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise ValueError('Incomplete result directory %s; missing %s' % (directory, missing))
    config, summary = read_json(directory / 'config.json'), read_json(directory / 'summary.json')
    expected_signature = trainer.signature(args)
    if config.get('signature') != expected_signature:
        saved = config.get('args', {})
        differences = {k: {'requested': v, 'saved': saved.get(k)}
                       for k, v in vars(args).items()
                       if k not in ('output', 'resume', 'overwrite')
                       and json.dumps(v, sort_keys=True) != json.dumps(saved.get(k), sort_keys=True)}
        raise ValueError('Configuration/source mismatch at %s. Argument differences: %s; '
                         'the seven numerical source files must also be identical.' % (directory, differences))
    status = 'SMOKE_ONLY' if args.smoke else 'TRAINING_COMPLETED'
    if summary.get('status') != status or summary.get('signature') != expected_signature:
        raise ValueError('Result status/signature mismatch at ' + str(directory))
    if summary.get('variant') != args.variant or summary.get('weight_scheme') != 'softmax_noise':
        raise ValueError('Variant or weighting scheme mismatch at ' + str(directory))
    steps, work = expected_budget(args)
    if summary.get('steps') != steps or summary.get('residual_points_processed') != work:
        raise ValueError('Completed update/work budget mismatch at ' + str(directory))
    stages = summary.get('stages', [])
    actual_grids = [(s['M'], s['N']) for s in stages]
    if actual_grids != [tuple(g) for g in args.grids] or any(s['epochs'] != args.epochs_per_stage for s in stages):
        raise ValueError('Stage schedule mismatch at ' + str(directory))
    return summary, config


def full_source(root, seed):
    root = Path(root).resolve()
    candidates = [destination(root, 'full_original', seed),
                  root / 'full_original' / ('seed_%d' % seed), root / ('seed_%d' % seed)]
    found = {p.resolve() for p in candidates if (p / 'config.json').is_file()}
    if len(found) != 1:
        raise ValueError('Expected one Full directory for seed=%d under %s; found %s. '
                         'Provide the original runs/C01_weights_budget2000 directory.' %
                         (seed, root, sorted(str(p) for p in found)))
    return next(iter(found))


def same_arrays(left, right):
    with np.load(left) as a, np.load(right) as b:
        return set(a.files) == set(b.files) and all(np.array_equal(a[k], b[k]) for k in a.files)


def json_write(path, value):
    trainer.json_write(Path(path), value)

