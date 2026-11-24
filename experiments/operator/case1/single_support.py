"""Case 1: one active parameter, independent holdouts, common metrics.

Paper [alpha, beta, gamma] corresponds to core [alpha, D, beta].
Parameter and evaluation-point generators never consume the training RNG.
"""
import contextlib
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

import training_support as s

PARAMETER_NAMES = ['alpha', 'beta_dispersion', 'gamma_nonlinearity']
LOWER = np.array([.6, .5, 1.], dtype=np.float64)
UPPER = np.array([.9, 1.5, 2.], dtype=np.float64)
METRICS = ['relative_l2', 'amplitude_relative_l2', 'phase_mae_radians',
           'mean_absolute_complex_error', 'max_absolute_complex_error']


def generator(seed, stream):
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(stream)]))


def mu_key(mu):
    if isinstance(mu, torch.Tensor):
        mu = mu.detach().cpu().numpy()
    return tuple(float(x) for x in np.asarray(mu, dtype=np.float32).reshape(3))


ACTIVE = {'alpha': 0, 'beta': 1, 'gamma': 2}


def parameter_row(operator, value, prefix, index):
    mu = [.8, 1., 1.]
    mu[ACTIVE[operator]] = value
    mu = mu_key(mu)
    return {'parameter_id': '%s_%04d' % (prefix, index), 'mu': list(mu),
            'active_value': mu[ACTIVE[operator]], **dict(zip(PARAMETER_NAMES, mu))}


def sample_parameters(operator, count, seed, stream, prefix, excluded):
    rng = generator(seed, stream)
    used = set(excluded)
    rows = []
    while len(rows) < count:
        # Canonicalize before computing source data: the network and PDE see
        # the same float32-representable parameter values, also saved in JSON.
        dim = ACTIVE[operator]
        candidate = LOWER[dim] + rng.random() * (UPPER[dim] - LOWER[dim])
        row = parameter_row(operator, candidate, prefix, len(rows))
        mu = mu_key(row['mu'])
        if mu in used or not LOWER[dim] < row['active_value'] < UPPER[dim]:
            continue
        used.add(mu)
        rows.append(row)
    return rows


def make_design(seed=20261017, counts=(4, 5, 6), validation_count=6,
                test_count=30, points=4096, grids=(5, 7, 9), operator='alpha'):
    if operator not in ACTIVE:
        raise ValueError('Choose alpha, beta or gamma.')
    counts = list(counts)
    if len(counts) != 3 or not 1 <= counts[0] < counts[1] < counts[2]:
        raise ValueError('Three strictly increasing training parameter counts required.')
    if min(validation_count, test_count, points) < 1:
        raise ValueError('Validation/test/point counts must be positive.')
    # Map the uploaded alpha order (.6,.7,.8,.9,.65,.85) to each active range.
    # Move endpoints inward only when float32 rounding would put them on or
    # outside a stated OPEN bound; interior levels retain the original order.
    if counts == [4, 5, 6]:
        dim = ACTIVE[operator]
        fractions = np.array([0, 1/3, 2/3, 1, 1/6, 5/6], dtype=float)
        values = LOWER[dim] + fractions * (UPPER[dim] - LOWER[dim])
        values = values.astype(np.float32)
        while float(values[0]) <= LOWER[dim]:
            values[0] = np.nextafter(values[0], np.float32(np.inf))
        while float(values[3]) >= UPPER[dim]:
            values[3] = np.nextafter(values[3], np.float32(-np.inf))
        train = [parameter_row(operator, v, 'train', i) for i, v in enumerate(values)]
    else:
        train = sample_parameters(operator, counts[-1], seed, 1, 'train', [])
    val = sample_parameters(operator, validation_count, seed, 2, 'validation',
                            [mu_key(r['mu']) for r in train])
    test = sample_parameters(operator, test_count, seed, 3, 'test',
                             [mu_key(r['mu']) for r in train + val])
    design = {
        'schema': 'case1_single_mu_holdout_v1', 'operator': operator,
        'active_index': ACTIVE[operator], 'parameter_names': PARAMETER_NAMES,
        'active_range_open': [float(LOWER[ACTIVE[operator]]), float(UPPER[ACTIVE[operator]])],
        'inactive_values_from': list(mu_key([.8, 1., 1.])),
        'sampling': 'train: fixed interior nested 4/5/6 design; validation/test: independent uniform active values; float32 disjoint',
        'split_seed': int(seed), 'parameter_rng_streams': {'training': 1, 'validation': 2, 'test': 3},
        'coordinate_rng_streams': {'validation': 11, 'test': 12, 'training_diagnostic': 13},
        'grids_intervals': list(grids), 'full_grid_points_per_axis': [g + 1 for g in grids],
        'stage_parameter_counts': counts, 'evaluation_points_per_parameter': int(points),
        'training': train, 'validation': val, 'test': test,
        'stage_training_ids': [[r['parameter_id'] for r in train[:n]] for n in counts],
        'replication': 'fixed parameter split and evaluation coordinates; independent training seeds',
        'test_use': 'final evaluation only, after validation selects and saves final.pt',
        'coordinate_order': ['t', 'x', 'y'],
        'source_family': 'u=v=t^(1+alpha)*sin(pi*beta*x)*sin(pi*gamma*y)',
        'core_mapping': '[alpha,D,beta] = paper [alpha,beta,gamma]',
    }
    validate_design(design)
    design['design_sha256'] = hashlib.sha256(
        json.dumps(design, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return design


def validate_design(design):
    all_keys = []
    for part in ['training', 'validation', 'test']:
        rows = design[part]
        values = np.array([r['mu'] for r in rows], dtype=np.float64)
        dim = ACTIVE[design['operator']]
        if not np.all((values[:, dim] > LOWER[dim]) & (values[:, dim] < UPPER[dim])):
            raise AssertionError('Active parameter outside the stated open interval.')
        if len(rows) > 1 and np.ptp(values[:, dim]) <= 0:
            raise AssertionError('The active parameter must vary.')
        fixed = np.asarray(mu_key([.8, 1., 1.]))
        for k in range(3):
            if k != dim and not np.all(values[:, k] == fixed[k]):
                raise AssertionError('An inactive parameter changed.')
        all_keys.extend(mu_key(r['mu']) for r in rows)
    if len(set(all_keys)) != len(all_keys):
        raise AssertionError('Training, validation and test parameter tuples overlap.')
    previous = set()
    for ids in design['stage_training_ids']:
        current = set(ids)
        if not previous.issubset(current):
            raise AssertionError('Training parameter sets are not nested.')
        previous = current
    return True


def coordinate_array(design, role):
    rng = generator(design['split_seed'], design['coordinate_rng_streams'][role])
    # All validation/test coordinates are fixed independent random points in
    # the open cube. Reject spatial coordinates on ANY training grid plane.
    # This also rules out coincidence with all sigma-residual grids, whose
    # spatial coordinates lie on those same grid planes.
    forbidden = set()
    for g in design['grids_intervals']:
        forbidden.update(float(x) for x in np.linspace(0., 1., g + 1).astype(np.float32))
    rows = []
    used = set()
    while len(rows) < design['evaluation_points_per_parameter']:
        p = tuple(float(v) for v in rng.random(3).astype(np.float32))
        if any(v <= 0 or v >= 1 for v in p) or p in used:
            continue
        if p[1] in forbidden or p[2] in forbidden:
            continue
        used.add(p)
        rows.append(p)
    return np.asarray(rows, dtype=np.float32)


def exact_real(tx, mu):
    # Reference arithmetic in float64 on the actual float32 input locations.
    t, x, y = [tx[:, i:i+1].double() for i in range(3)]
    alpha, beta, gamma = mu_key(mu)
    return t.pow(1 + alpha) * torch.sin(np.pi * beta * x) * torch.sin(np.pi * gamma * y)


def make_evaluation(rows, tx):
    return [{'parameter': row, 'tx': tx,
             'mu': tx.new_tensor([row['mu']]),
             'true': exact_real(tx, row['mu']).detach()} for row in rows]


def metric_values(u, v, true):
    result=s.metric_values(u,v,true)
    result['phase_mae_masked_radians']=result['phase_mae_radians']
    phase=torch.atan2(v.double(),u.double())-torch.atan2(true.double(),true.double())
    phase=torch.atan2(torch.sin(phase),torch.cos(phase)).abs()
    result['phase_mae_all_radians']=float(phase.mean())
    amp=np.sqrt(2)*true.double().abs()
    mask=amp>max(1e-3*float(amp.max()),1e-8)
    result['phase_mask_retained_fraction']=float(mask.double().mean())
    return result


def evaluate(model, data):
    if not data:
        raise ValueError('Empty evaluation set.')
    was_training = model.training
    model.eval()
    rows = []
    with torch.no_grad():
        for item in data:
            u, v = model(item['tx'], item['mu'])
            p = item['parameter']
            rows.append({'parameter_id': p['parameter_id'],
                         'active_value': p['active_value'],
                         **{key: p[key] for key in PARAMETER_NAMES},
                         **metric_values(u, v, item['true'])})
    model.train(was_training)
    keys = METRICS + ['phase_mae_masked_radians', 'phase_mask_retained_fraction']
    macro = {key: float(np.mean([r[key] for r in rows])) for key in keys}
    if not all(np.isfinite(value) for value in macro.values()):
        raise FloatingPointError('Nonfinite evaluation metric.')
    return macro, rows


def frozen_full_grid_targets(model, data_list):
    result = s.frozen_full_grid_targets(model, data_list)
    for teacher, data in zip(result, data_list):
        teacher['parameter_id'] = data['parameter_id']
        teacher['mu_key'] = list(mu_key(data['mu']))
    return result


def save_design(out, design, coordinates):
    out = Path(out)
    s.atomic_json(out / 'parameter_split.json', design)
    rows = []
    for part in ['training', 'validation', 'test']:
        for row in design[part]:
            rows.append({'partition': part, 'parameter_id': row['parameter_id'],
                         'active_value': row['active_value'],
                         **{key: row[key] for key in PARAMETER_NAMES}})
    s.write_csv(out / 'parameter_split.csv', rows)
    np.savez_compressed(out / 'evaluation_coordinates.npz', **coordinates)


@contextlib.contextmanager
def output_lock(out):
    import fcntl
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.run.lock').open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another process owns this output directory: %s' % out)
        handle.seek(0)
        handle.truncate()
        handle.write('pid=%d\n' % os.getpid())
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def test_distribution(rows):
    # Variation over parameters is distinct from variation over training seeds.
    return {key: {'mean': float(np.mean([r[key] for r in rows])),
                  'median': float(np.median([r[key] for r in rows])),
                  'p95': float(np.percentile([r[key] for r in rows], 95)),
                  'worst': float(max(r[key] for r in rows)),
                  'worst_parameter_id': max(rows, key=lambda r: r[key])['parameter_id']}
            for key in METRICS}
