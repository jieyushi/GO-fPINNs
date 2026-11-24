"""Collect completed runs and recompute the four manuscript metrics from saved predictions."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

METRICS = ('relative_l2', 'amplitude_relative_l2', 'phase_mae_radians', 'mean_abs_error')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def errors(prediction, exact):
    p, y = np.asarray(prediction, dtype=np.float64), np.asarray(exact, dtype=np.float64)
    if p.shape != y.shape or p.ndim != 2 or p.shape[1] != 2:
        raise ValueError('Expected matching N-by-2 complex-component arrays.')
    pc, yc = p[:, 0]+1j*p[:, 1], y[:, 0]+1j*y[:, 1]
    amp = np.abs(yc)
    threshold = max(1e-3*float(amp.max()), 1e-8)
    mask = amp > threshold
    wrapped = np.angle(pc*np.conj(yc))
    return {'relative_l2': float(np.linalg.norm(pc-yc)/max(np.linalg.norm(yc), 1e-30)),
            'amplitude_relative_l2': float(np.linalg.norm(np.abs(pc)-amp)/max(np.linalg.norm(amp), 1e-30)),
            'phase_mae_radians': float(np.abs(wrapped[mask]).mean()) if mask.any() else None,
            'phase_mae_all_radians': float(np.abs(wrapped).mean()),
            'mean_abs_error': float(np.abs(pc-yc).mean()),
            'max_abs_error': float(np.abs(pc-yc).max()),
            'phase_masked_point_count': int(mask.sum()), 'evaluation_point_count': len(pc)}


def completed(root, include_smoke=False):
    root = Path(root)
    planned = {}
    for plan_path in sorted(root.glob('plan_*.json')):
        for job in read_json(plan_path)['jobs']:
            key = str(Path(job['output']).resolve())
            if key in planned:
                roles = planned[key]['roles']+job.get('roles', [])
                planned[key]['roles'] = [dict(t) for t in {tuple(sorted(r.items())) for r in roles}]
            else:
                planned[key] = dict(job)
    for summary_path in root.rglob('summary.json'):
        folder = summary_path.parent
        if str(folder.resolve()) not in planned and (folder/'config.json').exists():
            summary = read_json(summary_path)
            config = read_json(folder/'config.json')
            experiment_id = summary.get('experiment_id') or ('ablation/'+summary.get('variant', 'unknown'))
            planned[str(folder.resolve())] = {'output': str(folder.resolve()), 'experiment_id': experiment_id,
                                              'seed': summary['seed'], 'roles': []}
    rows, stages, pending = [], [], []
    for directory, job in sorted(planned.items()):
        folder = Path(directory)
        if not (folder/'summary.json').exists():
            pending.append(job)
            continue
        summary, config = read_json(folder/'summary.json'), read_json(folder/'config.json')
        accepted = ['TRAINING_COMPLETED', 'SMOKE_ONLY'] if include_smoke else ['TRAINING_COMPLETED']
        if summary.get('status') not in accepted:
            pending.append(dict(job, reason=summary.get('status')))
            continue
        if summary['signature'] != config['signature']:
            raise ValueError('Summary/config signature mismatch: '+directory)
        required = ['final.pt', 'test_predictions.npz', 'evaluation_points.npz']
        if any(not (folder/name).is_file() for name in required):
            raise ValueError('Completed run is missing its model or predictions: '+directory)
        with np.load(folder/'test_predictions.npz') as data:
            values = errors(data['prediction'], data['exact'])
            test_digest = hashlib.sha256(data['tx'].tobytes()+data['exact'].tobytes()).hexdigest()
        for name in ('relative_l2', 'amplitude_relative_l2', 'mean_abs_error'):
            old_value = summary['test'].get(name)
            if old_value is not None and not np.isclose(old_value, values[name], rtol=1e-7, atol=1e-12):
                raise ValueError('Saved prediction/metric inconsistency: '+directory+' '+name)
        args = config['args']
        row = dict(experiment_id=job['experiment_id'], seed=summary['seed'], path=directory,
                   status=summary['status'], roles=json.dumps(job.get('roles', [])),
                   validation_l2=summary['validation_best'],
                   original_min_sigma_l2=summary['original_min_sigma_l2'],
                   selected_stage=summary['selected_stage'], selected_epoch=summary['selected_epoch'],
                   steps=summary['steps'], training_seconds=summary['training_seconds'],
                   residual_points_processed=summary['residual_points_processed'],
                   optimizer_registered_parameters=summary['parameter_count'],
                   peak_allocated_bytes=summary['peak_allocated_bytes'],
                   peak_reserved_bytes=summary['peak_reserved_bytes'],
                   test_points_sha256=test_digest, network_activation=args.get('network_activation'),
                   extra_hidden_layers=args['extra_hidden_layers'], hidden_blocks=args['extra_hidden_layers']+1,
                   width=args['width'], fractional_scheme=args.get('fractional_scheme'),
                   early_stopping_enabled=bool(summary.get('early_stopping_enabled', False)))
        row.update(values)
        rows.append(row)
        for stage in summary['stages']:
            stages.append(dict(stage, experiment_id=row['experiment_id'], seed=row['seed'], path=directory))
    digests = {r['test_points_sha256'] for r in rows}
    if len(digests) > 1:
        raise ValueError('Runs use different independent test points; collect each design separately.')
    return rows, stages, pending


def average(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['experiment_id']].append(row)
    result = []
    numeric = METRICS+('max_abs_error', 'validation_l2', 'original_min_sigma_l2',
                      'training_seconds', 'steps', 'residual_points_processed',
                      'optimizer_registered_parameters', 'peak_allocated_bytes', 'peak_reserved_bytes')
    for experiment_id, group in sorted(groups.items()):
        if len({r['seed'] for r in group}) != len(group):
            raise ValueError('Duplicate seeds: '+experiment_id)
        row = {'experiment_id': experiment_id, 'runs': len(group), 'seeds': ','.join(str(r['seed']) for r in group)}
        for metric in numeric:
            values = [float(r[metric]) for r in group if r.get(metric) is not None]
            row[metric+'_mean'] = float(np.mean(values)) if values else None
            row[metric+'_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        result.append(row)
    return result


def cell(row, key, scale):
    mean, sd = row.get(key+'_mean'), row.get(key+'_std')
    if mean is None:
        return '--'
    return ('$%.2f\\pm%.2f$' % (mean/scale, sd/scale)) if sd is not None else ('$%.2f$' % (mean/scale))


def write_tex(path, means):
    lines = [r'% Requires booktabs. Phase MAE uses the reference-amplitude mask.',
             r'\begin{table*}[t]', r'\centering\small',
             r'\caption{Case-1 independent-test errors. Entries are mean $\pm$ sample standard deviation; a single run has no standard deviation.}',
             r'\label{tab:c01_complete_errors}', r'\renewcommand{\arraystretch}{1.15}',
             r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccc@{}}', r'\toprule',
             r'Experiment & Runs & $\mathcal E_{L_2}$ $(10^{-3})$ & $\mathcal E_{\rm amp}$ $(10^{-3})$ & $\mathcal E_{\rm phase}$ $(10^{-2})$ & $\mathcal E_{\rm MAE}$ $(10^{-3})$ \\', r'\midrule']
    for row in means:
        label = row['experiment_id'].replace('_', r'\_')
        lines.append(' & '.join([label, str(row['runs'])]+[cell(row, k, scale) for k, scale in zip(METRICS, (1e-3, 1e-3, 1e-2, 1e-3))])+r' \\')
    lines += [r'\bottomrule', r'\end{tabular*}', r'\end{table*}']
    Path(path).write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('runs/C01_case1_complete'))
    p.add_argument('--include-smoke', action='store_true')
    a = p.parse_args()
    rows, stages, pending = completed(a.root, a.include_smoke)
    means = average(rows)
    dest = a.root/'collected'
    dest.mkdir(parents=True, exist_ok=True)
    write_csv(dest/'per_seed_results.csv', rows)
    write_csv(dest/'mean_std_results.csv', means)
    write_csv(dest/'stage_results.csv', stages)
    write_tex(dest/'all_error_results.tex', means)
    for suite in ('full', 'activation', 'discretization', 'transfer', 'gate', 'comparison', 'ablation'):
        members = {r['experiment_id'] for r in rows
                   if r['experiment_id'].startswith(suite+'/') or
                   any(role['suite'] == suite for role in json.loads(r['roles']))}
        write_tex(dest/(suite+'_errors.tex'), [r for r in means if r['experiment_id'] in members])
    report = {'completed': len(rows), 'pending': pending,
              'phase_primary': 'masked, reference amplitude > max(1e-3*max amplitude,1e-8)',
              'standard_deviation': 'sample SD across training seeds, ddof=1; absent for n=1',
              'smoke_included': a.include_smoke, 'all_planned_jobs_completed': not pending,
              'resource_unit': 'unique optimizer-registered parameters; CUDA training/backward peaks, not SOE recurrence state alone'}
    (dest/'collection_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
