"""Collect only completed matching Softmax+noise C01 ablations (Python 3.8+)."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path

import numpy as np

import ablation_protocol as protocol


METRICS = (
    'validation_best', 'original_min_sigma_l2', 'relative_l2',
    'amplitude_relative_l2', 'phase_mae_all_radians', 'phase_mae_radians',
    'mean_abs_error', 'max_abs_error', 'training_seconds',
    'peak_allocated_bytes', 'peak_reserved_bytes', 'process_peak_rss_bytes',
    'steps', 'residual_points_processed', 'parameter_count',
)


def csv_write(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def cell(row, metric, scale=1.0):
    mean, sd = row.get(metric + '_mean'), row.get(metric + '_std')
    if mean is None:
        return '--'
    return ('$%.2f\\pm%.2f$' % (mean / scale, sd / scale)
            if sd is not None else '$%.2f$' % (mean / scale))


def tex_label(variant):
    if variant == 'fixed_operator_branch':
        return r'\shortstack[l]{Fixed-parameter\\operator branch}'
    return protocol.LABELS[variant]


def tex_write(root, averaged, complete):
    lines = [
        r'% Requires \usepackage{booktabs}. All phase values below use ALL evaluation points.',
        r'\begin{table*}[t]', r'\centering', r'\small',
        r'\caption{C01 comparisons with the selected Softmax weighting and stochastic perturbation. '
        r'Results are mean $\pm$ sample standard deviation across independent runs. '
        r'All main metrics use the same independent test points and validation-selected checkpoints.}',
        r'\label{tab:c01_softmax_ablation}',
        r'\renewcommand{\arraystretch}{1.15}', r'\setlength{\tabcolsep}{4pt}',
        r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccc@{}}',
        r'\toprule',
        r'Variant & Runs & \shortstack{Rel. $L_2$\\$(\times10^{-3})$} '
        r'& \shortstack{Amp. Rel. $L_2$\\$(\times10^{-3})$} '
        r'& \shortstack{Phase MAE\\$(\times10^{-2}\,\mathrm{rad})$} '
        r'& \shortstack{MAE\\$(\times10^{-3})$} '
        r'& \shortstack{MaxAE\\$(\times10^{-3})$} \\', r'\midrule',
    ]
    for row in averaged:
        values = [tex_label(row['variant']), str(row['n_seeds']),
                  cell(row, 'relative_l2', 1e-3), cell(row, 'amplitude_relative_l2', 1e-3),
                  cell(row, 'phase_mae_radians', 1e-2), cell(row, 'mean_abs_error', 1e-3),
                  cell(row, 'max_abs_error', 1e-3)]
        lines.append(' & '.join(values) + r' \\')
    lines.extend([r'\bottomrule', r'\end{tabular*}', r'\end{table*}'])
    (root / 'ablation_table.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    resource_lines = [r'% GPU memory includes neural-network training/backpropagation, not SOE state alone.',
                      r'\begin{table*}[t]', r'\centering', r'\small',
                      r'\caption{Measured C01 training resources. Wall-clock time includes training, validation, '
                      r'and checkpoint I/O. Allocated and reserved CUDA memory are reported separately.}',
                      r'\label{tab:c01_softmax_resources}',
                      r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrrr@{}}',
                      r'\toprule',
                      r'Variant & Updates & \shortstack{Residual points\\$(\times10^6)$} '
                      r'& Parameters & Time (s) & \shortstack{Peak allocated\\(MiB)} '
                      r'& \shortstack{Peak reserved\\(MiB)} \\', r'\midrule']
    for row in averaged:
        resource_lines.append(' & '.join([
            tex_label(row['variant']), str(int(row['steps_mean'])),
            '$%.2f$' % (row['residual_points_processed_mean'] / 1e6),
            str(int(row['parameter_count_mean'])), cell(row, 'training_seconds'),
            cell(row, 'peak_allocated_bytes', 2**20), cell(row, 'peak_reserved_bytes', 2**20),
        ]) + r' \\')
    resource_lines.extend([r'\bottomrule', r'\end{tabular*}', r'\end{table*}'])
    (root / 'resources_table.tex').write_text('\n'.join(resource_lines) + '\n', encoding='utf-8')


def verify_group(runs):
    """Validate budgets/evaluation and unchanged data wherever grids are matched."""
    reference = next(iter(runs.values()))
    reference_summary, reference_config, reference_directory = reference
    common_keys = ('epochs_per_stage', 'budget_mode', 'eval_every', 'eval_points',
                   'validation_seed', 'test_seed', 'width', 'extra_hidden_layers', 'alpha',
                   'w_grad', 'kernel_mode', 'network_dtype', 'shared_update',
                   'noise_amplitude', 'weight_lower_bound', 'weight_floor',
                   'checkpoint_every', 'patience_evals', 'tol', 'safety_stage_cap',
                   'device', 'threads', 'smoke')
    for key, (summary, config, directory) in runs.items():
        if any(config['args'].get(k) != reference_config['args'].get(k) for k in common_keys):
            raise ValueError('Unmatched common protocol at ' + str(directory))
        if config.get('source_hashes') != reference_config.get('source_hashes'):
            raise ValueError('Mixed numerical source versions at ' + str(directory))
        if summary['steps'] != reference_summary['steps']:
            raise ValueError('Unmatched total optimizer-update budget at ' + str(directory))
        if not protocol.same_arrays(directory / 'evaluation_points.npz',
                                    reference_directory / 'evaluation_points.npz'):
            raise ValueError('Validation/test points differ at ' + str(directory))
        rule = config['weight_rule']
        expected_rule = protocol.trainer.description('softmax_noise',
                                                    config['args']['noise_amplitude'],
                                                    config['args']['weight_lower_bound'],
                                                    config['args']['weight_floor'])
        if rule != expected_rule:
            raise ValueError('Declared weight rule differs at ' + str(directory))
    # All variants with matching grids inherit identical NumPy collocation data per seed.
    by_seed = defaultdict(list)
    for (variant, seed), item in runs.items():
        by_seed[seed].append((variant, item))
    for seed, items in by_seed.items():
        for i, (variant, (summary, config, directory)) in enumerate(items):
            for other_variant, (other_summary, other_config, other_directory) in items[i+1:]:
                if config['args']['grids'] == other_config['args']['grids']:
                    if [s['stage_data_sha256'] for s in summary['stages']] != [
                            s['stage_data_sha256'] for s in other_summary['stages']]:
                        raise ValueError('Collocation data differ for seed=%d: %s vs %s.' %
                                         (seed, variant, other_variant))
                excluded = {'no_attention', 'coordinate_matched', 'fixed_operator_branch'}
                if variant not in excluded and other_variant not in excluded:
                    if config['initial_parameters_sha256'] != other_config['initial_parameters_sha256']:
                        raise ValueError('Predictor/MLP initialization differs for seed=%d.' % seed)


def collect(root, variants, seeds, allow_partial=False, include_smoke=False):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    runs, missing, rows = {}, [], []
    for variant in variants:
        for seed in seeds:
            directory = protocol.destination(root, variant, seed)
            if not (directory / 'summary.json').is_file():
                missing.append('%s/seed_%d' % (variant, seed))
                continue
            summary = protocol.read_json(directory / 'summary.json')
            allowed = {'TRAINING_COMPLETED', 'SMOKE_ONLY'} if include_smoke else {'TRAINING_COMPLETED'}
            if summary.get('status') not in allowed:
                missing.append('%s/seed_%d (%s)' % (variant, seed, summary.get('status')))
                continue
            config = protocol.read_json(directory / 'config.json')
            args = argparse.Namespace(**config['args'])
            protocol.validate_completed(directory, args)
            if summary['seed'] != seed or summary['variant'] != variant:
                raise ValueError('Directory label differs from run identity: ' + str(directory))
            runs[(variant, seed)] = (summary, config, directory)
            row = {'variant': variant, 'weight_scheme': summary['weight_scheme'], 'seed': seed,
                   'active_main_loss_weights': 'uniform' if variant in ('no_attention', 'coordinate_matched') else 'softmax_noise',
                   'status': summary['status'], 'path': str(directory),
                   'reused_full': variant == 'full_original' and directory.is_symlink(),
                   'selected_stage': summary['selected_stage'], 'selected_epoch': summary['selected_epoch']}
            for metric in METRICS:
                row[metric] = summary['test'].get(metric) if metric in summary['test'] else summary.get(metric)
                value = row[metric]
                if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value)):
                    raise ValueError('Nonfinite/non-numeric result %s at %s' % (metric, directory))
            rows.append(row)
    if runs:
        verify_group(runs)
    if missing and not allow_partial:
        report = {'status': 'PENDING', 'missing': missing,
                  'completed': len(runs), 'expected': len(variants) * len(seeds),
                  'note': 'No incomplete run is included. Use --allow-partial for explicitly labelled partial tables.'}
        protocol.json_write(root / 'collection_pending.json', report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return report
    averaged = []
    for variant in variants:
        group = [r for r in rows if r['variant'] == variant]
        if not group:
            continue
        row = {'variant': variant, 'n_seeds': len(group),
               'seeds': ','.join(str(r['seed']) for r in group)}
        for metric in METRICS:
            values = [r[metric] for r in group]
            if all(isinstance(v, (int, float)) for v in values):
                row[metric + '_mean'] = float(np.mean(values))
                row[metric + '_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        averaged.append(row)
    csv_write(root / 'per_seed_results.csv', rows)
    csv_write(root / 'mean_std_results.csv', averaged)
    tex_write(root, averaged, not missing)
    report = {
        'status': 'SMOKE_ONLY' if include_smoke else ('PARTIAL' if missing else 'COMPLETED'),
        'expected_variants': variants, 'expected_seeds': seeds, 'missing': missing,
        'completed': len(runs), 'weight_scheme': 'softmax_noise',
        'matched_updates_verified': bool(runs), 'same_validation_test_points_verified': bool(runs),
        'same_numerical_source_verified': bool(runs),
        'phase_main_table': 'phase_mae_radians; masked wrapped MAE',
        'phase_diagnostic': 'phase_mae_radians; excludes near-zero reference amplitude',
        'standard_deviation': 'sample SD (ddof=1); null for one run',
        'selection': 'independent validation only; no test-based checkpoint or hyperparameter selection',
        'no_grid_enrichment': 'fixed final grid with five LR/data phases; not a literal one-stage run',
        'resource_scope': 'neural training/backpropagation plus validation/checkpoint snapshots; not forward history alone',
    }
    protocol.json_write(root / 'collection_report.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='runs/C01_softmax_ablation_budget2000')
    p.add_argument('--expected-variants', default=','.join(protocol.ORDER))
    p.add_argument('--expected-seeds', default='1234,2024,3407')
    p.add_argument('--allow-partial', action='store_true')
    p.add_argument('--include-smoke', action='store_true')
    a = p.parse_args()
    variants, seeds = protocol.parse_list(a.expected_variants), protocol.parse_list(a.expected_seeds, int)
    if any(v not in protocol.VARIANTS for v in variants) or min(seeds) < 0:
        p.error('Unknown variant or negative seed.')
    collect(a.root, variants, seeds, a.allow_partial, a.include_smoke)


if __name__ == '__main__':
    main()
