"""Regenerate Figures 3--5 from saved runs, without embedded paper numbers."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, SymLogNorm
import numpy as np

from collect_case1 import completed, write_csv

COLORS = {'l1': '#9b77a6', 'fast_l1': '#8ea7bc', 'l21': '#58a39f',
          'fast_l21': '#145b8b', 'GO-fPINNs': '#145b8b', 'fPINNs': '#69b4d0',
          'FL1-fPINN': '#c33b4c', 'operator': '#8c68b0', 'fixed': '#8994a5'}
SCHEME_NAMES = {'l1': 'L1', 'fast_l1': 'Fast L1', 'l21': r'L2-$1_\sigma$',
                'fast_l21': r'Fast L2-$1_\sigma$'}


def style():
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 15,
                         'axes.labelsize': 17, 'axes.titlesize': 17,
                         'xtick.labelsize': 14, 'ytick.labelsize': 14,
                         'legend.fontsize': 12, 'savefig.dpi': 300,
                         'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})


def save(fig, out, name):
    fig.savefig(out/(name+'.pdf'), bbox_inches='tight')
    fig.savefig(out/(name+'.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)


def history(path):
    with (Path(path)/'training.csv').open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def stage_values(row):
    return json.loads((Path(row['path'])/'summary.json').read_text())['stages']


def empty(ax, message):
    ax.text(.5, .5, message, transform=ax.transAxes, ha='center', va='center', fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])


def pair_panel(ax, row_map, keys, labels, colors, exports):
    available = [(k, label, color) for k, label, color in zip(keys, labels, colors) if k in row_map]
    if not available:
        empty(ax, 'Awaiting completed runs')
        return
    right = ax.twinx()
    width = .7/len(available)
    handles, legend_labels = [], []
    for i, (key, label, color) in enumerate(available):
        data = stage_values(row_map[key])
        x = np.arange(len(data))
        times = [r['stage_training_seconds'] for r in data]
        values = [r['sigma_l2'] for r in data]
        ax.bar(x+(i-(len(available)-1)/2)*width, times, width=width,
               color=color, alpha=.43)
        line, = right.plot(x, values, marker='o', color=color, linewidth=2, label=label)
        handles.append(line)
        legend_labels.append(label)
        exports.extend(dict(experiment_id=key, stage=r['stage'], sigma_l2=r['sigma_l2'],
                            stage_training_seconds=r['stage_training_seconds']) for r in data)
    right.set_yscale('log')
    right.set_ylabel(r'Relative $L_2$ error')
    right.spines['right'].set_visible(True)
    ax.set_ylabel('Stage training time (s)')
    ax.set_xlabel('Stage')
    n = max(len(stage_values(row_map[k])) for k, _, _ in available)
    ax.set_xticks(np.arange(n), ['S%d' % i for i in range(n)])
    ax.legend(handles, legend_labels, loc='upper right', frameon=False)
    ax.grid(axis='y', alpha=.2)


def figure3(rows, out):
    row_map = {r['experiment_id']: r for r in rows}
    fig, axes = plt.subplots(2, 3, figsize=(17.8, 10.4), layout='constrained')
    exports = []
    activation = [r for r in rows if r['experiment_id'].startswith('activation/') or
                  any(s['suite'] == 'activation' for s in json.loads(r.get('roles', '[]')))]
    if activation:
        depths = sorted({r['extra_hidden_layers']+1 for r in activation})
        widths = sorted({r['width'] for r in activation})
        vmax = max(r['original_min_sigma_l2'] for r in activation)
        norm = Normalize(vmin=0, vmax=vmax)
        for act, ax, title in [('tanh', axes[0, 0], 'Tanh'),
                               ('dual_tanh', axes[1, 0], 'Adaptive dual-tanh')]:
            matrix = np.full((len(widths), len(depths)), np.nan)
            for r in activation:
                if r['network_activation'] == act:
                    matrix[widths.index(r['width']), depths.index(r['extra_hidden_layers']+1)] = r['original_min_sigma_l2']
                    exports.append(dict(experiment_id=r['experiment_id'], extra_hidden_layers=r['extra_hidden_layers'],
                                        hidden_layers=r['extra_hidden_layers']+1,
                                        width=r['width'], activation=act, sigma_l2=r['original_min_sigma_l2']))
            image = ax.imshow(matrix, origin='lower', aspect='auto', cmap='Blues', norm=norm)
            for (i, j), value in np.ndenumerate(matrix):
                if np.isfinite(value):
                    ax.text(j, i, '%.2f' % (value*1e3), ha='center', va='center',
                            fontsize=12, color='white' if value > .65*vmax else 'black')
            if np.isfinite(matrix).any():
                i, j = np.unravel_index(np.nanargmin(matrix), matrix.shape)
                ax.plot(j+.23, i+.24, marker='*', ms=13, color='#efb719', mec='#6c5510')
            ax.set_xticks(range(len(depths)), depths)
            ax.set_yticks(range(len(widths)), widths)
            ax.set_xlabel('Depth (hidden layers)')
            ax.set_ylabel('Width')
            ax.set_title(title)
        cb = fig.colorbar(image, ax=[axes[0, 0], axes[1, 0]], shrink=.82, pad=.03)
        cb.set_label(r'Relative $L_2$ error; cell labels $\times10^{-3}$', fontsize=13)
    else:
        empty(axes[0, 0], 'Awaiting activation runs')
        empty(axes[1, 0], 'Awaiting activation runs')
    for direction, ax in [('temporal', axes[0, 1]), ('spatial', axes[1, 1])]:
        right = ax.twinx()
        found = False
        for i, scheme in enumerate(('l1', 'fast_l1', 'l21', 'fast_l21')):
            key = 'discretization/%s_%s' % (direction, scheme)
            if key not in row_map:
                continue
            found = True
            data = stage_values(row_map[key])
            x = np.arange(len(data))
            times = [r['stage_training_seconds'] for r in data]
            err = [r['sigma_l2'] for r in data]
            ax.bar(x+(i-1.5)*.18, times, width=.17, color=COLORS[scheme], alpha=.45)
            right.plot(x, err, marker='o', linewidth=2, color=COLORS[scheme], label=SCHEME_NAMES[scheme])
            exports.extend(dict(experiment_id=key, stage=r['stage'], M=r['M'], N=r['N'],
                                sigma_l2=r['sigma_l2'], stage_training_seconds=r['stage_training_seconds']) for r in data)
        right.set_yscale('log')
        right.set_ylabel(r'Relative $L_2$ error')
        right.spines['right'].set_visible(True)
        if found:
            right.legend(frameon=False, fontsize=11, ncol=2)
            ax.set_xticks(range(5), [10, 20, 30, 40, 50])
        else:
            empty(ax, 'Awaiting '+direction+' runs')
        ax.set_title(direction.capitalize()+' grid')
        ax.set_xlabel('Time intervals' if direction == 'temporal' else 'Spatial intervals')
        ax.set_ylabel('Stage training time (s)')
        ax.grid(axis='y', alpha=.2)
    pair_panel(axes[0, 2], row_map, ['full/GO_fPINNs', 'transfer/fixed_grid'],
               ['Multi-stage', 'Fixed-grid control'], [COLORS['fast_l21'], COLORS['fixed']], exports)
    axes[0, 2].set_title('Transfer strategy')
    pair_panel(axes[1, 2], row_map, ['full/GO_fPINNs', 'gate/operator_g1'],
               [r'$g=0$', r'$g=1$'], [COLORS['fast_l21'], COLORS['operator']], exports)
    axes[1, 2].set_title('Gate mechanism')
    for label, ax in zip('abcdef', axes.ravel()):
        ax.text(.5, -.27, '('+label+')', transform=ax.transAxes, ha='center', va='top', fontsize=17)
    # Original ordering (a,d): activation; (b,e): schemes; (c,f): transfer/gate.
    save(fig, out, 'Figure_3_revise')
    write_csv(out/'figure3_data.csv', exports)


def figures45(rows, out):
    row_map = {r['experiment_id']: r for r in rows}
    methods = [('comparison/fPINNs', 'fPINNs'), ('comparison/FL1_fPINN', 'FL1-fPINN'),
               ('full/GO_fPINNs', 'GO-fPINNs')]
    available = [(k, label) for k, label in methods if k in row_map]
    if not available:
        return
    fig, axes = plt.subplots(1, 2, figsize=(15.6, 5.6), layout='constrained')
    exports = []
    for key, label in available:
        records = history(row_map[key]['path'])
        x = [int(r['global_step']) for r in records]
        y = [float(r['loss']) for r in records]
        axes[0].semilogy(x, y, color=COLORS[label], label=label, linewidth=1.3)
        exports.extend(dict(experiment_id=key, global_step=int(r['global_step']),
                            stage=int(r['stage']), loss=float(r['loss'])) for r in records)
    axes[0].set_xlabel('Optimizer updates')
    axes[0].set_ylabel('Training loss')
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=.2)
    if 'full/GO_fPINNs' in row_map:
        edges = stage_values(row_map['full/GO_fPINNs'])
        start = 0
        for i, stage in enumerate(edges):
            stop = stage['global_step']
            if i % 2 == 0:
                axes[0].axvspan(start, stop, color='grey', alpha=.10)
            axes[0].text((start+stop)/2, .98, 'S%d' % i,
                         transform=axes[0].get_xaxis_transform(), ha='center', va='top', fontsize=11)
            start = stop
    pair_panel(axes[1], row_map, [k for k, _ in available], [s for _, s in available],
               [COLORS[s] for _, s in available], [])
    for label, ax in zip('ab', axes):
        ax.text(.5, -.2, '('+label+')', transform=ax.transAxes, ha='center', va='top', fontsize=17)
    save(fig, out, 'Figure_4')
    write_csv(out/'figure4_loss_data.csv', exports)

    fields = {}
    for key, label in available:
        path = Path(row_map[key]['path'])/'plot_predictions.npz'
        if not path.exists():
            continue
        with np.load(path) as f:
            fields[label] = {name: f[name].copy() for name in f.files}
    if not fields:
        return
    first = next(iter(fields.values()))
    for f in fields.values():
        if not np.array_equal(f['tx'], first['tx']) or not np.array_equal(f['exact'], first['exact']):
            raise ValueError('Figure 5 needs identical dense coordinates and references.')
    error_max = max(float(np.abs(f['prediction']-f['exact']).max()) for f in fields.values())
    error_max = max(error_max, 1e-12)
    norm = SymLogNorm(linthresh=max(error_max*1e-4, 1e-12), vmin=0, vmax=error_max)
    vmax = max(float(np.abs(first['exact']).max()), 1e-12)
    labels = list(fields)
    fig, axes = plt.subplots(2, len(labels)+1, figsize=(3.65*(len(labels)+1)+1.3, 7.1),
                             squeeze=False, layout='constrained', sharex=True, sharey=True)
    extent = [0, 1, 0, 1]
    shape = (len(first['t']), len(first['x']))
    for comp in range(2):
        image_exact = axes[comp, 0].imshow(first['exact'][:, comp].reshape(shape), origin='lower',
                                           extent=extent, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[comp, 0].set_ylabel(r'$t$')
        axes[comp, 0].text(.04, .96, r'$u$' if comp == 0 else r'$v$',
                            transform=axes[comp, 0].transAxes, va='top', fontsize=18,
                            bbox=dict(facecolor='white', alpha=.75, edgecolor='none'))
        for j, label in enumerate(labels, start=1):
            values = np.abs(fields[label]['prediction'][:, comp]-fields[label]['exact'][:, comp]).reshape(shape)
            image_error = axes[comp, j].imshow(values, origin='lower', extent=extent,
                                               aspect='auto', cmap='Blues', norm=norm)
            axes[comp, j].text(.04, .96, r'$\mathcal{E}_u$' if comp == 0 else r'$\mathcal{E}_v$',
                               transform=axes[comp, j].transAxes, va='top', fontsize=18,
                               bbox=dict(facecolor='white', alpha=.75, edgecolor='none'))
    axes[0, 0].set_title('Exact solution')
    for j, label in enumerate(labels, start=1):
        axes[0, j].set_title(label)
    for ax in axes[-1]:
        ax.set_xlabel(r'$x$')
        ax.set_xticks([0, .5, 1])
    fig.colorbar(image_exact, ax=axes[:, 0], shrink=.92, pad=.03, label='Exact component')
    fig.colorbar(image_error, ax=axes[:, 1:].ravel().tolist(), shrink=.92, pad=.025,
                 label='Absolute component error')
    save(fig, out, 'Figure_5')
    with (out/'figure5_plot_settings.json').open('w') as handle:
        json.dump({'shared_error_vmax': error_max, 'shared_error_linthresh': norm.linthresh,
                   'model_selection': 'independent validation L2',
                   'dense_fields': {k: str(Path(row_map[j]['path'])/'plot_predictions.npz')
                                    for j, k in available if k in fields}}, handle, indent=2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('runs/C01_case1_complete'))
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--output', type=Path, default=None)
    p.add_argument('--include-smoke', action='store_true')
    a = p.parse_args()
    rows, _, pending = completed(a.root, a.include_smoke)
    rows = [r for r in rows if r['seed'] == a.seed and not r['experiment_id'].startswith('ablation/')]
    if not rows:
        raise SystemExit('No completed requested-seed runs. No paper values are fabricated.')
    out = a.output or a.root/'figures'/('seed_%d' % a.seed)
    out.mkdir(parents=True, exist_ok=True)
    style()
    figure3(rows, out)
    figures45(rows, out)
    (out/'figure_notes.txt').write_text(
        'Figure 3 and Figure 4 stage errors retain the original stage sigma-grid L2.\n'
        'Independent-test error tables are generated separately by collect_case1.py.\n'
        'Figure 3(c): fixed-grid control has five optimization phases, not one strict stage.\n'
        'The strict-single-stage result is in the transfer table.\n'
        'Figure 4(a): shading marks only GO-fPINNs stage boundaries; other methods stop independently.\n'
        'Figure 5: both components and all methods share the same absolute-error normalization.\n'
        'All plots use actual saved data. Missing configurations remain visibly blank.\n'
        'SMOKE outputs are not scientific accuracy results.\n', encoding='utf-8')
    print('Saved figures and plotting CSVs to '+str(out))


if __name__ == '__main__':
    main()
