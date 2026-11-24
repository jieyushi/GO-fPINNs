"""Complete Case-1 run design, usable without importing PyTorch."""
from collections import OrderedDict

SUITES = ('full', 'activation', 'discretization', 'transfer', 'gate', 'comparison', 'ablation')
ABLATIONS = ('full_original', 'no_dual_tanh', 'no_gradient', 'no_attention',
             'no_soe', 'no_grid_enrichment', 'coordinate_matched')
BASE_GRIDS = [(10, 10), (20, 20), (30, 30), (40, 40), (50, 50)]


def make_plan(selected, seeds, depths=(2, 3, 4, 5), widths=(20, 40, 60, 80),
              width=60, extra_hidden_layers=2, ablations=ABLATIONS):
    jobs = OrderedDict()

    def add(name, suite, label, **changes):
        settings = dict(fractional_scheme='fast_l21', network_activation='dual_tanh',
                        branch='coordinate', use_gradient=1, use_attention=1,
                        use_consistency=1, width=width, extra_hidden_layers=extra_hidden_layers,
                        grids=list(BASE_GRIDS))
        settings.update(changes)
        if name in jobs:
            if settings != jobs[name]['settings']:
                raise ValueError('Conflicting aliases for '+name)
            jobs[name]['roles'].append({'suite': suite, 'label': label})
        else:
            jobs[name] = {'experiment_id': name, 'settings': settings,
                          'roles': [{'suite': suite, 'label': label}], 'kind': 'early_stop'}

    for suite in selected:
        if suite in ('full', 'transfer', 'gate', 'comparison'):
            labels = {'full': 'GO-fPINNs', 'transfer': 'Multi-stage',
                      'gate': 'Coordinate branch (g=0)', 'comparison': 'GO-fPINNs'}
            add('full/GO_fPINNs', suite, labels[suite])
        if suite == 'activation':
            for act in ('tanh', 'dual_tanh'):
                for depth in depths:
                    if depth < 1:
                        raise ValueError('Total hidden-layer depth must be positive.')
                    extra_depth = depth - 1
                    for w in widths:
                        # Reuse the identical Full rather than train it a second time.
                        name = ('full/GO_fPINNs' if act == 'dual_tanh' and extra_depth == extra_hidden_layers and w == width
                                else 'activation/%s_depth%d_width%d' % (act, depth, w))
                        add(name, suite, '%s depth=%d width=%d' % (act, depth, w),
                            network_activation=act, extra_hidden_layers=extra_depth, width=w)
        elif suite == 'discretization':
            for direction in ('temporal', 'spatial'):
                grids = [(10, n) if direction == 'temporal' else (n, 10) for n in (10, 20, 30, 40, 50)]
                for scheme in ('l1', 'fast_l1', 'l21', 'fast_l21'):
                    add('discretization/%s_%s' % (direction, scheme), suite,
                        direction+' '+scheme, fractional_scheme=scheme, grids=grids)
        elif suite == 'transfer':
            add('transfer/fixed_grid', suite, 'Fixed-grid control',
                grids=[(50, 50)]*5, use_consistency=0)
            add('transfer/single_stage', suite, 'Strict single stage',
                grids=[(50, 50)], use_consistency=0)
        elif suite == 'gate':
            add('gate/operator_g1', suite, 'Fixed-parameter operator branch (g=1)', branch='operator')
        elif suite == 'comparison':
            for name, scheme in (('fPINNs', 'l1'), ('FL1_fPINN', 'fast_l1')):
                add('comparison/'+name, suite, name.replace('_', '-'),
                    fractional_scheme=scheme, network_activation='tanh',
                    use_gradient=0, use_attention=0, use_consistency=0,
                    grids=[(50, 50)], initial_lr=1e-4)
        elif suite == 'ablation':
            for variant in ablations:
                name = 'ablation/'+variant
                jobs[name] = {'experiment_id': name, 'kind': 'fixed_ablation',
                              'variant': variant, 'roles': [{'suite': suite, 'label': variant}]}
    result = []
    for job in jobs.values():
        for seed in seeds:
            result.append(dict(job, seed=seed))
    return result


def grids_string(grids):
    return ','.join('%dx%d' % tuple(g) for g in grids)
