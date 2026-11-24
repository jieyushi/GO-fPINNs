"""Equal parameter averaging per run, then mean and SAMPLE SD across seeds."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import single_support as p
import training_support as s


def read_csv(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))


def main(argv=None):
    q=argparse.ArgumentParser(description=__doc__)
    q.add_argument('--root',type=Path,required=True)
    q.add_argument('--operators',default='alpha,beta,gamma')
    q.add_argument('--seeds',default='1234,1235,1236')
    q.add_argument('--include-l1-baseline',action='store_true')
    q.add_argument('--allow-smoke',action='store_true')
    a=q.parse_args(argv);seeds=[int(v) for v in a.seeds.split(',')]
    if not seeds or len(set(seeds))!=len(seeds):q.error('Distinct seeds required.')
    methods=['go']+(['l1_reimplementation'] if a.include_l1_baseline else [])
    per_run=[];curves=[];statistics=[];split_by_operator={};statuses=set()
    for operator in a.operators.split(','):
        for method in methods:
            runs=[];by_value={}
            for seed in seeds:
                run=a.root/'operators'/operator/method/('seed_'+str(seed))
                summary=json.loads((run/'summary.json').read_text())
                split=json.loads((run/'parameter_split.json').read_text())
                p.validate_design(split)
                if split['operator']!=operator:raise ValueError('Wrong operator output.')
                if operator in split_by_operator and split_by_operator[operator]!=split['design_sha256']:
                    raise ValueError('Splits differ across seeds or methods.')
                split_by_operator[operator]=split['design_sha256']
                smoke=summary['status']=='SMOKE_ONLY';statuses.add('smoke' if smoke else 'main')
                if smoke and not a.allow_smoke:raise ValueError('Smoke results require --allow-smoke.')
                rows=read_csv(run/'test_per_parameter.csv')
                if len(rows)!=len(split['test']) or (not smoke and len(rows)!=30):
                    raise ValueError('Incomplete 30-parameter main test set.')
                expected={r['parameter_id']:r for r in split['test']}
                if len({r['parameter_id'] for r in rows})!=len(rows):raise ValueError('Duplicate test ID.')
                for r in rows:
                    if r['parameter_id'] not in expected:raise ValueError('Unexpected test ID.')
                    if abs(float(r['active_value'])-expected[r['parameter_id']]['active_value'])>1e-10:
                        raise ValueError('Test coordinate/ID mismatch.')
                macro={key:float(np.mean([float(r[key]) for r in rows])) for key in p.METRICS}
                if not all(np.isfinite(list(macro.values()))):raise ValueError('Nonfinite metric.')
                if not all(np.isclose(macro[k],summary['test_macro_over_parameters'][k],rtol=1e-12,atol=1e-14) for k in macro):
                    raise ValueError('Saved macro does not equal equal-weight parameter mean.')
                record={'operator':operator,'method':method,'seed':seed,'test_instances':len(rows),
                        'total_updates':summary['total_updates'],**macro,
                        'optimization_hours':summary['resources']['optimization_seconds']/3600,
                        'peak_gpu_mib':(summary['resources']['peak_training_allocated_bytes']/2**20
                                        if summary['resources']['peak_training_allocated_bytes'] is not None else None),
                        'residual_point_evaluations':summary['residual_point_evaluations']}
                runs.append(record);per_run.append(record)
                for row in rows:by_value.setdefault(float(row['active_value']),[]).append(row)
            for key in p.METRICS:
                values=[r[key] for r in runs]
                statistics.append({'operator':operator,'method':method,'metric':key,'seeds':len(values),
                                   'mean':float(np.mean(values)),
                                   'sample_sd':float(np.std(values,ddof=1)) if len(values)>1 else None})
            for value,rows in sorted(by_value.items()):
                if len(rows)!=len(seeds):raise ValueError('Missing seed/parameter entry.')
                values=[float(r['relative_l2']) for r in rows]
                curves.append({'operator':operator,'method':method,'active_value':value,'seed_count':len(values),
                               'relative_l2_mean':float(np.mean(values)),
                               'relative_l2_sample_sd':float(np.std(values,ddof=1)) if len(values)>1 else None})
    if len(statuses)>1:raise ValueError('Never mix smoke and main results.')
    out=a.root/'aggregated';out.mkdir(exist_ok=True,parents=True)
    s.write_csv(out/'per_run.csv',per_run);s.write_csv(out/'mean_std.csv',statistics)
    s.write_csv(out/'error_curves.csv',curves)
    meta={'status':'SMOKE_ONLY' if statuses=={'smoke'} else 'RECOMPUTED_FROM_SAVED_RESULTS',
          'operators':a.operators.split(','),'methods':methods,'seeds':seeds,'design_sha256':split_by_operator,
          'averaging':'equal average over test parameters per seed, then mean/sample SD across seeds',
          'single_seed_sample_sd':None,'noise_level':0.,'original_baseline_available':False,
          'reported_manuscript_numbers_hardcoded':False}
    s.atomic_json(out/'aggregation.json',meta)
    tex=['% Recomputed; a single seed has no sample SD. Primary phase metric uses all points.',
         r'\begin{tabular}{llrrrr}',r'\toprule',
         r'Operator & Method & $\mathcal E_{L_2}$ & $\mathcal E_{\rm amp}$ & $\mathcal E_{\rm phase}$ & $\mathcal E_{\rm MAE}$ \\',r'\midrule']
    for operator in a.operators.split(','):
        for method in methods:
            values=[]
            for key in p.METRICS[:4]:
                record=next(r for r in statistics if r['operator']==operator and r['method']==method and r['metric']==key)
                value='%.3e'%record['mean']
                if record['sample_sd'] is not None:value+=r'\pm %.3e'%record['sample_sd']
                values.append('$'+value+'$')
            label='GO-fPINNs' if method=='go' else 'L1-DeepONet (reimpl.)'
            tex.append(operator+' & '+label+' & '+' & '.join(values)+r' \\')
    tex += [r'\bottomrule',r'\end{tabular}']
    (out/'operator_metrics.tex').write_text('\n'.join(tex)+'\n')
    print(json.dumps(meta,indent=2));return meta


if __name__=='__main__':main()
