"""Export tables from completed-run CSV/LaTeX outputs; never fill missing results.

Existing case-generated LaTeX tables are copied verbatim. Numeric CSV columns
are exported with their existing row labels. A schema file records the
source and columns of every generated table. Source files remain unchanged.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
LABEL_COLUMNS = ('case', 'method', 'operator', 'experiment_id', 'variant',
                 'budget_mode', 'unknown', 'supervised_points', 'noise_level',
                 'problem', 'configuration', 'alpha', 'beta', 'gamma', 'stage',
                 'seed', 'metric')
METRIC_WORDS = ('relative_l2', 'phase_mae', 'mean_abs', 'mae', 'error',
                'seconds', 'time', 'peak', 'memory', 'parameter', 'residual',
                'updates', 'mean', 'std', 'sample_sd', 'count', 'seeds')
EXCLUDED_WORDS = ('curve', 'training_curve', 'training.csv', 'stage_test_errors',
                  'observation', 'collocation', 'training_history', 'figure')


def tex_escape(value):
    text = str(value)
    replacements = {'\\':r'\textbackslash{}', '&':r'\&', '%':r'\%', '$':r'\$',
                    '#':r'\#', '_':r'\_', '{':r'\{', '}':r'\}', '~':r'\textasciitilde{}'}
    return ''.join(replacements.get(c, c) for c in text)


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def csv_to_tex(source, output):
    with source.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        rows = list(reader)
    if not rows or len(rows) > 3000: return None
    labels = [c for c in LABEL_COLUMNS if c in fields]
    columns = [c for c in fields if c not in labels and any(w in c.lower() for w in METRIC_WORDS)
               and any(number(r.get(c)) is not None for r in rows)]
    if not columns: return None
    if not labels: labels = [fields[0]]
    # Case collectors own the statistical protocol. Preserve their rows rather
    # than guessing which configurations are repeated seeds of one experiment.
    formatted = [[tex_escape(row.get(c, '')) for c in labels] +
                 [f'{number(row.get(c)):.6g}' if number(row.get(c)) is not None else '--'
                  for c in columns] for row in rows]
    names = labels + columns
    lines = ['% Generated from completed-run outputs; requires booktabs.',
             '% Values retain the units used in the source CSV.',
             r'\begin{tabular}{' + 'l'*len(labels) + 'r'*len(columns) + '}',
             r'\toprule', ' & '.join(map(tex_escape, names)) + r' \\', r'\midrule']
    lines += [' & '.join(row) + r' \\' for row in formatted]
    lines += [r'\bottomrule', r'\end{tabular}', '']
    output.write_text('\n'.join(lines), encoding='utf-8')
    return {'columns':names,'rows':len(formatted),'units':'unchanged from source',
            'aggregation':'none; recorded rows and native collector statistics preserved'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    source_root = args.results_root.resolve()
    output = args.output.resolve()
    if source_root == output: parser.error('Use a separate table output directory.')
    if not source_root.is_dir(): parser.error('The results root does not exist.')
    files = sorted(p for p in source_root.rglob('*') if p.is_file()
                   and output not in p.parents)
    if any('smoke' in part.lower() for part in source_root.parts):
        parser.error('Smoke outputs cannot be exported as paper tables.')
    output.mkdir(parents=True, exist_ok=True)
    index = []
    for source in files:
        if any('smoke' in part.lower() for part in source.relative_to(source_root).parts): continue
        rel = source.relative_to(source_root)
        if source.suffix == '.tex':
            dest = output/'case_tables'/rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
            index.append({'source':str(rel),'output':str(dest.relative_to(output)),
                          'kind':'original case-generated LaTeX'})
        elif source.suffix == '.csv' and not any(w in source.name.lower() for w in EXCLUDED_WORDS):
            dest = output/'csv_tables'/rel.with_suffix('.tex')
            dest.parent.mkdir(parents=True, exist_ok=True)
            info = csv_to_tex(source, dest)
            if info:
                index.append({'source':str(rel),'output':str(dest.relative_to(output)),
                    'kind':'CSV numeric table','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),**info})
    static_path = ROOT/'configs/static_table_templates.json'
    if static_path.exists():
        for name, body in json.loads(static_path.read_text()).items():
            dest = output/'configuration_tables'/(name+'.tex')
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_text(body+'\n')
            index.append({'source':'configs/static_table_templates.json','output':str(dest.relative_to(output)),
                          'kind':'static manuscript configuration or definition table'})
    (output/'table_index.json').write_text(json.dumps(index, indent=2)+'\n')
    print(f'Exported {len(index)} tables to {output}')
    if not any(i['kind'] != 'static manuscript configuration or definition table' for i in index):
        print('No completed numerical result tables were found; missing values were not filled.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
