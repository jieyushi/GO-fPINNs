"""Collect existing generated figure files using the manuscript filename map.

Missing panels are reported rather than replaced. The long-time outputs retain
their GO-only names because the historical baseline was not supplied.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args(argv)
    results = args.results_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    mapping = json.loads((ROOT/'configs/reproduction_map.json').read_text())['figures']
    for item in mapping:
        candidates = [results/rel.format(seed=args.seed) for rel in item['source_candidates']]
        source = next((p for p in candidates if p.is_file()), None)
        record = {'manuscript_file':item['manuscript_file'], 'case':item['case'],
                  'status':'missing','note':item.get('note','')}
        if source:
            name = item.get('collection_name', item['manuscript_file'])
            dest = output/name
            shutil.copyfile(source, dest)
            record.update(status='copied',source=str(source.relative_to(results)),
                          output=name,sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
        entries.append(record)
    (output/'figure_index.json').write_text(json.dumps(entries, indent=2)+'\n')
    missing = [r['manuscript_file'] for r in entries if r['status']=='missing']
    print(f'Collected {len(entries)-len(missing)} existing figures to {output}')
    if missing: print('Not available: '+', '.join(missing))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
