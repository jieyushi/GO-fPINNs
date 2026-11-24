"""Check source syntax, local dependencies, English text, and anonymous packaging.

This lightweight check does not train neural networks or certify numerical
accuracy. It uses the standard library and can run before installing PyTorch.
"""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
TEXT_EXTENSIONS = {'.py','.json','.csv','.md','.txt'}
CJK = re.compile(r'[\u3400-\u9fff]')
PRIVATE_PATH = re.compile(r'(?:/home/|/Users/|/root/|/workspace/|[A-Za-z]:\\Users\\)')
EMAIL = re.compile(r'[A-Za-z0-9_.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
ACCOUNT_URL = re.compile(r'https?://(?:www\.)?(?:github|gitlab)\.com/[^\s\)"\']+')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    problems = []
    files = [p for p in root.rglob('*') if p.is_file()
             and '__pycache__' not in p.parts and p.suffix != '.pyc']
    readmes = [p for p in files if p.name.lower().startswith('readme')]
    if readmes != [root/'README.md']:
        if len(readmes) != 1 or readmes[0] != root/'README.md':
            problems.append('Exactly one root README.md is required.')
    python_files = [p for p in files if p.suffix == '.py']
    all_modules = {p.stem for p in python_files}
    registry = json.loads((root/'configs/experiments.json').read_text())
    for key, value in registry.items():
        if not (root/value['path']/value['entrypoint']).is_file():
            problems.append('Missing experiment entry point: '+key)
    for p in files:
        rel = str(p.relative_to(root))
        if CJK.search(rel): problems.append('Non-English filename: '+rel)
        if p.suffix in {'.pdf','.png','.svg','.jpg','.jpeg','.pt','.pth','.pkl','.zip'}:
            problems.append('Pre-generated asset or nested archive: '+rel)
        if p.suffix not in TEXT_EXTENSIONS: continue
        text = p.read_text(encoding='utf-8-sig')
        if CJK.search(text): problems.append('Non-English text: '+rel)
        # The validator's regular expressions themselves are not identifying data.
        if p.name != 'validate_package.py':
            if PRIVATE_PATH.search(text): problems.append('Machine-specific private path: '+rel)
            if EMAIL.search(text): problems.append('Email address: '+rel)
            if ACCOUNT_URL.search(text): problems.append('Identifying repository URL: '+rel)
        if p.suffix == '.json':
            try: json.loads(text)
            except ValueError as exc: problems.append(f'Invalid JSON: {rel}: {exc}')
        elif p.suffix == '.py':
            try:
                tree = ast.parse(text, filename=rel)
                compile(text, rel, 'exec')
            except (SyntaxError, ValueError) as exc:
                problems.append(f'Invalid Python: {rel}: {exc}'); continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    name = node.module.split('.')[0]
                    if name in all_modules and not any((directory/(name+'.py')).is_file()
                        for directory in [p.parent, p.parent/'legacy_ablation',
                                          p.parent/'legacy_alpha_ablation',p.parent/'c01_source']):
                        # Package imports and repository-local path injection may
                        # legitimately resolve a module from another case folder.
                        if name not in {'baseline_original','weight_schemes','direct_reference'}:
                            problems.append(f'Inspect local import: {rel}: {node.module}')
    index = root/'parameter_samples/index.csv'
    if index.exists():
        with index.open(newline='') as handle:
            for row in csv.DictReader(handle):
                p = root/row['file']
                if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != row['sha256']:
                    problems.append('Parameter design hash mismatch: '+row['file'])
    report = {'files':len(files),'python_files':len(python_files),
              'readmes':len(readmes),'problems':problems,
              'scope':'Packaging and syntax only; numerical training has not been validated by this command.'}
    print(json.dumps(report,indent=2))
    return 1 if problems else 0


if __name__ == '__main__':
    raise SystemExit(main())
