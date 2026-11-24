"""Run an original experiment in its own directory; pass its flags after --.

Examples:
  python scripts/run_experiment.py --list
  python scripts/run_experiment.py OP2 -- plan
  python scripts/run_experiment.py C01 --dry-run -- --suites full --seeds 1234
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    marker = argv.index('--') if '--' in argv else len(argv)
    own, forwarded = argv[:marker], argv[marker + 1:]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('case', nargs='?')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(own)
    root = Path(__file__).resolve().parents[1]
    registry = json.loads((root / 'configs/experiments.json').read_text(encoding='utf-8'))
    if args.list:
        for key, item in registry.items():
            print('%-4s %-9s %s' % (key, item['category'], item['title']))
        return 0
    aliases = {'case1': 'OP1', 'case2': 'OP2', 'case3': 'OP3', 'case4': 'OP4'}
    case = aliases.get(str(args.case).lower(), str(args.case).upper())
    if case not in registry:
        parser.error('Choose a case from --list.')
    item = registry[case]
    workdir = root / item['path']
    command = [sys.executable, '-u', str(workdir / item['entrypoint'])] + forwarded
    print('Working directory: ' + str(workdir), flush=True)
    print('Command: ' + shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    return subprocess.call(command, cwd=str(workdir))


if __name__ == '__main__':
    raise SystemExit(main())

