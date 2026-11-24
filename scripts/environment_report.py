"""Record the actual software/hardware environment without changing it."""
import argparse
import importlib
import json
from pathlib import Path
import platform
import subprocess
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('environment.local.json'))
    args = parser.parse_args(argv)
    info = {'python': sys.version, 'platform': platform.platform(), 'packages': {}}
    for name in ['numpy', 'scipy', 'matplotlib', 'torch', 'threadpoolctl']:
        try:
            module = importlib.import_module(name)
            info['packages'][name] = getattr(module, '__version__', 'installed')
        except ImportError:
            info['packages'][name] = None
    if info['packages']['torch']:
        import torch
        info['cuda_runtime'] = torch.version.cuda
        info['cuda_available'] = torch.cuda.is_available()
        info['gpus'] = []
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                prop = torch.cuda.get_device_properties(index)
                info['gpus'].append({'index': index, 'name': prop.name,
                                     'total_memory_bytes': prop.total_memory})
    try:
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=str(root), capture_output=True, text=True)
        info['git_commit'] = result.stdout.strip() if result.returncode == 0 else None
    except FileNotFoundError:
        info['git_commit'] = None
    info['shared_environment_python_requirement'] = 'Python >=3.10; per-case requirements also apply'
    info['python_requirement_satisfied'] = sys.version_info >= (3, 10)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(info, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(info, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

