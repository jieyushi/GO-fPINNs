# GO-fPINNs

Python 3.10, PyTorch 2.5, CUDA 11.8.

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
python scripts/validate_package.py
```

| ID | Experiment |
| --- | --- |
| C01 | Standard forward problem and component ablations |
| C02 | Large spatial domain |
| C03 | Long-time evolution |
| I01 | Constant-coefficient identification |
| I02 | One-dimensional potential reconstruction |
| I03 | Two-dimensional potential reconstruction |
| OP1 | Separate alpha, beta, and gamma operators |
| OP2 | Joint alpha-beta-gamma operator |
| OP3 | Fixed-data beta-gamma operator |
| OP4 | Bright-hump alpha operator |
| OPA | Controlled alpha-operator ablations |

```bash
python scripts/reproduce.py --list
python scripts/reproduce.py --stage plan
python -u scripts/reproduce.py C01 --stage all --device cuda:0
python -u scripts/reproduce.py OPA --stage train --device cuda:0
python scripts/reproduce.py --stage tables
```

Experiment settings are in `configs/` and each experiment's `DEFAULT_SETTINGS.json`. Parameter designs are in `parameter_samples/`. Use `--help` for individual training, reference, evaluation, and plotting options.
