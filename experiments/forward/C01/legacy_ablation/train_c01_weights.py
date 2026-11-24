"""C01 weighting comparisons and the later ablation foundation (Python 3.8+).

The original solver, local x/t regularizer, rolling consistency, activation,
stage learning rates and optimizer update order are imported unchanged.
Only the three task-loss weights are replaced.
"""
import argparse
import copy
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import signal
import time

import numpy as np
import torch

import baseline_original as b
import run_c01_ablation as original
from experiment_utils import environment, metrics as original_metrics, seed_all, synchronize, timed_inference
from weight_schemes import SCHEMES, description, install_weights, set_stage

FORMAT = 'C01_weight_comparison_v1'
STOP_REQUESTED = False
SOURCE_FILES = [
    'baseline_original.py', 'run_c01_ablation.py', 'experiment_utils.py',
    'operator_architecture.py', 'direct_reference.py', 'weight_schemes.py',
    'train_c01_weights.py',
]


def parse_args(argv=None, default_scheme=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weight-scheme', choices=SCHEMES, default=default_scheme or 'softmax_noise')
    p.add_argument('--variant', choices=original.VARIANTS, default='full_original',
                   help='Default is full only; the original ablations remain available for later use.')
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output', required=True)
    p.add_argument('--budget-mode', choices=['matched_steps', 'original_stop'], default='matched_steps')
    p.add_argument('--epochs-per-stage', type=int, default=2000)
    p.add_argument('--grids', type=b.parse_grids,
                   default=[(10, 10), (20, 20), (30, 30), (40, 40), (50, 50)])
    p.add_argument('--width', type=int, default=60)
    p.add_argument('--extra-hidden-layers', type=int, default=2)
    p.add_argument('--alpha', type=float, default=0.9)
    p.add_argument('--w-grad', type=float, default=0.001)
    p.add_argument('--eval-every', type=int, default=100)
    p.add_argument('--eval-points', type=int, default=4096)
    p.add_argument('--validation-seed', type=int, default=98101)
    p.add_argument('--test-seed', type=int, default=98102)
    p.add_argument('--kernel-mode', choices=['original', 'stable64'], default='original')
    p.add_argument('--network-dtype', choices=['float32', 'float64'], default='float32')
    p.add_argument('--shared-update', choices=['legacy_double', 'once'], default='legacy_double')
    p.add_argument('--noise-amplitude', type=float, default=0.05)
    p.add_argument('--weight-lower-bound', type=float, default=1e-6)
    p.add_argument('--weight-floor', type=float, default=0.2)
    p.add_argument('--patience-evals', type=int, default=10)
    p.add_argument('--tol', type=float, default=1e-8)
    p.add_argument('--safety-stage-cap', type=int, default=0,
                   help='Optional cap for original_stop only; zero retains the original uncapped rule.')
    p.add_argument('--checkpoint-every', type=int, default=100)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args(argv)
    positive = [a.epochs_per_stage, a.width, a.eval_every, a.eval_points,
                a.checkpoint_every, a.threads, a.patience_evals]
    if min(positive) < 1 or a.extra_hidden_layers < 0 or a.safety_stage_cap < 0:
        p.error('Invalid count or layer setting.')
    if min(a.seed, a.validation_seed, a.test_seed) < 0 or a.validation_seed == a.test_seed:
        p.error('Seed values must be nonnegative; validation and test seeds must differ.')
    if not 0 < a.alpha < 1 or not 0 < a.weight_floor < 1:
        p.error('alpha and weight-floor must lie in (0,1).')
    values = [a.w_grad, a.noise_amplitude, a.weight_lower_bound, a.weight_floor, a.tol]
    if not all(math.isfinite(v) for v in values):
        p.error('Weight settings must be finite.')
    if min(a.w_grad, a.noise_amplitude, a.tol) < 0 or a.weight_lower_bound <= 0:
        p.error('Invalid weight setting.')
    if a.network_dtype == 'float64' and a.kernel_mode == 'original':
        p.error('Use stable64 for a float64 network.')
    if a.smoke:
        a.grids = [(4, 4), (6, 6)]
        a.width = 8
        a.extra_hidden_layers = 1
        a.epochs_per_stage = 3
        a.eval_every = 1
        a.eval_points = 64
        a.checkpoint_every = 1
    # Keep the original module's additional configuration fields available.
    args = b.parse_args([])
    for key, value in vars(a).items():
        setattr(args, key, value)
    args.independent_points = a.eval_points
    if args.variant == 'no_grid_enrichment':
        args.epochs_per_stage *= len(args.grids)
        args.grids = [args.grids[-1]]
    return args


def json_write(path, obj):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(str(temporary), str(path))


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    return copy.deepcopy(value)


def data_pack(value):
    if isinstance(value, torch.Tensor):
        return {'tensor_value': value.detach().cpu().clone(), 'requires_grad': value.requires_grad}
    if isinstance(value, dict):
        return {k: data_pack(v) for k, v in value.items()}
    return copy.deepcopy(value)


def data_unpack(value, device):
    if isinstance(value, dict) and set(value) == {'tensor_value', 'requires_grad'}:
        return value['tensor_value'].to(device).requires_grad_(value['requires_grad'])
    if isinstance(value, dict):
        return {k: data_unpack(v, device) for k, v in value.items()}
    return value


def atomic_torch_save(value, path):
    temporary = Path(path).with_name(Path(path).name + '.tmp')
    torch.save(value, temporary)
    os.replace(str(temporary), str(path))


def load_own_checkpoint(path, device='cpu'):
    # Only load checkpoints generated by this package.
    return torch.load(path, map_location=device, weights_only=False)


def configure(args, device):
    state = original.configure(args, device)
    return install_weights(state, args.weight_scheme, args.noise_amplitude,
                           args.weight_lower_bound, args.weight_floor)


def capture(state, args, stage, epoch, M, N, values, previous):
    saved = original.snapshot(state, args, stage, epoch, M, N, values, previous)
    saved.update(format=FORMAT, weight_scheme=args.weight_scheme, weight_stage=stage)
    return cpu_tree(saved)


def restore(state, saved, with_optimizers=False):
    if saved.get('format') != FORMAT:
        raise ValueError('Use a checkpoint from this weighting-comparison package.')
    original.restore(state, saved)
    set_stage(state, saved['weight_stage'])
    if with_optimizers:
        for opt, saved_opt in zip(state['optimizers'], saved['optimizers']):
            opt.load_state_dict(saved_opt)


def load_for_prediction(path, device='cpu'):
    saved = load_own_checkpoint(path)
    args = argparse.Namespace(**saved['config'])
    state = configure(args, torch.device(device))
    restore(state, saved)
    return state, saved


def restore_rng(saved):
    random.setstate(saved['python'])
    np.random.set_state(saved['numpy'])
    torch.set_rng_state(saved['torch'].cpu())
    if saved['cuda'] is not None:
        if not torch.cuda.is_available():
            raise ValueError('This checkpoint was made on CUDA; continue on a CUDA device.')
        torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda']])


def tensor_digest(value):
    digest = hashlib.sha256()
    def add(item):
        if isinstance(item, torch.Tensor):
            digest.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item):
                digest.update(key.encode())
                add(item[key])
        elif isinstance(item, (list, tuple)):
            for part in item:
                add(part)
        else:
            digest.update(repr(item).encode())
    add(value)
    return digest.hexdigest()


def source_hashes():
    root = Path(__file__).resolve().parent
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def signature(args):
    values = vars(args).copy()
    for key in ['resume', 'output', 'overwrite']:
        values.pop(key, None)
    payload = {'args': values, 'sources': source_hashes()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def augmented_metrics(prediction, reference):
    result = original_metrics(prediction, reference)
    p = prediction.detach().double()
    y = reference.detach().double()
    complex_error = torch.linalg.vector_norm(p - y, dim=1)
    result['mean_abs_error'] = float(complex_error.mean())
    pc = torch.complex(p[:, 0], p[:, 1])
    yc = torch.complex(y[:, 0], y[:, 1])
    angle_difference = torch.angle(pc) - torch.angle(yc)
    wrapped = torch.atan2(torch.sin(angle_difference), torch.cos(angle_difference))
    result['phase_mae_all_radians'] = float(wrapped.abs().mean())
    amp = yc.abs()
    result['phase_masked_point_count'] = int((amp > max(float(amp.max()) * 1e-3, 1e-8)).sum())
    result['evaluation_point_count'] = len(p)
    return result


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    temporary = Path(path).with_name(Path(path).name + '.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print('Stop requested; saving after the current update. Re-run with --resume.', flush=True)


def run(args):
    global STOP_REQUESTED
    STOP_REQUESTED = False
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    seed_all(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    run_signature = signature(args)
    config_path = output / 'config.json'
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding='utf-8'))
        if existing.get('signature') != run_signature:
            raise ValueError('Existing output has a different configuration/source. Use a new --output.')
        if not args.resume:
            raise FileExistsError('Output already exists; use --resume or a new --output.')
        summary_path = output / 'summary.json'
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding='utf-8'))
            if summary.get('status') in ['TRAINING_COMPLETED', 'SMOKE_ONLY']:
                print('Matching run already completed; no updates repeated.', flush=True)
                return 0
    state = configure(args, device)
    val_tx, val_y = b.independent_points(args.validation_seed, args.eval_points, device, state['dtype'])
    test_tx, test_y = b.independent_points(args.test_seed, args.eval_points, device, state['dtype'])
    unique = {id(p): p for opt in state['optimizers'] for group in opt.param_groups
              for p in group['params'] if p.requires_grad}
    config = {
        'suite': 'C01_weight_comparison', 'signature': run_signature, 'args': vars(args),
        'weight_rule': description(args.weight_scheme, args.noise_amplitude,
                                   args.weight_lower_bound, args.weight_floor),
        'environment': environment(device), 'source_hashes': source_hashes(),
        'trainable_parameter_count': sum(p.numel() for p in unique.values()),
        'architecture': {'real_network': str(state['realnet']), 'imag_network': str(state['imagenet']),
                         'weight_MLP': str(state['hypernet']), 'hidden_blocks': args.extra_hidden_layers+1},
        'initial_parameters_sha256': tensor_digest({
            'real': state['realnet'].state_dict(), 'imag': state['imagenet'].state_dict(),
            'hyper': state['hypernet'].state_dict(), 'a1': state['a1'], 'a2': state['a2']}),
        'selection': 'minimum independent validation relative L2; compare schemes by seed-mean validation only',
        'test_used_for_training_or_scheme_selection': False,
        'unchanged': ['fast L2-1sigma residual/kernel mode', 'local x/t gradient regularizer',
                      'w_grad*exp(-3*detached_Lf)', 'rolling sigma-grid consistency with 1/(epoch+1)',
                      'stage-last transfer', 'stage LR=max(1e-5,1e-3*0.5**stage)',
                      'original real/imag/hyper optimizer update order',
                      'original sigma-grid console relative L2'],
        'phase_primary': 'original masked phase MAE: reference amplitude > max(1e-3*max_amplitude,1e-8)',
        'phase_secondary': 'phase_mae_all_radians: wrapped angle error over all evaluation points',
        'comparison_protocol': {
            'grids': args.grids, 'epochs_per_stage': args.epochs_per_stage, 'budget_mode': args.budget_mode,
            'eval_every': args.eval_every, 'eval_points': args.eval_points,
            'validation_seed': args.validation_seed, 'test_seed': args.test_seed,
            'width': args.width, 'extra_hidden_layers': args.extra_hidden_layers, 'alpha': args.alpha,
            'w_grad': args.w_grad, 'kernel_mode': args.kernel_mode, 'network_dtype': args.network_dtype,
            'shared_update': args.shared_update, 'patience_evals': args.patience_evals, 'tol': args.tol,
            'safety_stage_cap': args.safety_stage_cap, 'checkpoint_every': args.checkpoint_every,
            'device': str(device), 'threads': args.threads},
    }
    if not config_path.exists():
        json_write(config_path, config)
    else:
        config = existing
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    records, stages = [], []
    previous = (None, None, None)
    best_val = float('inf')
    best_sigma = float('inf')
    best_ckpt = None
    global_step = work = 0
    elapsed_before = 0.0
    allocated_before = reserved_before = 0
    start_stage = next_epoch = 0
    resume_data = None
    stage_best_loss, bad = float('inf'), 0
    resume_file = output / 'resume.pt'
    if args.resume and resume_file.exists():
        saved = load_own_checkpoint(resume_file)
        if saved.get('signature') != run_signature:
            raise ValueError('Resume checkpoint configuration/source mismatch.')
        restore(state, saved['state'], with_optimizers=True)
        records, stages = saved['records'], saved['stages']
        previous = tuple(x.to(device) if x is not None else None for x in saved['previous'])
        best_val, best_sigma, best_ckpt = saved['best_val'], saved['best_sigma'], saved['best_checkpoint']
        global_step, work = saved['global_step'], saved['residual_points_processed']
        elapsed_before = saved['elapsed_seconds']
        allocated_before, reserved_before = saved['peak_allocated_bytes'], saved['peak_reserved_bytes']
        start_stage, next_epoch = saved['next_stage'], saved['next_epoch']
        stage_best_loss, bad = saved['stage_best_loss'], saved['no_improvement_count']
        resume_data = data_unpack(saved['stage_data'], device) if saved['stage_data'] is not None else None
        restore_rng(saved['rng'])
        print('Resuming stage=%d next_epoch=%d updates=%d' %
              (start_stage, next_epoch, global_step), flush=True)
    elif args.resume and config_path.exists() and not records:
        print('No resume checkpoint yet; starting the matching run from epoch zero.', flush=True)
    synchronize(device)
    clock_start = time.perf_counter()

    def elapsed():
        return elapsed_before + time.perf_counter() - clock_start

    def peaks():
        allocated = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0
        reserved = torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else 0
        return max(allocated_before, allocated), max(reserved_before, reserved)

    def save_progress(stage, epoch, M, N, next_stage, next_epoch_value, data):
        peak_alloc, peak_reserved = peaks()
        state_checkpoint = capture(state, args, stage, epoch, M, N, {}, previous)
        progress = {
            'signature': run_signature, 'state': state_checkpoint,
            'next_stage': next_stage, 'next_epoch': next_epoch_value,
            'stage_data': data_pack(data) if data is not None else None,
            'previous': cpu_tree(previous), 'rng': cpu_tree(b.rng_state()),
            'records': records, 'stages': stages,
            'best_val': best_val, 'best_sigma': best_sigma, 'best_checkpoint': best_ckpt,
            'stage_best_loss': stage_best_loss, 'no_improvement_count': bad,
            'global_step': global_step, 'residual_points_processed': work,
            'elapsed_seconds': elapsed(), 'peak_allocated_bytes': peak_alloc,
            'peak_reserved_bytes': peak_reserved,
        }
        atomic_torch_save(progress, resume_file)
        write_csv(output / 'training.csv', records)
        json_write(output / 'progress.json', {
            'status': 'PAUSED' if STOP_REQUESTED else 'RUNNING',
            'next_stage': next_stage, 'next_epoch': next_epoch_value,
            'global_step': global_step, 'completed_stages': len(stages),
            'best_validation_l2': best_val if math.isfinite(best_val) else None,
            'elapsed_seconds': elapsed(), 'weight_scheme': args.weight_scheme,
        })

    print('C01 | variant=%s | weights=%s | budget=%s | epochs/stage=%d | grids=%s' %
          (args.variant, args.weight_scheme, args.budget_mode, args.epochs_per_stage, args.grids), flush=True)
    for sid in range(start_stage, len(args.grids)):
        M, N = args.grids[sid]
        set_stage(state, sid)
        continuing = sid == start_stage and resume_data is not None
        if continuing:
            data = resume_data
            first_epoch = next_epoch
        else:
            for optimizer in state['optimizers']:
                for group in optimizer.param_groups:
                    group['lr'] = b.get_stage_lr(sid)
            data = b.build_stage_data(M, N, args.alpha, device, state['dtype'], args.kernel_mode)
            stage_best_loss, bad = float('inf'), 0
            first_epoch = 0
            save_progress(sid, -1, M, N, sid, 0, data)
        print('\n========== Stage %d: Grid %d x %d ==========' % (sid, M, N), flush=True)
        print('[Stage %d] Learning rate set to %.2e' %
              (sid, state['optimizers'][0].param_groups[0]['lr']), flush=True)
        stage_data_hash = tensor_digest(data)
        cap = args.epochs_per_stage if args.budget_mode == 'matched_steps' else (args.safety_stage_cap or 10**9)
        for epoch in range(first_epoch, cap):
            active = (None, None, None) if sid == 0 or args.variant in [
                'no_consistency', 'no_grid_enrichment', 'coordinate_matched'] else previous
            if args.variant == 'coordinate_matched' and sid > 0:
                active = previous
            gradient_weight = 0.0 if args.variant in ['no_gradient', 'coordinate_matched'] else args.w_grad
            terms = b.training_step(state, data, M, N, args.alpha, gradient_weight, active, epoch)
            global_step += 1
            work += (M-1)*N
            row = {'stage': sid, 'epoch': epoch, 'global_step': global_step,
                   'weight_scheme': args.weight_scheme,
                   'learning_rate': state['optimizers'][0].param_groups[0]['lr'],
                   'elapsed_seconds': elapsed()}
            row.update({key: float(value.detach().cpu()) for key, value in terms.items()})
            evaluate = epoch % args.eval_every == 0 or epoch+1 == cap
            if evaluate:
                sigma_l2, tx, pred = b.uploaded_sigma_l2(state, data)
                validation_l2 = b.relative_l2(b.prediction(state, val_tx), val_y)
                current_teacher = (tx.detach(), pred[:, 0:1].detach(), pred[:, 1:2].detach())
                row.update(l2=sigma_l2, sigma_l2=sigma_l2, validation_l2=validation_l2)
                checkpoint = capture(state, args, sid, epoch, M, N,
                                     {'validation_l2': validation_l2, 'sigma_l2': sigma_l2}, previous)
                if validation_l2 < best_val:
                    best_val, best_ckpt = validation_l2, checkpoint
                    atomic_torch_save(checkpoint, output / 'best_validation.pt')
                if sigma_l2 < best_sigma:
                    best_sigma = sigma_l2
                    atomic_torch_save(checkpoint, output / 'original_best_sigma.pt')
                if row['loss'] < stage_best_loss - args.tol:
                    stage_best_loss, bad = row['loss'], 0
                else:
                    bad += 1
                row['no_improvement_count'] = bad
                print('[Stage %d] Epoch %6d | Loss %.3e | L2 %.3e | val %.3e | '
                      'w_f %.3e | w_ic %.3e | w_bc %.3e' %
                      (sid, epoch, row['loss'], sigma_l2, validation_l2,
                       row['w_f'], row['w_ic'], row['w_bc']), flush=True)
            records.append(row)
            by_stop = args.budget_mode == 'original_stop' and evaluate and bad >= args.patience_evals
            by_cap = epoch+1 == cap
            if by_stop or by_cap:
                reason = 'original_best_loss_patience' if by_stop else (
                    'matched_update_budget' if args.budget_mode == 'matched_steps' else 'explicit_safety_cap')
                atomic_torch_save(capture(state, args, sid, epoch, M, N, {}, previous),
                                  output / ('stage_%02d_last.pt' % sid))
                stages.append({'stage': sid, 'M': M, 'N': N, 'epochs': epoch+1,
                               'lr': b.get_stage_lr(sid), 'stop_reason': reason,
                               'stage_data_sha256': stage_data_hash})
                stage_best_loss, bad = float('inf'), 0
                previous=current_teacher
                save_progress(sid, epoch, M, N, sid+1, 0, None)
                break
            if (epoch+1) % args.checkpoint_every == 0 or STOP_REQUESTED:
                save_progress(sid, epoch, M, N, sid, epoch+1, data)
            if STOP_REQUESTED:
                print('Checkpoint saved to %s; resume with the same command and --resume.' % resume_file, flush=True)
                return 130
        resume_data = None
        if STOP_REQUESTED:
            return 130
    synchronize(device)
    seconds = elapsed()
    peak_alloc, peak_reserved = peaks()
    if best_ckpt is None:
        raise RuntimeError('No validation checkpoint was produced.')
    last_row = records[-1]
    M, N = args.grids[-1]
    last_checkpoint = capture(state, args, len(args.grids)-1, last_row['epoch'], M, N, {}, previous)
    atomic_torch_save(last_checkpoint, output / 'last.pt')
    last_metrics = augmented_metrics(b.prediction(state, test_tx), test_y)
    restore(state, best_ckpt)
    test_prediction = b.prediction(state, test_tx)
    selected_metrics = augmented_metrics(test_prediction, test_y)
    atomic_torch_save(best_ckpt, output / 'final.pt')
    np.savez(output / 'test_predictions.npz', tx=test_tx.cpu().numpy(),
             exact=test_y.cpu().numpy(), prediction=test_prediction.cpu().numpy())
    np.savez(output / 'evaluation_points.npz', validation_tx=val_tx.cpu().numpy(),
             validation_exact=val_y.cpu().numpy(), test_tx=test_tx.cpu().numpy(), test_exact=test_y.cpu().numpy())
    result = {
        'suite': 'C01_weight_comparison', 'signature': run_signature,
        'weight_scheme': args.weight_scheme, 'variant': args.variant, 'seed': args.seed,
        'status': 'SMOKE_ONLY' if args.smoke else 'TRAINING_COMPLETED',
        'validation_best': best_val, 'original_min_sigma_l2': best_sigma,
        'selected_stage': best_ckpt['stage'], 'selected_epoch': best_ckpt['epoch'],
        'test': selected_metrics, 'test_last': last_metrics, 'steps': global_step,
        'residual_points_processed': work, 'training_seconds': seconds,
        'training_time_scope': 'training loop, validation and checkpoint I/O; excludes time while paused and final test/inference',
        'parameter_count': config['trainable_parameter_count'], 'stages': stages,
        'peak_allocated_bytes': peak_alloc if device.type == 'cuda' else None,
        'peak_reserved_bytes': peak_reserved if device.type == 'cuda' else None,
        'cuda_peak_scope': 'this process including neural training/backward, validation and checkpoint snapshots; not forward SOE history alone',
        'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'inference_seconds_per_eval_set': timed_inference(lambda: b.prediction(state, test_tx), device),
        'model_selection': 'minimum independent validation L2; final.pt mirrors best_validation.pt',
        'test_used_for_training_or_scheme_selection': False,
    }
    write_csv(output / 'training.csv', records)
    json_write(output / 'summary.json', result)
    json_write(output / 'progress.json', {'status': result['status'], 'global_step': global_step,
                                        'completed_stages': len(stages), 'best_validation_l2': best_val,
                                        'weight_scheme': args.weight_scheme})
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


def main(argv=None, default_scheme=None):
    args = parse_args(argv, default_scheme)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'run.lock').open('a+', encoding='utf-8') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('This exact output is already owned by another process. Different schemes/seeds must use different output directories.')
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()) + '\n')
        handle.flush()
        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)
        return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
