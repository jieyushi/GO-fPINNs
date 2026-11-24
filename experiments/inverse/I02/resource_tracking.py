"""Measure optimization memory/time separately from streaming SOE claims."""
import os
import platform
try:import resource
except ImportError:resource=None  # Windows: peak RSS is unavailable.
import subprocess
import time
import numpy as np
import scipy
import torch


def sync(device):
    if device.type=='cuda':torch.cuda.synchronize(device)


def hardware_info(device):
    cpu=platform.processor()
    try:
        with open('/proc/cpuinfo') as handle:
            for line in handle:
                if line.startswith('model name'):cpu=line.split(':',1)[1].strip();break
    except OSError:pass
    info={'python':platform.python_version(),'os':platform.platform(),'cpu':cpu,
        'cpu_logical_count':os.cpu_count(),'torch':torch.__version__,'numpy':np.__version__,
        'scipy':scipy.__version__,'device':str(device),'torch_threads':torch.get_num_threads(),
        'network_dtype':'float32','history_dtype':'float64',
        'cuda_runtime':torch.version.cuda,'cudnn_version':torch.backends.cudnn.version(),
        'cuda_visible_devices':os.getenv('CUDA_VISIBLE_DEVICES'),
        'tf32_matmul_allowed':torch.backends.cuda.matmul.allow_tf32,
        'float32_matmul_precision':torch.get_float32_matmul_precision()}
    try:info['host_ram_bytes']=os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')
    except (OSError,ValueError,AttributeError):info['host_ram_bytes']=None
    if device.type=='cuda':
        props=torch.cuda.get_device_properties(device)
        info['gpu']={'name':props.name,'total_memory_bytes':props.total_memory,
            'compute_capability':[props.major,props.minor],'multiprocessors':props.multi_processor_count,
            'logical_index':device.index if device.index is not None else torch.cuda.current_device()}
        try:
            result=subprocess.run(['nvidia-smi','--query-gpu=index,name,driver_version,memory.total',
                '--format=csv,noheader'],capture_output=True,text=True,timeout=10)
            info['nvidia_smi_inventory']=result.stdout.strip() if result.returncode==0 else None
        except (OSError,subprocess.TimeoutExpired):info['nvidia_smi_inventory']=None
    else:info['gpu']=None
    return info


class TrainingMeter:
    """One scope = loss forward + autograd/backward + clip + Adam step.

    Static live tensors are included in allocated peaks. Validation, checkpoint
    serialization and dataset setup are excluded from the timed optimization.
    GPU peaks cover autograd; CUDA context/non-PyTorch allocations are excluded.
    """
    def __init__(self,device):
        self.device=device;self.seconds=0.;self.steps=0
        self.peak_allocated=None;self.peak_reserved=None;self.peak_increment=None
        self.max_step_seconds=0.
    def begin(self):
        sync(self.device)
        if self.device.type=='cuda':
            self.baseline=torch.cuda.memory_allocated(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self.started=time.perf_counter()
    def end(self):
        sync(self.device);duration=time.perf_counter()-self.started
        self.seconds+=duration;self.steps+=1;self.max_step_seconds=max(self.max_step_seconds,duration)
        if self.device.type=='cuda':
            allocated=torch.cuda.max_memory_allocated(self.device)
            reserved=torch.cuda.max_memory_reserved(self.device)
            self.peak_allocated=max(self.peak_allocated or 0,allocated)
            self.peak_reserved=max(self.peak_reserved or 0,reserved)
            self.peak_increment=max(self.peak_increment or 0,allocated-self.baseline)
        return duration
    def report(self):
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if resource is not None else None
        return {'optimization_seconds':self.seconds,'updates':self.steps,
            'max_measured_update_seconds':self.max_step_seconds,
            'peak_training_allocated_bytes':self.peak_allocated,
            'peak_training_reserved_bytes':self.peak_reserved,
            'peak_step_increment_allocated_bytes':self.peak_increment,
            'process_peak_rss_bytes':None if rss is None else int(rss*(1 if platform.system()=='Darwin' else 1024)),
            'memory_scope':'optimization steps including live data, model, Adam, automatic derivatives and backward; PyTorch allocator only',
            'timer_scope':'synchronized wall time of optimization steps; excludes data setup, validation and checkpoint IO'}


def history_inventory(core,data_list):
    rows=[]
    for data in data_list:
        alpha=float(data['alpha'].detach().cpu()); g=data['N'];ns=(data['M1']-1)*(data['M2']-1)
        coeff=core._get_fast_coeff(alpha,1-alpha/2,data['tau'],core.device)
        nexp=coeff[0].numel()
        rows.append({'alpha':alpha,'spatial_interior_points':ns,'time_intervals':g,'soe_terms':nexp,
            'ideal_streaming_two_component_state_bytes':2*ns*nexp*8,
            'two_component_full_grid_prediction_bytes':2*(data['M1']+1)*(g+1)*(data['M2']+1)*4,
            'warning':'streaming state formula is NOT a measurement of neural training/backward memory'})
    return rows
