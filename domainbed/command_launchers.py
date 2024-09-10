# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
A command launcher launches a list of commands on a cluster; implement your own
launcher to add support for your cluster. We've provided an example launcher
which runs all commands serially on the local machine.
"""

import subprocess
import time
import torch


def local_launcher(commands):
    """Launch commands serially on the local machine."""
    for cmd in commands:
        subprocess.call(cmd, shell=True)


def dummy_launcher(commands):
    """
    Doesn't run anything; instead, prints each command.
    Useful for testing.
    """
    for cmd in commands:
        print(f'Dummy launcher: {cmd}')


def multi_gpu_launcher(commands):
    """
    Launch commands on the local machine, using all GPUs in parallel.
    """
    print('WARNING: using experimental multi_gpu_launcher.')
    n_gpus = torch.cuda.device_count()
    procs_by_gpu = [None] * n_gpus

    while len(commands) > 0:
        for gpu_idx in range(n_gpus):
            proc = procs_by_gpu[gpu_idx]
            if (proc is None) or (proc.poll() is not None):
                # Nothing is running on this GPU; launch a command.
                cmd = commands.pop(0)
                new_proc = subprocess.Popen(
                    f'CUDA_VISIBLE_DEVICES={gpu_idx} {cmd}', shell=True)
                procs_by_gpu[gpu_idx] = new_proc
                break
        time.sleep(1)

    # Wait for the last few tasks to finish before returning
    for p in procs_by_gpu:
        if p is not None:
            p.wait()


def same_type_gpu_launcher(commands, first_gpu_id, n_gpus, jobs_per_gpu):
    """
    Launch commands on the local machine, using all GPUs in parallel.
    """
    print('WARNING: same_type_gpu_launcher.')
    procs_by_gpu = [None] * n_gpus * jobs_per_gpu

    while len(commands) > 0:
        for process_idx in range(n_gpus * jobs_per_gpu):
            proc = procs_by_gpu[process_idx]
            if (proc is None) or (proc.poll() is not None):
                # Nothing is running on this GPU; launch a command.
                cmd = commands.pop(0)
                new_proc = subprocess.Popen(
                    f'CUDA_VISIBLE_DEVICES={(process_idx % n_gpus) + first_gpu_id} {cmd}', shell=True)
                procs_by_gpu[process_idx] = new_proc
                break
        time.sleep(1)

    # Wait for the last few tasks to finish before returning
    for p in procs_by_gpu:
        if p is not None:
            p.wait()


def two_types_gpu_launcher(commands):
    """
    Launch commands on the local machine, using all GPUs in parallel.
    """
    print('WARNING: two_types_gpu_launcher.')
    first_type_1_gpu_id = 2
    n_type_1_gpus = 2
    process_per_type_1_gpu = 10

    first_type_2_gpu_id = 4
    n_type_2_gpus = 4
    process_per_type_2_gpu = 3
    procs_by_gpu = [None] * (n_type_1_gpus * process_per_type_1_gpu + n_type_2_gpus * process_per_type_2_gpu)

    while len(commands) > 0:
        for process_idx in range(len(procs_by_gpu)):
            proc = procs_by_gpu[process_idx]
            if (proc is None) or (proc.poll() is not None):
                # Nothing is running on this GPU; launch a command.
                cmd = commands.pop(0)

                if process_idx < (n_type_1_gpus * process_per_type_1_gpu):
                    new_proc = subprocess.Popen(
                        f'CUDA_VISIBLE_DEVICES={(process_idx % n_type_1_gpus) + first_type_1_gpu_id} {cmd}', shell=True)
                else:
                    new_proc = subprocess.Popen(
                        f'CUDA_VISIBLE_DEVICES={(process_idx % n_type_2_gpus) + first_type_2_gpu_id} {cmd}', shell=True)
                procs_by_gpu[process_idx] = new_proc
                break
        time.sleep(1)

    # Wait for the last few tasks to finish before returning
    for p in procs_by_gpu:
        if p is not None:
            p.wait()


REGISTRY = {
    'local': local_launcher,
    'dummy': dummy_launcher,
    'multi_gpu': multi_gpu_launcher,
    'same_type_gpu': same_type_gpu_launcher,
    'two_types_gpu': two_types_gpu_launcher
}

try:
    from domainbed import facebook

    facebook.register_command_launchers(REGISTRY)
except ImportError:
    pass
