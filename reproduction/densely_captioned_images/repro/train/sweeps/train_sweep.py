#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

from densely_captioned_images.repro.train.train_clip import CLIPAndNegConfig, CLIPTrainJob, get_dir_name
from densely_captioned_images.repro.config import MODEL_PATH
import itertools
import submitit
import time
import os

def makeGrid(pars_dict):  
    keys=pars_dict.keys()
    combinations=itertools.product(*pars_dict.values())
    ds=[dict(zip(keys,cc)) for cc in combinations]
    return ds

def main():
    base_only_sweep = {
        "lora_r": [1], 
        "lora_alpha": [2],
        "lora_dropout": [0.2],
        "train_count": [7800],
        "valid_count": [100],
        "use_subcaptions": [False],
        "caption_negative_source": ['swaps'],
        "caption_negative_strategy": ['rand'],
        "caption_selection": ['first', 'pick1', 'pick5'],
        "lr": [1e-3, 1e-4],
        "bs": [32],
        "loss_alpha": [1],
        "loss_beta": [1, 0],
        "sampler": ['rand'],
        "datasource": ['long_captions'],
        "epochs": [10],
    }

    base_only_baselines = {
        "lora_r": [1], 
        "lora_alpha": [2],
        "lora_dropout": [0.2],
        "train_count": [7800],
        "valid_count": [100],
        "use_subcaptions": [False],
        "caption_negative_source": ['spacy'],
        "caption_negative_strategy": ['rand'],
        "caption_selection": ['first'],
        "lr": [1e-3, 1e-4],
        "bs": [32],
        "loss_alpha": [1],
        "loss_beta": [1, 0],
        "sampler": ['rand'],
        "datasource": ['long_captions', 'coco', 'localized_narratives'],
        "epochs": [10],
    }

    base_only_sweep_params = makeGrid(base_only_sweep) + makeGrid(base_only_baselines)

    base_only_sweep = [CLIPAndNegConfig(**p) for p in base_only_sweep_params]

    final_sweep = {
        "lora_r": [1, 2, 4], 
        "lora_alpha": [2],
        "lora_dropout": [0.2],
        "train_count": [1e30],
        "valid_count": [1e30],
        "caption_negative_source": ['swaps'],
        "caption_negative_strategy": ['rand'],
        "caption_selection": ['pick1', 'pick5', 'first'],
        "lr": [1e-3, 1e-4],
        "bs": [32],
        "loss_alpha": [1],
        "loss_beta": [9, 1, 0],
        "sampler": ['rand'],
        "datasource": ['long_captions'],
        "epochs": [10],
    }

    final_sweep_baseline = {
        "lora_r": [1], 
        "lora_alpha": [2],
        "lora_dropout": [0.2],
        "train_count": [1e30],
        "valid_count": [1e30],
        "caption_negative_source": ['spacy'],
        "caption_negative_strategy": ['rand'],
        "caption_selection": ['first'],
        "lr": [1e-3, 1e-4],
        "bs": [32],
        "loss_alpha": [1],
        "loss_beta": [1],
        "sampler": ['rand'],
        "datasource": ['long_captions', 'coco', 'localized_narratives'],
        "epochs": [10],
    }

    final_sweep_params =  makeGrid(final_sweep_baseline) #+ makeGrid(final_sweep) 

    final_sweep = [CLIPAndNegConfig(**p) for p in final_sweep_params]

    log_base_folder = os.path.join(MODEL_PATH, 'logs', f"sweep-{time.time()}")
    os.makedirs(log_base_folder)
    log_folder = f"{log_base_folder}/%j"
    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(
        slurm_partition='learnfair',
        nodes=1,
        timeout_min=60*24,
        tasks_per_node=1,
        gpus_per_node=2,
        cpus_per_task=20,
        slurm_mem='100GB',
        slurm_constraint='volta32gb',
    )
    job_array = []
    existing_sweeps = set()
    for sweep_args in base_only_sweep + final_sweep:
        if get_dir_name(sweep_args) in existing_sweeps:
            continue
        existing_sweeps.add(get_dir_name(sweep_args))
        job = executor.submit(CLIPTrainJob(), sweep_args)
        job_array.append(job)
        print(f"Job {job.job_id} queued")
    
    for job in job_array:
        _ = job.result()
        print(f"Job {job.job_id} done")

if __name__ == '__main__':
    main()
