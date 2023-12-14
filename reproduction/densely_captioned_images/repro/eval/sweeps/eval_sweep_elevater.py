#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

from densely_captioned_images.repro.eval.run_full_evals import CLIPEvalConfig, CLIPEvalJob
from densely_captioned_images.repro.config import MODEL_PATH, DENSE_CAPS_DIR
import submitit
import os

DO_FULL = False # Whether to do full-shot

def main():
    with open(os.path.join(DENSE_CAPS_DIR, 'elevater_models.txt')) as f:
        MODEL_FILES = [
            os.path.join(MODEL_PATH, m.strip())
            for m in f.readlines()
        ]

    print("Launching elevater evals for: ", len(MODEL_FILES))

    eval_sweep = []
    if not DO_FULL:
        for shots in [0, 5, 20, 50]:
            eval_sweep += [
                CLIPEvalConfig(
                    run_aro=False,
                    run_vlc=False,
                    run_dense_cap=False,
                    run_winoground=False,
                    run_elevater=shots,
                    lora_weight_location=model_path,
                    model_name=model_path.split('/')[-2]
                ) for model_path in MODEL_FILES
            ]
    else:
        for dataset in range(20):
            eval_sweep += [
                CLIPEvalConfig(
                    run_aro=False,
                    run_vlc=False,
                    run_dense_cap=False,
                    run_winoground=False,
                    run_elevater=-1,
                    elevater_dataset=dataset,
                    lora_weight_location=model_path,
                    model_name=model_path.split('/')[-2]
                ) for model_path in MODEL_FILES
            ]

    log_base_folder = os.path.join(MODEL_PATH, 'logs', f"sweep-elevater-eval")
    os.makedirs(log_base_folder, exist_ok=True)
    log_folder = f"{log_base_folder}/%j"
    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(
        slurm_partition='learnfair',
        nodes=1,
        timeout_min=60*24*3,
        tasks_per_node=1,
        gpus_per_node=1,
        cpus_per_task=10,
        slurm_mem='20GB',
    )
    job_array = []
    for sweep_args in eval_sweep:
        job = executor.submit(CLIPEvalJob(), sweep_args)
        job_array.append(job)
        print(f"Job {job.job_id} queued")
    
    for job in job_array:
        _ = job.result()
        print(f"Job {job.job_id} done")

if __name__ == '__main__':
    main()
