#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

from mephisto.operations.operator import Operator
from mephisto.tools.scripts import (
    task_script,
    build_custom_bundle,
)

from mephisto.abstractions.blueprints.abstract.static_task.static_blueprint import (
    SharedStaticTaskState,
)

from omegaconf import DictConfig
from dataclasses import dataclass, field
import cv2
import base64
import random
from PIL import Image
from mephisto.data_model.qualification import QUAL_NOT_EXIST, QUAL_EXISTS
from mephisto.utils.qualifications import make_qualification_dict
from mephisto.operations.hydra_config import build_default_task_config

from typing import Any, Dict, List, Optional

import os
import json



AREA_CUTOFF = 5000 # Cutoff on minimum area size for masks to show for annotation

NUM_TASKS = 1
LOW=0
HIGH=100000

BASE_SOURCE_PATH = os.path.expanduser("~/diffusion_with_feedback/long_captions/crowdsourcing/annotate/assets/")
BASE_IMAGE_PATH = os.path.join(BASE_SOURCE_PATH, 'images')
BASE_MASK_PATH = os.path.join(BASE_SOURCE_PATH, 'masks')
PILOT_QUALIFICATION = 'long-caps-ready'
ALLOWLIST_QUALIFICATION = 'long-caps-approved'


@dataclass
class LongCapsConfig(build_default_task_config("test")):  # type: ignore
    idx_start: int = field(
        default=LOW,
        metadata={"help": "Index of image to process start"},
    )
    idx_end: int = field(
        default=HIGH,
        metadata={"help": "Index of image to process end"},
    )
    is_pilot: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether to pilot or go live"},
    )

def build_tasks(num_tasks, lo, hi):
    """
    Create a set of tasks you want annotated
    """
    # NOTE These form the init_data for a task
    tasks = []
    all_masks = os.listdir(BASE_MASK_PATH)
    use_masks = all_masks[lo:hi]
    # random.shuffle(use_masks)
    idx = 0
    while idx < len(use_masks) and len(tasks) < num_tasks:
        mask_name = use_masks[idx]
        idx += 1
        img_name = mask_name[:-len('-mask.json')]
        image_path = os.path.join(BASE_IMAGE_PATH, img_name)
        with open(image_path, "rb") as img_file:
            b64_image = base64.b64encode(img_file.read())
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_path = os.path.join(BASE_MASK_PATH, mask_name)
        
        try:
            with open(mask_path, 'r') as mask_file:
                mask_data = json.load(mask_file)
        except Exception as e:
            print("Exception raised!", e)
            continue

        masks_unrolled = {}
        curr_idx = 0
        
        def unroll_mask_into(curr_mask_layer, target):
            nonlocal curr_idx
            curr_mask_layer['idx'] = curr_idx
            curr_idx += 1
            if curr_mask_layer['area'] < AREA_CUTOFF:
                return
            curr_mask_layer['requirements'] = []
            curr_mask_layer['parent'] = -1
            for deeper_mask_layer in sorted(
                list(curr_mask_layer['subgroups'].values()), 
                key=lambda x: x['bounds'][0][0] * 5000 + x['bounds'][0][1],
            ):
                unroll_mask_into(deeper_mask_layer, target)
                deeper_mask_layer['parent'] = curr_mask_layer['idx']
                curr_mask_layer['requirements'].append(deeper_mask_layer['idx'])
            del curr_mask_layer['subgroups']
            target[curr_mask_layer['idx']] = curr_mask_layer
            (top, left), (bottom, right) = curr_mask_layer['bounds']
            assert top < bottom and left < right, "Dims don't make sense!!"
            curr_mask_layer['bounds'] = {'topLeft': {'x': left, 'y': top}, 'bottomRight': {'x': right, 'y': bottom}}
            curr_mask_layer['label'] = ''
            curr_mask_layer['caption'] = ''
            curr_mask_layer['mask_quality'] = 0

        try:
            for entry in mask_data.values():
                unroll_mask_into(entry, masks_unrolled)
        except AssertionError:
            continue

        image_data = {
            "image": "data:image/png;base64," + b64_image.decode('utf-8'), 
            "width": image.shape[1], 
            "height": image.shape[0],
        }

        tasks.append(
            {
                "mask": mask_name,
                "mask_data": masks_unrolled,
                "image_data": image_data,
            }
        )
    print(len(tasks), len(all_masks), len(use_masks))
    return tasks

@task_script(config=LongCapsConfig)
def main(operator: Operator, cfg: DictConfig) -> None:

    shared_state = SharedStaticTaskState(
        static_task_data=build_tasks(cfg.num_tasks, cfg.idx_start, cfg.idx_end),
    )
    
    if cfg.is_pilot is True:
        shared_state.qualifications = [
            make_qualification_dict(
                PILOT_QUALIFICATION,
                QUAL_EXISTS,
                None,
            ),
            make_qualification_dict(
                ALLOWLIST_QUALIFICATION,
                QUAL_NOT_EXIST,
                None,
            ),
        ]
    elif cfg.is_pilot is False:
        shared_state.qualifications = [
            make_qualification_dict(
                ALLOWLIST_QUALIFICATION,
                QUAL_EXISTS,
                None,
            ),
        ]


    task_dir = cfg.task_dir

    build_custom_bundle(
        task_dir,
        force_rebuild=cfg.mephisto.task.force_rebuild,
    )

    operator.launch_task_run(cfg.mephisto, shared_state)
    operator.wait_for_runs_then_shutdown(skip_input=True, log_rate=30)


if __name__ == "__main__":
    main()
