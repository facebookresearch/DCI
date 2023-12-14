#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import submitit
import os
from dataclasses import dataclass, field
import hydra
from omegaconf import MISSING
from peft import PeftModel
from hydra.core.config_store import ConfigStoreWithProvider  # type: ignore

from transformers import CLIPProcessor, CLIPModel

from densely_captioned_images.dataset.utils import print_trainable_parameters
from densely_captioned_images.repro.eval.run_aro import run_aro_evals
from densely_captioned_images.repro.eval.run_vlc import run_vlc_on_model
from densely_captioned_images.repro.eval.run_winoground import run_winoground
from densely_captioned_images.repro.eval.run_elevater import run_elevater_on
from densely_captioned_images.dataset.scripts.run_clip_dense_cap_eval import run_dense_cap_on_model


from typing import Any, Optional

# CONFIGURATION

HYDRA_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'hydra_configs')

@dataclass
class CLIPEvalConfig():
    run_aro: bool = field(
        default=False,
        metadata={"help": "Whether to run ARO evals"},
    )
    run_vlc: bool = field(
        default=False,
        metadata={"help": "Whether to run VLC evals"},
    )
    run_dense_cap: bool = field(
        default=False,
        metadata={"help": "Whether to run dense captions evals"},
    )
    run_winoground: bool = field(
        default=False,
        metadata={"help": "Whether to run winoground evals"},
    )
    run_elevater: int = field(
        default=-2,
        metadata={"help": "What elevator shot option to do, if any"},
    )
    elevater_dataset: Optional[int] = field(
        default=None,
        metadata={"help": "What elevator dataset to run, if just one"},
    )
    lora_weight_location: str = field(
        default=MISSING,
        metadata={"help": "The location for the model"},
    )
    model_name: str = field(
        default=MISSING,
        metadata={"help": "The name for the model"},
    )

config = ConfigStoreWithProvider("long_captions")
config.store(name="scriptconfig", node=CLIPEvalConfig)


class CLIPEvalJob:
    def __call__(self, *args, **kwargs):
        run_eval_clip(args[0])

    def checkpoint(self, *args: Any, **kwargs: Any) -> submitit.helpers.DelayedSubmission:
        return submitit.helpers.DelayedSubmission(self, *args, **kwargs)

@hydra.main(
    config_path=HYDRA_CONFIG_PATH, config_name="scriptconfig", version_base="1.2"
)
def run_eval_clip(cfg: CLIPEvalConfig):
    print('[-] Loading base CLIP')
    base_clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print_trainable_parameters(base_clip_model)

    print(f"[-] Loading from LoRA weights: {cfg.lora_weight_location}")
    loaded = PeftModel.from_pretrained(base_clip_model, cfg.lora_weight_location)
    print_trainable_parameters(loaded)
    loaded = loaded.merge_and_unload().to('cuda')

    if cfg.run_aro:
        print("[-] Running ARO")
        run_aro_evals(loaded, processor)
    if cfg.run_vlc:
        print("[-] Running VLC")
        run_vlc_on_model(loaded, processor, cfg.model_name)
    if cfg.run_winoground:
        print("[-] Running Winoground")
        run_winoground(loaded, processor)
    if cfg.run_elevater != -2:
        print(f"[-] Running Elevater shot {cfg.run_elevater}")
        run_elevater_on(
            loaded, 
            cfg.model_name, 
            run_full=True, 
            do_finetune=True, 
            shot_options=[cfg.run_elevater], 
            dataset_option=cfg.elevater_dataset,
        )
    if cfg.run_dense_cap:
        print("[-] Running Dense Cap")
        run_dense_cap_on_model(loaded, processor)


if __name__ == '__main__':
    run_eval_clip()
