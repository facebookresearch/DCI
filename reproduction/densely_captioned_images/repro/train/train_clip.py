#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import submitit
import os
from dataclasses import dataclass, field
import hydra
import math
from hydra.core.config_store import ConfigStoreWithProvider  # type: ignore

from peft import get_peft_model, LoraConfig
from transformers import CLIPProcessor, CLIPModel
from transformers import TrainingArguments

from densely_captioned_images.repro.train.trainer import compute_metrics, ClipAndNegTrainer
from densely_captioned_images.dataset.utils import print_trainable_parameters
from densely_captioned_images.dataset.impl import get_clip_ready_ds
from densely_captioned_images.repro.train.coco_wrap import COCODataset, get_dataset_source as get_coco_dataset_source
from densely_captioned_images.repro.train.localized_narratives_wrap import COCOLocalizedNarrativesDataset, get_dataset_source as get_loc_nar_dataset_source
from densely_captioned_images.repro.config import MODEL_PATH

from typing import Any

# CONFIGURATION

HYDRA_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'hydra_configs')

@dataclass
class CLIPAndNegConfig():
    lora_r: int = field(
        default=32,
        metadata={"help": "Lora r value, default 32"},
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "Lora alpha value, default 32"},
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "Lora dropout value, default 0.1"},
    )
    use_base_images: bool = field(
        default=True,
        metadata={"help": "Whether to include full image entries in data"},
    )
    use_subcaptions: bool = field(
        default=True,
        metadata={"help": "Whether to include submask entries in data"},
    )
    caption_negative_source: str = field(
        default='swaps',
        metadata={"help": "Source for negative prompt, basic|layout|swaps|any|spacy"},
    )
    caption_negative_strategy: str = field(
        default='rand',
        metadata={"help": "How to select negative option, rand|first|hardest"},
    )
    train_count: int = field(
        default=7800,
        metadata={"help": "How many images to use in train"},
    )
    valid_count: int = field(
        default=100,
        metadata={"help": "How many images to use in valid"},
    )
    lr: float = field(
        default=5e-4,
        metadata={"help": "Learning rate for training"},
    )
    bs: int = field(
        default=16,
        metadata={"help": "Batch size for training"},
    )
    loss_alpha: float = field(
        default=1,
        metadata={"help": "CLIP loss scaling factor"},
    )
    loss_beta: float = field(
        default=1,
        metadata={"help": "Negatives loss scaling factor"},
    )
    caption_selection: str = field(
        default='first',
        metadata={"help": "How to select captions, [first|pick<n>]"},
    )
    sampler: str = field(
        default='rand',
        metadata={"help": "How to sample batches: [rand|seq]"},
    )
    loss_pool_type: str = field(
        default='avg',
        metadata={"help": "How to pool caption bags (if present): [avg|max|min|hardest]"},
    )
    datasource: str = field(
        default="long_captions",
        metadata={"help": "What to train on. Options are long_captions, localized_narrative, or COCO"}
    )
    epochs: int = field(
        default=10,
        metadata={"help": "How many epochs to train for"},
    )

config = ConfigStoreWithProvider("long_captions")
config.store(name="scriptconfig", node=CLIPAndNegConfig)


def get_dir_name(cfg):
    base = ""
    if cfg.datasource == 'localized_narratives':
        base = "LOCNAR-BASELINE-"
    if cfg.datasource == 'coco':
        base = "COCO-BASELINE-"
    base += (
        f"lora-{cfg.lora_r}-{cfg.lora_alpha}-{cfg.lora_dropout}-"
        f"dci-{cfg.caption_negative_source}:{cfg.caption_negative_strategy}-"
        f"bs-{cfg.bs}-lr-{cfg.lr}-cl-{cfg.loss_alpha}-nl-{cfg.loss_beta}-"
        f"captions-{cfg.caption_selection}-sampling-{cfg.sampler}"
    )
    if cfg.caption_selection.startswith('pick') and cfg.caption_selection != 'pick1':
        base += f"-loss_pooling-{cfg.loss_pool_type}"
    if cfg.use_subcaptions is False:
        base += f"-base-count-{cfg.train_count}"
    return base

class CLIPTrainJob:
    def __call__(self, *args, **kwargs):
        run_train_clip(args[0])

    def checkpoint(self, *args: Any, **kwargs: Any) -> submitit.helpers.DelayedSubmission:
        return submitit.helpers.DelayedSubmission(self, *args, **kwargs)

@hydra.main(
    config_path=HYDRA_CONFIG_PATH, config_name="scriptconfig", version_base="1.2"
)
def run_train_clip(cfg: CLIPAndNegConfig):
    print('[-] Loading base CLIP')
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print_trainable_parameters(model)

    print("[-] Creating LoRA weights")
    l_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=[
            "k_proj", 
            "v_proj", 
            "q_proj", 
            "out_proj", 
            "fc1",
            "fc2",
            "visual_projection", 
            "text_projection"
        ],
        lora_dropout=cfg.lora_dropout,
        bias="lora_only",
    )
    lora_model = get_peft_model(model, l_config)
    print_trainable_parameters(lora_model)

    use_antonyms = cfg.caption_negative_source=='spacy-ant'
    caption_count = 0 # use first caption
    if cfg.caption_selection.startswith('pick'):
        caption_count = int(cfg.caption_selection[4:])
    if cfg.datasource == 'long_captions':
        print("[-] Loading long_captions dataset source")
        train_ds = get_clip_ready_ds(
            split='train',
            load_base_image=cfg.use_base_images,
            load_subcaptions=cfg.use_subcaptions,
            negative_source=cfg.caption_negative_source,
            negative_strategy=cfg.caption_negative_strategy,
            count=cfg.train_count, 
            caption_bag_size=caption_count,
        )
        eval_ds = get_clip_ready_ds(
            split='valid',
            load_base_image=cfg.use_base_images,
            load_subcaptions=cfg.use_subcaptions,
            negative_source=cfg.caption_negative_source,
            negative_strategy=cfg.caption_negative_strategy,
            count=cfg.valid_count, 
            caption_bag_size=caption_count,
        )

    elif cfg.datasource == 'localized_narratives':
        print("[-] Loading localized_narratives dataset source")
        train_source = get_loc_nar_dataset_source('train', cfg.train_count, use_antonyms=use_antonyms)
        valid_source = get_loc_nar_dataset_source('valid', cfg.valid_count, use_antonyms=True)

        train_ds = COCOLocalizedNarrativesDataset(train_source)
        eval_ds = COCOLocalizedNarrativesDataset(valid_source)
    elif cfg.datasource == 'coco':
        train_source = get_coco_dataset_source('train', cfg.train_count, use_antonyms=use_antonyms)
        valid_source = get_coco_dataset_source('valid', cfg.valid_count, use_antonyms=True)

        train_ds = COCODataset(train_source, caption_count)
        eval_ds = COCODataset(valid_source, 0)
    else:
        raise NotImplementedError(f'Datasource {cfg.datasource} not implemented')

    print(f"[-] Running training job for {get_dir_name(cfg)}")

    output_dir = os.path.join(MODEL_PATH, get_dir_name(cfg))
    epochs = cfg.epochs
    training_args = TrainingArguments(
        report_to='tensorboard',
        output_dir=output_dir,
        learning_rate=cfg.lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=cfg.bs,
        per_device_eval_batch_size=cfg.bs,
        save_total_limit=50,
        optim="adamw_torch",
        evaluation_strategy='steps',
        eval_steps=1/(3 * epochs),
        save_strategy='steps',
        save_steps=1/(3 * epochs),
        logging_steps=50,
        logging_dir=os.path.join(MODEL_PATH, 'tf_logs', get_dir_name(cfg)),
        remove_unused_columns=False,
        label_names=["input_ids", "attention_mask", "pixel_values"],
        dataloader_drop_last=True,
        do_train=True,
        do_eval=True,
        include_inputs_for_metrics=True,
        load_best_model_at_end=True,
    )

    trainer = ClipAndNegTrainer(
        model=lora_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        loss_alpha=cfg.loss_alpha,
        loss_beta=cfg.loss_beta,
        sampler=cfg.sampler,
        loss_pool_type=cfg.loss_pool_type,
    )

    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    from densely_captioned_images.dataset.scripts.run_clip_dense_cap_eval import run_dense_cap_test
    run_dense_cap_test(lora_model, processor)

if __name__ == '__main__':
    run_train_clip()
