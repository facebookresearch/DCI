#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import torch
from transformers import Trainer
import datasets
from transformers.trainer_utils import seed_worker

from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from densely_captioned_images.dataset.loss import clip_loss, negatives_loss
from densely_captioned_images.dataset.impl import DenseCaptionBatchSampler
from typing import Optional

# Available gpus is important for tiling
available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
total_gpus = len(available_gpus)

# Constants for clip loss and negatives loss weight in final loss, overridable
ALPHA = 1
BETA = 1


def compute_metrics(eval_pred):
    """Our model reports clip loss and negatives loss, which we return"""
    global last_eval_pred
    with torch.no_grad():
        c_loss = eval_pred.predictions[6].mean()
        n_loss = eval_pred.predictions[7].mean()
        return {'clip-loss': c_loss, 'neg-loss': n_loss}


class ClipAndNegTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self._la: float = kwargs.pop('loss_alpha', ALPHA)
        self._lb: float = kwargs.pop('loss_beta', BETA)
        self._sampler_choice: str = kwargs.pop('sampler', 'rand')
        self._loss_pool_type: str = kwargs.pop('loss_pool_type', 'avg')
        super().__init__(*args, **kwargs)

    def _a(self):
        """
        Opportunity for loss scheduling
        """
        return self._la
    
    def _b(self):
        """
        Opportunity for loss scheduling
        """
        return self._lb

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            sampler = self._get_train_sampler()
            if isinstance(sampler, torch.utils.data.BatchSampler):
                dataloader_params["batch_sampler"] = sampler
                del dataloader_params['batch_size']
            else:
                dataloader_params["sampler"] = sampler
                dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
        
        print(dataloader_params)

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self._sampler_choice == 'rand':
            return RandomSampler(self.train_dataset)
        elif self._sampler_choice == 'seq':
            return SequentialSampler(self.train_dataset)
        elif self._sampler_choice == 'rand_batch':
            return DenseCaptionBatchSampler(self.train_dataset, self._train_batch_size)
        else:
            raise NotImplementedError

    def compute_loss(self, model, inputs, return_outputs=False):
        # Tile text across gpu count to account for dataparallel
        if 'bag_input_ids' in inputs:
            bs, n, t = inputs['bag_input_ids'].shape
            unstacked_inputs = inputs['bag_input_ids'].reshape(bs*n, t)
            unstacked_attention = inputs['bag_attention_mask'].reshape(bs*n, t)
            pos_inputs = {
                'input_ids': unstacked_inputs.repeat(total_gpus, 1),
                'attention_mask': unstacked_attention.repeat(total_gpus, 1),
                'pixel_values': torch.squeeze(inputs['pixel_values'], axis=1),
            }
        else:
            pos_inputs = {
                'input_ids': torch.squeeze(inputs['input_ids'], axis=1).repeat(total_gpus, 1),
                'attention_mask': torch.squeeze(inputs['attention_mask'], axis=1).repeat(total_gpus, 1),
                'pixel_values': torch.squeeze(inputs['pixel_values'], axis=1),
            }
        

        # Get clip loss from positives
        outputs = model(**pos_inputs)
        logits = outputs.logits_per_image
        c_loss = clip_loss(logits, pool_type=self._loss_pool_type)
        

        if self._b() == 0 and not return_outputs:
            return self._a() * c_loss
        
        if 'bag_negative_input_ids' in inputs:
            bs, n, t = inputs['bag_negative_input_ids'].shape
            unstacked_inputs = inputs['bag_negative_input_ids'].reshape(bs*n, t)
            unstacked_attention = inputs['bag_negative_attention_mask'].reshape(bs*n, t)
            neg_inputs = {
                'input_ids': unstacked_inputs.repeat(total_gpus, 1),
                'attention_mask': unstacked_attention.repeat(total_gpus, 1),
                'pixel_values': torch.squeeze(inputs['pixel_values'], axis=1),
            }
        else:
            neg_inputs = {
                'input_ids': torch.squeeze(inputs['negative_input_ids'], axis=1).repeat(total_gpus, 1),
                'attention_mask': torch.squeeze(inputs['negative_attention_mask'], axis=1).repeat(total_gpus, 1),
                'pixel_values': torch.squeeze(inputs['pixel_values'], axis=1),
            }

        # Get negatives loss
        neg_outputs = model(**neg_inputs)
        n_loss = negatives_loss(logits, neg_outputs.logits_per_image, pool_type=self._loss_pool_type)

        # Get joint loss, but report both
        loss = self._a() * c_loss + self._b() * n_loss
        outputs['c_loss'] = c_loss
        outputs['n_loss'] = n_loss
        return (loss, outputs) if return_outputs else loss
