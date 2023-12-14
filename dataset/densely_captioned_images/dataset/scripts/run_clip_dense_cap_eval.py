#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import torch
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from densely_captioned_images.dataset.impl import get_clip_ready_ds, DenseCaptionedDataset
from densely_captioned_images.dataset.loss import get_pooled_groups, get_pooled_diag
from torch.utils.data import DataLoader
from typing import Optional, List


def run_dense_cap_test_on_model(
        model: CLIPModel, 
        dataset: DenseCaptionedDataset,
):
    clip_correct_tot = 0
    neg_correct_tot = 0
    exs = 0
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    for inputs in tqdm(loader):
        exs += inputs['input_ids'].shape[0]
        
        if 'bag_input_ids' in inputs:
            bs, n, t = inputs['bag_input_ids'].shape
            unstacked_inputs = inputs['bag_input_ids'].reshape(bs*n, t)
            unstacked_attention = inputs['bag_attention_mask'].reshape(bs*n, t)
            pos_inputs = {
                'input_ids': unstacked_inputs.to(model.device),
                'attention_mask': unstacked_attention.to(model.device),
                'pixel_values': torch.squeeze(inputs['pixel_values'], axis=1).to(model.device),
            }
        else:
            pos_inputs = {
                'input_ids': torch.squeeze(inputs['input_ids'], axis=1).to(model.device),
                'attention_mask': torch.squeeze(inputs['attention_mask'], axis=1).to(model.device),
                'pixel_values': torch.squeeze(inputs['pixel_values'], axis=1).to(model.device),
            }
        
        if 'bag_negative_input_ids' in inputs:
            bs, n, t = inputs['bag_negative_input_ids'].shape
            unstacked_inputs = inputs['bag_negative_input_ids'].reshape(bs*n, t)
            unstacked_attention = inputs['bag_negative_attention_mask'].reshape(bs*n, t)
            neg_inputs = {
                'input_ids': unstacked_inputs.to(model.device),
                'attention_mask': unstacked_attention.to(model.device),
                'pixel_values': torch.squeeze(inputs['pixel_values'], axis=1).to(model.device),
            }
        else:
            neg_inputs = {
                'input_ids': torch.squeeze(inputs['negative_input_ids'], axis=1).to(model.device),
                'attention_mask': torch.squeeze(inputs['negative_attention_mask'], axis=1).to(model.device),
                'pixel_values': torch.squeeze(inputs['pixel_values'], axis=1).to(model.device),
            }

        # Get clip loss from positives
        outputs = model(**pos_inputs)
        clip_logits = outputs.logits_per_image

        # Get max of negatives, min of positives
        similarity = get_pooled_groups(clip_logits, pool_type='max')
        min_sim_diag = get_pooled_diag(clip_logits, pool_type='min')
        similarity[range(len(min_sim_diag)), range(len(min_sim_diag))] = min_sim_diag
        maxes = torch.max(similarity, dim=1)
        clip_correct_tot += sum([int(i==m) for i, m in enumerate(maxes.indices)])

        # Get negatives loss
        neg_outputs = model(**neg_inputs)
        neg_logits = neg_outputs.logits_per_image
        pos_diag = get_pooled_diag(clip_logits, pool_type='min')
        neg_diag = get_pooled_diag(neg_logits, pool_type='max')
        neg_correct_tot += sum(pos_diag > neg_diag).item()

    return clip_correct_tot / exs, neg_correct_tot / exs


def run_dense_cap_on_model(
        model: CLIPModel, 
        processor: CLIPProcessor, 
        run_subtests: Optional[List[str]] = None
):
    """
    Run the (CLIP-ready) Dense Cap test on the provided model.

    Use run_subtests to specify specific keys, otherwise all of the
    standard subtests will be run.
    """
    subtests = {
        "all_swaps": {
            "load_base_image": True,
            "load_subcaptions": True,
            "negative_source":  'swaps',
            "negative_strategy": "first",
            "caption_bag_size": 0,
        },
        "all_swaps_pick5": {
            "load_base_image": True,
            "load_subcaptions": True,
            "negative_source":  'swaps',
            "negative_strategy": "first",
            "caption_bag_size": 5,
        },
        "base_swaps": {
            "load_base_image": True,
            "load_subcaptions": False,
            "negative_source":  'swaps',
            "negative_strategy": "first",
            "caption_bag_size": 0,
        },
        "all_hardest": {
            "load_base_image": True,
            "load_subcaptions": True,
            "negative_source":  'any',
            "negative_strategy": "hardest",
            "caption_bag_size": 0,
        },
    }
    if run_subtests is None:
        run_subtests = list(subtests.keys())
    
    for key, args in subtests.items():
        if key not in run_subtests:
            continue
        dataset = get_clip_ready_ds(split='test', **args)
        dataset.processor = processor
        with torch.no_grad():
            clip_correct_prop, neg_correct_prop = run_dense_cap_test_on_model(model, dataset)
            print(f"Test: {key}. CLIP Correct: {clip_correct_prop} neg_correct {neg_correct_prop}")


def run_dense_cap_on_lora(lora_weight_path: str):
    """
    Run the (CLIP-ready) Dense Cap test on the provided model.

    Assumes that the lora weights are to be applied to a standard CLIP vit/b32
    """
    from peft import PeftModel

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    base_clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    loaded = PeftModel.from_pretrained(base_clip_model, lora_weight_path)
    loaded = loaded.merge_and_unload()
    run_dense_cap_on_model(loaded, processor)


if __name__ == '__main__':
    from transformers import CLIPProcessor, CLIPModel
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    run_dense_cap_on_model(clip_model, clip_processor)
