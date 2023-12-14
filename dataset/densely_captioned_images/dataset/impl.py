#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

from tqdm import tqdm
import random
from collections import defaultdict

import torch
import os
import json

from typing import Dict, List, Any, Callable, Optional, Tuple
from densely_captioned_images.dataset.dense_image import (
    DenseCaptionedImage, get_dci_count, DCIEntry, NegativeEntry, DCINegatives, DCISummaries
)
from densely_captioned_images.dataset.utils import get_clip_processor
from densely_captioned_images.dataset.config import DATASET_BASE
from densely_captioned_images.dataset.spacy_negs import get_spacy_negative


def get_negative_by_strategy(
        negatives: NegativeEntry, 
        negative_source: str = 'swaps', # which LLM prompt to use for negs, swaps|layout|basic|any
        negative_strategy: str = 'rand', # how to select which neg of multiple options, rand|first|hardest
        clip_scores: Optional[Dict[str, float]] = None,
) -> str:
    negative_sources = []
    if negative_source != 'any':
        negative_cands = negatives[negative_source]
        negative_sources = [f"{negative_source}_{i}" for i in range(len(negative_cands))]
    else:
        negative_cands = []
        for neg_type, val in negatives.items():
            negative_cands += val
            negative_sources += [f"{neg_type}_{i}" for i in range(len(val))]
    
    if negative_strategy == 'first':
        return negative_cands[0]
    elif negative_strategy == 'rand':
        return random.choice(negative_cands)
    elif negative_strategy == 'hardest':
        zipped = list(zip(negative_cands, negative_sources))
        zipped.sort(key=lambda x: clip_scores[x[1]])
        return zipped[-1][0]


def get_complete_dataset_with_settings(
        split: str = 'train',
        load_base_image: bool = True,
        load_subcaptions: bool = True,
        count: Optional[int] = None,
) -> List[List[DCIEntry]]:
    assert split in ['train', 'valid', 'test'], f"Bad split {split}"

    with open(os.path.join(DATASET_BASE, 'splits.json')) as jsonf:
        split_metadata = json.load(jsonf)
        sources = split_metadata[split]

    entries_per_image = []
    for source_path in tqdm(sources, desc="Loading Dense Caps:"):
        if count is not None and len(entries_per_image) > count:
            break
        dci = DenseCaptionedImage(source_path)

        entries = []
        if load_base_image:
            entries += dci.get_formatted_complete_description()
        
        if load_subcaptions:
            all_masks = dci.filter_masks_by_size()
            entries += [dci.get_caption_with_subcaptions(m)[0] for m in all_masks]
        entries_per_image.append(entries)
    
    return entries_per_image


def get_summarized_dataset_with_settings(
        split: str = 'train',
        load_base_image: bool = True,
        load_subcaptions: bool = True,
        negative_source: str = 'swaps', # which LLM prompt to use for negs, swaps|layout|basic|any|spacy
        negative_strategy: str = 'rand', # how to select which neg of multiple options, rand|first|hardest
        count: Optional[int] = None,
) -> List[List[DCIEntry]]:
    assert split in ['train', 'valid', 'test'], f"Bad split {split}"
    assert negative_source in ['swaps', 'layout', 'basic', 'any', 'spacy', 'spacy-ant'], f"Bad neg source {negative_source}"
    assert negative_strategy in ['rand', 'first', 'hardest'], f"Bad neg strat {negative_strategy}"

    with open(os.path.join(DATASET_BASE, 'splits.json')) as jsonf:
        split_metadata = json.load(jsonf)
        sources = split_metadata[split]

    entries_per_image = []
    for source_path in tqdm(sources, desc="Loading Dense Caps:"):
        if count is not None and len(entries_per_image) > count:
            break
        try:
            dci = DenseCaptionedImage(source_path)
            summaries = dci.get_summaries()
            if summaries is None:
                continue

            negatives = dci.get_negatives()
            if negatives is None or len(negatives) == 1 and load_subcaptions:
                continue

            entries = []
            if load_base_image:
                entries += dci.get_formatted_complete_description()
            
            if load_subcaptions:
                all_masks = dci.filter_masks_by_size()
                entries += [dci.get_caption_with_subcaptions(m)[0] for m in all_masks]
            
            # Remap to summaries
            for entry in entries:
                if isinstance(summaries[entry['key']], str):
                    entry['caption'] = summaries[entry['key']]
                else:
                    entry['caption'] = summaries[entry['key']][0]
                    entry['captions'] = summaries[entry['key']]
            
            # Include negatives:
            for entry in entries:
                if negative_source == 'spacy' or negative_source == 'spacy-ant' :
                    use_antonyms = negative_source == 'spacy-ant'
                    entry['negative'] = get_spacy_negative(entry['caption'], use_antonyms=use_antonyms)
                    # if 'captions' in entry:
                    #     entry['negatives'] = [
                    #         get_spacy_negative(c) for c in entry['captions']
                    #     ]
                else:
                    negs = negatives[entry['key']]

                    entry['negative'] = get_negative_by_strategy(
                        negs,
                        negative_source, 
                        negative_strategy,
                        dci._data['clip_scores'][entry['key']],
                    )
            entries_per_image.append(entries)
        except Exception:
            import traceback
            traceback.print_exc()
            print(f"Skipping image {source_path} due to loading issue")
    
    return entries_per_image

def get_data_iterator(source, cap=1e30):
    def gen_dci():
        yielded_count = 0
        for i, dci_entry_list in enumerate(source):
            for entry in dci_entry_list:
                yield (i, entry)
                yielded_count += 1
                if yielded_count == cap:
                    return
    return gen_dci



class DenseCaptionedDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, caption_bag_size=3, processor=None, is_test=False):
        self.data_list = data_list
        if caption_bag_size > 0:
            self.data_list = [
                (i, item) for (i, item) in data_list if
                len(item.get('captions', [])) >= caption_bag_size
            ]
            if len(self.data_list) < len(data_list):
                print(f"{len(data_list) - len(self.data_list)} masks skipped for having too few captions for bag size {caption_bag_size}")
        self.processor = get_clip_processor() if processor is None else processor
        self.caption_bag_size = caption_bag_size
        self.is_test = is_test

    def __getitem__(self, idx):
        _, item = self.data_list[idx]
        bag_inputs = None
        bag_negatives = None
        inputs = self.processor(text=[item['caption']], images=[item['image']], return_tensors="pt", padding="max_length")
        negatives = self.processor(text=[item['negative']], images=[item['image']], return_tensors="pt", padding="max_length")
        res = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'negative_input_ids': negatives['input_ids'],
            'negative_attention_mask': negatives['attention_mask'],
            'pixel_values': inputs['pixel_values'],
        }
        if item.get('captions') is not None and self.caption_bag_size > 0:
            if self.is_test:
                use_captions = item['captions'][:self.caption_bag_size]
            else:
                use_captions = random.sample(item['captions'], self.caption_bag_size)
            bag_inputs = self.processor(text=use_captions, images=[item['image']], return_tensors="pt", padding="max_length")
            res['bag_input_ids'] = bag_inputs['input_ids']
            res['bag_attention_mask'] = bag_inputs['attention_mask']
            
        if item.get('negatives') is not None and self.caption_bag_size > 0:
            if self.is_test:
                use_negatives = item['negatives'][:self.caption_bag_size]
            else:
                use_negatives = random.sample(item['negatives'], self.caption_bag_size)
            bag_negatives = self.processor(text=use_negatives, images=[item['image']], return_tensors="pt", padding="max_length")
            res['bag_negative_input_ids'] = bag_negatives['input_ids']
            res['bag_negative_attention_mask'] = bag_negatives['attention_mask']

        return res

    def __len__(self):
        return len(self.data_list)


class DenseCaptionBatchSampler(torch.utils.data.sampler.BatchSampler):
    """
    Batch sampler for DenseCaptionedDataset that tries to fit images
    into the batches randomly, but all captions from an image
    in a row.
    """

    def __init__(self, dataset: DenseCaptionedDataset, batch_size: int):
        self.dataset = dataset
        self.bs = batch_size
        self._max_len = len(dataset) // batch_size

    def _generate_batch(self, dataset: DenseCaptionedDataset):
        all_data = dataset.data_list
        image_to_data_dict = defaultdict(lambda: [])
        for data_id, (image_idx, _) in enumerate(all_data):
            image_to_data_dict[image_idx].append(data_id)
        
        for data_id_array in image_to_data_dict.values():
            random.shuffle(data_id_array)
        
        image_to_data_dict_list = list(image_to_data_dict.items())
        random.shuffle(image_to_data_dict_list)
        
        batches = []
        curr_batch = []
        _, curr_data_list = image_to_data_dict_list.pop(0)
        while len(image_to_data_dict_list) > 0 and len(curr_data_list) > 0:
            if len(curr_batch) == self.bs:
                batches.append(curr_batch)
                curr_batch = []
            if len(curr_batch) < self.bs:
                curr_batch.append(curr_data_list.pop(0))
            if len(curr_data_list) == 0 and len(image_to_data_dict_list) > 0:
                _, curr_data_list = image_to_data_dict_list.pop(0)
        return batches

    def __iter__(self):
        batch_list = self._generate_batch(self.dataset)
        while len(batch_list) > 0:
            yield batch_list.pop(0)

    def __len__(self):
        return self._max_len


def get_clip_ready_ds(
        split: str = 'train',
        load_base_image: bool = True,
        load_subcaptions: bool = True,
        negative_source: str = 'swaps', # which LLM prompt to use for negs, swaps|layout|basic|any|spacy
        negative_strategy: str = 'rand', # how to select which neg of multiple options, rand|first|hardest
        count: int = 1e30,
        caption_bag_size: int = 0, # How many captions should be returned per load, 0 is always first
):
    """
    Load a DenseCaptionedDataset with the provided configuration. Example usage can 
    be found in densely_captioned_images.dataset.scripts.run_clip_dense_cap_eval

    load_base_image: True to include the base image
    load_subcaptions: True to include all submasks for images
    negative_source: which LLM prompt to use for negs, swaps|layout|basic|any, or spacy for spacy negs
    negative_strategy: how to select which neg of multiple options, rand|first|hardest
    count: how many example to load maximum. If larger than the dataset, return all
    caption_bag_size: int = 0, # How many captions should be returned per load, 0 is always first
    """
    train_source = get_summarized_dataset_with_settings(
        split=split, 
        load_base_image=load_base_image, 
        load_subcaptions=load_subcaptions,
        negative_source=negative_source,
        negative_strategy=negative_strategy,
        count=count
    )
    data_list = [n for n in get_data_iterator(train_source, count)()]
    return DenseCaptionedDataset(data_list, caption_bag_size)
