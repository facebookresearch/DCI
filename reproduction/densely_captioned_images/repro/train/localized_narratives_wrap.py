#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import torch
import os
from tqdm import tqdm


from typing import Dict, List, Any, Callable, Optional, Tuple

from densely_captioned_images.repro.eval.localized_narratives.localized_narratives import DataLoader as LocNarDataLoader

from PIL import Image
from densely_captioned_images.dataset.utils import get_clip_processor, get_clip_token_length
from densely_captioned_images.repro.config import COCO_TRAIN2017_DATAPATH, COCO_VALID2017_DATAPATH, LOCALIZED_NARRATIVES_DATAPATH
from densely_captioned_images.dataset.spacy_negs import get_spacy_negative


def get_dataset_source(split='train', count=1e10, use_antonyms=False):
    if split == 'train':
        source_dir = COCO_TRAIN2017_DATAPATH
        ln_split = 'coco_train'
    elif split == 'valid':
        source_dir = COCO_VALID2017_DATAPATH
        ln_split = 'coco_val'
    else:
        raise NotImplementedError('Must pull from train or valid, no test')

    loader = LocNarDataLoader(LOCALIZED_NARRATIVES_DATAPATH)

    res = []
    count_so_far = 0
    skipped = 0
    for n in tqdm(loader.load_annotations(ln_split)):
        toks = get_clip_token_length(n.caption)
        if toks > 75:
            skipped += 1
            continue
        image_path = os.path.join(source_dir, f"{n.image_id.zfill(12)}.jpg")
        res.append({
            "image_path": image_path,
            "caption": n.caption,
            "negative": get_spacy_negative(n.caption, use_antonyms=use_antonyms),
        })
        count_so_far += 1
        if count_so_far == count:
            break

    return res


class COCOLocalizedNarrativesDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_source):
        self.data_list = dataset_source
        self.processor = get_clip_processor()

    def __getitem__(self, idx):
        item = self.data_list[idx]
        image = Image.open(item['image_path'])
        inputs = self.processor(text=[item['caption']], images=[image], return_tensors="pt", padding="max_length")
        negatives = self.processor(text=[item['negative']], images=[image], return_tensors="pt", padding="max_length")
        res = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'negative_input_ids': negatives['input_ids'],
            'negative_attention_mask': negatives['attention_mask'],
            'pixel_values': inputs['pixel_values'],
        }

        return res

    def __len__(self):
        return len(self.data_list)
