#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import torch
import os
import json
import random
from tqdm import tqdm


from typing import Dict, List, Any, Callable, Optional, Tuple

from PIL import Image
from densely_captioned_images.dataset.utils import get_clip_processor, get_clip_token_length
from densely_captioned_images.repro.config import COCO_TRAIN2017_DATAPATH, COCO_VALID2017_DATAPATH, COCO_TRAIN2017_ANNOTATION_PATH, COCO_VALID2017_ANNOTATION_PATH
from densely_captioned_images.dataset.spacy_negs import get_spacy_negative


def get_dataset_source(split='train', count=1e10, use_antonyms=False):
    if split == 'train':
        source_dir = COCO_TRAIN2017_DATAPATH
        annotation_dir = COCO_TRAIN2017_ANNOTATION_PATH
    elif split == 'valid':
        source_dir = COCO_VALID2017_DATAPATH
        annotation_dir = COCO_VALID2017_ANNOTATION_PATH
    else:
        raise NotImplementedError('Must pull from train or valid, no test')

    with open(annotation_dir) as coco_fp:
        coco_annotations = json.load(coco_fp)
    
    coco_by_img_id = {v['id']: v for v in coco_annotations['images']}
    for v in coco_by_img_id.values():
        v['caption'] = []
    for captions in coco_annotations['annotations']:
        coco_by_img_id[captions['image_id']]['caption'].append(captions['caption'])
    
    res = []
    count_so_far = 0
    skipped = 0
    for n in tqdm(coco_by_img_id.values()):
        all_good = True
        for caption in n['caption']:
            toks = get_clip_token_length(caption)
            if toks > 75:
                all_good = False
                break
        if not all_good:
            skipped += 1
            continue
        image_path = os.path.join(source_dir, n['file_name'])
        res.append({
            "image_path": image_path,
            "caption": n['caption'][0],
            "captions": n['caption'],
            "negative": get_spacy_negative(n['caption'][0], use_antonyms=use_antonyms),
        })
        count_so_far += 1
        if count_so_far == count:
            break

    return res


class COCODataset(torch.utils.data.Dataset):
    def __init__(self, dataset_source, caption_bag_size=0):
        self.data_list = dataset_source
        self.processor = get_clip_processor()
        assert caption_bag_size <= 5, "Max 5 captions per image"
        self.caption_bag_size = caption_bag_size

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
        if item.get('captions') is not None and self.caption_bag_size > 0:
            use_captions = random.sample(item['captions'], self.caption_bag_size)
            bag_inputs = self.processor(text=use_captions, images=[image], return_tensors="pt", padding="max_length")
            res['bag_input_ids'] = bag_inputs['input_ids']
            res['bag_attention_mask'] = bag_inputs['attention_mask']

        return res

    def __len__(self):
        return len(self.data_list)
