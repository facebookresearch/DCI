#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import json
from densely_captioned_images.dataset.utils import get_clip_token_length
from densely_captioned_images.repro.config import COCO_TRAIN2017_ANNOTATION_PATH, COCO_VALID2017_ANNOTATION_PATH

def get_clip_token_lengths_for_coco_split(split='train'):
    if split == 'train':
        annotation_dir = COCO_TRAIN2017_ANNOTATION_PATH
    elif split == 'valid':
        annotation_dir = COCO_VALID2017_ANNOTATION_PATH
    else:
        raise NotImplementedError('Must pull from train or valid, no test')

    with open(annotation_dir) as coco_fp:
        coco_annotations = json.load(coco_fp)
    
    caption_count = 0
    token_count = 0
    for annotation in coco_annotations['annotations']:
        caption = annotation['caption']
        caption_count += 1
        token_count += get_clip_token_length(caption)
    return token_count, caption_count, (token_count / caption_count), len(coco_annotations['images'])
    
def get_clip_token_lengths_for_coco():
    toks, caps, prop, imgs = get_clip_token_lengths_for_coco_split('train')
    print(f"Train: {toks} / {caps} = {prop} for {imgs} images")
    toks, caps, prop, imgs = get_clip_token_lengths_for_coco_split('valid')
    print(f"Valid: {toks} / {caps} = {prop} for {imgs} images")

if __name__ == '__main__':
    get_clip_token_lengths_for_coco()