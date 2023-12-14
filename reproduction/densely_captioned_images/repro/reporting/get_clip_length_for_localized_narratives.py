#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

from densely_captioned_images.repro.config import LOCALIZED_NARRATIVES_DATAPATH
from densely_captioned_images.dataset.utils import get_clip_token_length
from densely_captioned_images.repro.eval.localized_narratives.localized_narratives import DataLoader as LocNarDataLoader


def get_clip_token_lengths_for_localized_narratives_split(split='train'):
    if split == 'train':
        ln_split = 'coco_train'
    elif split == 'valid':
        ln_split = 'coco_val'
    else:
        raise NotImplementedError('Must pull from train or valid, no test')

    loader = LocNarDataLoader(LOCALIZED_NARRATIVES_DATAPATH)
    caption_count = 0
    token_count = 0
    full_caption_count = 0
    full_token_count = 0
    for n in loader.load_annotations(ln_split):
        toks = get_clip_token_length(n.caption)
        full_caption_count += 1
        full_token_count += toks
        if toks > 76:
            continue
        caption_count += 1
        token_count += toks
    return token_count, caption_count, full_token_count, full_caption_count
    

def get_clip_token_lengths_for_localized_narratives():
    toks, caps, full_toks, full_caps = get_clip_token_lengths_for_localized_narratives_split('train')
    print(f"Train: {toks} / {caps} with {full_toks} / {full_caps} not skipped")
    toks, caps, full_toks, full_caps = get_clip_token_lengths_for_localized_narratives_split('valid')
    print(f"Valid: {toks} / {caps} with {full_toks} / {full_caps} not skipped")


if __name__ == '__main__':
    get_clip_token_lengths_for_localized_narratives()