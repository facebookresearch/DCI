#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
from transformers import CLIPProcessor, CLIPModel
from long_captions.dense_image import DenseCaptionedImage, get_key_for, get_dci_count
from long_captions.config import DATASET_COMPLETE_PATH

from tqdm import tqdm
from typing import Tuple

clip_processor = None
clip_model = None
def get_clip() -> Tuple[CLIPModel, CLIPProcessor]:
    global clip_processor, clip_model
    if clip_processor is None:
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    if clip_model is None:
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    return clip_model, clip_processor

def get_clip_scores(dci: DenseCaptionedImage):
    did_nothing = True
    scores = dci._data.get('clip_scores', {})
    summaries = dci.get_summaries()
    negatives = dci.get_negatives()

    mask_exs = [dci.get_caption_with_subcaptions(m)[0] for m in dci.filter_masks_by_size()]
    exs = dci.get_formatted_complete_description() + mask_exs

    clip_model, clip_processor = get_clip()

    for ex in exs:
        key = ex['key']
        if key in scores:
            continue

        summary = summaries[key]
        negs = negatives[key]

        subkeys = ['sum']
        texts = [summary]

        for neg_type, typed_negs in negs.items():
            for idx, typed_neg in enumerate(typed_negs):
                texts.append(typed_neg)
                subkeys.append(f"{neg_type}_{idx}")
        
        inputs = clip_processor(
            text=texts, 
            images=[ex['image']], 
            return_tensors="pt", 
            padding=True
        )
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image.detach().cpu().numpy()[0].tolist()

        scores[key] = {sk: logit for sk, logit in zip(subkeys, logits_per_image)}
        did_nothing = False
    assert not did_nothing
    return scores

def generate_and_write_clip_scores():
    count = get_dci_count()

    for i in tqdm(range(count)):
        key = get_key_for(i)
        source_path = os.path.join(DATASET_COMPLETE_PATH, key)
        if not os.path.exists(source_path):
            continue # not a complete entry
        dci = DenseCaptionedImage(i)
        try:
            clip_scores = get_clip_scores(dci)
        except Exception:
            continue
        with open(source_path) as jsonf:
            base_data = json.load(jsonf)
        
        base_data['clip_scores'] = clip_scores

        with open(source_path, 'w') as jsonf:
            json.dump(base_data, jsonf)

if __name__ == "__main__":
    generate_and_write_clip_scores()