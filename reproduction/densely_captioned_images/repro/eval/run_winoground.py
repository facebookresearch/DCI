#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

from datasets import load_dataset
from tqdm import tqdm
import torch


def run_winoground(model, processor):
    winoground = load_dataset("facebook/winoground")['test'] # Note, need to be logged into HF Hub
    winoground_clip_scores = []
    with torch.no_grad():
        for example in tqdm(winoground):
            # Note that some images in winoground are RGBA and some are RGB. Need to convert all to RGB with .convert('RGB')
            # Note that we could run this example through CLIP as a batch, but I want to drive the point home that we get four independent image-caption scores for each example
            input_c0_i0 = processor(text=[example["caption_0"]], images=[example["image_0"].convert("RGB")], return_tensors="pt")
            input_c1_i0 = processor(text=[example["caption_1"]], images=[example["image_0"].convert("RGB")], return_tensors="pt")
            input_c0_i1 = processor(text=[example["caption_0"]], images=[example["image_1"].convert("RGB")], return_tensors="pt")
            input_c1_i1 = processor(text=[example["caption_1"]], images=[example["image_1"].convert("RGB")], return_tensors="pt")
            output_c0_i0 = model(**input_c0_i0.to(model.device))
            output_c1_i0 = model(**input_c1_i0.to(model.device))
            output_c0_i1 = model(**input_c0_i1.to(model.device))
            output_c1_i1 = model(**input_c1_i1.to(model.device))
            clip_score_c0_i0 = output_c0_i0.logits_per_image.item()
            clip_score_c1_i0 = output_c1_i0.logits_per_image.item()
            clip_score_c0_i1 = output_c0_i1.logits_per_image.item()
            clip_score_c1_i1 = output_c1_i1.logits_per_image.item()
            winoground_clip_scores.append({"id" : example["id"], "c0_i0": clip_score_c0_i0, "c0_i1": clip_score_c0_i1, "c1_i0": clip_score_c1_i0, "c1_i1": clip_score_c1_i1})

    def text_correct(result):
        return result["c0_i0"] > result["c1_i0"] and result["c1_i1"] > result["c0_i1"]

    def image_correct(result):
        return result["c0_i0"] > result["c0_i1"] and result["c1_i1"] > result["c1_i0"]

    def group_correct(result):
        return image_correct(result) and text_correct(result)

    text_correct_count = 0
    image_correct_count = 0
    group_correct_count = 0
    for result in winoground_clip_scores:
        text_correct_count += 1 if text_correct(result) else 0
        image_correct_count += 1 if image_correct(result) else 0
        group_correct_count += 1 if group_correct(result) else 0

    denominator = len(winoground_clip_scores)
    print("text score:", text_correct_count/denominator)
    print("image score:", image_correct_count/denominator)
    print("group score:", group_correct_count/denominator)

    return {
        'text_score': text_correct_count/denominator, 
        'image_score': image_correct_count/denominator,
        'group_score': group_correct_count/denominator,
    }

if __name__ == '__main__':
    from transformers import CLIPProcessor, CLIPModel
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    run_winoground(clip_model, clip_processor)