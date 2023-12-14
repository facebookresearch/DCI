#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

from densely_captioned_images.repro.eval.ARO.dataset_zoo import VG_Relation, VG_Attribution, COCO_Order, Flickr30k_Order
from densely_captioned_images.repro.eval.clip_aro_wrap import AROtoHFCLIPWrap
from densely_captioned_images.repro.config import ARO_DIR, COCO_DIR
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import DataLoader
import pandas as pd

import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def run_aro_evals(model: CLIPModel, processor: CLIPProcessor):
    # Load data
    vgr_dataset = VG_Relation(image_preprocess=processor.image_processor, download=True, root_dir=ARO_DIR)
    vga_dataset = VG_Attribution(image_preprocess=processor.image_processor, download=True, root_dir=ARO_DIR)
    coco_order_dataset = COCO_Order(image_preprocess=processor.image_processor, download=True, root_dir=COCO_DIR) 
    flickr_order_dataset = Flickr30k_Order(image_preprocess=processor.image_processor, split='test', root_dir=ARO_DIR)

    # wrap into loaders
    vgr_loader = DataLoader(vgr_dataset, batch_size=16, shuffle=False)
    vga_loader = DataLoader(vga_dataset, batch_size=16, shuffle=False)
    coco_loader = DataLoader(coco_order_dataset, batch_size=16, shuffle=False)
    flickr_loader =  DataLoader(flickr_order_dataset, batch_size=16, shuffle=False)

    # wrap the eval model
    aro_wrap = AROtoHFCLIPWrap(model, processor)

    # get scores for VG-R
    vgr_scores = aro_wrap.get_retrieval_scores_batched(vgr_loader)
    vgr_records = vgr_dataset.evaluate_scores(vgr_scores)
    symmetric = ['adjusting', 'attached to', 'between', 'bigger than', 'biting', 'boarding', 'brushing', 'chewing', 'cleaning', 'climbing', 'close to', 'coming from', 'coming out of', 'contain', 'crossing', 'dragging', 'draped over', 'drinking', 'drinking from', 'driving', 'driving down', 'driving on', 'eating from', 'eating in', 'enclosing', 'exiting', 'facing', 'filled with', 'floating in', 'floating on', 'flying', 'flying above', 'flying in', 'flying over', 'flying through', 'full of', 'going down', 'going into', 'going through', 'grazing in', 'growing in', 'growing on', 'guiding', 'hanging from', 'hanging in', 'hanging off', 'hanging over', 'higher than', 'holding onto', 'hugging', 'in between', 'jumping off', 'jumping on', 'jumping over', 'kept in', 'larger than', 'leading', 'leaning over', 'leaving', 'licking', 'longer than', 'looking in', 'looking into', 'looking out', 'looking over', 'looking through', 'lying next to', 'lying on top of', 'making', 'mixed with', 'mounted on', 'moving', 'on the back of', 'on the edge of', 'on the front of', 'on the other side of', 'opening', 'painted on', 'parked at', 'parked beside', 'parked by', 'parked in', 'parked in front of', 'parked near', 'parked next to', 'perched on', 'petting', 'piled on', 'playing', 'playing in', 'playing on', 'playing with', 'pouring', 'reaching for', 'reading', 'reflected on', 'riding on', 'running in', 'running on', 'running through', 'seen through', 'sitting behind', 'sitting beside', 'sitting by', 'sitting in front of', 'sitting near', 'sitting next to', 'sitting under', 'skiing down', 'skiing on', 'sleeping in', 'sleeping on', 'smiling at', 'sniffing', 'splashing', 'sprinkled on', 'stacked on', 'standing against', 'standing around', 'standing behind', 'standing beside', 'standing in front of', 'standing near', 'standing next to', 'staring at', 'stuck in', 'surrounding', 'swimming in', 'swinging', 'talking to', 'topped with', 'touching', 'traveling down', 'traveling on', 'tying', 'typing on', 'underneath', 'wading in', 'waiting for', 'walking across', 'walking by', 'walking down', 'walking next to', 'walking through', 'working in', 'working on', 'worn on', 'wrapped around', 'wrapped in', 'by', 'of', 'near', 'next to', 'with', 'beside', 'on the side of', 'around']
    df = pd.DataFrame(vgr_records)
    df = df[~df.Relation.isin(symmetric)]
    vgr_metric = df.Accuracy.mean()
    print(f"VG-Relation Macro Accuracy: {vgr_metric}")

    # get scores for VG-A
    vga_scores = aro_wrap.get_retrieval_scores_batched(vga_loader)
    vga_records = vga_dataset.evaluate_scores(vga_scores)
    df = pd.DataFrame(vga_records)
    vga_metric = df.Accuracy.mean()
    print(f"VG-Attribution Macro Accuracy: {vga_metric}")

    # get scores for COCO
    coco_scores = aro_wrap.get_retrieval_scores_batched(coco_loader)
    coco_records = coco_order_dataset.evaluate_scores(coco_scores)
    df = pd.DataFrame(coco_records)
    coco_metric = df['Precision@1'].mean()
    print(f"COCO Precision@1: {coco_metric}")

    # get scores for COCO
    flickr_scores = aro_wrap.get_retrieval_scores_batched(flickr_loader)
    flickr_records = flickr_order_dataset.evaluate_scores(flickr_scores)
    df = pd.DataFrame(flickr_records)
    flickr_metric = df['Precision@1'].mean()
    print(f"Flickr Precision@1: {flickr_metric}")

    return {
        'vgr_metric': vgr_metric,
        'vga_metric': vga_metric,
        'coco_metric': coco_metric,
        'flickr_metric': flickr_metric,
    }


def run_aro_on_lora(lora_weight_path):
    from peft import PeftModel

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    base_clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    loaded = PeftModel.from_pretrained(base_clip_model, lora_weight_path)
    loaded = loaded.merge_and_unload()
    run_aro_evals(loaded, processor)


if __name__ == '__main__':
    from transformers import CLIPProcessor, CLIPModel
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    run_aro_evals(clip_model, clip_processor)
