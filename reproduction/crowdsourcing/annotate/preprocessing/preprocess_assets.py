#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import time
import sys
from segment_anything import sam_model_registry, SamPredictor
from .mask_creation_utils import compute_group_tree, FinalGrouping, FinalGroup
from PIL import Image
import numpy as np
import os
import base64
from io import BytesIO
import cv2
import json

LOW = 5000 # Low value into the images array to start at
HIGH = 12000 # High value in images array to go to
SETEV_MODEL_ROOT = 'FILL_ME' # TODO fill in
ANNOTATE_ROOT = os.path.dirname(os.path.dirname(__file__))
SOURCE_DIR = os.path.join(ANNOTATE_ROOT, "assets/images")
OUT_DIR = os.path.join(ANNOTATE_ROOT, "assets/masks")

def fold_group_tree(g: FinalGrouping):
    def fold_group(subg: FinalGroup):
        outer_mask = subg['outer_mask']
        mask_img = Image.fromarray(np.uint8(outer_mask.mask * 255)) # type: ignore
        mask_img = mask_img.convert('1')
        maskbuf = BytesIO()
        mask_img.save(maskbuf, format='png', bits=1, optimize=True)
        mask_bytes = maskbuf.getvalue()
        as_base64 = base64.b64encode(mask_bytes)
        as_str = as_base64.decode('utf-8')
        (t, l), (b, r) = subg['outer_mask'].get_tlbr()
        
        return {
            'outer_mask': as_str,
            'area': outer_mask.get_size(),
            'bounds': ((int(t), int(l)), (int(b), int(r))),
            'subgroups': {
                idx: fold_group(subsubg) for (idx, subsubg) in subg['subgroups'].items()
            }
        }
    return {
        idx: fold_group(subg) for (idx, subg) in g.items()
    }

def main():
    all_images = os.listdir(SOURCE_DIR)
    target_images = all_images[LOW:HIGH]

    sam_checkpoint = SETEV_MODEL_ROOT
    model_type = "vit_h"
    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)

    for idx, img in enumerate(target_images):
        start_time = time.time()
        path = os.path.join(SOURCE_DIR, img)
        img_array = cv2.imread(path)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        result = compute_group_tree(
            predictor, 
            img_array, 
            score_cutoff=0.65,
            outer_sim_thresh=0.9, 
            mutual_sim_thresh=0.95, 
            retain_best=False,
        )
        folded = fold_group_tree(result)
        with open(os.path.join(OUT_DIR, img + "-mask.json"), 'w+') as json_outf:
            json.dump(folded, json_outf)

        print(f"[{time.time()}] Total compute time for image {idx+LOW} : {time.time() - start_time}")

if __name__ == '__main__':
    main()
