#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import time
import sys
from segment_anything import sam_model_registry
from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
from .mask_creation_utils import get_groups_simple, refine_groups_simple, FinalGrouping, FinalGroup, get_points_from_canny_greedy
from .efficient_mask import EfficientMask
from PIL import Image
import numpy as np
import os
import base64
from io import BytesIO
import cv2
import json
from typing import TypedDict, List

LOW = 5000 # Low value into the images array to start at
HIGH = 12000 # High value in images array to go to
SETEV_MODEL_ROOT = 'FILL_ME' # TODO fill in
ANNOTATE_ROOT = os.path.dirname(os.path.dirname(__file__))
SOURCE_DIR = os.path.join(ANNOTATE_ROOT, "assets/images")
OUT_DIR = os.path.join(ANNOTATE_ROOT, "assets/masks")

class SAMResult(TypedDict):
    segmentation: np.ndarray # the mask itself
    bbox: List[float] #XYWH of the mask
    area: int # area of the mask
    predicted_iou: float # model predicted quality
    point_coords: List[List[float]] # coords of this point
    stability_score: float # model stability score
    crop_box: List[float] # image crop used to generate this mask, XYWH


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
            'area': int(outer_mask.get_size()),
            'bounds': ((int(t), int(l)), (int(b), int(r))),
            'subgroups': {
                idx: fold_group(subsubg) for (idx, subsubg) in subg['subgroups'].items()
            }
        }
    return {
        idx: fold_group(subg) for (idx, subg) in g.items()
    }

def group_outputs(outputs: List[SAMResult]) -> FinalGrouping:
    as_efficient_masks: List[EfficientMask] = [
        EfficientMask(
            res['segmentation'], 
            res['predicted_iou'] * (res['stability_score'] ** 2), 
            size=res['area'],
        ) for res in outputs
    ]
    
    in_order = sorted(as_efficient_masks, key=lambda x: x.get_size(), reverse=True)
    return get_groups_simple(in_order)

def main():
    all_images = os.listdir(SOURCE_DIR)
    target_images = all_images[LOW:HIGH]

    sam_checkpoint = SETEV_MODEL_ROOT
    model_type = "vit_h"
    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side = 50,
        points_per_batch = 64,
        pred_iou_thresh = 0.8,
        stability_score_thresh = 0.94,
        stability_score_offset = 1.0,
        box_nms_thresh = 0.97,
        min_mask_region_area = 1000,
        output_mode = "binary_mask",
    )

    first_start = time.time()
    for idx, img in enumerate(target_images):
        try:
            start_time = time.time()
            path = os.path.join(SOURCE_DIR, img)
            img_array = cv2.imread(path)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            
            canny_points = get_points_from_canny_greedy(img_array, distance_threshold=12, jitter_amount=35, num_extra=8)
            if len(canny_points) == 0:
                canny_results = []
                print(f"[{time.time() - first_start}] No canny points for image {idx+LOW} : {time.time() - start_time}")
            else:
                points_for_sam = np.array([
                    [pt[1]/img_array.shape[1], pt[0]/img_array.shape[0]] for pt in canny_points
                ])
                canny_generator = SamAutomaticMaskGenerator(
                    sam,
                    points_per_side=None,
                    point_grids=points_for_sam,
                    points_per_batch = 64,
                    pred_iou_thresh = 0.8,
                    stability_score_thresh = 0.94,
                    stability_score_offset = 1.0,
                    box_nms_thresh = 0.97,
                    min_mask_region_area = 1000,
                    output_mode = "binary_mask",
                )
                canny_results = canny_generator.generate(img_array)
                print(f"[{time.time() - first_start}] SA canny compute time for image {idx+LOW} : {time.time() - start_time}")

            result = generator.generate(img_array)
            print(f"[{time.time() - first_start}] SA compute time for image {idx+LOW} : {time.time() - start_time}")

            result += canny_results
            grouped = group_outputs(result)
            refined = refine_groups_simple(grouped)
            folded = fold_group_tree(refined)
            with open(os.path.join(OUT_DIR, img + "-mask.json"), 'w+') as json_outf:
                json.dump(folded, json_outf)

            print(f"[{time.time() - first_start}] Total compute time for image {idx+LOW} : {time.time() - start_time}")
        except Exception as e:
            print(f"[{time.time() - first_start}] Error on image {idx+LOW} : {e} - skipping")

if __name__ == '__main__':
    main()
