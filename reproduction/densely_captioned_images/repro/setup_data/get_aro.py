#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

from densely_captioned_images.repro.eval.ARO.dataset_zoo import VG_Relation, VG_Attribution, COCO_Order
from densely_captioned_images.repro.config import COCO_DIR, ARO_DIR
import os

def run_downloads():
    # ARO-specific
    if not os.path.exists(ARO_DIR):
        os.makedirs(ARO_DIR, exist_ok=True)

    _ = VG_Relation(image_preprocess=lambda x: x, download=True, root_dir=ARO_DIR)
    _ = VG_Attribution(image_preprocess=lambda x: x, download=True, root_dir=ARO_DIR)

    # COCO
    if not os.path.exists(COCO_DIR):
        raise Exception("Downoad COCO through the hake setup first!")

    _ = COCO_Order(image_preprocess=lambda x: x, download=True, root_dir=COCO_DIR) 


if __name__ == '__main__':
    run_downloads()