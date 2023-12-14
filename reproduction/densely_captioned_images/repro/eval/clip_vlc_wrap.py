#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

from densely_captioned_images.repro.eval.VLChecklist.vl_checklist.vlp_model import VLPModel

import torch
from PIL import Image


class VLCtoHFCLIPWrap(VLPModel):
    """Wrapping HF CLIP model into the format for VLC"""
    def __init__(self, model_id, model, processor, device=None):
        self.model = model
        self.processor = processor
        if device is None:
            device = model.device
        self.device = device
        self.model_id = model_id
        self.batch_size=16

    def model_name(self):
        return self.model_id

    def _load_data(self, src_type, data):
        pass

    def predict(self,
                images: list,
                texts: list,
                src_type: str = 'local'
                ):
        loaded_images = [Image.open(p) for p in images]
        with torch.no_grad():
            inputs = self.processor(text=texts, images=loaded_images, return_tensors='pt', padding=True)
            res = self.model(**inputs.to(self.device))
        logits_per_image = res.logits_per_image
        probs = []
        probs.extend(logits_per_image.cpu().diag().numpy())
        return {"probs":[(None, p) for p in probs]}
