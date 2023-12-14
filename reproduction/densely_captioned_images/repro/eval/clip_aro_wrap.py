#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from tqdm import tqdm

class AROtoHFCLIPWrap():
    def __init__(self, model, processor, device=None):
        self.model = model
        self.processor = processor
        if device is None:
            device = model.device
        self.device = device
    
    @torch.no_grad()
    def process_batch(self, b):
        width = len(b['caption_options'])
        bs = len(b['caption_options'][0])

        all_entries = []
        for cap_tuple in b['caption_options']:
            all_entries += list(cap_tuple)
        entries_tokenized = self.processor.tokenizer(all_entries, return_tensors='pt', padding=True).to(self.device)
        pixel_values = b['image_options'][0]['pixel_values'][0]
        all_logits = self.model(input_ids=entries_tokenized['input_ids'], attention_mask=entries_tokenized['attention_mask'], pixel_values=pixel_values.to(self.device))

        def do_keep(a):
            rowsize = width*bs

            curr_row = a // rowsize
            curr_col = a % bs
            return curr_col == curr_row

        index_np = np.arange(width*bs*bs).reshape((bs,width*bs))
        grouped = all_logits.logits_per_image.cpu().numpy()[do_keep(index_np)]

        scores = grouped.reshape((bs,1,width))
        return scores

    @torch.no_grad()
    def get_retrieval_scores_batched(self, joint_loader):
        """Computes the scores for each image_option / caption_option pair in the joint loader.

        Args:
            joint_loader (DataLoader): batches have "image_options" and "caption_options" fields.
            "image_options" is a list of images, and "caption_options" is a list of captions.

        Returns:
            all_scores: A numpy array containing the scores of the shape NxKxL,
            where N is the number of test cases, K is the number of image options per the test case,
            and L is the number of caption options per the test case.
        """
        scores = []
        tqdm_loader = tqdm(joint_loader)
        tqdm_loader.set_description("Computing retrieval scores")
        for batch in tqdm_loader:
            batch_score = self.process_batch(batch)
            scores.append(batch_score)
        
        all_scores = np.concatenate(scores, axis=0) # N x K x L
        return all_scores