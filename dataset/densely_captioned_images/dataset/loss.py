#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn

"""
Main loss functions for CLIP and negatives loss
"""

def get_grouped_diag(logits: torch.Tensor) -> torch.Tensor:
    """Convert an (B, N*B) matrix into an (B, N) diag"""
    dims = logits.shape # (B, N*B)
    bs = dims[0]
    N = dims[1] // bs

    def do_keep(a):
        rowsize = N*bs

        curr_row = a // rowsize
        curr_col = (a % rowsize) // N
        return curr_col == curr_row

    index_ten = torch.arange(N*bs*bs).reshape((bs,N*bs))
    return logits[do_keep(index_ten)].reshape(bs, N)

def get_pooled_groups(logits: torch.Tensor, pool_type='avg') -> torch.Tensor:
    """Convert an (B, N*B) matrix into an (B, B)"""
    dims = logits.shape # (B, N*B)
    bs = dims[0]
    N = dims[1] // bs

    grouped = logits.reshape((bs, bs, N))
    if pool_type == 'avg':
        return torch.mean(grouped, dim=2)
    elif pool_type == 'max':
        return torch.max(grouped, dim=2).values
    elif pool_type == 'min':
        return torch.min(grouped, dim=2).values

def get_pooled_diag(logits: torch.Tensor, pool_type='avg') -> torch.Tensor:
    """Convert a (B, N*B) matrix ino (B, 1)"""
    return get_pooled_groups(logits, pool_type=pool_type).diag()

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

def clip_loss(similarity: torch.Tensor, pool_type='avg') -> torch.Tensor:
    """Average contrastive loss of captions -> images and images -> captions"""
    if pool_type == 'hardest':
        # Get max of negatives, min of positives
        similarity = get_pooled_groups(similarity, pool_type='max')
        min_sim_diag = get_pooled_diag(similarity, pool_type='min')
        similarity[range(len(min_sim_diag)), range(len(min_sim_diag))] = min_sim_diag
    else: # pool_type == 'avg':
        similarity = get_pooled_groups(similarity)
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0

def negatives_loss_pairwise(scores: torch.Tensor) -> torch.Tensor:
    """Pairwise BCE for positive/negative labelled images"""
    label = torch.tensor([[1, 0]], device=scores.device).float()
    labels = label.tile((scores.size()[0], 1))
    neg_loss = nn.functional.binary_cross_entropy_with_logits(scores, labels)
    return neg_loss

def negatives_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor, pool_type='avg') -> torch.Tensor:
    pos_pool_type = 'min' if pool_type == 'hardest' else pool_type
    neg_pool_type = 'max' if pool_type == 'hardest' else pool_type
    pos_diag = get_pooled_diag(pos_scores, pool_type=pos_pool_type)
    neg_diag = get_pooled_diag(neg_scores, pool_type=neg_pool_type)
    res = torch.stack((pos_diag, neg_diag), axis=-1)
    return negatives_loss_pairwise(res)