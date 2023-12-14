#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import matplotlib.pyplot as plt # type: ignore

from .mask_creation_utils import FinalGrouping
from typing import Optional

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.2])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_masks(masks, ax):
    sorted_masks = sorted(masks, key=lambda m: np.sum(m*1), reverse=True)
    for mask in sorted_masks:
        show_mask(mask, ax, random_color=True)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='green', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def display_all_levels(image: np.ndarray, full_group_tree: FinalGrouping, label: Optional[str]=None):
    unexplored_depth = list(full_group_tree.values())
    depth_levels = []
    
    while len(unexplored_depth) > 0:
        depth_levels.append([r['outer_mask'].mask for r in unexplored_depth])
        next_depth = []
        for elem in unexplored_depth:
            next_depth += list(elem['subgroups'].values())
        unexplored_depth = next_depth
    
    for idx, level in enumerate(depth_levels):
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        plt.title(f"All Submasks depth {idx}, {label}", fontsize=18)
        show_masks(level, plt.gca())
        plt.axis('off')
        plt.show()
