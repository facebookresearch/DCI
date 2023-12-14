#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

"""
This module contains utilities for operating on the np-based masks
produced by the SegmentAnything setup.
"""

import numpy as np
import math
from typing import List, Optional, Tuple, NewType, cast

Xdm = NewType("Xdm", int)
Ydm = NewType("Ydm", int)

# Operators on regular np masks 

def mask_size(mask: np.ndarray) -> int:
    """Determine the number of pixels in this mask"""
    return np.sum(mask*1)

def masks_overlap(mask1: np.ndarray, mask2: np.ndarray) -> bool:
    """Return true if any pixels are shared between the two masks"""
    return np.any(mask1*1 + mask2*1 == 2)

def mask_union(mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
    """Return the union of the provided masks"""
    return cast(np.ndarray, mask1*1 + mask2*1 > 0)

def all_mask_union(masks: List[np.ndarray]) -> np.ndarray:
    """Union a full list of masks"""
    base = masks[0]
    for mask in masks[1:]:
        base = mask_union(base, mask)
    return base

def subtract_mask(mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
    """Provide the difference removing the second mask from the first"""
    return cast(np.ndarray, mask1*1 - mask2*1 == 1)


# Other utilities

Point = Tuple[Ydm, Xdm]

def point_in_box(y: Ydm, x: Xdm, tlbr: Tuple[Point, Point]) -> bool:
    """Return true if the given y,x point is in the provided bbox"""
    (t, l), (b, r) = tlbr
    return t <= y and y <= b and l <= x and x <= r


# Efficient operator for special repeated steps
class EfficientMask():
    """Class for more efficient mask mask over full numpy ndarrays"""
    def __init__(self, mask: np.ndarray, score: float, size: Optional[int] = None):
        self.mask = mask
        self.score = score
        self._size: Optional[int] = size
        self._tlbr: Optional[Tuple[Point, Point]] = None
    
    def __repr__(self) -> str:
        return f"<EM : {self.get_size()}, {self.get_tlbr()}>"
    
    def _reset_cache(self):
        self._tlbr = None
        self._size = None
    
    def set_to(self, other: "EfficientMask"):
        """Set this mask's values to that of other"""
        self.mask = other.mask
        self.score = other.score
        self._size = other._size
        self._tlbr = other._tlbr
    
    def get_tlbr(self) -> Tuple[Point, Point]:
        """Return the top left and bottom right bounds of this mask"""
        if self._tlbr is None:
            try:
                np_where = np.where(self.mask == True)
                left = np.min(np_where[1])
                right = np.max(np_where[1]) + 1
                top = np.min(np_where[0])
                bottom = np.max(np_where[0]) + 1
            except ValueError:
                top, left, bottom, right = (0, 0, 0, 0)
            self._tlbr = ((cast(Ydm, top), cast(Xdm, left)), (cast(Ydm, bottom), cast(Xdm, right)))
        return self._tlbr
    
    def get_size(self) -> int:
        """Return the total number of true pixels in this mask"""
        if self._size is None:
            (top, left), (bottom, right) = self.get_tlbr()
            self._size = np.sum(self.mask[top:bottom,left:right]*1)
        return self._size
    
    def get_density(self) -> float:
        """Provide rough density with number of pixels and bbox size"""
        size = self.get_size()
        (t, l), (b, r) = self.get_tlbr()
        area = (b-t) * (r-l) + 1
        return size / area
    
    def dense_score(self) -> float:
        """Return the score times the density, a heuristic for quality"""
        return self.score * math.sqrt(self.get_density())
    
    def _bbox_overlaps(self, other: "EfficientMask") -> bool:
        """Check points of opposite diagonals in each other bbox"""
        (t1, l1), (b1, r1) = self.get_tlbr()
        (t2, l2), (b2, r2) = other.get_tlbr()
        return (
            point_in_box(t1, l1, other.get_tlbr()) or 
            point_in_box(b1, r1, other.get_tlbr()) or 
            point_in_box(t2, r2, self.get_tlbr()) or 
            point_in_box(b2, l2, self.get_tlbr()) 
        )
    
    def _get_overlap_submask(self, other: "EfficientMask") -> np.ndarray:
        """Get a classic ndarray of pixels in the overlap between this and other"""
        if not self._bbox_overlaps(other):
            return np.array([])
        (t1, l1), (b1, r1) = self.get_tlbr()
        (t2, l2), (b2, r2) = other.get_tlbr()
        maxt, maxl = max(t1, t2), max(l1, l2)
        minb, minr = min(b1, b2), min(r1, r2)
        return (self.mask[maxt:minb,maxl:minr]*1 + other.mask[maxt:minb,maxl:minr]*1 == 2)
    
    def _get_xor_submask(self, other: "EfficientMask") -> np.ndarray:
        """Get a classic ndarray of pixels in the xor between this and other"""
        if not self._bbox_overlaps(other):
            return np.array([])
        (t1, l1), (b1, r1) = self.get_tlbr()
        (t2, l2), (b2, r2) = other.get_tlbr()
        mint, minl = min(t1, t2), min(l1, l2)
        maxb, maxr = max(b1, b2), max(r1, r2)
        return (self.mask[mint:maxb,minl:maxr]*1 + other.mask[mint:maxb,minl:maxr]*1 == 1)
    
    def intersect(self, other: "EfficientMask") -> "EfficientMask":
        """Return an efficient mask of the overlap between this and other"""
        res = np.full(self.mask.shape, False)
        submask = self._get_overlap_submask(other)
        if len(submask) != 0:
            (t1, l1), (b1, r1) = self.get_tlbr()
            (t2, l2), (b2, r2) = other.get_tlbr()
            maxt, maxl = max(t1, t2), max(l1, l2)
            minb, minr = min(b1, b2), min(r1, r2)
            res[maxt:minb,maxl:minr] = submask
        return EfficientMask(res, (self.score + other.score)/2)

    def mostly_contained_in(self, out_mask: "EfficientMask", thresh: float = 0.95) -> bool:
        """Returns True if thresh of self's pixels are in out_mask"""
        size_in = self.get_size() + 1
        overlap = mask_size(self._get_overlap_submask(out_mask))
        return overlap / size_in > thresh
    
    def overlaps_threshold(self, other: "EfficientMask", thresh: float = 0.50) -> bool:
        """Returns true if over thresh of either mask is contained in the other"""
        size_1 = self.get_size() + 1
        size_2 = other.get_size() + 1
        overlap = mask_size(self._get_overlap_submask(other))
        return overlap / size_1 > thresh or overlap / size_2 > thresh
    
    def near_equivalent_to(self, other: "EfficientMask", thresh: float = 0.96) -> bool:
        """Return true if these two masks have prop overlapping pixels > thresh"""
        size_1 = self.get_size() + 1
        size_2 = other.get_size() + 1
        if size_1 / size_2 < thresh or size_2 / size_1 < thresh:
            return False
        difference = mask_size(self._get_xor_submask(other))
        if (difference / size_1) > (1-thresh) or (difference / size_2) > (1-thresh):
            return False
        return True
    
    def union(self, other: "EfficientMask") -> "EfficientMask":
        """Return a new efficient mask unioning these"""
        new_mask = self.mask * 1
        (t2, l2), (b2, r2) = other.get_tlbr()
        new_mask[t2:b2,l2:r2] += other.mask[t2:b2,l2:r2]*1
        return EfficientMask(
            mask=cast(np.ndarray, new_mask > 0),
            score=(self.score + other.score) / 2, # may be more appropriate as weighted mask sizes
        )

    def subtract(self, other: "EfficientMask") -> "EfficientMask":
        """Subtract the other mask from this one"""
        new_mask = self.mask * 1
        (t2, l2), (b2, r2) = other.get_tlbr()
        new_mask[t2:b2,l2:r2] -= other.mask[t2:b2,l2:r2]*1
        return EfficientMask(
            mask=cast(np.ndarray, new_mask == 1),
            score=self.score,
        )
