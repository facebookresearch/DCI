#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import random
import numpy as np
import json
from tqdm import tqdm # type: ignore
import cv2

from .efficient_mask import EfficientMask, Point, Xdm, Ydm, all_mask_union

from typing import Optional, List, Dict, Literal, Union, TypedDict, Any, Tuple, cast, TYPE_CHECKING
if TYPE_CHECKING:
    from segment_anything import SamPredictor #type: ignore

TARGET_STEP = 100
SKIP_LOGGING = True

# Types
MaskMergeKey = Union[Literal['best'], Literal['largest']]

class GroupItem(TypedDict):
    outer_mask: EfficientMask
    points: List[Point]
    submasks: List[EfficientMask]
    

GroupDictKey = Union[int, Literal['empty']]
GroupDict = Dict[GroupDictKey, GroupItem]

class FinalGroup(TypedDict):
    outer_mask: EfficientMask
    subgroups: Dict[Union[int, str], "FinalGroup"]

FinalGrouping = Dict[Union[int, str], FinalGroup]


def jitter(size: float) -> float:
    """Get some jitter from -size/2 to size/2"""
    return (0.5 - random.random()) * size

def bound(v, lo, hi):
    """Return a value for v after bounding into lo, hi"""
    return int(max(min(hi, v), lo))

def _load_final_group_from_json(json_dict) -> FinalGroup:
    from PIL import Image
    from io import BytesIO
    import base64

    as_str = json_dict['outer_mask']
    im = Image.open(BytesIO(base64.b64decode(as_str)))
    mask_array = np.array(im)

    return {
        'outer_mask': EfficientMask(mask_array, -1),
        'subgroups': {
            idx: _load_final_group_from_json(val) for (idx, val) in json_dict['subgroups'].items()
        },
    }

def load_final_group_from_json(json_dict) -> FinalGrouping:
    res: FinalGrouping = {
        idx: _load_final_group_from_json(val) for (idx, val) in json_dict.items()
    }
    return res


#####
# Candidate point methods
#####

def get_grid(
    step: int, 
    top_left: Point, 
    bottom_right: Point, 
    noise: Optional[float] = None
) -> List[Point]:
    """
    Get a perturbed grid in the given dimensions, with spacing of step between points
    then shifted by noise (default step/4)
    """
    top, left = top_left
    bottom, right = bottom_right
    
    if noise is None:
        noise = step / 4
    
    # Calculate the grid points approximately at step size
    height = bottom - top
    width = right - left
    height_steps = height // step - 1
    width_steps = width // step - 1
    height_step_size = (height - step) / height_steps
    width_step_size = (width - step) / width_steps
    points: List[Point] = []
    for j in range(width_steps + 1):
        for i in range(height_steps + 1):
            points.append((
                cast(Ydm, int(top + step / 2 + i*height_step_size + jitter(noise))),
                cast(Xdm, int(left + step / 2 + j*width_step_size + jitter(noise)))
            ))
    return points

def get_missing_points_greedy(mask: np.ndarray, min_size: int) -> List[Point]:
    """
    Given an existing mask, find any points with min_size radius out that are missing
    Suggested min_size at a stage is ~step/4
    """
    curr_mask = np.copy(mask)
    np_where = cast(Tuple[List[Ydm], List[Xdm], Any], np.where(curr_mask == False))
    possible_points = list(zip(np_where[0], np_where[1]))
    found_points: List[Point] = []
    step = int(min_size / 2)
    checkpoints: List[Tuple[int, int]] = [(0, step), (0, -step), (step, 0), (-step, 0)]
    for (y, x) in possible_points:
        clear = True
        for (dy, dx) in checkpoints:
            try:
                if curr_mask[y+dy][x+dx] != False:
                    clear = False
                    break
            except IndexError: # out of bounds
                clear = False
                break
        if clear:
            curr_mask[y-step:y+step,x-step:x+step] = True
            found_points.append((cast(Ydm, y), cast(Xdm, x)))
    return found_points

def get_points_from_canny_greedy(
    image: np.ndarray, 
    distance_threshold: int = 40, 
    jitter_amount: int = 40,
    num_extra: int = 3,
) -> List[Point]:
    """
    Uses a canny edge detection output to determine good candidate points for
    additional mask generation. First detects edges, then selects detected
    edge points at random, removing other possible candidates within the provided
    distance threshold. Adds num_extra points jitter_amount around each selected
    candidate for coverage.
    """
    # Create a blurred image to run canny on
    blur_image = cv2.bilateralFilter(image, 11, 61, 39)
    edges = cv2.Canny(image=blur_image, threshold1=100, threshold2=250) 
    point_remask = edges == 255

    np_where = cast(Tuple[List[Ydm], List[Xdm], Any], np.where(point_remask == True))
    possible_points = list(zip(np_where[0], np_where[1]))
    random.shuffle(possible_points)
    found_points: List[Point] = []
    step = int(distance_threshold)
    while len(possible_points) > 0:
        (y, x) = possible_points.pop()
        if x < 10 or y < 10 or x > image.shape[1] - 10 or y > image.shape[0] - 10:
            continue # skip points too close to image edge
        if point_remask[y, x]:
            found_points.append((cast(Ydm, bound(y, 10, image.shape[1]-10)), cast(Xdm, bound(x, 10, image.shape[0]-10))))
            for _ in range(num_extra):
                found_points.append((
                    cast(Ydm, bound(y+jitter(jitter_amount), 10, image.shape[1]-10)), 
                    cast(Xdm, bound(x+jitter(jitter_amount), 10, image.shape[0]-10))
                ))
            # clear out nearby candidates
            point_remask[max(0,y-step):y+step, max(0,x-step):x+step] = False
    return found_points


######
# SAM-based predictor methods
######

def predict_all(
    predictor: "SamPredictor", 
    image: np.ndarray, 
    step: int = TARGET_STEP, 
    top_left: Optional[Point] = None, 
    bottom_right: Optional[Point] = None, 
    containing_mask: Optional[np.ndarray] = None
) -> Dict[Point, List[EfficientMask]]:
    """
    Predict a grid-based series of masks from SAM for the currently
    set image, assuming it's already set
    """
    if top_left is None:
        top_left = cast(Point, (0, 0))
    if bottom_right is None:
        bottom_right = cast(Point, (image.shape[0], image.shape[1]))
    
    grid_points = get_grid(step, top_left, bottom_right, noise=0)
    if containing_mask is not None:
        grid_points = [pt for pt in grid_points if containing_mask[pt]]

    return predict_for_points(predictor, grid_points)

def predict_for_points(
    predictor: "SamPredictor", 
    points: List[Point],
) -> Dict[Point, List[EfficientMask]]:
    """
    Predict a grid-based series of masks from SAM for the currently
    set image, assuming it's already set, from a list of points
    in the format [y, x]
    """
    results: Dict[Point, List[EfficientMask]] = {}
    for pt in points:
        masks, scores, _ = predictor.predict(
            point_coords=np.array([[pt[0], pt[1]]]),
            point_labels=np.array([1]),
            multimask_output=True,
        )
        results[pt] = [EfficientMask(m, s) for m, s in zip(masks, scores)]
    return results

def predict_for_bounded_points(
    predictor: "SamPredictor", 
    image: np.ndarray, 
    points: List[Point], 
    mask: EfficientMask,
) -> Dict[Point, List[EfficientMask]]:
    """Produce new masks based on bounding the given image before predicting"""
    (top, left), (bottom, right) = mask.get_tlbr()
    bounded_image = image[top:bottom,left:right,:]
    predictor.set_image(bounded_image)
    results: Dict[Point, List[EfficientMask]] = {}
    for pt in points:
        masks, scores, _ = predictor.predict(
            point_coords=np.array([[pt[0]-top, pt[1]-left]]),
            point_labels=np.array([1]),
            multimask_output=False,
        )
        new_masks = []
        for new_mask in masks:
            base_mask = np.full(bounded_image.shape[:2], False)
            base_mask[top:bottom,left:right] = new_mask
            new_masks.append(base_mask)
        results[(pt[0], pt[1])] = [EfficientMask(m, s) for m, s in zip(new_masks, scores)]
    return results

def get_canny_masks(
    predictor: "SamPredictor", 
    image: np.ndarray, 
    distance_threshold: int = 40, 
    jitter_amount: int = 40
):
    """Determine masks for predicted canny edge points"""
    points = get_points_from_canny_greedy(image, distance_threshold=distance_threshold, jitter_amount=jitter_amount)
    return predict_for_points(predictor, points)

#####
# Processing masks into groups
#####

def process_best_largest(
    results: Dict[Point, List[EfficientMask]], 
    penalty_gap: float = 0.2,
) -> Dict[Point, Dict[MaskMergeKey, EfficientMask]]:
    """
    Return the best and largest mask for each point, skipping large masks that are 
    more than the penalty gap worse than the best mask
    """
    processed_results: Dict[Point, Dict[MaskMergeKey, EfficientMask]] = {}
    for pt, ems in results.items():
        best_mask = ems[0]
        largest_mask = ems[0]
        for oem in ems[1:]:
            if oem.score > best_mask.score:
                best_mask = oem
        for oem in ems[1:]:
            if oem.get_size() > largest_mask.get_size():
                if oem.score < largest_mask.score and oem.score < best_mask.score - penalty_gap:
                    continue # skip if that much worse than the best
                largest_mask = oem
        processed_results[pt] = {'best': best_mask, 'largest': largest_mask}
    return processed_results


def get_groups(
    processed_results: Dict[Point, Dict[MaskMergeKey, EfficientMask]], 
    merge_key: MaskMergeKey = 'best', 
    groups: Optional[GroupDict] = None,
) -> GroupDict:
    """Break down processed results into groups, potentially extending a given set of groups"""
    if groups is None:
        groups = {}
        curr_idx = 0
    else:
        curr_idx = max([k for k in groups.keys() if isinstance(k, int)]) + 1
    for pt, masks in processed_results.items():
        if len(groups) == 0:
            groups[curr_idx] = {'points': [pt], 'outer_mask': masks[merge_key], 'submasks': []}
            curr_idx += 1
        else:
            overlaps = []
            for other_idx, other_group in groups.items():
                if other_group['outer_mask'].overlaps_threshold(masks[merge_key]):
                    overlaps.append(other_idx)
                
            if len(overlaps) == 0:
                # New group, add in
                groups[curr_idx] = {'points': [pt], 'outer_mask': masks[merge_key], 'submasks': []}
                curr_idx += 1
            else:
                # merge into first overlapping group
                merge_idx = overlaps[0]
                groups[merge_idx]['points'].append(pt)
                groups[merge_idx]['outer_mask'] = groups[merge_idx]['outer_mask'].union(masks[merge_key])
                if len(overlaps) > 1:
                    if not SKIP_LOGGING:
                        print("Forcing merge on overlap score", masks[merge_key].score)
                    # merge any other overlapping groups into same
                    for from_idx in overlaps[1:]:
                        groups[merge_idx]['points'] += groups[from_idx]['points']
                        groups[merge_idx]['outer_mask'] = groups[merge_idx]['outer_mask'].union(groups[from_idx]['outer_mask'])
                        del groups[from_idx]
    return groups


def get_groups_simple(
    sam_results: List[EfficientMask],
) -> FinalGrouping:
    """Convert a list of masks into a tree of masks, assuming a sorted list"""
    if len(sam_results) == 0:
        return {}

    groups: FinalGrouping = {}
    sublists: Dict[Union[int, str], List[EfficientMask]] = {}

    curr_idx = 0
    for mask in sam_results:
        if len(groups) == 0:
            groups[curr_idx] = {'outer_mask': mask, 'subgroups': {}}
            sublists[curr_idx] = []
            curr_idx += 1
        else: 
            has_overlap = None

            for other_idx, other_group in groups.items():
                if other_group['outer_mask'].near_equivalent_to(mask, 0.95):
                    has_overlap = other_idx
                    break
            if has_overlap is not None:
                # take the best of the overlapping masks
                own_score = mask.score
                other_score = groups[has_overlap]['outer_mask'].score
                if own_score > other_score:
                    groups[has_overlap]['outer_mask'] = mask
                continue # skip as one mask was selected
            
            is_contained = None
            too_close = False
            for other_idx, other_group in groups.items():
                if mask.mostly_contained_in(other_group['outer_mask'], 0.90):
                    is_contained = other_idx
                    # contained masks should be at least 10% smaller
                    if mask.get_size() / other_group['outer_mask'].get_size() > 0.90:
                        too_close = True
                    break
            
            if too_close:
                own_score = mask.score
                other_score = groups[is_contained]['outer_mask'].score
                if own_score > other_score:
                    groups[is_contained]['outer_mask'] = mask
                continue
                
            if is_contained is None:
                # New group, add in
                groups[curr_idx] = {'outer_mask': mask, 'subgroups': {}}
                sublists[curr_idx] = []
                curr_idx += 1
            else:
                # add as subgroup
                sublists[is_contained].append(mask)

    for group_idx, group in groups.items():
        group['subgroups'] = get_groups_simple(sublists[group_idx])

    return groups

def print_groups(groups: FinalGrouping) -> None:
    """Print out a representation of this grouping"""
    def _get_group_map(curr_g: FinalGrouping) -> Dict[Union[int, str], Any]:
        return {idx: _get_group_map(g['subgroups']) for idx, g in curr_g.items()}
    
    print(json.dumps(_get_group_map(groups), indent=4))

def refine_groups_simple(groups: FinalGrouping, merge_thresh = 0.03) -> FinalGrouping:
    """Takes a final grouping and attempts to merge overlapping masks where possible"""
    
    new_final_group: Dict[Union[str, int], FinalGroup] = {}
    curr_idx = 0
    for group in groups.values():
        group['subgroups'] = refine_groups_simple(group['subgroups'], merge_thresh)

        found_overlaps = []
        for other_idx, other_group in new_final_group.items():
            if group['outer_mask'].overlaps_threshold(other_group['outer_mask'], merge_thresh):
                found_overlaps.append(other_idx)

        if len(found_overlaps) == 0:
            new_final_group[curr_idx] = group
            curr_idx += 1
        else:
            new_subgroup: FinalGroup = {'outer_mask': group['outer_mask'], 'subgroups': {curr_idx: group}}
            for other_idx in found_overlaps:
                other_group = new_final_group[other_idx]
                new_subgroup['subgroups'].update(other_group['subgroups'])
                new_subgroup['outer_mask'] = new_subgroup['outer_mask'].union(other_group['outer_mask'])
                del new_final_group[other_idx]
            new_final_group[curr_idx] = new_subgroup
            curr_idx += 1

    # clean up non-mergers, recurse
    for sg in new_final_group.values():
        if len(sg['subgroups']) == 1 and sg['outer_mask'].near_equivalent_to(list(sg['subgroups'].values())[0]['outer_mask'], 0.93):
            sg['subgroups'] = list(sg['subgroups'].values())[0]['subgroups']

    return new_final_group

# First round

def first_iteration_groups(
    predictor: "SamPredictor",
    processed_results: Dict[Point, Dict[MaskMergeKey, EfficientMask]], 
    step: int, 
    merge_key: MaskMergeKey = "largest",
) -> GroupDict:
    """Get the first level of groups provided the initial best-largest masks"""
    groups = get_groups(processed_results, merge_key)
    group_mask = all_mask_union([p['outer_mask'].mask for p in groups.values()])
    missing_points = get_missing_points_greedy(group_mask, int(step/4))
    processed_missing_predictions = process_best_largest(predict_for_points(predictor, missing_points))
    processed_results.update(processed_missing_predictions)
    final_groups = get_groups(processed_missing_predictions, merge_key, groups)
    group_mask = all_mask_union([p['outer_mask'].mask for p in final_groups.values()])
    final_groups["empty"] = {
        'points': [], 
        'outer_mask': EfficientMask(cast(np.ndarray, group_mask == False), 0),
        "submasks": [],
    } # empty exists for any additional points added later
    return final_groups

def get_subgroup_mask_lists(
    groups: GroupDict, 
    base_masks: Dict[Point, List[EfficientMask]], 
    canny_masks: Dict[Point, List[EfficientMask]], 
    score_cutoff: float = 0.7, 
    retain_best: bool = False,
) -> GroupDict:
    """Given the level 0 groups, break out the masks into their respective subgroups"""
    subgroup_mask_lists: Dict[GroupDictKey, GroupItem] = {}
    joined_group: Dict[Point, List[EfficientMask]] = {}
    joined_group.update(base_masks)
    joined_group.update(canny_masks)
    for group_idx, group in groups.items():
        if not SKIP_LOGGING:
            print(f"Processing group {group_idx}...")
        target_mask = group['outer_mask']
        used_masks: List[EfficientMask] = []
        total_skipped = 0
        old_joined_group = joined_group
        joined_group = {}
        for (y, x), ems in tqdm(old_joined_group.items(), disable=SKIP_LOGGING):
            if target_mask.mask[y,x] != True:
                # See on another iteration
                joined_group[(y,x)] = ems
            else:
                for em in ems:
                    if em.dense_score() < score_cutoff:
                        total_skipped+=1
                        continue
                    pos_mask = em.intersect(target_mask)
                    if pos_mask.near_equivalent_to(target_mask, thresh=0.90):
                        total_skipped+=1
                        continue
                    used_masks.append(pos_mask)
        if not SKIP_LOGGING:
            print(f"Found {len(used_masks)} possible masks, skipped {total_skipped}. Filtering...")

        used_masks.sort(key=lambda x: x.get_size(), reverse=True)

        post_filtered_masks: List[EfficientMask] = []
        for mask_elem in tqdm(used_masks, disable=SKIP_LOGGING):
            too_similar = False
            for existing_mask in post_filtered_masks:
                if mask_elem.near_equivalent_to(existing_mask, thresh=0.85):
                    # take the better scoring mask
                    if retain_best and mask_elem.dense_score() > existing_mask.dense_score():
                        existing_mask.set_to(mask_elem)
                    too_similar = True
                    break
            if not too_similar:
                post_filtered_masks.append(mask_elem)
        points: List[Point] = []
        subgroup_mask_lists[group_idx] = {
            'outer_mask': target_mask,
            'submasks': post_filtered_masks,
            'points': points,
        }
    return subgroup_mask_lists

# Rest of rounds

def compute_subgroups(
    group_mask_item: GroupItem, 
    contained_in_thresh: float = 0.90, 
    outer_sim_thresh: float = 0.77, 
    mutual_sim_thresh: float = 0.85, 
    retain_best: bool = False,
) -> GroupDict:
    """
    Given an item of masks, break the existing masks into the highest level split and 
    create a new GroupDict
    """
    submasks = group_mask_item['submasks']
    if len(submasks) == 0:
        return {}
    
    # filter similar masks 
    outer_mask = group_mask_item['outer_mask']
    group_mask_list: List[EfficientMask] = []
    if not SKIP_LOGGING:
        print(f"Filtering {len(submasks)} submasks:")
    for mask_elem in tqdm(submasks, disable=SKIP_LOGGING):
        if mask_elem.get_size() == 0:
            continue
        # Filter out too similar to outer mask
        pos_mask = mask_elem.intersect(outer_mask)
        if pos_mask.near_equivalent_to(outer_mask, thresh=outer_sim_thresh):
            continue
        # Filter out too similar to another (smaller) mask.
        too_similar = False
        for existing_mask in group_mask_list:
            if mask_elem.near_equivalent_to(existing_mask, thresh=mutual_sim_thresh):
                # take the better scoring mask
                if retain_best and mask_elem.dense_score() > existing_mask.dense_score():
                    existing_mask.set_to(mask_elem)
                too_similar = True
                break
            elif (
                (not mask_elem.mostly_contained_in(existing_mask, thresh=contained_in_thresh)) 
                and mask_elem.overlaps_threshold(existing_mask, thresh=0.7)
            ):
                existing_mask.set_to(mask_elem.union(existing_mask))
                too_similar = True
                break
        if not too_similar:
            group_mask_list.append(mask_elem)
    
    if len(group_mask_list) == 0:
        return {}
    
    # Group masks into submask categories
    new_groups: GroupDict = {}
    group_idx = 0
    if not SKIP_LOGGING:
        print(f"Grouping {len(group_mask_list)} submasks:")
    for mask_item in tqdm(group_mask_list, disable=SKIP_LOGGING):
        found = False
        for group in new_groups.values():
            if mask_item.mostly_contained_in(group['outer_mask'], thresh=contained_in_thresh):
                group['submasks'].append(mask_item)
                found = True
                break
        if not found:
            new_groups[group_idx] = {
                'outer_mask': mask_item,
                'submasks': [],
                'points': [],
            }
            group_idx += 1

    # remove the smallest masks from the larger ones
    if not SKIP_LOGGING:
        print(f"Cleaning separation of {group_idx} groups:")
    for small_idx in tqdm(reversed(range(group_idx)), disable=SKIP_LOGGING):
        for big_idx in range(group_idx):
            if big_idx == small_idx:
                break
            smaller_group = new_groups[small_idx]
            bigger_group = new_groups[big_idx]
            bigger_group['outer_mask'] = bigger_group['outer_mask'].subtract(smaller_group['outer_mask'])
            bigger_group['submasks'] = [
                m.subtract(smaller_group['outer_mask']) for m in bigger_group['submasks']
            ]
    
    all_masks = all_mask_union([g['outer_mask'].mask for g in new_groups.values()])
    empty_mask = outer_mask.subtract(EfficientMask(all_masks, 0))
    if not empty_mask.near_equivalent_to(outer_mask, thresh=outer_sim_thresh):
        new_groups['empty'] = {'outer_mask': empty_mask, 'submasks': [], 'points': []}
    
    return new_groups

def add_points_in_mask(
    predictor: "SamPredictor", 
    image: np.ndarray, 
    item: GroupItem, 
    score_cutoff: float = 0.7,
    num_points = 5,
) -> GroupItem:
    """Pick some points inside this mask, then add them to this GroupItem"""
    top_left, bottom_right = item['outer_mask'].get_tlbr()
    (t, l), (b, r) = top_left, bottom_right
    step = int(min(b-t, r-l)/3)
    if step == 0:
        return item
    points = get_grid(step, top_left, bottom_right)
    random.shuffle(points)
    points = points[:num_points]
    point_submasks = predict_for_bounded_points(predictor, image, points, item['outer_mask'])
    for more_submasks in point_submasks.values():
        for mask in more_submasks:
            if mask.score < score_cutoff:
                continue
            if mask.get_density() < 0.15:
                continue
            item['submasks'].append(mask.intersect(item['outer_mask']))
    item['submasks'] = sorted(item['submasks'], key=lambda x: x.get_size(), reverse=True)
    return item

def compute_subgroup_recursively(
    predictor: "SamPredictor", 
    image: np.ndarray, 
    group_mask_item: GroupItem, 
    score_cutoff: float = 0.7, 
    contained_in_thresh: float = 0.90, 
    outer_sim_thresh: float = 0.77, 
    mutual_sim_thresh: float = 0.85, 
    retain_best: bool = False, 
    depth: int = 0,
) -> FinalGroup:
    final_subgrouping: FinalGroup = {
        'outer_mask': group_mask_item['outer_mask'],
        'subgroups': {}
    }
    if depth < 6 and group_mask_item['outer_mask'].get_density() > 0.75:
        group_mask_item = add_points_in_mask(predictor, image, group_mask_item, score_cutoff=score_cutoff)
    subgroup_mapping = compute_subgroups(
        group_mask_item, 
        contained_in_thresh=contained_in_thresh, 
        outer_sim_thresh=outer_sim_thresh, 
        mutual_sim_thresh=mutual_sim_thresh, 
        retain_best=retain_best,
    )
    if not SKIP_LOGGING:
        print(f"d-{depth}:{len(group_mask_item['submasks'])}:{len(subgroup_mapping)}", end =" ")
    if len(subgroup_mapping) != 0:
        final_subgrouping['subgroups'] = {
            idx: compute_subgroup_recursively(
                predictor,
                image,
                subgroup_mapping[idx],
                contained_in_thresh=contained_in_thresh, 
                outer_sim_thresh=outer_sim_thresh, 
                mutual_sim_thresh=mutual_sim_thresh, 
                retain_best=retain_best,
                depth = depth + 1
            ) for idx in subgroup_mapping.keys()
        }
    return final_subgrouping

def compute_group_tree(
    predictor: "SamPredictor", 
    image: np.ndarray, 
    score_cutoff: float = 0.7, 
    contained_in_thresh: float = 0.9, 
    outer_sim_thresh: float = 0.8, 
    mutual_sim_thresh: float = 0.9, 
    retain_best: bool = False,
) -> FinalGrouping:
    """Compute a full mask tree for the given image"""
    blur_image = cv2.bilateralFilter(image, 11, 61, 39)
    predictor.set_image(blur_image)
    result = predict_all(predictor, image, step=150)
    processed_results = process_best_largest(result)
    groups = first_iteration_groups(predictor, processed_results, step=150)
    canny_masks = get_canny_masks(predictor, image, distance_threshold=30, jitter_amount=20)
    subgroup_mask_lists = get_subgroup_mask_lists(groups, result, canny_masks, score_cutoff=score_cutoff, retain_best=retain_best)
    full_group_tree = {idx: compute_subgroup_recursively(
        predictor,
        image,
        subgroup_mask_lists[idx],
        score_cutoff=score_cutoff,
        contained_in_thresh=contained_in_thresh, 
        outer_sim_thresh=outer_sim_thresh, 
        mutual_sim_thresh=mutual_sim_thresh, 
        retain_best=retain_best,
    ) for idx in subgroup_mask_lists.keys()}

    return full_group_tree
