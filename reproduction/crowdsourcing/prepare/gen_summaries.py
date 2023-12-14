#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

from long_captions.dense_image import DenseCaptionedImage, get_key_for
from long_captions.prepare.remote_language_model import RemoteLanguageModel
from long_captions.utils import get_clip_token_length, truncate_long_captions
from long_captions.config import OUTPUT_SUMMARY_PATH

import os
import json
import threading
from queue import Queue
from typing import Dict, Any, Optional, Tuple

IDX_TARGET = 8028
DEBUG = False
SKIP_EXISTING = False
NUM_THREADS = 40

def gen_summaries_for(idx, rlm, max_fails=500, existing=None) -> Tuple[Optional[Dict[str, Any]], bool, bool]:
    failures = 0
    dci = DenseCaptionedImage(idx)
    if existing is not None:
        summaries = existing
    else:
        summaries = {
            'base': None,
        }
    created_any = False
    # get base summary
    while summaries['base'] is None or get_clip_token_length(summaries['base']) > 77:
        try:
            base_caption = dci.get_formatted_complete_description()[0]['caption']
            base_caption = truncate_long_captions(base_caption)
            summary_caption = rlm.get_summary(base_caption)
            assert get_clip_token_length(summary_caption) < 140, "Too long, retry outright"
            summary_caption_reduced = rlm.reduce_length(summary_caption)
            summaries['base'] = summary_caption_reduced
            created_any = True
        except AssertionError as e:
            if DEBUG:
                import traceback
                traceback.print_exc()
            print('e', end='', flush=True)
            failures += 1
            if failures >= max_fails:
                return summaries, created_any, False

    # get summaries for long masks
    all_masks = dci.filter_masks_by_size()
    for m in all_masks:
        entry = dci.get_caption_with_subcaptions(m)[0]
        key = entry['key']
        caption = entry['caption']
        if key in summaries and get_clip_token_length(summaries[key]) <= 77:
            continue

        mask_sum_caption = None
        short = get_clip_token_length(caption) <= 75

        while mask_sum_caption is None:
            try:
                caption = truncate_long_captions(caption)
                mask_sum_caption_long = rlm.get_summary(caption, short=short)
                assert get_clip_token_length(mask_sum_caption_long) < 140, "Too long, retry outright"
                mask_sum_caption = rlm.reduce_length(mask_sum_caption_long)
                summaries[key] = mask_sum_caption
                created_any = True
            except AssertionError as e:
                if DEBUG:
                    import traceback
                    traceback.print_exc()
                print('e', end='', flush=True)
                failures += 1
                if failures >= max_fails:
                    return summaries, created_any, False

    if DEBUG:
        print(summaries)
    return summaries, created_any, True

def thread_entry(idx_pool: Queue, rlm):
    while not idx_pool.empty():
        i, existing = idx_pool.get()
        key = get_key_for(i)
        if int(i) % 500 == 0:
            print(f"\n{i}") 
        target_path = os.path.join(OUTPUT_SUMMARY_PATH, key)
        summaries, created_any, success = gen_summaries_for(i, rlm, existing=existing)
        if created_any is False:
            print('n', end='', flush=True)
        else:
            if success is False:
                print('F', end='', flush=True)
            with open(target_path, 'w+') as jsonf:
                json.dump(summaries, jsonf)
            print('S', end='', flush=True)
        continue

def run_gen_summary():
    # get model connection
    rlm = RemoteLanguageModel('meta-llama/Llama-2-70b-hf')

    idx_pool = Queue()

    # generate summaries
    for i in range(IDX_TARGET):
        key = get_key_for(i)
        target_path = os.path.join(OUTPUT_SUMMARY_PATH, key)
        existing = None
        if os.path.exists(target_path):
            if SKIP_EXISTING:
                print('s', end='', flush=True)
                continue
            else:
                with open(target_path) as jsonf:
                    existing = json.load(jsonf)
    
        idx_pool.put((i, existing))
    
    thread_pool = []
    for _ in range(NUM_THREADS):
        t = threading.Thread(target=thread_entry, args=(idx_pool, rlm))
        t.start()
        thread_pool.append(t)
    
    for thread in thread_pool:
        thread.join()


if __name__ == '__main__':
    run_gen_summary()