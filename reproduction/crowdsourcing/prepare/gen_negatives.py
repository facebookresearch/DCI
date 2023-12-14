#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

from long_captions.dense_image import DenseCaptionedImage, get_key_for
from long_captions.prepare.remote_language_model import RemoteLanguageModel
from long_captions.utils import get_clip_token_length
from long_captions.config import OUTPUT_SUMMARY_PATH, OUTPUT_NEGATIVE_PATH

import os
import json
import threading
from queue import Queue
from typing import Dict, Any, Optional, Tuple

IDX_TARGET = 8028
DEBUG = False
SKIP_EXISTING = False
NUM_THREADS = 40

def get_n_negatives_per(
        input_caption: str, 
        rlm: RemoteLanguageModel, 
        n: int = 3, 
        max_fails: int = 20
):
    if DEBUG:
        print(input_caption)

    res = {
        'basic': [],
        'layout': [],
        'swaps': [],
    }
    targets = list(res.keys()) * n
    failures = 0

    caption_tokens = get_clip_token_length(input_caption)

    for target in targets:
        added = False
        while added is False:
            try:
                negative_caption = rlm.get_negative(input_caption, target)
                if DEBUG:
                    print(negative_caption)
                assert get_clip_token_length(negative_caption) > caption_tokens - 15, "Too short, retry outright"
                assert get_clip_token_length(negative_caption) < caption_tokens + 50, "Too long, retry outright"
                negative_caption_reduced = rlm.reduce_length(negative_caption, target_tokens=caption_tokens+5)
                res[target].append(negative_caption_reduced)
                added = True
            except Exception as e:
                if DEBUG:
                    import traceback
                    traceback.print_exc()
                print('e', end='', flush=True)
                failures += 1
                if failures >= max_fails:
                    raise
    if DEBUG:
        print(res)
    return res


def negatives_len_fit(negs):
    for subkeys in negs.values():
        for generated_negative in subkeys:
            if get_clip_token_length(generated_negative) > 77:
                return False
    return True

def gen_negatives_for(
        idx: int, 
        rlm: RemoteLanguageModel, 
        n: int = 3, 
        max_fails: int = 14, 
        existing: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[Dict[str, Any]], bool, bool]:
    failures = 0
    dci = DenseCaptionedImage(idx)
    had_edit = False

    if existing is None:
        negatives = {
            'base': None,
        }
    else:
        negatives = existing

    # load summaries
    key = get_key_for(idx)
    summary_path = os.path.join(OUTPUT_SUMMARY_PATH, key)
    with open(summary_path) as jsonf:
        summaries = json.load(jsonf)

    # get base negative
    while negatives['base'] is None or not negatives_len_fit(negatives['base']):
        try:
            negatives['base'] = get_n_negatives_per(summaries['base'], rlm, n)
            had_edit = True
        except Exception as e:
            if DEBUG:
                import traceback
                traceback.print_exc()
            print('e', end='', flush=True)
            failures += 1
            if failures >= max_fails:
                return negatives, had_edit, False

    # get negatives for all masks
    all_masks = dci.get_all_masks()
    for m in all_masks:
        entry = dci.get_caption_with_subcaptions(m)[0]
        key = entry['key']
        caption = entry['caption']
        if key in negatives:
            if negatives_len_fit(negatives[key]):
                continue # We've already done this one
        if key in summaries:
            caption = summaries[key]
        else:
            continue # we'll do all negatives when we have all summaries

        res = None
        while res is None:
            try:
                res = get_n_negatives_per(caption, rlm, 1)
                negatives[key] = res
                had_edit = True
            except Exception as e:
                if DEBUG:
                    import traceback
                    traceback.print_exc()
                print('e', end='', flush=True)
                failures += 1
                if failures >= max_fails:
                    return negatives, had_edit, False

    return negatives, had_edit, True


def thread_entry(idx_pool: Queue, rlm: RemoteLanguageModel):
    while not idx_pool.empty():
        i, existing = idx_pool.get()
        key = get_key_for(i)
        if int(i) % 500 == 0:
            print(f"\n{i}") 
        target_path = os.path.join(OUTPUT_NEGATIVE_PATH, key)
        negatives, edited, success = gen_negatives_for(i, rlm, existing=existing)
        
        if edited:
            with open(target_path, 'w+') as jsonf:
                json.dump(negatives, jsonf)
            if success is False:
                print('F', end='', flush=True)
            else:
                print('S', end='', flush=True)
                with open('new_negs.out', 'a+') as neg_file:
                    neg_file.write(target_path + "/n")
        else:
            print('n', end='', flush=True)

        continue

def run_gen_summary():
    # get model connection
    rlm = RemoteLanguageModel('meta-llama/Llama-2-70b-hf')

    idx_pool = Queue()

    # generate summaries
    for i in range(IDX_TARGET):
        key = get_key_for(i)
        target_path = os.path.join(OUTPUT_NEGATIVE_PATH, key)
        existing = None
        if os.path.exists(target_path):
            if SKIP_EXISTING:
                print('s', end='', flush=True)
                continue
            else:
                with open(target_path) as jsonf:
                    existing = json.load(jsonf)

        summary_path = os.path.join(OUTPUT_SUMMARY_PATH, key)
        if not os.path.exists(summary_path):
            print('S', end='', flush=True)
            continue
    
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