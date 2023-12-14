#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
from long_captions.prepare.remote_language_model import RemoteLanguageModel
from long_captions.dense_image import DenseCaptionedImage, get_key_for, get_dci_count
from long_captions.config import DATASET_COMPLETE_PATH

import threading
from queue import Queue
from typing import Tuple, Dict, List

NUM_THREADS = 40

def get_new_summaries(
        rlm: RemoteLanguageModel, dci: DenseCaptionedImage, max_retry = 15
        ) -> Tuple[Dict[str, List[str]], bool, bool]:
    did_nothing = True

    summaries = dci.get_summaries()

    while not isinstance(summaries['base'], list):
        try:
            summaries['base'] = [summaries['base']] + rlm.get_new_captions_for_dci(dci)
            did_nothing = False
        except AssertionError:
            max_retry -= 1
            if max_retry == 0:
                return summaries, did_nothing, False
    
    all_masks = dci.filter_masks_by_size()
    for m in all_masks:
        key = f'm-{m["idx"]}-sc'
        while not isinstance(summaries[key], list):
            try:
                summaries[key] = [summaries[key]] + rlm.get_new_captions_for_mask(dci, m)
                did_nothing = False
            except AssertionError:
                max_retry -= 1
                if max_retry == 0:
                    return summaries, did_nothing, False

    return summaries, did_nothing, True


def thread_entry(idx_pool: Queue, rlm):
    while not idx_pool.empty():
        i = idx_pool.get()
        if int(i) % 500 == 0:
            print(f"\n{i}") 
        dci = DenseCaptionedImage(i)
        key = get_key_for(i)
        source_path = os.path.join(DATASET_COMPLETE_PATH, key)
        if not os.path.exists(source_path):
            continue # not a complete entry
        new_summaries, did_nothing, success = get_new_summaries(rlm, dci)
        if did_nothing is True:
            print(".", end='', flush=True)
        else:
            with open(source_path, 'r') as jsonf:
                orig_data = json.load(jsonf)
            orig_data['summaries'] = new_summaries

            if success is False:
                print('F', end='', flush=True)
            with open(source_path, 'w+') as jsonf:
                json.dump(orig_data, jsonf)
            print('S', end='', flush=True)
        continue

def run_gen_additional_summary():
    # get model connection
    rlm = RemoteLanguageModel('meta-llama/Llama-2-70b-hf')

    idx_pool = Queue()
    count = get_dci_count()

    # generate summaries
    for i in range(count):
        idx_pool.put(i)
    
    thread_pool = []
    for _ in range(NUM_THREADS):
        t = threading.Thread(target=thread_entry, args=(idx_pool, rlm))
        t.start()
        thread_pool.append(t)
    
    for thread in thread_pool:
        thread.join()


if __name__ == '__main__':
    run_gen_additional_summary()