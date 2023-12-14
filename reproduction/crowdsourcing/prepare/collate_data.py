#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
from long_captions.utils import get_clip_token_length
from long_captions.config import OUTPUT_SUMMARY_PATH, OUTPUT_NEGATIVE_PATH, DATASET_ANNOTATION_DUMP, DATASET_COMPLETE_PATH

"""
Helper script to load all of the generated and collected data into 
combined files for easy data loading.
"""

ANNOTATION_PATH = DATASET_ANNOTATION_DUMP

IDX_TARGET = 8028
DEBUG = False
SKIP_EXISTING = False
NUM_THREADS = 40

def ensure_size(text, max_size=77):
    text = text.strip()
    assert get_clip_token_length(text) <= max_size, f"Given text {text} over max size {max_size}"
    return text

def run_collate():
    """
    For every entry in the ANNOTATION_PATH, create a joined version if summaries and negatives exist
    """
    entries = os.listdir(ANNOTATION_PATH)
    for e in entries:
        try:
            source_path = os.path.join(ANNOTATION_PATH, e)
            summary_path = os.path.join(OUTPUT_SUMMARY_PATH, e)
            negative_path = os.path.join(OUTPUT_NEGATIVE_PATH, e)
            output_path = os.path.join(DATASET_COMPLETE_PATH, e)

            if os.path.exists(output_path):
                print('.', end='', flush=True)
                continue

            if not os.path.exists(summary_path):
                print('S', end='', flush=True)
                continue

            if not os.path.exists(negative_path):
                print('N', end='', flush=True)
                # os.unlink(summary_path)
                continue

            with open(source_path) as jsonf:
                source_dict = json.load(jsonf)
            
            with open(summary_path) as jsonf:
                summary_dict = json.load(jsonf)
                
            with open(negative_path) as jsonf:
                negative_dict = json.load(jsonf)
            
            if len(negative_dict) == 1:
                print('M', end='', flush=True)
                continue


            # Remove old mask labels
            if 'masks' in summary_dict:
                del summary_dict['masks']
            if 'masks' in negative_dict:
                del negative_dict['masks']

            # Run strip on all entries
            summary_dict = {k: ensure_size(summary) for k, summary in summary_dict.items()}
            negative_dict = {
                k: {
                    neg_type: [
                        ensure_size(n.strip()) for n in sub_negatives
                    ] for neg_type, sub_negatives in negatives.items()
                } for k, negatives in negative_dict.items()
            }

            source_dict['summaries'] = summary_dict
            source_dict['negatives'] = negative_dict

            assert len(source_dict['summaries']) == len(source_dict['negatives'])

            with open(output_path, 'w+') as jsonf:
                json.dump(source_dict, jsonf)
                print('+', end='', flush=True)
        except Exception as _:
            print('e', end='', flush=True)
            with open('missing.out', 'a+') as fn:
                fn.write(e + "\n")
            # if os.path.exists(summary_path):
            #     os.unlink(summary_path)

            # if os.path.exists(negative_path):
            #     os.unlink(negative_path)
            # print(f"Exception on entry {e}")
            # import traceback
            # traceback.print_exc()
            continue
        

if __name__ == '__main__':
    run_collate()