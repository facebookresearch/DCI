#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

"""
Utility script to print out all the elevater scores saved
"""

import os
import json


def print_all_elevater_scores():
    LOG_DIR = input("Input full dir for elevater sweep logs >> ")
    elevater_dir = LOG_DIR + '/elevater/{}/predictions'

    MODELS = os.listdir(os.path.join(LOG_DIR,'elevater'))
    for model_dir in MODELS:
        print(model_dir)
        full_model = elevater_dir.format(model_dir)
        for shot_dir in os.listdir(full_model):
            results = []
            full_shot_dir = os.path.join(full_model, shot_dir)
            for dataset_file in os.listdir(full_shot_dir):
                with open(os.path.join(full_shot_dir, dataset_file)) as jsonf:
                    dat = json.load(jsonf)
                results.append(dat["best_acc"])
            print(f"Shot dir: {shot_dir}")
            print(f"Avg Acc: {sum(results)/len(results)} over {len(results)}")
            print(f"Results: {' '.join([f'{r:0.1f}' for r in results])}")
        print()

if __name__ == '__main__':
    print_all_elevater_scores()
