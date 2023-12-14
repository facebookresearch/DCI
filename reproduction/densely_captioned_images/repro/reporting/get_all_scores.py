# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import parse
import os

FORMAT_STRING = """{}
[-] Loading from LoRA weights: {}
{}
VG-Relation Macro Accuracy: {}
VG-Attribution Macro Accuracy: {}
COCO Precision@1: {}
Flickr Precision@1: {}
[-] Running VLC
Scores:
{}
Overall Scores:
[({}, 'Object'), ({}, 'Attribute'), ({}, 'Relation')]
[-] Running Winoground
text score: {}
image score: {}
group score: {}
[-] Running Dense Cap
Test: all_swaps. CLIP Correct: {} neg_correct {}
Test: all_swaps_pick5. CLIP Correct: {} neg_correct {}
Test: base_swaps. CLIP Correct: {} neg_correct {}
Test: all_hardest. CLIP Correct: {} neg_correct {}
{}"""


def extract_numbers(fpath):
    with open(fpath) as f:
        dat = f.read()
        parsed = parse.parse(FORMAT_STRING, dat)
    mn = parsed[1]
    vgr = parsed[3]
    vga = parsed[4]
    coco = parsed[5]
    flickr = parsed[6]
    vlco = parsed[8]
    vlca = parsed[9]
    vlcr = parsed[10]
    wgt = parsed[11]
    wgi = parsed[12]
    wgg = parsed[13]
    dcas = parsed[14]
    dcasn = parsed[15]
    dcp5 = parsed[16]
    dcp5n = parsed[17]
    dcbn = parsed[19]
    dch = parsed[20]
    dchn = parsed[21]
    print(f"{mn}\n{vgr},{vga},{coco},{flickr},{dcas},{dcasn},{dcp5},{dcp5n},{dcbn},{dchn},{vlco},{vlca},{vlcr},0,0,{wgt},{wgi},{wgg}\n")

def print_all_scores():
    evals_dir = input("Provide full path to eval sweep output logdir >> ")
    for subdir in os.listdir(evals_dir):
        outpath = os.path.join(evals_dir, subdir)
        outfile = os.path.join(outpath, [n for n in os.listdir(outpath) if ".out" in n][0])
        extract_numbers(outfile)

if __name__ == '__main__':
    print_all_scores()
