#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

# Download images for HAKE Dataset
import os

import os
from urllib.request import urlretrieve
from tqdm import tqdm
import json as js


def image_url_download(url_file, to_folder):
    count = 0
    if not os.path.exists(to_folder):
        os.mkdir(to_folder)
    contents = js.load(open(url_file, 'r'))
    for img in tqdm(contents):
        if not os.path.exists(os.path.join(to_folder, img)):
            try:
                urlretrieve(contents[img], os.path.join(to_folder, img))
            except:
                count += 1
    print(count)


if __name__=='__main__':
    import sys
    if len(sys.argv) != 3:
        print("Usage: python download.py url_base imgname to_folder")
    else:
        image_url_download(sys.argv[1], sys.argv[2])