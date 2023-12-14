#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import os
import yaml

DENSE_CAPS_DATASET_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DENSE_CAPS_DIR = os.path.dirname(DENSE_CAPS_DATASET_PACKAGE_DIR)
CONFIG_FILE = os.path.join(DENSE_CAPS_DATASET_PACKAGE_DIR, 'config.yaml')

def init_config():
    config = {'initialized': True}
    print("Initializing the dense captions dataset")
    default_data_dir = os.path.join(DENSE_CAPS_DIR, 'data')
    data_dir = input(f"Data directory: (default {default_data_dir})\n>> ")
    if data_dir.strip() == "":
        config['data_dir'] = default_data_dir
    else:
        config['data_dir'] = data_dir.strip()
    with open(CONFIG_FILE, 'w+') as config_file:
        yaml.dump(config, config_file)
    return config

def get_config():
    if not os.path.exists(CONFIG_FILE):
        return init_config()
    with open(CONFIG_FILE, 'r') as config_file:
        config = yaml.safe_load(config_file)
    return config


DCI_CONFIG = get_config()
DATASET_BASE = os.path.join(DCI_CONFIG['data_dir'], 'densely_captioned_images')
MODEL_BASE = os.path.join(DENSE_CAPS_DIR, 'models')
DATASET_PHOTO_PATH = os.path.join(DATASET_BASE, 'photos')
DATASET_ANNOTATIONS_PATH = os.path.join(DATASET_BASE, 'annotations')
DATASET_COMPLETE_PATH = os.path.join(DATASET_BASE, 'complete')
