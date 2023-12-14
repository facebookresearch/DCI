#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import os
import yaml

DENSE_CAPS_REPRODUCTION_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DENSE_CAPS_DIR = os.path.dirname(DENSE_CAPS_REPRODUCTION_PACKAGE_DIR)
CONFIG_FILE = os.path.join(DENSE_CAPS_REPRODUCTION_PACKAGE_DIR, 'config.yaml')

def init_config():
    config = {'initialized': True}
    print("Initializing the dense captions reproduction")
    default_data_dir = os.path.join(DENSE_CAPS_DIR, 'data')
    data_dir = input(f"Data directory: (default {default_data_dir})\n>> ")
    if data_dir.strip() == "":
        config['data_dir'] = default_data_dir
    else:
        config['data_dir'] = data_dir.strip()

    coco_default = os.path.join(data_dir, 'hake/vcoco')
    coco_dir = input(f"COCO Dataset Path: (default {coco_default})\n>> ")
    if coco_dir.strip() == "":
        config['coco_dir'] = coco_default
    else:
        config['coco_dir'] = coco_dir.strip()
    aro_default = os.path.join(data_dir, 'ARO')
    aro_dir = input(f"ARO Dataset Path: (default {aro_default})\n>> ")
    if aro_dir.strip() == "":
        config['aro_dir'] = aro_default
    else:
        config['aro_dir'] = aro_dir.strip()

    vlc_default = os.path.join(DENSE_CAPS_REPRODUCTION_PACKAGE_DIR, 'eval', 'VLChecklist')
    vlc_dir = input(f"VLChecklist Install Path: (default {vlc_default})\n>> ")
    if vlc_dir.strip() == "":
        config['vlc_dir'] = vlc_default
    else:
        config['vlc_dir'] = vlc_dir.strip()

    ln_default = os.path.join(DENSE_CAPS_REPRODUCTION_PACKAGE_DIR, 'eval', 'localized_narratives')
    ln_dir = input(f"Localized Narratives Install Path: (default {ln_default})\n>> ")
    if ln_dir.strip() == "":
        config['ln_dir'] = ln_default
    else:
        config['ln_dir'] = ln_dir.strip()

    elevater_default = os.path.join(DENSE_CAPS_REPRODUCTION_PACKAGE_DIR, 'eval/ElevaterIC/vision_benchmark')
    elevater_dir = input(f"ElevaterIC Install Path: (default {elevater_default})\n>> ")
    if elevater_dir.strip() == "":
        config['elevater_dir'] = elevater_default
    else:
        config['elevater_dir'] = elevater_dir.strip()

    model_default = os.path.join(DENSE_CAPS_DIR, 'models')
    model_dir = input(f"Model Checkpoint Path: (default {model_default})\n>> ")
    if elevater_dir.strip() == "":
        config['model_dir'] = model_default
    else:
        config['model_dir'] = model_dir.strip()

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


EVAL_DATASET_PATH = DCI_CONFIG['data_dir']
VLC_ROOT_PATH = DCI_CONFIG['vlc_dir']
COCO_DIR = DCI_CONFIG['coco_dir']
ARO_DIR = DCI_CONFIG['aro_dir']
MODEL_PATH = DCI_CONFIG['model_dir']

ELEVATER_EVAL_ROOT = DCI_CONFIG['elevater_dir']
ELEVATER_MODEL_CONFIG = os.path.join(ELEVATER_EVAL_ROOT, 'resources/model/custom_clip.yaml')
ELEVATER_DATASET_CONFIG_ROOT = os.path.join(ELEVATER_EVAL_ROOT, 'resources/datasets')

ELEVATER_DATASET_ROOT = os.path.join(EVAL_DATASET_PATH, 'ElevaterDownload/classification/data')
LOCALIZED_NARRATIVES_DATAPATH =  os.path.join(DCI_CONFIG['ln_dir'], 'data')

COCO_TRAIN2017_DATAPATH = os.path.join(COCO_DIR, 'train2017')
COCO_TRAIN2017_ANNOTATION_PATH = os.path.join(COCO_DIR, 'annotations/captions_train2017.json')
COCO_VALID2017_DATAPATH = os.path.join(COCO_DIR, 'val2017')
COCO_VALID2017_ANNOTATION_PATH = os.path.join(COCO_DIR, 'annotations/captions_val2017.json')

EVAL_LOG_PATH = os.path.join(DENSE_CAPS_REPRODUCTION_PACKAGE_DIR, 'eval/logs')
