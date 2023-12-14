#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

from uuid import uuid4

import os
import json
from transformers import CLIPProcessor, CLIPModel
from densely_captioned_images.repro.eval.VLChecklist.vl_checklist.evaluate import Evaluate
from densely_captioned_images.repro.config import VLC_ROOT_PATH, EVAL_LOG_PATH
from densely_captioned_images.repro.eval.clip_vlc_wrap import VLCtoHFCLIPWrap


ATTRIBUTE_YAML = """
MAX_NUM: 2000
MODEL_NAME: "{model_name}"
BATCH_SIZE: 4
TASK: "{task}"
DATA:
  TYPES: ["Attribute/action", "Attribute/color", "Attribute/material", "Attribute/size", "Attribute/state"]
  TEST_DATA: ["vg","vaw"]
OUTPUT:
  DIR: "{output_dirname}"
  NUM: 5
"""

OBJECT_YAML = """
MAX_NUM: 2000
MODEL_NAME: "{model_name}"
BATCH_SIZE: 4
TASK: "{task}"
DATA:
  TYPES: ["Object/Location/center", "Object/Location/margin", "Object/Location/mid", "Object/Size/large", "Object/Size/medium", "Object/Size/small"]
  TEST_DATA: ["hake","swig_agent","swig_destination", "swig_item", "swig_tool", "vg_obj", "vg_subj"]
OUTPUT:
  DIR: "{output_dirname}"
  NUM: 5
"""

RELATION_SPATIAL_YAML = """
MAX_NUM: 2000
MODEL_NAME: "{model_name}"
BATCH_SIZE: 4
TASK: "{task}"
DATA:
  TYPES: ["Relation/action"]
  TEST_DATA: ["vg", "swig", "hake"]
OUTPUT:
  DIR: "{output_dirname}"
  NUM: 0
"""

RELATION_ACTION_YAML = """
MAX_NUM: 2000
MODEL_NAME: "{model_name}"
BATCH_SIZE: 4
TASK: "{task}"
DATA:
  TYPES: ["Relation/spatial"]
  TEST_DATA: ["vg"]
OUTPUT:
  DIR: "{output_dirname}"
  NUM: 0
"""


CORPUS_PATH = os.path.join(VLC_ROOT_PATH, 'corpus.json')
LOG_PATH = os.path.join(EVAL_LOG_PATH, 'vlc')

def score_vlc(model_name, swig_only=False):
    m = json.load(open(CORPUS_PATH))
    if swig_only:
        m = {
            task_name: {
                ds_name: ds_split 
                for ds_name, ds_split in task_split.items() 
                if 'swig' in ds_name
            } for task_name, task_split in m.items()
        }
    score_list = []
    filepath = os.path.join(LOG_PATH, model_name, 'itc')
    for item in m.keys():
        data_num = len(m[item].keys())
        data_score = []
        if data_num == 0:
            score_list.append(0)
            continue
        for data in m[item].keys():
            score = 0
            file_num = len(m[item][data])
            if file_num == 0:
                data_score.append(0)
                continue
            for file in m[item][data]:
                json_name = os.path.join(filepath,f"{file}.json")
                if not os.path.exists(json_name):
                    print(f"{file} has not been evaluated.")
                    return
                else:
                    m1 = json.load(open(json_name))
                    score += m1["total_acc"]
            data_score.append(score/file_num)
        score_list.append(sum(data_score)/data_num)
    print("Scores:")
    print(list(zip(score_list, ['O-Large', 'O-Medium', 'O-Small', 'O-Center', 'O-Mid', 'O-Margin', 'A-Color', 'A-Material', 'A-Size',"A-State", "A-Action", "R-action", "R-spatial"])))
    overall_scores = [sum(score_list[0:6])/6, sum(score_list[6:11])/5, sum(score_list[11:])/2]
    print("Overall Scores:")
    print(list(zip(overall_scores, ["Object", "Attribute", "Relation"])))


def run_vlc_on_model(model: CLIPModel, processor: CLIPProcessor, model_name=None):
    if model_name is None:
        model_name = str(uuid4())

    wrap_model = VLCtoHFCLIPWrap(model_name, model, processor)

    tasks = ['itc']
#    tasks = ['itm']
    output_dirname = os.path.join(EVAL_LOG_PATH, 'vlc', model_name)
    os.makedirs(output_dirname, exist_ok=True)
    for task in tasks:
        for BASE_YAML in [ATTRIBUTE_YAML, OBJECT_YAML, RELATION_SPATIAL_YAML, RELATION_ACTION_YAML]:
#        for BASE_YAML in [RELATION_ACTION_YAML]:
            yaml = BASE_YAML.format(model_name=model_name, task=task, output_dirname=output_dirname)
            yaml_path = os.path.join(EVAL_LOG_PATH, f"vlc-{model_name}-{task}.yaml")
            with open(yaml_path, 'w+') as yaml_file:
                yaml_file.write(yaml)

            evaluator = Evaluate(config_file=yaml_path, model=wrap_model)
            evaluator.start()
            os.unlink(yaml_path)
    score_vlc(model_name)


if __name__ == '__main__':
    from transformers import CLIPProcessor, CLIPModel
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    run_vlc_on_model(clip_model, clip_processor, model_name='clip-baseline')
