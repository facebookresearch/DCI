
#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import torch
import os
import sys
from densely_captioned_images.repro.config import MODEL_PATH, ELEVATER_MODEL_CONFIG, ELEVATER_DATASET_CONFIG_ROOT, ELEVATER_DATASET_ROOT, EVAL_LOG_PATH
from densely_captioned_images.repro.eval.ElevaterIC.vision_benchmark.commands.linear_probe import main as linear_probe
from densely_captioned_images.repro.eval.ElevaterIC.vision_benchmark.commands.zeroshot import main as zeroshot

DATASETS_LIST = [
    # 'imagenet-1k', TODO download?
    'cifar100', 'caltech101', 'cifar10',  'country211', 
    'dtd', 'eurosat-clip', 'fer2013', 'fgvc-aircraft-2013b', 
    'food101', 'gtsrb', 'hateful-memes', 'kitti-distance', 'mnist', 
    'flower102', 'oxford-iiit-pets', 'patchcamelyon', 'rendered-sst2', 
    'resisc45-clip', 'stanfordcar', 'voc2007classification'
]

DATASETS_NAME_MAP = {
    # 'imagenet-1k', TODO download?
    'cifar100': 'cifar-100', 'caltech101': 'caltech-101', 'cifar10': 'cifar-10',  'country211': 'country211', 
    'dtd': 'dtd', 'eurosat-clip': 'eurosat_clip', 'fer2013': 'fer-2013', 'fgvc-aircraft-2013b': 'fgvc-aircraft-2013b-variants102', 
    'food101': 'food-101', 'gtsrb': 'gtsrb', 'hateful-memes': 'hateful-memes', 'kitti-distance': 'kitti-distance', 'mnist': 'mnist', 
    'flower102': 'oxford-flower-102', 'oxford-iiit-pets': 'oxford-iiit-pets', 'patchcamelyon': 'patch-camelyon', 'rendered-sst2': 'rendered-sst2', 
    'resisc45-clip': 'resisc45_clip', 'stanfordcar': 'stanford-cars', 'voc2007classification': 'voc-2007-classification'
}

SHOT_OPTIONS = [
    0, 5, 20, 50, -1
]

def run_elevater_on(
        model, 
        path_key, 
        run_full=False, 
        do_finetune=True, 
        shot_options=None, 
        dataset_option=None,
):
    submodel_path = os.path.join(MODEL_PATH, path_key, 'compiled.pth')
    if not os.path.exists(submodel_path):
        torch.save(model.state_dict(), submodel_path)
    run_elevater(
        submodel_path, 
        path_key, 
        run_full=run_full, 
        do_finetune=do_finetune, 
        shot_options=shot_options, 
        dataset_option=dataset_option,
    )

def run_elevater(
        model_path, 
        path_key, 
        run_full=False, 
        do_finetune=True, 
        shot_options = None, 
        rerun_existing=False, 
        dataset_option=None
):
    if shot_options is None:
        shot_options = SHOT_OPTIONS
    use_datasets = DATASETS_LIST
    if dataset_option is not None:
        use_datasets = [use_datasets[dataset_option]]
    if run_full is False:
        shot_options = shot_options[:1]
        use_datasets = use_datasets[:1]
    
    def dataset_already_run_for_all_shots(d):
        for shot in shot_options:
            if shot == 0:
                subpath = "zeroshot_eval_wiki_False_wnh_False_wnd_False_gpt3_Falseagg_WIKI_AND_GPT3_gpt3count_0"
            elif shot == -1:
                subpath = 'linear_probe_full'
            else:
                subpath = f'linear_probe_{shot}'
            full_path = os.path.join(EVAL_LOG_PATH, 'elevater', path_key, 'predictions', subpath)
            if not os.path.exists(full_path):
                return False
            all_tasks = '\n'.join(os.listdir(full_path))
            if DATASETS_NAME_MAP[d] not in all_tasks:
                return False
        return True
    
    if not rerun_existing:
        use_datasets = [d for d in use_datasets if not dataset_already_run_for_all_shots(d)]

    print(f"Running elevater on {len(use_datasets)} datasets and {len(shot_options)} shot options")
    os.makedirs(f'{EVAL_LOG_PATH}/elevater/{path_key}/', exist_ok=True)
    for dataset in use_datasets:
        for num_shots in shot_options:
            if num_shots != 0:
                args_list = [
                    '--model', ELEVATER_MODEL_CONFIG, 
                    '--ds', f'{ELEVATER_DATASET_CONFIG_ROOT}/{dataset}.yaml',
                    '--no-tuning', f'{not do_finetune}',
                    '--lr', '1e-6',
                    '--l2', '1e-5',
                    'MODEL.CLIP_FP32', 'True',
                    'DATASET.NUM_SAMPLES_PER_CLASS', f'{num_shots}',
                    'DATASET.ROOT', f'{ELEVATER_DATASET_ROOT}/',
                    'OUTPUT_DIR', f'{EVAL_LOG_PATH}/elevater/{path_key}/',
                    'DATASET.RANDOM_SEED_SAMPLING', '0',
                    'TRAIN.FREEZE_IMAGE_BACKBONE', 'True',
                    'TRAIN.INIT_HEAD_WITH_TEXT_ENCODER', 'True', 
                    'TRAIN.MERGE_ENCODER_AND_HEAD_PROJ', 'False',
                    'KNOWLEDGE.WORDNET.USE_HIERARCHY', 'False',
                    'KNOWLEDGE.WORDNET.USE_DEFINITION', 'False',
                    'KNOWLEDGE.WIKITIONARY.USE_DEFINITION', 'False',
                    'KNOWLEDGE.GPT3.USE_GPT3', 'False',
                    'KNOWLEDGE.AGGREGATION.NUM_GPT3_ITEMS', '0',
                    'TEST.MODEL_FILE', f'{model_path}', 
                ]

                saved_argv = sys.argv
                try:
                    sys.argv = ['./vision_benchmark/commands/linear_probe.py'] + args_list
                    linear_probe()
                finally:
                    sys.argv = saved_argv
            else:
                args_list = [
                    '--model', ELEVATER_MODEL_CONFIG, 
                    '--ds', f'{ELEVATER_DATASET_CONFIG_ROOT}/{dataset}.yaml',
                    'MODEL.CLIP_FP32', 'True',
                    'DATASET.NUM_SAMPLES_PER_CLASS', f'{num_shots}',
                    'DATASET.ROOT', f'{ELEVATER_DATASET_ROOT}/',
                    'OUTPUT_DIR', f'{EVAL_LOG_PATH}/elevater/{path_key}/',
                    'DATASET.RANDOM_SEED_SAMPLING', '0',
                    'TRAIN.FREEZE_IMAGE_BACKBONE', 'True',
                    'TRAIN.INIT_HEAD_WITH_TEXT_ENCODER', 'True', 
                    'TRAIN.MERGE_ENCODER_AND_HEAD_PROJ', 'False',
                    'KNOWLEDGE.WORDNET.USE_HIERARCHY', 'False',
                    'KNOWLEDGE.WORDNET.USE_DEFINITION', 'False',
                    'KNOWLEDGE.WIKITIONARY.USE_DEFINITION', 'False',
                    'KNOWLEDGE.GPT3.USE_GPT3', 'False',
                    'KNOWLEDGE.AGGREGATION.NUM_GPT3_ITEMS', '0',
                    'TEST.MODEL_FILE', f'{model_path}', 
                ]

                saved_argv = sys.argv
                try:
                    sys.argv = ['./vision_benchmark/commands/zeroshot.py'] + args_list
                    zeroshot()
                finally:
                    sys.argv = saved_argv


def run_elevator_from_lora_checkpoint(lora_path_key, run_full=False, shot_options=None):
    from transformers import CLIPModel
    from peft import PeftModel

    if shot_option is not None and isinstance(shot_option, int):
        shot_option = [shot_option]
    submodel_path = os.path.join(MODEL_PATH, lora_path_key)
    base_clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    loaded = PeftModel.from_pretrained(base_clip_model, submodel_path)
    loaded = loaded.merge_and_unload()
    run_elevater_on(loaded, lora_path_key, run_full=run_full, shot_options=shot_options)


if __name__ == '__main__':
    clip_path = os.path.join(MODEL_PATH, 'CLIP.pth')
    if not os.path.exists(clip_path):
        from transformers import CLIPModel

        base_clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        torch.save(base_clip_model.state_dict(), clip_path)

    run_elevater(clip_path, 'clip-baseline', run_full=True, shot_options=[0, 5])
