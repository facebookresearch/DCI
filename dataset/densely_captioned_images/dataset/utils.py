#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

from transformers import CLIPProcessor

# Utilities

processor = None
def get_clip_processor() -> CLIPProcessor:
    global processor
    if processor is None:
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return processor

def get_clip_token_length(in_str: str) -> int:
    processor = get_clip_processor()
    return len(processor.tokenizer(in_str)['input_ids'])


def truncate_long_captions(in_str: str, max_len: int = 2000):
    clip_len = get_clip_token_length(in_str)
    if clip_len > max_len:
        full_len = len(in_str)
        target_len = int((max_len / clip_len) * full_len)
        in_str = in_str[:target_len]
    return in_str

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )
