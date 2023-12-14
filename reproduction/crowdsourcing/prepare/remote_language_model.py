#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import json
import math
from typing import List, Dict, Any
import requests
from fastchat.model.model_adapter import get_conversation_template
from long_captions.utils import get_clip_token_length
from long_captions.dense_image import DenseCaptionedImage, MaskDataEntry

ALTER_PROMPTS = {
    'basic': (
        "You are given the description for an image. You should provide a mostly similar description, "
        "changing the original one slightly, but introducing enough significant differences such that "
        "the two descriptions could not possibly be for the same image. Keep the description length "
        "the same. Provide just the updated description."
    ),
    'layout': (
        "You are given the description for an image. You should provide a mostly similar description, "
        "with one part of it changed in a way that visibly alters the structure, layout, or "
        "content of the image. The change should introduce enough difference such that "
        "the two descriptions could not be for the same image. Keep the description length "
        "the same. Provide just the updated description. "
    ),
    'swaps': (
        "You are given a description. Selecting ONLY from the same words, "
        "construct a random response with the same words in a completely new order. The new description "
        "should not remain accurate to the first. "
        "Try to use all of the words from the original description, and keep the length the same, "
        "but use the words nearly randomly such that the scene makes less sense. Try to pair nouns "
        "and adjectives differently than in the original. "
    )
}

MULTI_CAPTION_PROMPT = (
    "You are providing descriptions of an image. The goal is to create "
    "a list of summarized descriptions that all accurately describe the same "
    "image, but may pay attention to slightly different details. A complete "
    "description will be provided. Complete 5 entries in this list, each a "
    "few sentences long. Provide in the format:\n1. <description>\n2. ..."
)


class RemoteLanguageModel:
    def __init__(
        self,
        model_path: str,
        worker_addr: str = "http://localhost:21002",
        debug: bool = False
    ) -> None:
        self.model_path = model_path
        self.worker_addr = worker_addr
        self.debug = debug

    def generate(
        self,
        message: str,
        system: str = None,
        temperature: float = 0.72, # 0.58
        max_new_tokens: int = 256
    ) -> Dict[str, Any]:
        conv = get_conversation_template(self.model_path)
        if system is not None:
            conv.system_message = f"{system}"
        conv.append_message(conv.roles[0], message)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        if self.debug:
            print(prompt)
        headers = {"User-Agent": "FastChat Client"}
        gen_params = {
            "model": self.model_path,
            "prompt": prompt,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "stop": conv.stop_str,
            "stop_token_ids": conv.stop_token_ids,
            "echo": False,
        }
        response = requests.post(
            self.worker_addr + "/worker_generate",
            headers=headers,
            json=gen_params
        )
        return json.loads(response.text)

    def get_short_enough_full_caption(self, dci: DenseCaptionedImage, tok_bound=1000) -> str:
        """Get a full captions for the image, within the token bounds"""
        full_caption = dci.get_formatted_complete_description()[0]['caption']
        while get_clip_token_length(full_caption) > tok_bound:
            # Chop lines off until it's the right size
            full_caption = "\n".join(full_caption.split("\n")[:-1])
        return full_caption

    def get_extracted_captions(self, resp) -> List[str]:
        _, content = resp.split("\n1. ")
        c1, content = content.split("\n2. ")
        c2, content = content.split("\n3. ")
        c3, content = content.split("\n4. ")
        c4, content = content.split("\n5. ")
        c5 = content.split("\n")[0]
        return [c1.strip(), c2.strip(), c3.strip(), c4.strip(), c5.strip()]

    def get_new_captions_for_dci(
            self, dci: DenseCaptionedImage, 
            max_len=77, target_count=8, max_retry=5) -> List[str]:
        new_captions = []
        full_caption = self.get_short_enough_full_caption(dci)
        while len(new_captions) < target_count and max_retry > 0:
            max_retry -= 1
            try:
                message = f"Full description: {full_caption}\n\nSummaries:"
                res = self.generate(
                    message=message,
                    system=MULTI_CAPTION_PROMPT,
                    max_new_tokens=1400,
                )
                for c in self.get_extracted_captions(res['text']):
                    if get_clip_token_length(c) < max_len:
                        new_captions.append(c)
            except:
                pass
        assert len(new_captions) > target_count
        return new_captions

    def get_short_enough_mask_caption(
            self, dci: DenseCaptionedImage, 
            mask: MaskDataEntry, tok_bound=1000) -> str:
        depth = dci._get_max_depth(mask)
        caption_length = 1000000
        while caption_length > tok_bound:
            full_caption = dci.get_caption_with_subcaptions(mask, max_depth=depth)[0]['caption']
            depth -= 1
            caption_length = get_clip_token_length(full_caption)
        return full_caption

    def get_new_captions_for_mask(
            self, dci: DenseCaptionedImage, 
            mask: MaskDataEntry, max_len=77, 
            target_count=None, max_retry=5) -> list[str]:
        mask_subcaption = self.get_short_enough_mask_caption(dci, mask)
        caption_length = get_clip_token_length(mask_subcaption)
        if target_count is None:
            target_count = max(4, math.log(caption_length))
        new_captions = []
        while len(new_captions) < target_count and max_retry > 0:
            max_retry -= 1
            try:
                message = f"Full description: {mask_subcaption}\n\nSummaries:"
                res = self.generate(
                    message=message,
                    system=MULTI_CAPTION_PROMPT,
                    max_new_tokens=1400,
                )
                for c in self.get_extracted_captions(res['text']):
                    if get_clip_token_length(c) < max_len:
                        new_captions.append(c)
            except:
                pass
        assert len(new_captions) > target_count
        return new_captions

    def get_summary(self, message: str, short=False):
        if not short:
            SYSTEM_PROMPT = (
                "You are given a full-text description of an image. You should summarize "
                "it into about 65 words, being sure to include as much salient visual "
                "information as possible given the 65 word constraint, especially information from "
                "the start of the original description. The new description "
                "should apply for the original image. Respond with only the summary, in one line. "
            )
        else:
            SYSTEM_PROMPT = (
                "You are given a description of part of an image. You should summarize "
                "it into a single line no longer than 65 words, being sure to include as much salient  "
                "visual information as possible given the 65. Don't include any details not present "
                "in the provided description. Respond with only the summary, in one line. "
            )
        res = self.generate(message, system=SYSTEM_PROMPT, max_new_tokens=250)
        if "\n" in res['text']: 
            assert "\n" not in res['text'][100:], f"Response had enter: {res['text']}"
            res['text'] = res['text'].split("\n")[-1]
        return res['text'].strip()
    
    def get_negative(self, message: str, negative_type='base'):
        alter_prompt = ALTER_PROMPTS[negative_type]
        res = self.generate(message, system=alter_prompt, max_new_tokens=150)
        if "\n" in res['text']: 
            assert "\n" not in res['text'][100:], f"Response had enter: {res['text']}"
            res['text'] = res['text'].split("\n")[-1]
        return res['text'].strip()

    def reduce_length(self, message: str, target_tokens=75, failures=3):
        last_len = get_clip_token_length(message)
        
        while (last_len) > target_tokens:
            print(f"r", end='', flush=True)
            if last_len - target_tokens < 10:
                ALTER_PROMPT = (
                    "You are given the description for a scene. Summarize and rephrase the description such that the "
                    "new description is a few words shorter, but retains key information. "
                )
            else:
                ALTER_PROMPT = (
                    "You are given the description for a scene. Summarize and rephrase the description such "
                    "that the new description is shorter, but retains key information. "
                )
            res = self.generate(message, system=ALTER_PROMPT, max_new_tokens=150)
            new_len = get_clip_token_length(res['text'].strip())
            if new_len >= last_len or "\n" in res['text']:
                assert failures > 0, f"New message was not shorter, {last_len} -> {new_len}, failed many times" 
                failures -= 1
            else:
                last_len = new_len
                message = res['text'].strip()
        return message
