#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import random
import spacy
from densely_captioned_images.dataset.utils import get_clip_token_length

_loaded_en_nlp = None

ADJ_ANTONYM_RATE = 0.5

def get_english():
    global _loaded_en_nlp
    if _loaded_en_nlp is None:
        _loaded_en_nlp = spacy.load("en_core_web_sm")
    return _loaded_en_nlp

def get_wordnet():
    """Import wordnet as function, as this can be time consuming"""
    from nltk.corpus import wordnet

    return wordnet


def get_antonyms(phrase):
    wordnet = get_wordnet()

    antonyms = []

    for syn in wordnet.synsets(phrase):
        for l in syn.lemmas():
            if l.antonyms():
                antonyms.append(l.antonyms()[0].name())
    return list(set(antonyms))


def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def different_enough(s1, s2):
    if s1.lower() in s2.lower() or s2.lower() in s1.lower():
        return False
    return levenshtein_distance(s1, s2) / min(len(s1), len(s2)) > 0.5

def process_swaps_single_pass(in_text, swaps):
    """Given in text and a list of index pairs to swap, swaps all in one pass"""
    chunks = {}
    starts = []
    ends = []
    for ((s1, e1), tuple_or_word) in swaps:
        if isinstance(tuple_or_word, str):
            chunks[(s1, e1)] = tuple_or_word
            starts.append(s1)
            ends.append(e1)
        else:
            s2, e2 = tuple_or_word
            chunks[(s1, e1)] = in_text[s2:e2]
            chunks[(s2, e2)] = in_text[s1:e1]
            starts += [s1, s2]
            ends += [e1, e2]

    starts.sort()
    ends.sort()
    if starts[0] != 0:
        ends = [starts[0]] + ends
        starts = [0] + starts
    if ends[-1] != len(in_text):
        starts = starts + [ends[-1]]
        ends = ends + [len(in_text)]
    for s in starts:
        if s != 0 and s not in ends:
            ends.append(s)
    for e in ends:
        if e != len(in_text) and e not in starts:
            starts.append(e)
    
    starts.sort()
    ends.sort()
    
    res_text = ""
    for (s, e) in zip(starts, ends):
        if (s, e) in chunks:
            to_add = chunks[(s, e)]
        else:
            to_add = in_text[s:e]
        # print(f"({s}, {e}) Orig: {in_text[s:e]}, adding {to_add}")
        res_text += to_add
    return res_text

def get_spacy_negative(in_text, swap_count=2, use_antonyms=False):
    nlp = get_english()
    out_text = in_text
    doc = nlp(out_text)
    
    def get_possible_swaps(swap_source, max_swaps=2):
        """Select two elems at a time from the swap source to change"""
        used_elems = set()
        swaps = []
        while len(swaps) < max_swaps and len(swap_source) > 0:
            elem = swap_source.pop(0)
            if elem.text in used_elems:
                continue
            used_elems.add(elem.text)
            if use_antonyms and hasattr(elem, 'pos_') and elem.pos_ == "ADJ" and random.random() < ADJ_ANTONYM_RATE:
                antonyms = get_antonyms(elem.text)
                if len(antonyms) > 0:
                    swaps.append(((elem.idx, elem.idx + len(elem)), random.choice(antonyms)))
                    continue
            possible_swaps = [n for n in swap_source if different_enough(n.text, elem.text) and n.text not in used_elems]
            if len(possible_swaps) == 0:
                continue
            swap_elem = random.choice(possible_swaps)
            used_elems.add(swap_elem.text)
            try:
                swaps.append(((elem.start_char, elem.end_char), (swap_elem.start_char, swap_elem.end_char)))
            except:
                swaps.append((
                    (elem.idx, elem.idx + len(elem)), 
                    (swap_elem.idx, swap_elem.idx + len(swap_elem))
                ))
            # print(f"Adding swap {elem} => {swap_elem}")
        return swaps
    
    # Lets swap some noun phrases
    noun_phrases = [noun_phrase for noun_phrase in doc.noun_chunks]
    random.shuffle(noun_phrases)
    swaps = get_possible_swaps(noun_phrases, swap_count)
    
    # Reprocess text after swaps
    if len(swaps) > 0:
        out_text = process_swaps_single_pass(out_text, swaps)
        doc = nlp(out_text)

    # Lets swap adjectives too
    adjectives = [tok for tok in doc if tok.pos_ == "ADJ"]
    random.shuffle(adjectives)
    swaps = get_possible_swaps(adjectives, swap_count)
    
    # Lets swap verbs as well
    verbs = [tok for tok in doc if tok.pos_ == "VERB"]
    random.shuffle(verbs)
    swaps += get_possible_swaps(verbs, swap_count)
    
    if len(swaps) == 0:
        return out_text
    
    # Return result
    out_text = process_swaps_single_pass(out_text, swaps)
    if get_clip_token_length(out_text) >= 77:
        # Small chance of over-increase on token count when using antonyms
        # retry without
        return get_spacy_negative(in_text, swap_count=swap_count, use_antonyms=False)
    return out_text
        