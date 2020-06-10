# -*- coding: utf-8 -*-
###############################################################################
# MODULE: similarity
# this is a repo of tools to compare texts and eliminate repeats
#
# Author:   William Kinsman
# Created:  10.10.2015
###############################################################################

import re
import math
from collections import Counter


def jaccard(textA, textB, mode=2):
    """
    @param textA & textB: strings
    @param mode: 0 as textA based denomitor; 1 as textB based and 2 as both
    return: jaccard similarity of unique strings
    """
    tokA = set(textA.split())
    tokB = set(textB.split())
    if mode == 0:
        return len(tokA.intersection(tokB)) / float(len(tokA))
    elif mode == 1:
        return len(tokA.intersection(tokB)) / float(len(tokB))
    else:
        return len(tokA.intersection(tokB)) / float(len(tokA | tokB))


def cosine(textA, textB):
    """
    @param textA & textB: strings
    return: cosine similarity
    """
    re_token = re.compile(r"[a-zA-Z]")
    vec1 = Counter(re_token.findall(textA))
    vec2 = Counter(re_token.findall(textB))
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
    sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def perc_length_difference(textA, textB):
    """
    @param textA/textB: strings
    return: length difference (smaller relative to larger)
    """
    if len(textA) == 0 and len(textB) == 0:
        return 0
    if len(textA) == 0 or len(textB) == 0:
        return 1
    return abs(len(textA) - len(textB)) / float(max(len(textA), len(textB)))


def unique_by_jaccard(texts, threshold=0.95):
    """
    @param texts: a list of strings
    @param threshold: decimal %. if >= this similarty, a repeat is removed
    return: the list of strings with repeats removed. First instances saved.
    NOTE: this function looks for <10% length similarity first for speed
    """
    texts = list(texts)
    tokens = [set(i.split()) for i in texts]
    i = 0
    while i < len(texts) - 1:
        j = i + 1
        while j < len(texts):

            # IF length_diff<=5% difference AND jaccard high, drop second
            # version
            if perc_length_difference(texts[i], texts[j]) <= 0.05:

                try:  # this try statement needs to be dropped. FIGURE IT OUT.
                    if (
                        len(tokens[i].intersection(tokens[j]))
                        / float(len(tokens[i] | tokens[j]))
                        >= threshold
                    ):
                        del texts[j]
                        del tokens[j]
                        continue
                except:
                    pass
            j += 1
        i += 1
    return texts


def repeated_find(texts, min_cluster_size=1):
    """
    @param texts: a list of strings
    @param min_cluster_size: minimum size of repeated group
    @param drop_timestamps: if true, drops timestamps before evaluation
    return: tuples (repeated page, origin index)
    """
    # initialization
    if not isinstance(texts, list) or len(texts) <= min_cluster_size:
        return []

    out = []
    repeat_pages = []
    i = 0
    while i < len(texts) - 1:
        if len(texts[i]) != 0 and i not in repeat_pages:
            j = i + 1
            while j < len(texts):
                if (
                    j not in repeat_pages
                    and len(texts[j]) != 0
                    and texts[i] == texts[j]
                ):
                    out.append((j, i))
                    repeat_pages.append(j)
                j += 1
        i += 1
    out.sort()

    # eliminate non-clusters
    if min_cluster_size > 1:
        i = 0
        acceptable = []
        while i < len(out) - 1:
            j = i + 1
            while j < len(out):
                if (
                    out[j][0] == out[j - 1][0] + 1
                    and out[j][1] == out[j - 1][1] + 1
                ):
                    j += 1
                else:
                    break
            if j - i >= min_cluster_size:
                acceptable = acceptable + out[i:j]
            i = j
        return acceptable
    return out


def repeated_delete(texts, min_cluster_size=1, whiteout=True):
    """
    remove repeats in a list, retaining earliest instance
    @param texts: list of strings
    @param min_cluster_size: minimum size of repeated group
    @param drop_timestamps: if true, drops timestamps before evaluation
    @param whiteout: if true, doesnt delete the list index but replaces it with
        empty string
    """
    repeats = repeated_find(texts, min_cluster_size)
    if not repeats:
        return texts
    repeats = [i[0] for i in repeats]

    # if not whiteout, delete
    if not whiteout:
        return [texts[i] for i in range(len(texts)) if i not in repeats]

    # if whiteout, replace with whitespace
    else:
        return [
            texts[i] if i not in repeats else "" for i in range(len(texts))
        ]
