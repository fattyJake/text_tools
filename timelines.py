# -*- coding: utf-8 -*-

import re
import numpy as np
import matplotlib.pyplot as plt
from text_tools import vocab_tools
from text_tools.extract.resolve_entities import fetch_capped_chains


def timeline(text, names, windowsize=1000):
    """
    returns a vector of number of occurrences in each sliding window
    @param text: string
    @param names: a name or a list of names (aliases)
    @param windowsize: size of sliding window
    """
    # initialize
    if isinstance(text, list):
        text = "\n".join(text)
    locs = [
        int((i.end() - i.start()) / 2) + i.start()
        for i in re.finditer(vocab_tools.vocab_regex(names), text)
    ]
    if windowsize % 2 == 0:
        windowsize = max(0, windowsize - 1)
    radius = (windowsize - 1) / 2
    textlen = len(text)
    vector = np.zeros(textlen)

    # place into vector
    for i in locs:
        vector[
            int(max(0, i - radius)) : int(min(i + radius + 1, textlen))
        ] += 1
    return vector


def timeline_dict(text, windowsize=1000):
    """
    @param text: text
    """
    if isinstance(text, list):
        text = "\n".join(text)
    entities = fetch_capped_chains(text)
    entity_dict = {i: timeline(text, i, windowsize) for i in entities}
    totals = sum(entity_dict.values())
    entity_dict = {k: v / totals for k, v in entity_dict.items()}
    return entity_dict


def timeline_ranking(entity_dict):
    """
    @param text: text
    """
    entity_dict = [(k, np.mean(v)) for k, v in entity_dict.items()]
    entity_dict.sort(key=lambda x: x[1], reverse=True)
    return entity_dict


def plot_timeline(entity_dict, entity):
    """
    @param entity_dict: entity_dict
    @param entity: entity or list of entities
    """
    # initialize
    if isinstance(entity, str):
        entity = [entity]
    entity = [i for i in entity if i in entity_dict]
    if not entity:
        for i in entity_dict:
            print(i)
        print("Please choose from the above list of entities. Aborting.")
        return

    # plot
    plt.figure(figsize=(12, 3))
    for i in entity:
        plt.plot(range(len(entity_dict[i])), entity_dict[i], label=i)
    plt.xlabel("Index")
    plt.ylabel("Confidence")
    plt.ylim((0, 1))
    plt.grid()
    plt.legend(loc="upper right")
    plt.show()
