# -*- coding: utf-8 -*-
###############################################################################
# MODULE: vocab_tools
# this is a repo of tools to evaluate and jump between words
#
# Assumptions: ASCII character set
# Author:   William Kinsman
# Created:  10.10.2015
###############################################################################


def has_upper(text):
    for i in text:
        if i.isupper():
            return True
    return False


def has_lower(text):
    for i in text:
        if i.islower():
            return True
    return False


def fullword(text, index):
    textlen = len(text)
    if index == None or not text[index].isalpha():
        return None
    start = index
    end = index

    # step backwards
    while start - 1 > 0 and text[start - 1].isalpha():
        start -= 1

    # step forwards
    while end + 1 < textlen and text[end + 1].isalpha():
        end += 1
    return text[start : end + 1]


def nextwordindex(text, index):
    textlen = len(text)
    punct = "~|\\!*\"'()+,./`[]^;:{}<>?\n\t\r\f"
    if index == None:
        return None

    # step to end of current word
    if text[index].isalpha():
        while index < textlen and text[index].isalpha():
            index += 1
    else:
        index += 1
    if index >= textlen or text[index] in punct:
        return None

    # step to start of next word
    while index < textlen and text[index] not in punct:
        if index >= textlen or text[index] in punct:
            return None
        if text[index].isalpha():
            return index
        index += 1
    return None


def prevwordindex(text, index):
    punct = "~|\\!*\"'()+,./`[]^;:{}<>?\n\t\r\f"
    if index == None:
        return None

    # find start of current word
    if text[index].isalpha():
        while index >= 0 and text[index].isalpha():
            index -= 1
    else:
        index -= 1
    if index < 0 or text[index] in punct:
        return None

    # find end of previous word
    while text[index] not in punct and not text[index].isalpha():
        if index < 0 or text[index] in punct:
            return None
        index -= 1
    if index < 0 or text[index] in punct:
        return None

    # find start of previous word
    while index >= 0 and text[index].isalpha():
        index -= 1
        if index == -1:
            return 0
        if not text[index].isalpha():
            return index + 1
    return None


def nextword(text, index):
    return fullword(text, nextwordindex(text, index))


def prevword(text, index):
    return fullword(text, prevwordindex(text, index))


def wordstartindex(text, index):
    if index == None or not text[index].isalpha():
        return None

    # find start of previous word
    while index >= 0 and text[index].isalpha():
        index -= 1
        if index == -1:
            return 0
        if not text[index].isalpha():
            return index + 1
    return None


def wordendindex(text, index):
    """
    given : text, index as the start of a word
    return: the ending index of a word
    """
    if index == None:
        return None
    textlen = len(text)
    punct = "~|\\!*\"'()+,./`[]^;:{}<>?\n\t\r\f"

    # step to end of current word
    if text[index].isalpha():
        while index < textlen and text[index].isalpha():
            index += 1
    else:
        index += 1
    index -= 1
    if index >= textlen or text[index] in punct:
        return None
    return index
