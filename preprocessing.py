# -*- coding: utf-8 -*-
# module for basic preprocessing of text

import os
import re
import random
import pickle
import text_tools
from copy import copy
from stemming.porter2 import stem
import pyap
from itertools import chain


def preprocess(text, negex=False, stem=True):
    """
    fully preprocess a string or list of strings
    """
    text = force_lower(text)  # does not impact space count
    text = force_punct(text)  # does not impact space count
    text = remove_false_periods(text)  # does not impact space count
    text = force_abbr(text)  # does not impact space count
    if negex:
        text = drop_negex(text)  # does not impact space count
    if stem:
        text = stem_all(text)  # does not impact space count
    return text


def force_ascii(texts):
    texts = copy(texts)
    strBOOL = False
    if isinstance(texts, str):
        texts = [texts]
        strBOOL = True

    # check hashmap for each char else use "_"
    charmap = pickle.load(
        open(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "pickles",
                "charmap",
            ),
            "rb",
        )
    )
    for i in range(len(texts)):
        texts[i] = "".join([charmap.get(i, "_") for i in texts[i]])

    if strBOOL == True:
        return texts[0]
    return texts


def force_lower(texts):
    texts = copy(texts)
    strBOOL = False
    if isinstance(texts, str):
        texts = [texts]
        strBOOL = True

    for i in range(len(texts)):
        texts[i] = texts[i].lower()

    if strBOOL == True:
        return texts[0]
    return texts


def force_punct(texts, all_punct=False):
    texts = copy(texts)
    strBOOL = False
    if isinstance(texts, str):
        texts = [texts]
        strBOOL = True

    # precompile
    re_ws = re.compile(
        r"[-_/<>](?=[a-z])|(?<=[a-z])[-_/<>]"
    )  # replace underscores/dashes/slashes/html tags with nothing
    re_punct = re.compile(
        "[%s]" % re.escape("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~•—")
    )
    re_spaces = re.compile(r"[ ]{2,}")  # simplify multi-spaces
    re_dots = re.compile(r"[.]{2,}")  # simplify multi-periods
    re_poss = re.compile(r"\b'\b")  # drop posessions/contractions

    # perform
    for i in range(len(texts)):
        texts[i] = re.sub(re_ws, r" ", texts[i])
        if all_punct:
            texts[i] = re.sub(re_punct, " ", texts[i])
        texts[i] = re.sub(re_spaces, r" ", texts[i])
        texts[i] = re.sub(re_poss, r" ", texts[i])
        texts[i] = re.sub(re_dots, r".", texts[i])

    if strBOOL == True:
        return texts[0]
    return texts


def force_abbr(texts):
    texts = copy(texts)
    strBOOL = False
    if isinstance(texts, str):
        texts = [texts]
        strBOOL = True

    abbr_dict = pickle.load(
        open(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                r"pickles",
                "abbr_dict",
            ),
            "rb",
        )
    )
    abbr_pattern = pickle.load(
        open(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                r"pickles",
                "abbr_pattern",
            ),
            "rb",
        )
    )
    texts = [
        abbr_pattern.sub(lambda x: abbr_dict[x.group()], text)
        for text in texts
    ]

    if strBOOL:
        return texts[0]
    return texts


def force_number(texts, keep=True):
    texts = copy(texts)
    strBOOL = False
    if isinstance(texts, str):
        texts = [texts]
        strBOOL = True

    texts = [
        re.sub(
            r"\b\-?\d+\.?\d*((e|E)(\+|\-)\d+)?\b",
            "NUM" if keep else "",
            text
        )
        for text in texts
    ]

    if strBOOL:
        return texts[0]
    return texts


def force_demographic(texts, keep=True):
    texts = copy(texts)
    strBOOL = False
    if isinstance(texts, str):
        texts = [texts]
        strBOOL = True

    # find all dates
    re_date = r"(\b\d{4}[-]\d{1,2}[-]\d{1,2}\b|\b\d{1,2}[-]\d{1,2}[-]\d{4}\b|"\
        r"\b\d{1,2}[-]\d{1,2}[-]\d{2}\b|\b\d{4}[\.]\d{1,2}[\.]\d{1,2}\b|\b\d"\
        r"{1,2}[\.]\d{1,2}[\.]\d{4}\b|\b\d{1,2}[\.]\d{1,2}[\.]\d{2}\b|\b\d{4}"\
        r"[/]\d{1,2}[/]\d{1,2}\b|\b\d{1,2}[/]\d{1,2}[/]\d{4}\b|\b\d{1,2}[/]\d"\
        r"{1,2}[/]\d{2}\b)|((\b\d{1,2}\D{0,3})?\b(?:(J|j)an(?:uary)?|(F|f)eb"\
        r"(?:ruary)?|(M|m)ar(?:ch)?|(A|a)pr(?:il)?|(M|m)ay|(J|j)un(?:e)?|(J|"\
        r"j)ul(?:y)?|(A|a)ug(?:ust)?|(S|s)ep?(?:tember)?|(O|o)ct(?:ober)?|(("\
        r"N|n)ov|(D|d)ec)(?:ember)?)\D{1,2}(\d{1,2}(st|nd|th)?\D?)?\D?(\d{4}))"
    for i in range(len(texts)):
        if keep:
            date_list = [
                (d.start(), d.end())
                for d in re.finditer(re_date, texts[i], flags=re.IGNORECASE)
            ]
            for start, end in zip(date_list):
                texts[i] = (
                    texts[i][:start]
                    + re.sub(r"\S", "X", texts[i][start:end])
                    + texts[i][end:]
                )
        else:
            texts[i] = re.sub(re_date, "", texts[i], flags=re.IGNORECASE)

    # find all times
    re_time = r"(([0]?[1-9]|1[0-2])((\:|\.)[0-5][0-9]){1,2}( )?(am|pm))|(([0]"\
        r"?[0-9]|1[0-9]|2[0-3])((\:|\.)[0-5][0-9]){1,2})"
    for i in range(len(texts)):
        if keep:
            time_list = [
                (t.start(), t.end())
                for t in re.finditer(re_time, texts[i], flags=re.IGNORECASE)
            ]
            for start, end in zip(time_list):
                texts[i] = (
                    texts[i][:start]
                    + re.sub(r"\S", "X", texts[i][start:end])
                    + texts[i][end:]
                )
        else:
            texts[i] = re.sub(re_time, "", texts[i], flags=re.IGNORECASE)

    # find all address
    for i in range(len(texts)):
        addr_list = [str(j) for j in pyap.parse(texts[i], country="US")]
        for addr in addr_list:
            if keep:
                addr_match = re.search(re.escape(addr), texts[i])
                start, end = addr_match.start(), addr_match.end()
                texts[i] = (
                    texts[i][:start]
                    + re.sub(r"\S", "X", texts[i][start:end])
                    + texts[i][start:]
                )
            else:
                texts[i] = re.sub(re.escape(addr), "", texts[i])

    # find all phone numbers
    for i in range(len(texts)):
        re_phone = re.compile(
            r"(\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}"
        )
        if keep:
            phone_list = [
                (t.start(), t.end()) for t in re.finditer(re_phone, texts[i])
            ]
            for start, end in zip(phone_list):
                texts[i] = (
                    texts[i][:start]
                    + re.sub(r"\S", "X", texts[i][start:end])
                    + texts[i][end:]
                )
        else:
            texts[i] = re.sub(re_phone, "", texts[i])

    if strBOOL:
        return texts[0]
    return texts


def split_into_sentence(texts, chaining=False):
    texts = copy(texts)
    strBOOL = False
    if isinstance(texts, str):
        texts = [texts]
        strBOOL = True

    texts = [
        re.split(
            r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<!\s\.)(?<=\.|\?|\!)\s", text
        )
        for text in texts
    ]

    if strBOOL:
        return texts[0]
    if chaining:
        texts = chain(*texts)
    return texts


def strip_line_returns(texts):
    """
    @param text: string
    return: text without in-par line returns
    WHY: parsed PDFs have line returns at the end of every line
    """
    texts = copy(texts)
    strBOOL = False
    if isinstance(texts, str):
        texts = [texts]
        strBOOL = True

    # precompile and perform
    re_LR = re.compile(
        r"(?<!\r\n)\r\n(?!\r\n)"
    )  # replace underscores and dashes
    for i in range(len(texts)):
        texts[i] = re.sub(re_LR, r" ", texts[i])

    if strBOOL:
        return texts[0]
    return texts


def strip_parenthesized(texts):
    """
    @param text: string
    return: text without parenthesized text
    """
    texts = copy(texts)
    strBOOL = False
    if isinstance(texts, str):
        texts = [texts]
        strBOOL = True

    # precompile and perform
    re_parenthesized = re.compile(r"\(.*\)")
    for i in range(len(texts)):
        texts[i] = re.sub(re_parenthesized, r"", texts[i])

    if strBOOL == True:
        return texts[0]
    return texts


def remove_false_periods(texts):
    """
    @param texts: string or list of strings
    return: strings without false periods (e.g. etc. and so on)
    """
    texts = copy(texts)
    strBOOL = False
    if isinstance(texts, str):
        texts = [texts]
        strBOOL = True

    # precompile and perform
    re_acronym = re.compile(r"(?<=\.[a-zA-Z])\.")
    re_periods = re.compile(r"\.\s*[a-zA-Z]")
    prefixes = [
        "amb", "bgen", "brigen", "capt", "col", "dr", "gen", "gov", "hon",
        "inc", "jr", "lieut", "lt", "maj", "mdme", "mr", "mrs", "ms", "msgr",
        "messrs", "no", "prof", "rep", "rev", "sen", "sgt", "sr"
    ] + list("abcdefghijklmnopqrstuvwxyz")
    for i in range(len(texts)):
        texts[i] = re.sub(re_acronym, r"", texts[i])
        locs = [m.start() for m in re.finditer(re_periods, texts[i])]
        for j in range(len(locs) - 1, -1, -1):
            if (
                locs[j] - 1 != -1
                and texts[i][locs[j] - 1].isalpha()
                and text_tools.words.prevword(texts[i], locs[j]).lower()
                in prefixes
            ):
                texts[i] = texts[i][: locs[j]] + texts[i][locs[j] + 1 :]

    if strBOOL == True:
        return texts[0]
    return texts


def drop_negex(texts, comma_delimit=False):
    """
    @param text: text string
    @param n: if None, remove until sentance ending punct, else # of words max
    return: text with all negated phrases replaced with capped terms (not
        detected by pipeline but still visible)
    NOTE: see negex.txt for items used
    """
    # load all locations
    punct = "!()+.;:?\n\t\r\f\-\\/"  # comma was removed from delimit list
    if comma_delimit:
        punct = punct + ","
    with open(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "pickles", "negex"
        ),
        "rb",
    ) as fp:
        negex = pickle.load(fp)
        negex_false = pickle.load(fp)
        negex_pre = pickle.load(fp)

    strBOOL = False
    if isinstance(texts, str):
        texts = [texts]
        strBOOL = True

    # for each text
    for i in range(len(texts)):
        textlength = len(texts[i])

        # find all locations sans false locations
        locs = [[m.start(), m.group()] for m in re.finditer(negex, texts[i])]
        locs_false = [m.start() for m in re.finditer(negex_false, texts[i])]
        locs = [m for m in locs if m[0] not in locs_false]
        locs.sort(reverse=True)

        # negex post
        for m in locs:
            idx = m[0] + len(m[1])
            while idx < textlength and texts[i][idx] not in punct:
                idx += 1
            texts[i] = (
                texts[i][0 : m[0]]
                + texts[i][m[0] : idx].upper()
                + texts[i][idx:textlength]
            )

        # negex pre
        locs_pre = [
            [m.end(), m.group()] for m in re.finditer(negex_pre, texts[i])
        ]
        locs_pre.sort(reverse=True)
        for m in locs_pre:
            idx = m[0] - len(m[1])
            while idx > 0 and texts[i][idx] not in punct:
                idx -= 1
            texts[i] = (
                texts[i][0:idx]
                + texts[i][idx : m[0]].upper()
                + texts[i][m[0] : textlength]
            )

    if strBOOL == True:
        return texts[0]
    return texts


def stem_all(texts):
    texts = copy(texts)
    strBOOL = False
    if isinstance(texts, str):
        texts = [texts]
        strBOOL = True

    # tokenize and stem if alpha >= 3 chars
    re_splits = re.compile(r"\w+|\W+")
    for i in range(len(texts)):
        tokens = re.findall(re_splits, texts[i])
        for j in range(len(tokens)):
            if len(tokens[j]) >= 3 and tokens[j].isalpha():
                tokens[j] = stem(tokens[j])
        texts[i] = "".join(tokens)

    if strBOOL == True:
        return texts[0]
    return texts


def stem_map(doc, doc_stemmed=None):
    """
    @param doc: original document
    @param doc_stemmed: document that has been stemmed completely
    Return: dict mapping per page of K=stemmed_loc V=unstemmed_loc
    """
    if isinstance(doc, str):
        doc = [doc]
    if not doc_stemmed:
        doc_stemmed = stem_all(doc)
    map_dict = dict.fromkeys(list(range(len(doc))))
    regex_tokens = re.compile(r"\b[a-zA-Z0-9]+\b")

    # iterate over pages
    for page in range(len(doc)):
        stemmed_loc = [
            i.start() for i in re.finditer(regex_tokens, doc_stemmed[page])
        ]
        orig_loc = [i.start() for i in re.finditer(regex_tokens, doc[page])]
        map_dict[page] = dict(zip(stemmed_loc, orig_loc))
    return map_dict


def shuffle_tokens(text, vocab=None):
    """
    scrambles tokens within a string
    @param text: a string
    @param vocab: if available use it to maintain multigrams
    """
    regex = r"\b[a-zA-Z0-9]+\b|(?<=[\n\r ])[^ a-zA-Z0-9]+(?=[ \r\n])|\b[^ a-z"\
        r"A-Z0-9]+\b"
    if vocab:
        vocab = text_tools.vocab_tools.vocab_regex(vocab).pattern + "|" + regex
    strings = re.findall(regex, text)
    random.shuffle(strings)
    return " ".join(strings)


#################################### GGK TOOLS ################################


def GGK_strip_nonalphanum(text):
    if len(text) == 0:
        return text
    while len(text) > 0:
        if not text[0].isalnum():
            text = text[1:]
        else:
            break
    while len(text) > 0:
        if not text[-1].isalnum():
            text = text[:-1]
        else:
            break
    return text


def GGK_preprocessor(texts):
    """
    given:  a string
    return: a string reduced to be letter-bound ('abc-'=>'abc')
    """
    strBOOL = False
    if isinstance(texts, str):
        texts = [texts]
        strBOOL = True

    # precompile and perform
    for i in range(len(texts)):
        texts[i] = texts[i].lower()
        words = texts[i].split()
        for j in range(len(words)):
            words[j] = GGK_strip_nonalphanum(words[j])
        texts[i] = " ".join((" ".join(words)).split())

    if strBOOL == True:
        return texts[0]
    return texts
