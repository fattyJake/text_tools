# -*- coding: utf-8 -*-
# This is a set of python tools for constructing a vocabulary and IDF

import re
from copy import copy
import text_tools.words
import text_tools.preprocessing
import text_tools.vectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def vocab_regex(vocab, ignorecase=False):
    """
    @param vocab: a list of phrases
    return: a regular expression to search for all phrases 
    NOTE: enabling ignore case is significantly slower
    """
    if isinstance(vocab, str):
        vocab = [vocab]
    assert len(vocab) == len(
        list(set(vocab))
    ), "Error: repeat phrase found in vocabulary"
    vocab = copy(vocab)
    vocab.sort(key=len, reverse=True)
    vocab = r"\b(" + r"|".join(vocab) + r")\b"
    vocab = re.sub(r"""\-""", r"""\-""", vocab)
    vocab = re.sub(r"""\.""", r"""\.""", vocab)
    if ignorecase == True:
        return re.compile(vocab, re.IGNORECASE)
    return re.compile(vocab)


def build_vocab(text, multigram=False, vocab_init=[]):
    """
    @param text: string
    @param multigram: intelligently pull multigrams
    @param vocab_init: seperately confirmed vocab
    return: vocab, in alphabetical order. Accepts alphas>=3 letters.
    """
    text = text_tools.preprocessing.force_ascii(text)
    text = text_tools.preprocessing.remove_false_periods(text)
    abbrevs = [
        i.lower() for i in re.findall(r"\b[A-Z]{3,}\b", text)
    ]  # pulls out abbrevs and add them to delimiter list
    text = text_tools.preprocessing.force_lower(text)
    text = text_tools.preprocessing.force_punct(text)

    stop = list(ENGLISH_STOP_WORDS)
    stop_stemmed = text_tools.preprocessing.stem_all(stop)
    vocab = list(
        set(abbrevs + re.findall(r"\b[a-z]{2,40}\b", text) + vocab_init)
    )
    vocab = [i for i in vocab if i not in stop]
    vocab = text_tools.preprocessing.stem_all(vocab)

    if multigram:

        # split text by stopwords and punct delimiters
        stopwords = [i for i in ENGLISH_STOP_WORDS] + abbrevs
        stopwords.sort(key=len, reverse=True)
        stopwords = r"\b" + r"\b|\b".join(stopwords) + r"\b"
        punct = r"(?![a-z]+\b)[a-z0-9]+\b|\b[a-z]\b|[^a-z0-9 ]+"
        regex_stops = re.compile(stopwords + r"|" + punct)
        subtexts = re.split(regex_stops, text) + abbrevs
        subtexts = [i.strip() for i in subtexts]
        subtexts = text_tools.preprocessing.stem_all(subtexts)
        subtexts.sort()
        vocab = vocab + subtexts

    # for each substring, tokenize chains up to max-ngram length
    vocab = [i for i in list(set(vocab)) if i and i not in stop_stemmed]
    vocab.sort()
    return vocab


def vocab_counts(text, vocab):
    """
    return a list of tuples describing the frequency of vocab terms
    Note: text should be preprocessed if vocab is as well
    """
    counts = sum(text_tools.vectorizer.count_vectorizer(text, vocab))
    counts = [(vocab[i], int(counts[0, i])) for i in range(len(vocab))]
    counts.sort(key=lambda x: x[1], reverse=True)
    return counts


def vocab_preprocess(vocab, max_length=40):
    """
    @param vocab: a list or line-return seperated list of vocab
    @param max_length: the maximum length of a term
    return: list of preprocessed multigram candidates
    """
    if isinstance(vocab, str):
        vocab = vocab.splitlines()
    vocab = text_tools.preprocessing.force_lower(vocab)
    vocab = text_tools.preprocessing.strip_parenthesized(vocab)
    vocab = text_tools.preprocessing.force_punct(vocab)
    vocab = r"\r\n".join(vocab)
    stopwords = [i for i in ENGLISH_STOP_WORDS]
    stopwords.sort(key=len, reverse=True)
    stopwords = r"\b" + r"\b|\b".join(stopwords) + r"\b"
    regex_stops = re.compile(
        stopwords
        + r"|\b[a-z]*\d+[a-z]*\b|\%|\~|\||\\|!|\*|\"|\'|\(|\)|\+|\,|\.|\|/`|"\
            r"\[|\]|\^|\;|\:|\{|\}|\<|\>|\?|\n|\t|\r|\f"
    )
    vocab = re.split(regex_stops, vocab)
    vocab = [i.strip() for i in vocab]
    vocab = [i for i in vocab if len(i.split()) > 1]
    vocab = text_tools.preprocessing.stem_all(vocab)
    vocab = list(set(vocab))
    vocab = [i.strip() for i in vocab if len(i) <= max_length]
    vocab.sort()
