# -*- coding: utf-8 -*-
###############################################################################
# MODULE: readability
# this is a function used to define the readability of texts
#
# Author:   BAC
# Created:  09.01.2015
###############################################################################

import os
import re
import pickle
import numpy as np
from copy import copy
from langdetect import detect
from text_tools import vectorizer

def check_by_vocab(text,vocab,min_tokens=10):
    """
    return: readability as percent of tokens in the vocabulary
    @param text: a preprocessed document
    @param vocab: a list of vocabulary
    @param min_tokens: min tokens required for output to be non-zero
    """
    if isinstance(text,list): text = '\n'.join(text)
    doc_words = re.findall(r'\b[a-zA-Z]+\b',text)
    doc_words_count = len(doc_words)
    if doc_words_count<min_tokens: return 0.0
    doc_words = [i for i in doc_words if len(i)>1 and i in vocab]    
    return len(doc_words)/float(doc_words_count)

def check_by_trigrams(doc,minimum=False):
    """
    return: readability by ML, or mean readability if list
    @param doc: a string or list of strings
    @param minimum: take minimum readability instead of mean
    trained using OCR
    """
    text = copy(doc)
    if not text: return 1.0
    classifier = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickles','readability_classifier_trigram'),'rb'))
    if isinstance(text,list):
        results = classifier.predict_proba(vectorizer.trif_vectorizer(text))[:,1]
        
        # empty string case returns 1
        if minimum: return np.min([results[i] if text[i].strip() else 1.0 for i in range(len(results))])
        return np.mean([results[i] if text[i].strip() else 1.0 for i in range(len(results))])
    
    # empty string case returns 1
    if not text.strip(): return 1.0
    return classifier.predict_proba(vectorizer.trif_vectorizer(text))[0,1]

def is_english(doc):
    """
    return: if its english
    """
    text = copy(doc)
    if isinstance(text,list): text = '\n'.join(text)
    try:
        if detect(text)=='en': return True
    except: pass
    return False

def is_spanish(doc):
    """
    return: if its english
    """
    text = copy(doc)
    if isinstance(text,list): text = '\n'.join(text)
    try:
        if detect(text)=='es': return True
    except: pass
    return False