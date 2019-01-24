# -*- coding: utf-8 -*-

import os
import re
import scipy
import pickle
import platform
import numpy as np
import text_tools.vocab_tools
from copy import copy
from tqdm import tqdm
from collections import Counter
from itertools import islice,tee
from sklearn.preprocessing import normalize
from multiprocessing import Pool, cpu_count

def count_vectorizer(texts,vocab,regex=None):
    """
    @param text: string or list of preprocessed strings
    @param vocab: vocab
    return: sparse matrix (rows are input texts, columns are vocab)
    """
    # initialize
    if isinstance(texts,str): texts = [texts]
    hashdict = {k: v for v, k in enumerate(vocab)}
    if not regex: regex = text_tools.vocab_tools.vocab_regex(copy(vocab))
    
    # iterate through texts
    output = []
    for i in range(len(texts)):
        vector = np.zeros((1,len(vocab)))
        for m in re.findall(regex,texts[i]): vector[0,hashdict[m]]+=1
        output.append(scipy.sparse.csr_matrix(vector))
    return scipy.sparse.vstack(output)

def tb_vectorizer(texts,vocab):
    """
    returns sparse boolean matrix; rows are texts, columns are vocab terms
    @param texts: a string or list of strings
    @param vocab: a list of vocab
    """
    #initialize
    if not texts: return scipy.sparse.csr_matrix((0,len(vocab)))
    if isinstance(texts,str): texts = [texts]
    global regex
    regex = text_tools.vocab_tools.vocab_regex(vocab)
    global hashdict
    hashdict = dict({k:v for v,k in enumerate(vocab)})
    
    # parallel if linux
    if platform.system()=='Linux':
        threadpool = Pool(cpu_count())
        output = threadpool.map(_tb_unix_thread,texts)
        threadpool.close()
        threadpool.join()
    
    # serial if not linux
    else:
        output = []
        for text in texts:
            vector = np.zeros((1,len(hashdict)))
            for m in set(re.findall(regex,text)): vector[0,hashdict[m]]=1
            output.append(scipy.sparse.csr_matrix(vector))

    # delete globals and return
    del regex
    del hashdict
    return scipy.sparse.vstack(output)

def _tb_unix_thread(text):
    vector = np.zeros((1,len(hashdict)))
    for m in set(re.findall(regex,text)): vector[0,hashdict[m]]=1
    return scipy.sparse.csr_matrix(vector)

def tf_vectorizer(texts,vocab,regex=None):
    """
    @param text: string or list of preprocessed strings
    @param vocab: preprocesed vocab
    @param regex: regular expression if available (higher speed)
    return: sparse matrix (rows are texts, columns are tf)
    """
    #initialize
    if isinstance(texts,str): texts = [texts] 
    hashdict = {k: v for v, k in enumerate(vocab)}
    if not regex: regex = text_tools.vocab_tools.vocab_regex(copy(vocab))
    
    # iterate through texts
    output = []
    for i in tqdm(range(len(texts))):
        vector = np.zeros((1,len(vocab)))
        for m in re.findall(regex,texts[i]): vector[0,hashdict[m]]+=1

        # return nothing if not enough information
        vars_detected = np.count_nonzero(vector)
        if vars_detected==0: output.append(scipy.sparse.csr_matrix((1,len(vocab))))
        
        # else sum of squares operation
        else:
            denominator = np.sqrt(np.sum(vector**2))
            vector      = vector/float(denominator)
            output.append(scipy.sparse.csr_matrix(vector))
    return scipy.sparse.vstack(output)

def trif_vectorizer(texts):
    """
    return a normalized vector of trigram character frequencies
    @param text: string or list of preprocessed strings
    # trigrams from 'abcdefghijklmnopqrstuvwxyz P' where P is punct
    """
    if isinstance(texts,str): texts = [texts]
    re_punct    = re.compile(r'[^a-z0-9\. ]')
    re_nonalpha = re.compile(r'[^a-zP\.]+')
    trigrams = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickles/trigrams'), 'rb'))
    hashdict = {k: v for v, k in enumerate(trigrams)}
    
    # iterate through texts
    output = []
    for i in range(len(texts)):
        
        # preprocessing for variable reduction
        texts[i] = re.sub(re_punct,r'P',texts[i].lower())
        texts[i] = re.sub(re_nonalpha,r'  ',texts[i])
        
        # counting
        vector = np.zeros((1,len(hashdict)),dtype='float64')
        counts = Counter(_ngrams(texts[i],3))
        for k,v in counts.items():
            if k in hashdict: vector[0,hashdict[k]]=v
        vector = normalize(vector).ravel()
        output.append(scipy.sparse.csr_matrix(vector))
    return scipy.sparse.vstack(output)


def _ngrams(text, n):
    while True:
        a, b = tee(text)
        l = tuple(islice(a, n))
        if len(l) == n:
            yield l
            next(b)
            text = b
        else: break
