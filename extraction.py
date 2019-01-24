# -*- coding: utf-8 -*-
###############################################################################
# MODULE: convolution
# this is a repo of tools designed extract text spans from a text
# 
# Author:   William Kinsman
# Created:  10.10.2015
###############################################################################

import re
import math
from bisect import bisect_left, bisect_right
from text_tools.vocab_tools import vocab_regex
from text_tools.resolution import fetch_dates

def build_convolutions(texts,windowsize=100,step=50,metadata=False):
    """
    return: list of tuples (page,wordid_start,wordid_end,text)
    @param text: string or list of strings
    @param windowsize: windowsize of sliding window in characters
    @param step: step size used
    """
    # initialize
    if isinstance(texts,str): texts = [texts]
    assert windowsize>2,"Error: windowsize must be at least 3. Aborting."
    assert step>0,      "Error: step must be greater than 0. Aborting."
    
    # for each page
    output = []
    token_count = 0
    for page_num in range(len(texts)):
        
        # define the delimiters
        text = texts[page_num].strip()
        if not text: continue
        textlen = len(text)
        delimiters = [m.start()+1 for m in re.finditer(r' ',text)]
        if 0 not in delimiters: delimiters = [0] + delimiters
        
        # shortcircuit on small text
        if textlen<=windowsize and text:
            output.append((page_num+1,token_count+1,token_count+len(delimiters),text))
            token_count+=len(delimiters)
            continue
        
        # define fence conditions
        if windowsize%2==0: windowsize-=1
        radius = int((windowsize-1)/2)
        idx_s = radius
        idx_e = radius + step*(1+math.floor((textlen-(2*radius)-1)/step))
        
        # get span candidates and their center indices
        spans = [(max(i-radius,0),min(i+radius+1,textlen)) for i in range(idx_s,idx_e,step)]
        
        # delimit to defined delimiters
        span_output = []
        for i in range(len(spans)):
            s = bisect_left(delimiters,spans[i][0])
            if s>=len(delimiters): continue
            s = delimiters[s]
            e = delimiters[bisect_right(delimiters,spans[i][1])-1]
            if s<e: span_output.append((s,e))
        span_output = _remove_duplicates_maintain_order(span_output)
        
        # append findings for that page
        token_dict = dict(map(lambda t: (t[1], t[0]), enumerate(delimiters,start=token_count+1)))
        output += [(page_num+1,token_dict[i[0]],token_dict[i[1]]-1,text[i[0]:i[1]].strip()) for i in span_output]
        token_count+=len(delimiters)
    
    # if metadata requested
    if metadata:
        if output: return output
        else: return [(0,0,0,'')]
    
    # else just text windows
    return [i[3] for i in output]


def windows_by_index(text,indices,radius,return_index=False):
    """
    return: windows of text centered on index
    @param text: string or list of texts
    @param indices: int or list of ints
    @param radius: character radius to use
    @param return_index: if true returns (text,start_token_index)
    """
    #initialize
    if isinstance(indices,int): indices = [indices]
    wordbounds = [m.start() for m in re.finditer(r'\b',text)]
    textmax = len(text)
    indices = [i for i in indices if i>=0 and i<textmax]
    
    # for each index, move to word boundaries
    output = []
    for j in indices:
        start = max(j-radius,0)
        end   = min(j+radius,textmax)
        while start not in wordbounds and start!=textmax: start+=1
        while end not in wordbounds and end!=0: end-=1
        if return_index: output.append((text[start:end],start))
        else: output.append(text[start:end])
    return output

def windows_by_terms(texts,terms,radius=50,ignore_case=False):
    """
    return: windows of text centered on the term
    @param text: string or list of texts
    @param term: term to use
    @param radius: character radius to use
    @param punct_delimit: If true, stops on case of first stop punct
    """
    #initialize
    if isinstance(texts,str): texts = [texts]
    regex = vocab_regex(terms,ignorecase=False)
    regex_bounds = re.compile(r'\b')
    output = []
    
    # for each text input
    for i in range(len(texts)):
        
        # get start indices
        textmax = len(texts[i])
        locs = re.finditer(regex,texts[i])
        locs = [(m.start(),m.end()) for m in locs]
        wordbounds = [m.start() for m in re.finditer(regex_bounds,texts[i])]
        
        # for each index, move to word boundaries
        for j in locs:              
            start = max(j[0]-radius,0)
            end   = min(j[1]+radius,textmax)
            while start not in wordbounds and start!=textmax: start+=1
            while end not in wordbounds and end!=0: end-=1
            output.append(texts[i][start:end])        
    return output

def windows_by_dates(texts,radius,return_dates=False):
    """
    return: windows of text centered on dates
    @param text: string or list of texts
    @param radius: character radius to use
    """
    #initialize
    if isinstance(texts,str): texts = [texts]
    regex_bounds    = re.compile(r'\b')
    output_spans    = []
    output_dateobjs = []
    
    # for each text input
    for i in range(len(texts)):
        
        textmax    = len(texts[i])
        dateobjs,locs = fetch_dates(texts[i],return_ci=True)
        wordbounds = [m.start() for m in re.finditer(regex_bounds,texts[i])]
        output_dateobjs+=dateobjs
        
        # for each index, move to word boundaries
        for j in locs:
            start = max(j-radius,0)
            end   = min(j+radius,textmax)
            while start not in wordbounds and start!=textmax: start+=1
            while end not in wordbounds and end!=0: end-=1
            output_spans.append(texts[i][start:end])
    if return_dates: return list(zip(output_spans,output_dateobjs))
    return output_spans

def _remove_duplicates_maintain_order(seq):
    # removed duplicates maintaining order
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]