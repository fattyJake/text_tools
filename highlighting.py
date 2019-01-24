# -*- coding: utf-8 -*-

import re
import itertools
from text_tools import vocab_tools

def highlight_by_term(text,terms,color='#6cbbf7'):
    """
    return: windows of text centered on the term
    @param text: string or list of texts
    @param terms: terms to use
    @param color: hex color to use
    """
    #initialize
    lower_text = text.lower()
    regex = vocab_tools.vocab_regex(terms)
            
    # get start indices
    locs = [(m.start(),m.end()) for m in re.finditer(regex,lower_text)]
    locs = sorted(_merge_tuples(locs), key=lambda x: x[0], reverse=True)
    for j in locs:
        text = text[:j[0]] + """<span style="background-color: """+color+"""">""" + text[j[0]:j[1]] + """</span>""" + text[j[1]:]
    return text

def highlight_by_token(text,locs,color='#6cbbf7', title='Test'):
    """
    return: windows of text centered on the term
    @param text: string or list of texts
    @param locs: list of token start and end, along with display tag (optional)
    @param color: hex color to use
    @param title: HTML title
    """
    # get token dict
    text = text.strip()
    delimiters = [m.start()+1 for m in re.finditer(r'\s+', text)]
    if 0 not in delimiters: delimiters = [0] + delimiters
    if len(text) not in delimiters: delimiters = delimiters + [len(text)]
    token_dict = {delimiters.index(d):d for d in delimiters}
    
    # get start indices
    locs.sort(reverse=True)
    locs = list(locs for locs, _ in itertools.groupby(locs))
    locs = _cluster_highlights(locs)

    if len(locs[0]) == 2:
        for j in locs: text = text[: token_dict[j[0]]] + """<span class="tooltip" style="background-color: """+color+"""">""" + text[token_dict[j[0]]: token_dict[j[1]]] + """</span>""" + text[token_dict[j[1]]:]
    if len(locs[0]) == 3:
        for j in locs: text = text[: token_dict[j[0]]] + """<span class="tooltip" style="background-color: """+color+"""">""" + text[token_dict[j[0]]: token_dict[j[1]]] + """<span class="tooltiptext">"""+str(j[2]).upper()+"""</span></span>""" + text[token_dict[j[1]]:]
    css = """<!DOCTYPE html>
        <html>
        <style>
        .tooltip {
            position: relative;
            display: inline;
            border-bottom: 1px dotted black;
        }
        
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 120px;
            background-color: #555;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 5px 0;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -60px;
            opacity: 0;
            transition: opacity 0.3s;
        }
        
        .tooltip .tooltiptext::after {
            content: "";
            position: absolute;
            top: 100%;
            left: 50%;
            margin-left: -5px;
            border-width: 5px;
            border-style: solid;
            border-color: #555 transparent transparent transparent;
        }
        
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        </style>
        
        <h3>"""+title+"""</h3>
        
        <p style="white-space: pre-line">
        """
    text = css + text + '\n</p>\n</body>\n</html>'
    return text

def _merge_tuples(tuples):
    if not tuples: return []
    tuples = sorted(tuples, key=lambda x: x[0])
    out = []
    saved = list(tuples[0])
    for st, en, tag in sorted([sorted(t) for t in tuples]):
        if st <= saved[1]:
            saved[1] = max(saved[1], en)
        else:
            out.append(tuple(saved))
            saved[0] = st
            saved[1] = en
    out.append(tuple(saved))
    return out

def _cluster_highlights(locs):
    """
    @param locs
    """
    # for each finding walk backwards and merge if overlapping
    for i in range(len(locs)-2,-1,-1):
        # if overlapping findings, then merge them
        if locs[i][0] < locs[i+1][1]:
            cluster_list = [locs[i][0], locs[i][1], locs[i+1][0], locs[i+1][0]]
            locs[i][0] = min(cluster_list)
            locs[i][1] = max(cluster_list)
            if len(locs[i]) == 3:
                locs[i][2] = ', '.join(list(set([locs[i][2], locs[i+1][2]])))
            locs.pop(i+1)
    return locs

    



