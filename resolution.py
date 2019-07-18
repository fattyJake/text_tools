# -*- coding: utf-8 -*-

import re
import pyap
from datetime import datetime
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

import text_tools.extraction
import text_tools.vocab_tools
from text_tools import preprocessing,words

def fetch_capped_chains(text):
    """
    @param text: str
    return: all unique capitalized chains
    """
    # get all unique capped chain starts
    if isinstance(text,list): text = '\n'.join(text)
    text = preprocessing.remove_false_periods(text)
    locs = [i.start() for i in re.finditer(r'\b[A-Z][a-zA-Z]+\b',text) if i.group().lower() not in ENGLISH_STOP_WORDS]
    for i in range(len(locs)-1,-1,-1):
        if words.prevwordindex(text,locs[i]) in locs: del locs[i]
    
    # append new words
    output = set()
    for i in range(len(locs)):
        phrase = words.fullword(text,locs[i])
        locs[i] = words.nextwordindex(text,locs[i])
        while locs[i]!=None and text[locs[i]].isupper() and words.fullword(text,locs[i]).lower() not in ENGLISH_STOP_WORDS:
            phrase = phrase + r" " + words.fullword(text,locs[i])
            locs[i] = words.nextwordindex(text,locs[i])
        output.add(phrase)
    output = list(output)
    output.sort()
    return output

def fetch_addresses(text):
    """
    Pull USA addresses from text
    """
    # initialize
    if isinstance(text,list): text = '\n'.join(text)
    
    # find strings that are addresses that have a zip code in them
    return [str(i) for i in pyap.parse(text,country='US') if re.search(r'\b\d\d\d\d\d\b',str(i))]

def fetch_dates(text,century_reversion=None,return_ci=False):
    """
    return the dates a date objects
    @param text: a string
    @param century_reversion: if 2-digit year and yy>century_reversion, uses previous century
                              int, or datetimeobj. IF none ignores, only seeks YYYY not YY
    @param return_ci: if true returns tuples of (date_obj,center_index)
    NOTE: does not capture '7 Feb 18' or '7 Feb 2018'
    """
    #  initialize
    if isinstance(text,list): text = '\n'.join(text)
    if isinstance(century_reversion,int):        century_reversion=century_reversion+int(str(datetime.now().year)[-2:])
    elif isinstance(century_reversion,datetime): century_reversion=int(str(century_reversion.year)[-2:])
    
    # find all explicitely numeric dates
    dateobjs = []
    date_list = re.finditer(r'\b\d{4}[-]\d{1,2}[-]\d{1,2}\b|\b\d{1,2}[-]\d{1,2}[-]\d{4}\b|\b\d{1,2}[-]\d{1,2}[-]\d{2}\b|\b\d{4}[\.]\d{1,2}[\.]\d{1,2}\b|\b\d{1,2}[\.]\d{1,2}[\.]\d{4}\b|\b\d{1,2}[\.]\d{1,2}[\.]\d{2}\b|\b\d{4}[/]\d{1,2}[/]\d{1,2}\b|\b\d{1,2}[/]\d{1,2}[/]\d{4}\b|\b\d{1,2}[/]\d{1,2}[/]\d{2}\b',text)
    date_list = [(i.group().lower(),int(i.start()+((i.end()-i.start())/2))) for i in date_list]
    ci        = [i[1] for i in date_list]
    date_list = [i[0] for i in date_list]
    re_nums = re.compile(r'[\d]+')
    for date in date_list:
        nums = re.findall(re_nums,date)
        date = '/'.join(nums)
        
        # try exists so we can use datetime parser for success/failure
        try:
            if len(nums[0])==4: dateobjs.append(datetime.strptime(date,"%Y/%m/%d").date())        # if 4-year leading
            elif len(nums[2])==4: dateobjs.append(datetime.strptime(date,"%m/%d/%Y").date())      # if 4-year lagging
            elif century_reversion is not None:                                            # if 2-year lagging, assume century depending on current date
                
                # if 'mm.-/dd.-/yy' form lagging form, and year >= century revision year, drop a century
                if int(nums[2])>century_reversion: nums[2] = '19' + nums[2]
                else: nums[2] = '20' + nums[2]
                date = '/'.join(nums)
                dateobjs.append(datetime.strptime(date,"%m/%d/%Y").date())
        except: continue

    # language date search (SEP 30th, 2011 -> long or abbrev, day, 4digit year)
    date_list = re.finditer(r'(\b\d{1,2}\D{0,3})?\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep?(?:tember)?|oct(?:ober)?|(nov|dec)(?:ember)?)\D{1,2}(\d{1,2}(st|nd|th)?\D?)?\D?(\d{4})',text,flags=re.IGNORECASE)
    date_list = [(i.group().lower(),int(i.start()+((i.end()-i.start())/2))) for i in date_list]
    ci        = ci + [i[1] for i in date_list]
    date_list = [i[0] for i in date_list]
    for date in date_list:
        try:
            date = date.split()
            date[1] = re.findall(re_nums,date[1])[0]
            datestr = ' '.join(date)
            if len(date[0])==3:
                dateobjs.append(datetime.strptime(datestr,"%b %d %Y").date())
            else:
                dateobjs.append(datetime.strptime(datestr,"%B %d %Y").date())
        except: continue
    
    # return results
    if return_ci:   return dateobjs,ci
    else:           return dateobjs

def fetch_date_by_terms(texts,terms,radius=15):
    """
    return single most common dates assosciated with terms
    @param texts: input a text or list of texts
    @param terms: term or list of terms to use (e.g. DOB)
    @param radius: the character radius to use
    """
    # initialize
    if isinstance(terms,str): terms = [terms]
    regex   = text_tools.vocab_tools.vocab_regex(terms)
    windows = text_tools.extraction.windows_by_dates(texts,radius=radius,return_dates=True)
    dates   = [i[1] for i in windows if re.search(regex,i[0])]
    
    # get the most common date with the term
    if dates: return max(set(dates), key=dates.count)
    else: return None