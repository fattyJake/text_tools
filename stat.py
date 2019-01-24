import re
import numpy as np

def clust_coef(text,term='word'):
    """
    @param text: a string
    @param term: 'alpha','num','alphanum','punct','word' or custom
    return: clustering coefficient (between 0 and 1)
    """
    # 1. find max possible
    if term=='word':
        text = re.sub(r'\b[a-zA-Z]+\b','w',text)
        text = re.sub(r' w','w',text)
        term = '[w]'
    elif term=='alpha': term = r'[a-zA-Z]'
    elif term=='num': term = r'[0-9]'
    elif term=='punct':
        text = re.sub(r' ','',text)
        term = r'[\~\|\_\\\!\*\"\'\(\)\+\,\.\/\`\[\]\^\;\:\{\}\<\>\?\n\t\r\f]'
    denominator = (len(re.findall(term,text))*2)-2
    if denominator==0: return 1.0
    bef = len(re.findall(r'(?<=' + term + ')' + term,text))
    aft = len(re.findall(term + '(?=' + term + ')',text))
    return (aft+bef)/float(denominator)

def numeric_percent(text):
    return sum([i.isdigit() for i in text])/len(text)

def alphanumeric_percent(text):
    return len(re.findall(r'[0-9a-zA-Z]',text))/float(len(text))

def alpha_count(text):
    return sum([i.isalpha() for i in text])

def alpha_percent(text):
    return sum([i.isalpha() for i in text])/len(text)


def numericpunct_percent(text):
    return len(re.findall(r'[\~\|\_\\\!\*\"\'\(\)\+\,\.\/\`\[\]\^\;\:\{\}\<\>\?\n\t\r\f 0-9]',text))/float(len(text))
    
def word_count(text):
    return len(re.findall(r'\b[a-zA-Z]+\b',text))

def avg_word_length(text):
    out = re.findall(r'\b[a-zA-Z]+\b',text)
    out = [i for i in map(len,out)]
    if not out: return None
    else: return np.mean(out)

def word_count_unique(text):
    return len(set(re.findall(r'\b[a-zA-Z]+\b',text)))

def ttr(text):
    words = re.findall(r'\b[a-zA-Z]+\b',text)
    return len(words)/len(set(words))

  
  