import re
import collections
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

def upper_percent(text):
    total = len(re.findall(r'[A-Za-z]',text))
    if total==0: return 0.0
    return len(re.findall(r'[A-Z]',text))/total

def space_percent(text):
    if len(text)==0: return 0.0
    return len(re.findall(r'\s',text))/len(text)

def alpha_count(text):
    if len(text)==0: return 0.0
    return sum([i.isalpha() for i in text])

def alpha_percent(text):
    if len(text)==0: return 0.0
    return sum([i.isalpha() for i in text])/len(text)

def alphanumeric_percent(text):
    if len(text)==0: return 0.0
    return len(re.findall(r'[0-9a-zA-Z_]',text))/float(len(text))

def numeric_percent(text):
    if len(text)==0: return 0.0
    return sum([i.isdigit() for i in text])/len(text)

def punct_percent(text):
    if len(text)==0: return 0.0
    return len(re.findall(r'[\~\|\\\!\*\"\'\(\)\+\,\.\/\`\[\]\^\;\:\{\}\<\>\?\’\-\.]',text))/float(len(text))

def word_count(text):
    return len(re.findall(r'\b[a-zA-Z_]+\b',text))

def word_count_unique(text):
    return len(set(re.findall(r'\b[a-zA-Z_]+\b',text)))

def short_word_percent(text):
    words = re.findall(r'\b[a-zA-Z_]+\b',text)
    if len(words)==0: return 0.0
    return sum([1 for i in words if len(i)<4])/len(words)

def one_letter_word_percent(text):
    words = re.findall(r'\b[a-zA-Z_]+\b',text)
    if len(words)==0: return 0.0
    return sum([1 for i in words if len(i)==1])/len(words)

def word_length_average(text):
    words = re.findall(r'\b[a-zA-Z_]+\b',text)
    words = list(map(len,words))
    if not words: return 5.1 # english average
    else: return np.mean(words)

def word_length_frequencies(text,n=20):
    """
    @param n: frequency of wordlengths from 1 to n
    """
    word_lengths = list(map(len,re.findall(r'\b[a-zA-Z_]+\b',text)))
    if not word_lengths: return n*[0.0]
    return [word_lengths.count(i)/len(word_lengths) for i in range(1,n+1)]

def ttr(text):
    words = re.findall(r'\b[a-z_]+\b',text.lower())
    if len(words)==0: return 1.0
    return len(words)/len(set(words))

def letter_frequencies(text):
    letters = list('abcdefghijklmnopqrstuvwxyz_')
    total   = len(re.findall(r'[a-z_]',text.lower()))
    if total==0: return [0.0]*len(letters)
    counts = collections.Counter(text.lower())
    return [counts[i]/total for i in letters]

def punct_frequencies(text):
    punct = list(r"!?,()'-.\"")
    if len(text)==0: return [0.0]*len(punct)
    counts = collections.Counter(text)
    return [counts[i]/len(text) for i in punct]

def punct_delimited_word_frequencies(text):
    if len(text)==0: 0.0
    count = len(re.findall(r"""(?<=[a-zA-Z])'(?=[^a-zA-Z])|(?<=[^a-zA-Z])'(?=[a-zA-Z])""",text))
    return count

def hapax(text,n=1):
    # percent of words appearing n times
    words = re.findall(r'\b[a-zA-Z_]+\b',text)
    if len(words)==0: return 0.0
    return len([w for w in words if words.count(w)==n])/len(words)

def sentance_count(text):
    count = len(re.findall(r"""(?<=[a-zA-Z_])(\!|\?|\.)|(\!|\?|\.)(?=[a-zA-Z_])""",text))
    if count==0: return 1.0
    return count

def words_per_sentance(text):
    return word_count(text)/sentance_count(text)

def words_per_sentance_inverse(text):
    wordcount = word_count(text)
    if wordcount==0: return 1.0
    return sentance_count(text)/wordcount

def yules(text):
    """ 
    Yule's K
    (cf. Oakes, M.P. 1998. Statistics for Corpus Linguistics.
    International Journal of Applied Linguistics, Vol 10 Issue 2)
    In production this needs exception handling.
    """
    tokens = collections.Counter(re.findall(r'\b[a-zA-Z_]+\b',text.lower()))
    m1 = sum(tokens.values())
    m2 = sum([freq**2 for freq in tokens.values()])
    if m2-m1==0: return 0.0
    return (m1*m1)/(m2-m1)
