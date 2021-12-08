#!/usr/bin/env/python3

import string
import nltk
from nltk import pos_tag
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.stem import WordNetLemmatizer
from nltk.metrics import jaccard_distance
from nltk.probability import FreqDist
from nltk.collocations import BigramCollocationFinder
from nltk.collocations import TrigramCollocationFinder
from nltk.wsd import lesk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.corpus import sentiwordnet
from nltk.corpus import wordnet_ic

# here we will load the preprocessing utils 

# removing the punctuation and stop-words 

#first we want to extract all the words in lowercase without any punctuation marks
def strip_punctuation(sentence):
    '''
    Function which removes the punctuations taking it from the String library 
    
    Input:type: str
    Output:type: list(str)
    
    '''
    return [x for x in nltk.word_tokenize(sentence) if x not in string.punctuation]

# now we can remove the stopwords 

def strip_stopwords(sentence):
    '''
    Function which removes the punctuations and the stopwords from the nltk.stopwords.words('english') taking it from the String library 
    
    Input:type: str
    Output:type: list(str)
    
    '''
    return [x for x in nltk.word_tokenize(sentence) if x not in string.punctuation and x not in stopwords.words('english')]