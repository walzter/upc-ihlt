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
nltk.download('stopwords')

# defining the stopwords 
stopwords = nltk.corpus.stopwords.words('english')

#first we want to extract all the words in lowercase without any punctuation marks
def strip_punctuation(sentence):
    '''
    Function which removes the punctuations taking it from the String library 
    
    Input:type: str
    Output:type: list(str)
    
    '''
    return [x for x in nltk.word_tokenize(sentence) if x not in string.punctuation]

# now we can remove the stopwords 

def strip_stopwords_punctuation(sentence):
    '''
    Function which removes the punctuations and the stopwords from the nltk.stopwords.words('english') taking it from the String library 
    
    Input:type: str
    Output:type: list(str)
    
    '''
    return [x for x in nltk.word_tokenize(sentence) if x not in string.punctuation and x not in stopwords]


# POS-TAG Converter
# Mapping of Stanford POS-tag to WordNet type

def penn2morphy(penntag, returnNone=False):
    '''
    Function which converts the Stanford POS-tag to the corresponding Wordnet one
    input:type: str
    output:type: str 
    
    '''
    morphy_tag = {'NN':wordnet.NOUN, 'JJ':wordnet.ADJ,
                  'VB':wordnet.VERB, 'RB':wordnet.ADV}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return None if returnNone else ''
    
    
# POS TAGGING
    
def get_pos_tag(tokenized_sentence):
    '''
    Function which given a tokenized sentence will return the corresponding POS tag 
    
    input:type: list(tokenized words )
    output:type: list (tuple (str, str)) -> list(tuple (token, POS-TAG))
    
    '''
    return nltk.pos_tag(tokenized_sentence)
    
def get_lemmas(POS_TAG):
    '''
    Function which "lemmatizes" the word with a given POS_tag
    
    input:type: tuple (word, POS-Tag)
    output:type: list of lemmas
    
    '''
    list_of_lemmas = []
    wnl = WordNetLemmatizer()
    for token_pos_pair in POS_TAG:
        morphed_tag = penn2morphy(token_pos_pair[1]).lower()
        if morphed_tag in {'n','v','r','a'}:
            list_of_lemmas.append(wnl.lemmatize(token_pos_pair[0],pos=morphed_tag))
        else:
            list_of_lemmas.append(token_pos_pair[0])
    return list_of_lemmas


def clean_replace_unwanted_chars(tokenized_sentence_list):
    '''
    Function which removes any unwanted characters from the tokens:
    - apostrophes when possessive is shown (Peter's) or (Peters')
    
    input:type:list (tokenized sentence)
    output:type:list (cleaned tokenized sentence)
    
    '''
    for idx, token in enumerate(tokenized_sentence_list):
        ctoken = token.translate(str.maketrans('','',string.punctuation))
        tokenized_sentence_list[idx] = ctoken
    return tokenized_sentence_list
        
