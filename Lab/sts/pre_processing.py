#!/usr/bin/env/python3

import numpy as np
import string, nltk
from nltk import pos_tag, word_tokenize, pos_tag, ne_chunk
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
    return [x for x in word_tokenize(sentence) if x not in string.punctuation]

# now we can remove the stopwords 

def strip_stopwords_punctuation(sentence):
    '''
    Function which removes the punctuations and the stopwords from the nltk.stopwords.words('english') taking it from the String library 
    
    Input:type: str
    Output:type: list(str)
    
    '''
    return [x for x in word_tokenize(sentence) if x not in string.punctuation and x not in stopwords]


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
    return pos_tag(tokenized_sentence)
    
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
        

def get_tfidf(list_of_sentences):
    '''
    Function which returns the term-frequency inverse-data-frequency in a given corpus 
    The corpus her is a list of all the sentences. 
    Mainly, the higher frequencies have less meaning, and the less frequent are more important because they can transmit more information. 
    
    input:type: list
    output:type: dict
    
    '''
    
    def get_tf(list_of_sentences):
        # initiatie a new frequency distribution

        frequency_distribution = FreqDist()
        total_freq = 0
        
        for sentence in list_of_sentences:
            all_words = strip_punctuation(sentence)
            
            for word in all_words:
                # Sum 1 to the freq of the word 
                frequency_distribution[word.lower()] += 1
                
                # Sum 1 to the total nb of words
                total_freq += 1
        return frequency_distribution, total_freq
    
    def get_idf(freq_dist, total_frequency):
        # get the inverse document frequency in order to "normalize" the data 
        idf = dict()
        for key in freq_dist.keys():
            idf[key] = np.log((total_frequency / freq_dist[key]))
        return idf
    
    
    frequency_distribution, total_freq = get_tf(list_of_sentences)
    tfidf = get_idf(frequency_distribution, total_freq)
    return tfidf


def get_bleu_score(list_of_sentences):
    '''
    Function which returns the BLEU score of a list of sentences S, where the hypothesis is S[idx] for S-1 references
    
    input:type: list(str)
    output:type: list(float)
    
    '''
    bleu_score = []
    for idx, sentences in enumerate(list_of_sentences):
        hypothesis = list_of_sentences[idx]
        references = list_of_sentences[idx:]
        try:
            BLEU_SCORE = nltk.translate.bleu_score.sentence_bleu(references, hypothesis)
        except Exception as e: 
            BLEU_SCORE = 0
            continue
        bleu_score.append(BLEU_SCORE)
        
    return bleu_score


def get_processed_sentences(TRAINING_DATAFRAME):
    
    '''
    Function which creates a single dataframe for each sentence and then applies the preprocessing to it. 
    
    input:type: pandas.DataFrame
    output:type: pandas.DataFrame
    
    
    '''
    
    # sentence A 
    SA = TRAINING_DATAFRAME.copy().drop('SentB',axis=1)
    SA['SentA_nopunc'] = SA.SentA.apply(lambda x: strip_punctuation(x))
    # removing stopwords
    SA['SentA_nopunc_stop'] = SA.SentA.apply(lambda x: strip_stopwords_punctuation(x))
    # getting the pos-tag
    SA['SentA_pos'] = SA.SentA_nopunc_stop.apply(lambda x: get_pos_tag(x))
    # getting the lemmas
    SA['SentA_lemmas'] = SA.SentA_pos.apply(lambda x: get_lemmas(x))
    
    # Sentence B
    SB = TRAINING_DATAFRAME.copy().drop('SentA',axis=1)
    SB['SentB_nopunc'] = SB.SentB.apply(lambda x: strip_punctuation(x))
    # removing stopwords
    SB['SentB_nopunc_stop'] = SB.SentB.apply(lambda x: strip_stopwords_punctuation(x))
    # getting the pos-tag
    SB['SentB_pos'] = SB.SentB_nopunc_stop.apply(lambda x: get_pos_tag(x))
    # getting the lemmas
    SB['SentB_lemmas'] = SB.SentB_pos.apply(lambda x: get_lemmas(x))
    
    return SA, SB