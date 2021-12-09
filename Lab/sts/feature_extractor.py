#!/usr/bin/env/python3

from nltk.metrics import jaccard_distance
import Levenshtein as lev
from fuzzywuzzy import fuzz

def jd(sentence1, sentence2,is_set=False):
    '''
    Function which calculates the jaccard-distance 
    input:type: list(tokens)
    output:type: float
    '''
    if is_set:
        return 1-jaccard_distance(set(sentence1), set(sentence2))
    else:
        return 1-jaccard_distance(sentence1, sentence2) 
    

def jd_fuzz_lev(SENTENCE_A, SENTENCE_B, X_TRAIN):
    '''
    Function which calculates: 
    
    Jaccard Distance for: (nltk.metrics.jaccard_distance)
        - No Punctuation, no punctuation or stopwords, and lemmas 
        
    Fuzzy String Matching Ratio: (fuzzywuzzy)
        - original sentence 
    
    Levenshtein Distance and Ratio: (Levenshtein)
        - Original Sentence 
        
        
    input:type: pd.DataFrame
    output:type: pd.DataFrame
    

    '''


    # creating a copy 
    X_features = X_TRAIN.copy()
    #dropping one of the columns 
    X_features = X_features.drop('SentA',axis=1)
    # empty column to append the jacc distance 
    X_features['jacc_nopunc'] = ''
    X_features['jacc_nopunc_stop'] = ''
    X_features['jacc_lemmas'] = ''
    X_features['fuzzy_ratio'] = ''
    X_features['lev_ratio'] = ''
    X_features['lev_distance'] = ''
    for index in X_features.index:
        # no punctuation 
        X_features['jacc_nopunc'][index] = jd(SENTENCE_A['SentA_nopunc'][index], SENTENCE_B['SentB_nopunc'][index],is_set=True)
        # no punctuation or stopwords 
        X_features['jacc_nopunc_stop'][index] = jd(SENTENCE_A['SentA_nopunc_stop'][index], SENTENCE_B['SentB_nopunc_stop'][index], is_set=True)
        #lemmas 
        X_features['jacc_lemmas'][index] = jd(SENTENCE_A['SentA_lemmas'][index], SENTENCE_B['SentB_lemmas'][index],is_set=True)
        # FuzzyWuzzy String Matching
        X_features['fuzzy_ratio'][index] = fuzz.ratio(SENTENCE_A['SentA'][index].lower(), SENTENCE_B['SentB'][index].lower())
        # Levenshtein Ratio 
        X_features['lev_ratio'][index] = lev.ratio(SENTENCE_A['SentA'][index].lower(), SENTENCE_B['SentB'][index].lower())
        # Levenshtein Distance -> Number of edits for them to be the same 
        X_features['lev_distance'][index] = lev.distance(SENTENCE_A['SentA'][index].lower(), SENTENCE_B['SentB'][index].lower())
        
    #dropping one of the columns 
    X_features = X_features.drop('SentB',axis=1)
    return X_features
    