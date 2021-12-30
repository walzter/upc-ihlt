#!/usr/bin/env/python3
import numpy as np
import os
import glob
import Levenshtein as lev
from fuzzywuzzy import fuzz
import pandas as pd
import string
import nltk
from nltk import pos_tag, word_tokenize, pos_tag, ne_chunk
from nltk.corpus import wordnet,wordnet_ic
from nltk.corpus import sentiwordnet
from nltk.corpus import stopwords
from nltk.wsd import lesk
from nltk.collocations import BigramCollocationFinder
from nltk.collocations import TrigramCollocationFinder
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from nltk.wsd import lesk
from nltk.metrics import jaccard_distance

# downloads
nltk.download('stopwords')
nltk.download('maxent_ne_chunker')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('words')
nltk.download('sentiwordnet')
nltk.download('wordnet_ic')
# setting the wordnet_ic 
brown_ic = wordnet_ic.ic('ic-brown.dat')
# defining the stopwords 
stopwords = nltk.corpus.stopwords.words('english')

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
        X_features['lev_distance'][index] = levv.distance(SENTENCE_A['SentA'][index].lower(), SENTENCE_B['SentB'][index].lower())
        
    #dropping one of the columns 
    X_features = X_features.drop('SentB',axis=1)
    return X_features
    
    

def get_ngram_features(SA, SB):
    '''
    Function which extracts unigram, bigram, and trigram for:
    - lemmas, words_no_punct, words_no_punct_no_stop
    
    input:type: list
    output:type: pandas.DataFrame
    
    '''
    
    la = SA.SentA.tolist()
    lb = SB.SentB.tolist()
    IMPORTANCE = get_tfidf(la + lb)
    MAX_IMPORTANCE = max(IMPORTANCE.values())
    min_importance= min(IMPORTANCE.values())
    
    ngrams_dict = {
                'unigram_lemmas':[],
                'unigram_words':[],
                'unigram_words_filt':[],
                'unigram_lemmas_imp':[],
                'unigram_words_imp':[],
                'unigram_words_imp_filt':[],
                'bigram_lemmas':[],
                'bigram_words':[],
                'bigram_words_filt':[],
                'trigram_lemmas':[],
                'trigram_words':[],
                'trigram_words_filt':[]
                }
    for idx, _ in enumerate(SA['SentA_lemmas']):
        #unigram
        unigram_lemmas = unigram_similarity(SA['SentA_lemmas'][idx],SB['SentB_lemmas'][idx])
        unigram_words = unigram_similarity(SA['SentA_nopunc'][idx],SB['SentB_nopunc'][idx])
        unigram_words_filt = unigram_similarity(SA['SentA_nopunc_stop'][idx],SB['SentB_nopunc_stop'][idx])
        
        #unigram importance
        unigram_lemmas_imp = unigram_similarity_importance(SA['SentA_lemmas'][idx],SB['SentB_lemmas'][idx],IMPORTANCE,MAX_IMPORTANCE)
        unigram_words_imp = unigram_similarity_importance(SA['SentA_nopunc'][idx],SB['SentB_nopunc'][idx],IMPORTANCE,MAX_IMPORTANCE)
        unigram_words_imp_filt = unigram_similarity_importance(SA['SentA_nopunc_stop'][idx],SB['SentB_nopunc_stop'][idx],IMPORTANCE,MAX_IMPORTANCE)
        
        #bigrams
        bigram_lemmas = bigram_similarity(SA['SentA_lemmas'][idx],SB['SentB_lemmas'][idx])
        bigram_words = bigram_similarity(SA['SentA_nopunc'][idx],SB['SentB_nopunc'][idx])
        bigram_words_filt = bigram_similarity(SA['SentA_nopunc_stop'][idx],SB['SentB_nopunc_stop'][idx])
        
        trigrams
        trigram_lemmas = trigram_similarity(SA['SentA_lemmas'][idx],SB['SentB_lemmas'][idx])
        trigram_words = trigram_similarity(SA['SentA_nopunc'][idx],SB['SentB_nopunc'][idx])
        trigram_words_filt = trigram_similarity(SA['SentA_nopunc_stop'][idx],SB['SentB_nopunc_stop'][idx])
        
        #Updating dict
        ngrams_dict['unigram_lemmas'].append(unigram_lemmas)
        ngrams_dict['unigram_words'].append(unigram_words)
        ngrams_dict['unigram_words_filt'].append(unigram_words_filt)
        ngrams_dict['unigram_lemmas_imp'].append(unigram_lemmas_imp)
        ngrams_dict['unigram_words_imp'].append(unigram_words_imp)
        ngrams_dict['unigram_words_imp_filt'].append(unigram_words_imp_filt)
        ngrams_dict['bigram_lemmas'].append(bigram_lemmas)
        ngrams_dict['bigram_words'].append(bigram_words)
        ngrams_dict['bigram_words_filt'].append(bigram_words_filt)
        ngrams_dict['trigram_lemmas'].append(trigram_lemmas)
        ngrams_dict['trigram_words'].append(trigram_words)
        ngrams_dict['trigram_words_filt'].append(trigram_words_filt)
    df = pd.DataFrame(ngrams_dict)
    return df

# Lexical similarities
# getting the synset similarity
# we use the lemmas of SA and SB

def get_max_sim_synset(lemmaA, lemmaB,brown_ic):
  d = dict()
  # getting the synsets: 
  syn1 = wordnet.synsets(lemmaA)
  syn2 = wordnet.synsets(lemmaB)
  for asynset in syn1:
    for bsynset in syn2:  
      sims = get_similarities(asynset, bsynset,brown_ic)
      max_key = max(sims, key=sims.get)
      d[(lemmaA,lemmaB,max_key)] = sims[max_key]
  return d

def get_length_features(SA,SB):
    '''
    Function which extracts length of:
     - lemmas 
     - words 
    
    input:type: list
    output:type: pandas.DataFrame
    
    '''
    length_dict = {
            'ld_lemma':[],
            'ld_words':[]
            }
    for idx, lemma in enumerate(SA['SentA_lemmas']):
        ld_lemma = length_difference(lemma,SB['SentB_lemmas'][idx])
        ld_words = length_difference(SA['SentA_nopunc'][idx],SB['SentB_nopunc'][idx])
        length_dict['ld_lemma'].append(ld_lemma)
        length_dict['ld_words'].append(ld_words)
    df = pd.DataFrame(length_dict)
    return df

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
        cdtoken = token.translate(str.maketrans('','',string.punctuation))
        tokenized_sentence_list[idx] = cdtoken
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
        references = list_of_sentences[idx]
        try:
            BLEU_SCORE = nltk.translate.bleu_score.sentence_bleu(references, hypothesis)
        except Exception as e: 
            BLEU_SCORE = 0
            continue
        bleu_score.append(BLEU_SCORE)
        
    return bleu_score



computed_synsets_sim = {}
def wordnet_similarity(s1, s2, sim):
    if sim == "path" and s1 is not None and s2 is not None:
        return s1.path_similarity(s2)
    
    elif sim == "lch" and s1 is not None and s2 is not None and s1.pos == s2.pos:
        return s1.lch_similarity(s2)
    
    elif sim == "wup" and s1 is not None and s2 is not None:
        return s1.wup_similarity(s2)
    
    elif sim == "lin" and s1 is not None and s2 is not None and s1.pos == s2.pos and s1.pos in {'n', 'v', 'r', 'a'}:
        return s1.lin_similarity(s2)
    
    else:
        return None

def max_similarity_synsets(l1, l2, sim):
    # If are the same we return the max value
    if l1 == l2:
        if sim == "lch":
            return 3
        else:
            return 1
        
    # If we have computed before the similarity we don't compute anything
    elif (l1,l2,sim) in computed_synsets_sim:
        return computed_synsets_sim[(l1,l2,sim)]
    synsets1 = wordnet,synsets(l1)
    synsets2 = wordnet.synsets(l2)
    similarities = []
    for s1 in synsets1:
        for s2 in synsets2:
            similarity = wordnet_similarity(s1, s2, sim)
            if similarity is not None:
                similarities.append(similarity)
    if len(similarities) > 0:
        computed_synsets_sim[(l1,l2,sim)] = max(similarities)
        return max(similarities)
    else:
        computed_synsets_sim[(l1,l2,sim)] = 0
        return 0
def get_similarities(word1, word2,brown_ic,ret_dict=None):

  '''
  input --> Word1, Word2
  Output --> dict of similarities 

  '''
  simil_dict = dict()

  def path_sim(word1,word2): 
    return word1.path_similarity(word2)
  def lch_sim(word1,word2): 
    return word1.lch_similarity(word2)
  def wup_sim(word1,word2): 
    return word1.wup_similarity(word2)
  def lin_sim(word1,word2,brown_ic): 
    '''
    needs information content (IC) of LCS (least common subsumer)
    '''
    return word1.lin_similarity(word2,brown_ic)

  simil_dict['PATH_SIMIL'] = path_sim(word1,word2)
  simil_dict['LCdH_SIMIL'] = lch_sim(word1,word2)
  simil_dict['WUP_SIMIL'] = wup_sim(word1,word2)
  simil_dict['LINw_SIMIL'] = lin_sim(word1,word2,brown_ic)
  all_sims = [path_sim(word1,word2),lch_sim(word1,word2),wup_sim(word1,word2),lin_sim(word1,word2,brown_ic)]
  if ret_dict==True:
    return simil_dict
  elif ret_dict==False:
    return all_sims

def extract_features(DATASET,scaled=None):
    '''
    Function which extracts features: 
    
    - jaccard distance, fuzzywuzzy string match, levenshtein distance 
    - ngram features (unigram, bigram, trigram )
    - length features (length of lemmas, length of words_no_punct)
    
    input:type: pandas.DataFrame
    output:type: pandas.DataFrame    
    '''
    SAK, SBK = get_processed_sentences(DATASET)
    # Jaccard_Fuzzy_Lev
    feature_df = jd_fuzz_lev(SAK, SBK, DATASET)
    #unigram, bigram and trigram features
    ngram_features = get_ngram_features(SAK, SBK)
    #features related to the length
    length_features = get_length_features(SAK, SBK)
    #similarity measurements 
    #synset_sim_feat = get_similarity_measure_dict(SAK, SBK)
    # combining all the features 
    FINAL_DATASET = pd.concat([feature_df,ngram_features, length_features],axis=1)
    
    if scaled == True:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(FINAL_DATASET.values)
        SCALED_VALS = scaler.transform(FINAL_DATASET.values)
        return SCALED_VALS
    else: 
        return FINAL_DATASET

def get_similarity_measure_dict(SA, SB):
    '''
        Given a list of lemmas it will get their synsets and calculate different similarity metrics: 
        - LCH 
        - Wu-Palmer
        - Lin
        - Path

        input:type: list
        output:type: pandas.DataFrame
    '''

    similarity_dict = {
                        "sym_lch":[],
                        "sym_wp":[],
                        "sym_lin":[],
                        "sym_path":[]
                        }
    # calculate all the similarities with all the methods for all the words and their synsets 
    for idx, lemma in enumerate(SA['SentA_lemmas']):
        # metrics
        sym_lch = synsets_similarity(lemma,SB['SentB_lemmas'][idx],sim='lch')
        sym_wp = synsets_similarity(lemma,SB['SentB_lemmas'][idx],sim='wup')
        sym_lin = synsets_similarity(lemma,SB['SentB_lemmas'][idx],sim='lin')
        sym_path = synsets_similarity(lemma,SB['SentB_lemmas'][idx],sim='path')
        # appending to the dictionary
        similarity_dict['sym_lch'].append(sym_lch)
        similarity_dict['sym_wp'].append(sym_wp)
        similarity_dict['sym_lin'].append(sym_lin)
        similarity_dict['sym_path'].append(sym_path)
    df = pd.DataFrame(similarity_dict)

    return df

def synsets_similarity(lemmas1, lemmas2, sim):
    sum_sim1 = 0
    for l1 in lemmas1:
        sum_sim1 += max([max_similarity_synsets(l1, l2, sim) for l2 in lemmas2])
    mean_sim1 = sum_sim1 / len(lemmas1)
    
    sum_sim2 = 0
    for l2 in lemmas2:
        sum_sim2 += max([max_similarity_synsets(l2, l1, sim) for l1 in lemmas1])
    mean_sim2 = sum_sim2 / len(lemmas2)
    
    if mean_sim1 > 0 or mean_sim2 > 0:
        return (2 * mean_sim1 * mean_sim2)/(mean_sim1+mean_sim2)
    else:
        return 0
 
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


def same_num_entities(ne1, ne2, entity):
    num_ent_a = 0 
    for p1 in ne1:
        if isinstance(p1, nltk.tree.Tree) and p1.label()==entity:
            num_ent_a += 1
            
    num_ent_b = 0    
    for p2 in ne2:
        if isinstance(p2, nltk.tree.Tree) and p2.label()==entity:
            num_ent_b += 1
        
    if num_ent_a == num_ent_b:
        return 1
    else:
        return 0

def get_sentiment_score(lemma):
    '''
    Function: 
    
    input:type:
    output:type
    
    
    '''
    synsets = wordnet.synsets(lemma)
    score = 0
    for s in synsets:
        senti_synset = sentiwordnet.senti_synset(s.name())
        if senti_synset is not None:
            score += senti_synset.pos_score() - senti_synset.neg_score()
    return score

def get_sim_synset_max(SA,SB):
    '''
    Function: 
    
    input:type:
    output:type
    
    
    '''
    sA = SA.SentA_lemmas[0]
    sB = SB.SentB_lemmas[0]
    d = dict()
    l = []
    for idx, x in enumerate(sA):
        #print(x, wordnet.synsets(x))
        syn1 = wordnet.synsets(x)
        #print("syn1   ",syn1)
        syn2 = wordnet.synsets(sB[idx])
        for xx in syn1:
            for yy in syn2:  
                try:  
                    sims = get_similarities(xx, yy, brown_ic,ret_dict=False)
                    l.append(sims)
                except:
                    continue
                #max_val = max(sims, key=sims.get)
                d[xx, yy] = sims
    return d

def sentiment_similarity(lemmas1, lemmas2):
    '''
    Function: 
    
    input:type:
    output:type
    
    
    '''
    polarity1 = 0
    for l1 in lemmas1:
        polarity1 += get_sentiment_score(l1) 
    polarity2 = 0
    for l2 in lemmas2:
        polarity2 += get_sentiment_score(l2)
    if polarity1 > 0 or polarity2 > 0:
        return abs(polarity1-polarity2) / max(polarity1, polarity2)
    else:
        return 0  

def load_gs(PATH):
    
    '''
    Function which given a PATH will read the files and load the golden standard (gs) into a pandas 
    DataFrame for easier manipulation of the data. 
    
    Inputs: 
    PATH: 
    :type str
    Location of the golden standard (gs) to be read in 
    
    Output: 
    DataFrame
    :type pandas.DataFrame
    A DataFrame with the Column "gs", for all the files in the given PATH
    
    '''
    kk = []
    sum_file = PATH + '/summed_gs.csv'
    fn = 'summed_gs.csv'
    if fn in os.listdir(PATH):
        df = pd.read_csv(sum_file)
        df = df.drop('Unnamed: 0',axis=1)
        df.columns = ['gs']
        return df
    else:
        for file in os.listdir(PATH):
            val = pd.read_csv(PATH + '/' + file, sep='\t',header=None)
            kk.append(val)
        df2 = pd.concat(kk)
        df2.columns = ['gs']
        df2 = df2.dropna(axis=0)
        return df2

def lesk_similarity(words_no_punc_a, words_no_punc_b):
    '''
    Function: 
    
    input:type:
    output:type
    
    
    '''
    pos_tags1 = pos_tag(words_no_punc_a)
    pos_tags2 = pos_tag(words_no_punc_b)
    lesk_synsets1 = []
    for i in range(0, len(words_no_punc_a)):
        if(pos_tags1[i] in {'n', 'v', 'r', 'a'}):
            lesk_synsets1.append(lesk(words_no_punc_a, words_no_punc_a[i], pos_tags1[i]))
    lesk_synsets2 = []
    for i in range(0, len(words_no_punc_b)):
        if(pos_tags2[i] in {'n', 'v', 'r', 'a'}):
            lesk_synsets2.append(lesk(words_no_punc_b, words_no_punc_b[i], pos_tags2[i]))
    
    if len(lesk_synsets1) > 0 and len(lesk_synsets2) > 0:
        return 1-jaccard_distance(set(lesk_synsets1), set(lesk_synsets2))
    else:
        return 0
def length_difference(words_no_punc_a, words_no_punc_b):
    '''
    Function: 
    
    input:type:
    output:type
    
    
    '''
    return abs(len(words_no_punc_a)-len(words_no_punc_b)) / max(len(words_no_punc_a), len(words_no_punc_b))

def unigram_similarity(words_no_punc_a, words_no_punc_b):
    '''
    Function: 
    
    input:type:
    output:type
    
    
    '''
    count_same = 0
    for w in words_no_punc_a:
        count_same += min(words_no_punc_a.count(w), words_no_punc_b.count(w))
    
    if len(words_no_punc_a) > 0 or len(words_no_punc_b) > 0:
        return 2*count_same/(len(words_no_punc_a)+len(words_no_punc_b))
    else:
        return 0
    
def unigram_similarity_importance(words_no_punc_a, words_no_punc_b,IMPORTANCE,MAX_IMPORTANCE):
    '''
    Function: 
    
    input:type:
    output:type
    
    
    '''
    count_same = 0
    for w in words_no_punc_a:
        count_same += min(words_no_punc_a.count(w), words_no_punc_b.count(w)) * IMPORTANCE.get(w, MAX_IMPORTANCE)
    if len(words_no_punc_a) > 0 or len(words_no_punc_b) > 0:
        return 2*count_same/(len(words_no_punc_a)+len(words_no_punc_b))
    else:
        return 0

def bigram_similarity(words_no_punc_a, words_no_punc_b):
    '''
    Function: 
    
    input:type:
    output:type
    
    
    '''
    bigram_searcher_a = BigramCollocationFinder.from_words(words_no_punc_a)
    bigram_searcher_b = BigramCollocationFinder.from_words(words_no_punc_b)
    all_bigrams_a = []
    freq1 = []
    for b1 in bigram_searcher_a.ngram_fd.items():
        all_bigrams_a.append(b1[0])
        freq1.append(b1[1])
    all_bigrams_b = []
    freq2 = []
    for b2 in bigram_searcher_b.ngram_fd.items():
        all_bigrams_b.append(b2[0])
        freq2.append(b2[1])
    count = 0
    for i in range(len(all_bigrams_a)):
        if all_bigrams_a[i] in all_bigrams_b:
            count += min(freq1[i], freq2[all_bigrams_b.index(all_bigrams_a[i])])
    if len(words_no_punc_a) > 0 or len(words_no_punc_b) > 0:
        return 2*count/(len(words_no_punc_a)+len(words_no_punc_b))
    else:
        return 0

def load_sentences(PATH):
    '''
    Function which given a PATH will read the files and load the sentence pairs into a pandas 
    DataFrame for easier manipulation of the data. 
    
    Inputs: 
    PATH: 
    :type str
    Location of the sentence pairs to be read in 
    
    Output: 
    DataFrame
    :type pandas.DataFrame
    A DataFrame with the Columns "SentA" and "SentB", for all the files in the given PATH
    
    '''
    # regex to find all the files
    ABSPATH = PATH + "*"
    l=[]
    # iterating through all the files and: 1) reading 2) stripping and splitting 3) appending to final list
    for FILE in glob.glob(ABSPATH):
        with open(FILE, encoding='utf-8') as f: 
            for sentence_pairs in f:
                sentences = sentence_pairs.strip().split('\t')
                l.append((sentences[0],sentences[1]))
    # separating the two sentences 
    SentA = [x[0].lower() for x in l]
    SentB = [y[1].lower() for y in l]
    #creating the DataFrame 
    df = pd.DataFrame({
                        "SentA":SentA,
                        "SentB":SentB
                      })
    return df

def trigram_similarity(words_no_punc_a, words_no_punc_b):
    '''
    Function: 
    
    input:type:
    output:type
    
    
    '''
    finder1 = TrigramCollocationFinder.from_words(words_no_punc_a)
    finder2 = TrigramCollocationFinder.from_words(words_no_punc_b)
    
    # We get the bigrams of first sentence and its frequency
    trigrams1 = []
    freq1 = []
    for t1 in finder1.ngram_fd.items():
        trigrams1.append(t1[0])
        freq1.append(t1[1])
    
    # We get the trigrams of second sentence and its frequency
    trigrams2 = []
    freq2 = []
    for t2 in finder2.ngram_fd.items():
        trigrams2.append(t2[0])
        freq2.append(t2[1])
    
    count = 0
    for i in range(len(trigrams1)):
        if trigrams1[i] in trigrams2:
            # Count number of same trigrams
            count += min(freq1[i], freq2[trigrams2.index(trigrams1[i])])
            
    if len(words_no_punc_a) > 0 or len(words_no_punc_b) > 0:
        return 2*count/(len(words_no_punc_a)+len(words_no_punc_b))
    else:
        return 0