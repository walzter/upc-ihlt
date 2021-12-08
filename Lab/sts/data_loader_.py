#!/usr/bin/env/python3
import glob
import pandas as pd

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
    ABSPATH = PATH + "*"
    kk = []
    for gs in glob.glob(ABSPATH):
        val = pd.read_csv(gs, sep='\t',header=None)
        kk.append(val)
    df = pd.concat(kk)
    df.rename({0:"gs"})
    
    return df

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