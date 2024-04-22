#!/usr/bin/python3
import re
import nltk
import sys
import getopt

from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.corpus import stopwords

import string
import pickle
import math
from math import sqrt, log10
Important_courts = {"SG Court of Appeal", "SG Privy Council", "UK House of Lords", "UK Supreme Court", "High Court of Australia", "CA Supreme Court"}

def usage():
    print("usage: " + sys.argv[0] + " -d dictionary-file -p postings-file -q file-of-queries -o output-file-of-results")
  
def process_to_tokens(line):
    # Initialise stemmer, lemmatizer & stopwords
    p_stemmer = PorterStemmer()
    lemm = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    punctuation_chars = set(string.punctuation)
    
    processed_tokens = {}
    for token in nltk.word_tokenize(line):
        token = re.sub(r'\d+', '', token) # Apply number removal
        if all(char in punctuation_chars for char in token): # Remove punctuation tokens
            continue
        token = p_stemmer.stem(token) # Apply stemming
        token = token.lower() # Case-folding
        # token = lemm.lemmatize(token, "v") # Apply lemmatization
        if token in stop_words: # Apply stopword removal
            continue
        
        # Update token count
        if token not in processed_tokens:
            processed_tokens[token] = 1
        else:
            processed_tokens[token] += 1
    
    return processed_tokens
