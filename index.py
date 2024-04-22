#!/usr/bin/python3
import re
import nltk
import sys
import getopt

import csv
from collections import defaultdict

from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.corpus import stopwords

import pickle
import string
import math

def usage():
    print("usage: " + sys.argv[0] + " -i directory-of-documents -d dictionary-file -p postings-file")

class Doc:
    def __init__(self, docID, title, content, date, court):
        self.docID = docID
        self.title = title
        self.content = content
        self.date = date
        self.court = court

def process_to_tokens(text):
    # Initialise stemmer, lemmatizer & stopwords
    p_stemmer = PorterStemmer()
    lemm = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    punctuation_chars = set(string.punctuation)

    processed_tokens = {}

    for sentence in nltk.sent_tokenize(text):
        # Case-folding
        sentence = sentence.lower()

        # Tokenizing each line
        tokens = nltk.word_tokenize(sentence)

        for token in tokens:
            # Remove punctuation tokens
            if all(char in punctuation_chars for char in token):
                continue

            # Apply stemming
            token = p_stemmer.stem(token)

            # Case-folding
            token = token.lower()

            # Apply stopword removal
            if token in stop_words:
                continue

            # Update token count
            if token not in processed_tokens:
                processed_tokens[token] = 1
            else:
                processed_tokens[token] += 1

    return processed_tokens    

    
def build_index(in_dir, out_dict, out_postings):
    """
    build index from documents stored in the input directory,
    then output the dictionary file and postings file
    """
    print('indexing...')
    #Dataset: ['document_id', 'title', 'content', 'date_posted', 'court']
    
    corpus = csv.reader(open(in_dir, 'r'))
    next(corpus) # Skip header
    
    collectionsize = 0
    docfreq = {}
    posting = {}
    # Iterate through the corpus
    for entry in corpus:
        doc = Doc(entry[0], entry[1], entry[2], entry[3], entry[4])
        tokens = process_to_tokens(doc.content)
        
    


input_directory = output_file_dictionary = output_file_postings = None

try:
    opts, args = getopt.getopt(sys.argv[1:], 'i:d:p:')
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == '-i': # input directory
        input_directory = a
    elif o == '-d': # dictionary file
        output_file_dictionary = a
    elif o == '-p': # postings file
        output_file_postings = a
    else:
        assert False, "unhandled option"

if input_directory == None or output_file_postings == None or output_file_dictionary == None:
    usage()
    sys.exit(2)

build_index(input_directory, output_file_dictionary, output_file_postings)
