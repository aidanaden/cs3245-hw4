#!/usr/bin/python3
import re
import nltk
import sys
import getopt

import csv
import os
from collections import defaultdict

from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.corpus import stopwords

import pickle
import string
import math

class Doc:
    def __init__(self, docID, title, content, date, court):
        self.docID = docID
        self.title = title
        self.content = content
        self.date = date
        self.court = court

def process_to_tokens(text, zone):
    p_stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))

    # Tokenize the text into words
    processed_tokens = {}
    for sentence in nltk.sent_tokenize(text):
        words = nltk.word_tokenize(sentence)

        # Process each word
        for word in words:
            # Apply stemming
            word = p_stemmer.stem(word)
            # Remove punctuation and convert to lowercase
            word = ''.join(char.lower() for char in word if char.isalpha())

            # Skip stop words and empty words
            if word and word not in stop_words:
                processed_tokens[(word, zone)] = processed_tokens.get((word, zone), 0) + 1

    return processed_tokens

def process_to_biwords(text, zone):
    p_stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))

    # Tokenize the text into words
    processed_biwords = {}
    for sentence in nltk.sent_tokenize(text):
        words = nltk.word_tokenize(sentence)

        # Process each word
        for i in range(len(words) - 1):
            word1 = words[i]
            word2 = words[i + 1]

            # Apply stemming
            word1 = p_stemmer.stem(word1)
            word2 = p_stemmer.stem(word2)

            # Remove punctuation and convert to lowercase
            word1 = ''.join(char.lower() for char in word1 if char.isalpha())
            word2 = ''.join(char.lower() for char in word2 if char.isalpha())

            # Skip stop words and biwords containing numbers
            if word1 and word2 and word1 not in stop_words and word2 not in stop_words and not any(char.isdigit() for char in word1 + word2):
                biword = word1 + " " + word2
                processed_biwords[(biword, zone)] = processed_biwords.get((biword, zone), 0) + 1

    return processed_biwords

def build_index(in_dir, out_dict, out_postings):
    """
    build index from documents stored in the input directory,
    then output the dictionary file and postings file
    """
    print('indexing...')
    
    # keys are terms(strings) and values are tuples of (docfreq, idf)
    csv.field_size_limit(100000000)
    corpus = csv.reader(open(in_dir, 'r', encoding="utf-8"))
    next(corpus)

    raw_tf = {}
    norm_tf = {}
    docfreq = {}
    posting = {}
    
    #limit = 100
    #count = 0
    for entry in corpus:
        #if count == limit:
        #    break
        doc = Doc(entry[0], entry[1], entry[2], entry[3], entry[4])
        tokens = {}
        mono_tokens = {}
        mono_tokens.update(process_to_tokens(doc.title, "title"))
        mono_tokens.update(process_to_tokens(doc.content, "content"))
        mono_tokens[doc.date.split()[0], "date"] = 1
        mono_tokens.update(process_to_tokens(doc.court, "court"))

        bi_tokens = {}
        bi_tokens.update(process_to_biwords(doc.title, "title"))
        bi_tokens.update(process_to_biwords(doc.content, "content"))
        bi_tokens[(doc.date.split()[0], "date")] = 1
        bi_tokens.update(process_to_biwords(doc.court, "court"))

        # merge tokens
        # Tokens = {(processed_token, zone): raw_termfreq}
        tokens.update(mono_tokens)
        tokens.update(bi_tokens)

        for key in tokens:
            term, zone = key
            if term not in raw_tf:
                raw_tf[term] = 1
            else:
                raw_tf[term] += 1
                
        # log and normalize (log-frequency weighting, w t,d)
        for term in raw_tf:
            norm_tf[key] = 1 + math.log10(raw_tf[term])
        
        # Calculating Euclidean norm for cosine normalization
        square_sum = 0
        for value in norm_tf.values():
            square_sum += value ** 2
        lengthN = math.sqrt(square_sum)
        
        for key, value in norm_tf.items():
            # Cosine normalization
            norm_tf[key] = value/lengthN
            for key in tokens:
                if key not in posting:
                    posting[key] = {}
                posting[key][doc.docID] = value

        print(f"FileID: {doc.docID} indexed.")
        count += 1

    print("postings generated...")
                    
    # Write all postings as a string
    # for each term: "$ docID  docNormalizedTF nextdocID  nextdocTF..."
    pointerpos = 0
    final_dict = {}
    postingfile = open(out_postings, "w")
    for key, doc_dict in posting.items():
        postingstring = "$" # mark start of posting
        term, zone = key
        for docID, normtf in sorted(doc_dict.items(), key=lambda item: int(item[0])):
            postingstring += " " + str(docID) + " " +  str(normtf)
        
        postingfile.write(postingstring)
        # pointerpos added to track position in postings file
        final_dict[(term, zone)] = (pointerpos)
        pointerpos += len(postingstring)
    postingfile.close()
    pickle.dump(final_dict, open(out_dict, "wb"))
    print("postings written to disk...")

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
