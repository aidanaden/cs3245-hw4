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
    punctuation_chars = set(string.punctuation)

    text = re.sub(r'^[0-9,]+$', '', text)
    # Tokenize the text into words
    words = nltk.word_tokenize(text)

    processed_tokens = {}
    processed_biwords = {}

    cur_biword = []
    for word in words:
        if word in stop_words or not word.isalpha() or word in punctuation_chars:
            cur_biword = []
            continue
        word = p_stemmer.stem(word, to_lowercase=True)

        # process word
        if (word,zone) not in processed_tokens:
            processed_tokens[(word, zone)] = 1
        else:
            processed_tokens[(word, zone)] += 1

        # process biword
        cur_biword.append(word)

        if len(cur_biword) == 2:
            biword = cur_biword[0] + " " + cur_biword[1]

            if (biword,zone) not in processed_biwords:
                processed_biwords[(biword, zone)] = 1
            else:
                processed_biwords[(biword, zone)] += 1

            cur_biword[0] = cur_biword.pop()

    return [processed_tokens, processed_biwords]

def build_index(in_dir, out_dict, out_postings):
    """
    build index from documents stored in the input directory,
    then output the dictionary file and postings file
    """
    print('indexing...')
    
    csv.field_size_limit(100000000)
    corpus = csv.reader(open(in_dir, 'r', encoding="utf-8"))
    next(corpus)

    raw_tf = {}
    norm_tf = {}
    posting = {}
    collection_size = 0
    
    for entry in corpus:
        
        doc = Doc(entry[0], entry[1], entry[2], entry[3], entry[4])
        tokens = {}
        for entries in process_to_tokens(doc.title, "title"):
            tokens.update(entries)
        for entries in process_to_tokens(doc.title, "content"):
            tokens.update(entries)
        for entries in process_to_tokens(doc.title, "court"):
            tokens.update(entries)
        tokens[doc.date.split()[0], "date"] = 1

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
            normalized_val = value/lengthN
            norm_tf[key] = normalized_val
            for key in tokens:
                if key not in posting:
                    posting[key] = {}
                posting[key][doc.docID] = normalized_val

        collection_size += 1
        print(f"FileID: {doc.docID} indexed.")

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
    final_dict[("*", "*")] = (collection_size) # add collection size to dict
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
