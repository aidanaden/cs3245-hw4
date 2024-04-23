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

ZONES = {
        'title': 0.2, 
        'court': 0.2,
        'date_posted': 0.1,
        'content': 0.5
        }

RELEVANCE_THRESHOLD = 0.6

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
        token = re.sub(r'^[0-9,]+$', '', token) # Apply number removal
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

def run_search(dictionary_file, postings_file, query_file, results_file):
    dictionary = get_dictionary()

    results = ""
    with open(query_file, "r") as f:
        query = f.readline()
        relevant_docs = []
        rel_doc = f.readline()
        while rel_doc:
            relevant_docs.append(int(rel_doc))

        terms = parse_query(query)

        # terms = refine_query(terms)

        query_weights = get_query_weights(terms, dictionary)
        document_scores = get_document_scores(query_weights, dictionary)
        [relevant_docs, irrelevant_docs] = get_relevant_docs(document_scores)

        results += " ".join(relevant_docs) + "\n"

    with open(results_file, "w") as f:
        f.write(results)


# return map{ term: doc_count }
def get_dictionary():
    return {}

# get total number of documents
def get_collection_size():
    return 1

# get normalization val for doc
def get_document_normalization(doc):
    return 1

# accumulate all term.title, term.content, ...
# return [doc1, doc2, ...]
def get_postings_docs(term):
    return []

# get term.zone tf val
def get_doc_term_zone_tf(doc, term, zone):
    return 0

# e.g. "fertility treatment" AND damages => ["fertility treatment", "damages"]
# e.g. "quiet phone call" => ["quiet", "phone", "call"]
def parse_query(query):
    terms = []
    if 'AND' in query:
        terms = [x.strip().strip('"') for x in query.split("AND")]
    else:
        terms = [x.strip() for x in query.split(" ")]
        
    return tokenize_terms(terms)

def tokenize_terms(terms):
    res = []
    for x in terms:
        # TODO: follow index steming
        if x not in res:
            res.append(x)
        
    return res

# get td-idf of query, cosine normalized
# return map{ word: td-idf }
def get_query_weights(terms, dictionary):
    weights = {}
    collection_size = get_collection_size()
    normalize = 0
    for term in terms: # tf = 1 as no repeated terms in query
        idf = 0
        if term in dictionary:
            idf = math.log10(collection_size / dictionary[term])

        weights[term] = idf
        normalize += idf * idf

    if normalize > 0:
        normalize = sqrt(normalize)
        for term in weights:
            weights[term] /= normalize

    return weights

# get td-idf document scores
# return map{ doc: score }
def get_document_scores(query_weights, dictionary):
    document_scores = {}
    for term, weight in query_weights.items():
        postings = get_postings_docs(term)
        for doc in postings:
            if doc not in document_scores:
                document_scores[doc] = 0
            document_scores[doc] += get_doc_term_weight(doc) * weight

    return document_scores
            

# get zone weighted tf of term in doc, cosine normalized
def get_doc_term_weight(doc, term):
    normalize = get_document_normalization(doc)
    score = 0
    for zone, weight in ZONES.items():
        score += get_doc_term_zone_tf(doc, term, zone) * weight

    return score / normalize

# return [relevant_docs_arr, irrelevant_docs_arr], both sorted most to least relevant
def get_relevant_docs(doc_scores):
    relevant = []
    irrelevant = []
    for doc, score in doc_scores.items():
        if score > RELEVANCE_THRESHOLD:
            relevant.append([doc, score])
        else:
            irrelevant.append([doc, score])
            
    relevant.sort(key=lambda x: x[1], reverse=True)
    irrelevant.sort(key=lambda x: x[1], reverse=True)
    
    return [[x[0] for x in relevant], [x[0] for x in irrelevant]]