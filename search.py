#!/usr/bin/python3
import nltk
import sys
import getopt
import math

from query import (
    categorise_and_stem_query,
    get_words_from_clauses,
    intersect_document_ids,
    tag_results,
    union_document_ids,
)
from query_expand import expand_clause
from retrieve import (
    dictionary,
    set_dictionary,
    set_posting_file,
    term_in_dict,
    get_term_doc_count,
    get_collection_size,
    get_posting_list,
    get_postings_docs,
    get_doc_term_zone_tf,
    ZONES,
)

RELEVANCE_THRESHOLD = 0.6


def usage():
    print(
        "usage: "
        + sys.argv[0]
        + " -d dictionary-file -p postings-file -q file-of-queries -o output-file-of-results"
    )


def run_search(dictionary_file, postings_file, query_file, results_file):
    set_dictionary(dictionary_file)
    set_posting_file(postings_file)

    results = ""
    with open(query_file, "r") as f:
        query = f.readline()
        relevant_docs = []
        rel_doc = f.readline()
        while rel_doc:
            relevant_docs.append(int(rel_doc))
            rel_doc = f.readline()

        clauses = categorise_and_stem_query(query)
        query_list = get_words_from_clauses(clauses)
        expanded_words = []
        for query in query_list:
            expanded_words.extend(expand_clause(query))
            query_list.extend(expanded_words)
        query_list = list(set(query_list))
        final_result = run_query(query_list)
        results = " ".join(str(doc_id) for doc_id in final_result)

    with open(results_file, "w") as f:
        f.write(results)


def run_query(terms: list[str]):
    query_weights = get_query_weights(terms)
    document_scores = get_document_scores(query_weights)
    relevant_docs = get_relevant_docs(document_scores)
    return relevant_docs


# get td-idf of query, cosine normalized
# return map{ word: tf-idf-normalized }
def get_query_weights(terms: list[str]) -> dict[str, float]:
    query_weights = {}
    collection_size = get_collection_size()
    normalize = 0

    term_counts = {}

    # term_counts = dict{ term: raw_count }
    for term in terms:
        if term not in term_counts:
            term_counts[term] = 0
        term_counts[term] += 1

    # query_weights = dict{ term: tf-idf }
    for term, count in term_counts.items():
        tf = 1 + math.log10(count)
        if term_in_dict(term):
            term_doc_count = get_term_doc_count(term)
            idf = math.log10(collection_size / term_doc_count)
        else:
            idf = 0

        tf_idf = tf * idf
        query_weights[term] = tf_idf
        normalize += tf_idf * tf_idf

    # query_weights = dict{ term: tf-idf-normalized }
    if normalize > 0:
        normalize = math.sqrt(normalize)
        for term in query_weights:
            query_weights[term] /= normalize

    return query_weights


# get lnc.ltc document scores
# return map{ doc: score }
def get_document_scores(query_weights) -> dict[int, float]:
    document_scores = {}
    for term, weight in query_weights.items():
        postings = get_postings_docs(term)
        for doc in postings:
            if doc not in document_scores:
                document_scores[doc] = 0
            document_scores[doc] += get_doc_term_weight(doc, term) * weight

    return document_scores


# get zone weighted tf of term in doc, cosine normalized
def get_doc_term_weight(doc, term):
    score = 0
    for zone, weight in ZONES.items():
        score += get_doc_term_zone_tf(doc, term, zone) * weight

    return score


# return relevant docs sorted most to least relevant
def get_relevant_docs(doc_scores):
    relevant = []
    for doc, score in doc_scores.items():
        if score > RELEVANCE_THRESHOLD:
            relevant.append([doc, score])

    relevant.sort(key=lambda x: x[1], reverse=True)

    return [x[0] for x in relevant]


try:
    opts, args = getopt.getopt(sys.argv[1:], "d:p:q:o:")
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == "-d":
        dictionary_file = a
    elif o == "-p":
        postings_file = a
    elif o == "-q":
        file_of_queries = a
    elif o == "-o":
        file_of_output = a
    else:
        assert False, "unhandled option"

if (
    dictionary_file == None
    or postings_file == None
    or file_of_queries == None
    or file_of_output == None
):
    usage()
    sys.exit(2)

run_search(dictionary_file, postings_file, file_of_queries, file_of_output)
