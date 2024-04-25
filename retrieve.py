import pickle

ZONES = {"title": 0.2, "court": 0.2, "date": 0.1, "content": 0.5}

dictionary = {}

postings_path = ""


# return map{ (term, zone): pointer }
def set_dictionary(dictionary_file):
    global dictionary
    with open(dictionary_file, "rb") as handle:
        dictionary = pickle.load(handle)


def set_posting_file(postings_file):
    global postings_path
    postings_path = postings_file


def term_in_dict(term):
    for zone in ZONES:
        if (term, zone) in dictionary:
            return True
    return False


def get_term_doc_count(term):
    docs = set()
    for zone in ZONES:
        if (term, zone) in dictionary:
            postings = get_posting_list(term, zone)
            postings = [x[0] for x in postings]
            docs = docs.union(set(postings))

    return len(docs)


# get total number of documents
def get_collection_size():
    return dictionary[("*", "*")]


# return arr[ [docID, tf-norm] ]
def get_posting_list(term, zone):
    key = (term, zone)
    if key not in dictionary:
        return []

    posting_str = ""
    with open(postings_path, "r") as f:
        pointer = dictionary[key]
        f.seek(pointer + 2)  # skip start '$ ' char
        while True:
            cur = f.read(1)
            if cur == "$":
                break
            posting_str += cur

        posting_str.strip()

    posting = []
    cur = []
    for count, x in enumerate(posting_str.split(" ")):
        cur.append(x)
        if count % 2 == 1:
            posting.append(cur)
            cur = []

    return posting


# accumulate all term.title, term.content, ...
# return [doc1, doc2, ...]
def get_postings_docs(term):
    docs = set()
    for zone in ZONES:
        if (term, zone) in dictionary:
            postings = get_posting_list(term, zone)
            postings = [x[0] for x in postings]
            docs = docs.union(set(postings))

    return docs


# get term.zone tf val
def get_doc_term_zone_tf(doc, term, zone):
    key = (term, zone)
    if key not in dictionary:
        return 0

    posting_str = ""
    with open(postings_path, "r") as f:
        pointer = dictionary[key]
        f.seek(pointer + 2)  # skip start '$ ' char
        while True:
            cur = f.read(1)
            if cur == "$":
                break
            posting_str += cur

        posting_str.strip()

    split_posting = posting_str.split(" ")
    tf = 0
    for count, x in enumerate(split_posting):
        if x == doc:
            tf = split_posting[count + 1]
            break

    return float(tf)
