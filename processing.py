from nltk.corpus import wordnet as wn
from collections import defaultdict
from constants import *
from nltk.corpus import stopwords

import nltk
import regex

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

lemmatizer = nltk.stem.WordNetLemmatizer()
stemmer = nltk.stem.PorterStemmer()
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

def lemmatize(token_list: list[str], include_stem: bool = False) -> list[str]:
    return [lemmatizer.lemmatize(token.lower(), tag_map[tag[0]]) if not include_stem else stemmer.stem(lemmatizer.lemmatize(token.lower(), tag_map[tag[0]])) for token, tag in nltk.pos_tag(token_list)]

def stem(token_list: list[str]) -> list[str]:
    return [stemmer.stem(token.lower()) for token in token_list]

def tokenize_str(raw_string: str, remove_stopwords: bool = True, use_lemmatize: bool = True, use_stem: bool = True) -> list[str]:
    word_list = nltk.tokenize.word_tokenize(raw_string)
    cleaned_list = [clean_word(string) for string in word_list]
    token_list = " ".join(cleaned_list).split()

    if (remove_stopwords):
        token_list = [token for token in token_list if token not in stopwords.words('english')]

    if use_lemmatize:
        token_list = lemmatize(token_list, use_stem)
    elif use_stem:
        token_list = stem(token_list)

    return token_list

def clean_word(string: str) -> str:
    if not is_numeric(string):
        return regex.sub(r'[^a-zA-Z0-9\_\-\p{Sc}]', ' ', string)
    else:
        return string

def is_numeric(string: str) -> bool:
    if regex.match(r'[0-9]+[^0-9][0-9]+', string):
        return True
    else:
        return False