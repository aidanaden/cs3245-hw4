import nltk
from nltk.corpus import wordnet
from nltk.corpus import WordNetCorpusReader

from constants import *
from processing import stem


def expand_clause(
    expression: str, use_stemmer: bool = True, num_expand_synonyms: int = 3
) -> str:
    """
    Args:
        expression (str): The expression to expand
        use_stemmer (bool): Whether to stem
        num_expand_synonyms (int): The number of synonyms to expand
    Returns:
        str: The expanded expression
    """
    # Tokenise and get all possible synonyms
    token_list = nltk.word_tokenize(expression)
    synsets_token = get_synsets(token_list)

    expanded_tokens = []
    for i in range(len(synsets_token)):
        # Add 1 since the first synonym is always the word itself
        synonyms = get_top_k_synonyms(synsets_token[i], num_expand_synonyms + 1)

        # Make sure original word is included, add as first element
        synonym_names = [synonym.lemma_names()[0].lower() for synonym in synonyms]
        if token_list[i] not in synonym_names:
            synonym_names.insert(0, token_list[i])

        if use_stemmer:
            synonym_names = stem(synonym_names)

        # Concat everything
        expanded_token = " ".join(synonym_names)
        expanded_tokens.append(expanded_token)

    return " ".join(expanded_tokens)


def pos_to_wordnet(tag: str) -> str:
    """
    Args:
        tag (str): The POS tag
    Returns:
        A wordnet tag corresponding to the provided tag
    """
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    elif tag.startswith("V"):
        return wordnet.VERB
    return None


def get_synsets(tokens: list[str]) -> list[list]:
    """
    Args:
        tokens (str): The list of tokens
    Returns
        list(list): List of list of synsets
    """
    tagged = nltk.pos_tag(tokens)
    synsets = []

    for token in tagged:
        # Assign tag
        word, tag = token
        wn_tag = pos_to_wordnet(tag)
        if not wn_tag:
            continue

        # Format is to synset format, remove duplicate and add it to the list
        synsets.append(remove_duplicate_synsets(wordnet.synsets(word, pos=wn_tag)))

    return synsets


def remove_duplicate_synsets(synsets):
    """
    Args:
        synsets (list): List of synsets
    Returns:
        list(list): list of synsets with duplicates removed
    """
    words_encountered = {}
    unique_synsets = []

    for synset in synsets:
        word_name = synset.lemma_names()[0]
        if word_name in words_encountered:
            continue

        words_encountered[word_name] = True
        unique_synsets.append(synset)

    return unique_synsets


def get_top_k_synonyms(synsets: list, k: int) -> list[str]:
    """
    Args:
        synsets (list): A list of synsets
        k (int): The number of synonyms to extract
    Returns:
        list(str): list of synonyms
    """
    if not synsets:
        return []

    # Ordered by frequency, first element is most probable word without context
    syn_to_compare = synsets[0]

    # Create a new list where each element is (syn_set, similarity score)
    sim_score = [
        (synsets[i], syn_to_compare.wup_similarity(synsets[i]))
        for i in range(len(synsets))
    ]

    # Sort in descending score
    sorted(sim_score, key=lambda syn: syn[1] if syn[1] != None else 0, reverse=True)

    # Return top k synonyms
    return list(map(lambda x: x[0], sim_score[:k]))
