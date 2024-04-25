import re
from constants import Keywords, QueryType
from processing import tokenize_str

def categorise_and_stem_query(query: str) -> list[list[tuple[str, QueryType]]]:
    """
    Split a query into a list of of list clauses.
    Args: 
        query(str): The raw query
    Returns:
        list(list(clause, QueryType)): The list of subqueries resulting from splitting the query,
                                       where each subquery is a list of clauses
    """
    if (len(query) < 1):
        return

    and_clauses = re.split(Keywords.AND, query)
    query_clauses = []

    for i in range(len(and_clauses)):
        processed_and_clause = []
        and_clause = and_clauses[i]
        curr_str_idx = 0
        is_last_keyword_quote = False

        while curr_str_idx != -1 and curr_str_idx < len(and_clause):
            spliced_query = and_clause[curr_str_idx:]
            clause = ""
            closest_keyword_pos = 0

            # Finding closest double quote
            closest_keyword_pos = spliced_query.find(Keywords.DOUBLE_QUOTE)

            if (is_last_keyword_quote):
                is_last_keyword_quote = False
                clause_type = QueryType.PHRASAL
            else:
                is_last_keyword_quote = True
                clause_type = QueryType.FREE

            # If there is no more keyword, we splice until the end of the string
            next_idx = len(spliced_query) if closest_keyword_pos == -1 else closest_keyword_pos
            clause = spliced_query[:next_idx].strip()

            if (len(clause) > 0):
                stemmed_clause = " ".join(tokenize_str(clause))
                processed_and_clause.append((stemmed_clause, clause_type))

            # Update position
            curr_str_idx = closest_keyword_pos if closest_keyword_pos == -1 else curr_str_idx + next_idx + len(Keywords.DOUBLE_QUOTE)

        query_clauses.append(processed_and_clause)

    return query_clauses


def tag_results(results: list[int], tag: QueryType) -> list[tuple[int, QueryType]]:
    """
    Arguments:
        results (list(int)): A list of doc Id's
        tag (QueryType): The tag to tag with
    Returns:
        list(int, tag): The list of tagged results
    """
    return list(map(lambda x: (x, tag), results))

def union_document_ids(doc_list1: list[tuple[int, QueryType]], doc_list2: list[tuple[int, QueryType]]) -> list[tuple[int, QueryType]]:
    """
    Args:
        doc_list1 (list(doc_id, QueryType)): The first list of clauses
        doc_list2 (list(doc_id, QueryType)): The second list of clauses
    Returns:
        list(doc_id, QueryType): The merged list of clauses
    """
    doc_list1.sort(key=lambda x: x[0])
    doc_list2.sort(key=lambda x: x[0])

    result = []
    idx0, idx1 = 0, 0

    # Iterate while both lists are alive
    while idx0 < len(doc_list1) and idx1 < len(doc_list2):
        # Matching element
        if doc_list1[idx0][0] == doc_list2[idx1][0]:
            is_phrasal = doc_list1[idx0][1] == QueryType.PHRASAL or doc_list2[idx1][1] == QueryType.PHRASAL
            clause_type = QueryType.PHRASAL if is_phrasal else QueryType.FREE
            result.append((doc_list1[idx0][0], clause_type))
            idx0 += 1
            idx1 += 1

        # doc_list1[pointer] is smaller, advance the pointer
        elif doc_list1[idx0][0] < doc_list2[idx1][0]:
            result.append(doc_list1[idx0])
            idx0 += 1

        # doc_list2[pointer] is smaller, advance the pointer
        else:
            result.append(doc_list2[idx1])
            idx1 += 1

    # List one has still elements
    if idx0 < len(doc_list1):
        result.extend(doc_list1[idx0:])

    # List two has still elements
    if idx1 < len(doc_list2):
        result.extend(doc_list2[idx1:])

    return result

def intersect_document_ids(doc_list1: list[tuple[int, QueryType]], doc_list2: list[tuple[int, QueryType]]) -> list[tuple[int, QueryType]]:
    """
    Args:
        doc_list1 (list(doc_id, QueryType)): The first list of clauses
        doc_list2 (list(doc_id, QueryType)): The second list of clauses
    Returns:
        list(doc_id, QueryType): The merged list of clauses
    """
    doc_list1.sort(key=lambda x: x[0])
    doc_list2.sort(key=lambda x: x[0])

    result = []
    idx0, idx1 = 0, 0
    while idx0 < len(doc_list1) and idx1 < len(doc_list2):
        # Matching element
        if doc_list1[idx0][0] == doc_list2[idx1][0]:
            is_phrasal = doc_list1[idx0][1] == QueryType.PHRASAL or doc_list2[idx1][1] == QueryType.PHRASAL
            clause_type = QueryType.PHRASAL if is_phrasal else QueryType.FREE
            result.append((doc_list1[idx0][0], clause_type))
            idx0 += 1
            idx1 += 1

        # doc_list1[pointer] is smaller, advance the pointer
        elif doc_list1[idx0][0] < doc_list2[idx1][0]:
            idx0 += 1

        # doc_list2[pointer] is smaller, advance the pointer
        else:
            idx1 += 1
    return result

def get_words_from_clauses(query_clauses: list[list[tuple[str, QueryType]]]) -> list[str]:
    """
    Args:
        query_clauses (list(list(string, QueryType))): A list of list of tuples 
    Returns:
        str: A string consisting of all the words in the query clauses
    """
    list_of_words = []
    for and_clause in query_clauses:
        and_clause_words = " ".join([clause_word for clause_word, clause_type in and_clause]).split(" ")
        list_of_words.extend(and_clause_words)
    return list_of_words