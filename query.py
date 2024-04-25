import re
from constants import Keywords
from processing import tokenize_str


def categorise_and_stem_query(query: str) -> list[list[str]]:
    """
    Split a query into a list of of list clauses.
    Args:
        query(str): The raw query
    Returns:
        list(list(clause)): The list of subqueries resulting from splitting the query,
                                       where each subquery is a list of clauses
    """
    if len(query) < 1:
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

            if is_last_keyword_quote:
                is_last_keyword_quote = False
            else:
                is_last_keyword_quote = True

            # If there is no more keyword, we splice until the end of the string
            next_idx = (
                len(spliced_query) if closest_keyword_pos == -1 else closest_keyword_pos
            )
            clause = spliced_query[:next_idx].strip()

            if len(clause) > 0:
                stemmed_clause = " ".join(tokenize_str(clause))
                processed_and_clause.append(stemmed_clause)

            # Update position
            curr_str_idx = (
                closest_keyword_pos
                if closest_keyword_pos == -1
                else curr_str_idx + next_idx + len(Keywords.DOUBLE_QUOTE)
            )

        query_clauses.append(processed_and_clause)

    return query_clauses


def get_words_from_clauses(query_clauses: list[list[str]]) -> list[str]:
    """
    Args:
        query_clauses (list(list(string))): A list of list of str
    Returns:
        str: A string consisting of all the words in the query clauses
    """
    list_of_words = []
    for and_clause in query_clauses:
        and_clause_words = " ".join([clause_word for clause_word in and_clause]).split(
            " "
        )
        list_of_words.extend(and_clause_words)
    return list_of_words
