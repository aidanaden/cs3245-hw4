from enum import Enum

class Keywords(str, Enum):
    AND = "AND"
    DOUBLE_QUOTE = "\""
    DOC_LENGTH = "LENGTH"
    IMPORTANT = "IMPORTANT"

class Zone(str, Enum):
    TITLE = "title"
    CONTENT = "content"
    COURT = "court"
    DATE = "date"

class QueryType(Enum):
    FREE = 0
    PHRASAL = 1