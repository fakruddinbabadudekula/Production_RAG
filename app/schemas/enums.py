from enum import Enum


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    
class FileType(str,Enum):
    PDF="pdf"
    
class ErrorType(str,Enum):
    INCORRECT_CREDENTIALS="incorrect_credentials"
    UNKOWN_ERROR="un_known_error",
    INTIGIRITY_ERROR="intigrity_constraint_error"
    NOT_FOUND_ERR="not_found_error"