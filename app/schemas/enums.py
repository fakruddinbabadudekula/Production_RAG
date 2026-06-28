"""Schema Module for enums where it have enums class models"""

from enum import Enum


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    
class FileType(str,Enum):
    PDF="pdf"
    