from app.models.message import Message
from typing import List
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    BaseMessage,
)
from app.schemas.enums import MessageRole

# later we change this into proccess messages instead of only conversion it also truncate,summarization... 
def convert_msg_to_langchain_msg(messages:List[Message])->List[BaseMessage]:
    langchain_msgs:List[BaseMessage]=[]
    role_mapper={
        MessageRole.SYSTEM:SystemMessage,
        MessageRole.USER:HumanMessage,
        MessageRole.ASSISTANT:AIMessage
    }
    
    for message in messages:
        mapper_class=role_mapper.get(message.role)
        if not mapper_class:
            raise ValueError(
                f"Unsupported message role: {message.role}"
            )
        langchain_msgs.append(mapper_class(
            content=message.content
        ))
    return langchain_msgs