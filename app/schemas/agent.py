from langgraph.graph import MessagesState
from typing import List,Dict

class GraphState(MessagesState):
    """
    'messages': where stored the history of messages
    'retrieved_docs': stored retrieved metadata of the messages
    """
    retrieved_docs:List[Dict] | None