from langgraph.graph import MessagesState
from typing import List,Dict

class GraphState(MessagesState):
    """
    'messages': where stored the history of messages
    'metadata': stored retrieved metadata of the messages
    """
    metadata:List[Dict] | None