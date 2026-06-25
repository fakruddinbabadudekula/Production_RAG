
from typing import List,Dict

from pydantic import BaseModel
from typing import Optional
import uuid
class ChatRequest(BaseModel):
    session_id:Optional[uuid.UUID]=None
    query:str
    
class ChatRespose(BaseModel):
    query:str
    response:str
    top_k_docs:List[Dict]