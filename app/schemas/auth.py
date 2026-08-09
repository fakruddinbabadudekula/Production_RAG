from pydantic import BaseModel
from datetime import datetime
import uuid
class TokenData(BaseModel):
    access_token:str
    refresh_token:str
    access_expire_time:datetime
    refresh_expire_time:datetime
    
    
class LoginData(BaseModel):
    email:str
    password:str
    
class RefreshSchema(BaseModel):
    user_id: uuid.UUID
    hashed_token: str
    family_id: uuid.UUID
    expires_at: datetime
