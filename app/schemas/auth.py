from pydantic import BaseModel
from datetime import datetime

class TokenData(BaseModel):
    access_token:str
    refresh_token:str
    access_expire_time:datetime
    refresh_expire_time:datetime
    
    
class LoginData(BaseModel):
    email:str
    password:str