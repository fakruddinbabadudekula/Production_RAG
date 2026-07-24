"""Module for schemas which are specific for auth router"""

from pydantic import BaseModel, ConfigDict


class BaseUser(BaseModel):
    email:str
    name:str
    # what model_config means, how we create the instance of this class like this right BaseUser(key_value), this allows BaseUser(dict/another_dataclass_instance)
    model_config = ConfigDict(from_attributes=True)
    
    
class LoginRequest(BaseModel):
    email:str
    password:str
    
class AccessTokenResponse(BaseModel):
    token:str
    token_type:str="bearer"
        
class DeleteUser(BaseModel):
    email:str
    password:str

    