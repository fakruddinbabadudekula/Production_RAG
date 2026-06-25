from pydantic import BaseModel, ConfigDict,EmailStr,field_validator


class BaseUser(BaseModel):
    email:str
    name:str
    
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

    