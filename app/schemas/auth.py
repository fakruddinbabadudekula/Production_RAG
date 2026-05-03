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

class RegisterUser(BaseModel):
    name:str
    password:str
    email:EmailStr
    @field_validator('password')
    @classmethod
    def password_validator(cls,value:str)->str:
        if len(value) <8:
            raise ValueError("Password must be at least 8 characters")
        return value
        # we can also do more validations like atleast one upper and number
        
class DeleteUser(BaseModel):
    email:str
    password:str

    