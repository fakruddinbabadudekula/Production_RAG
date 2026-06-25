
from pydantic import BaseModel,field_validator,EmailStr
class RegisterUser(BaseModel):
    name:str
    password:str
    email:EmailStr
    @field_validator('password')
    @classmethod
    def password_validator(cls,value:str)->str:
        if len(value) <7:
            raise ValueError("Password must be at least 8 characters")
        return value