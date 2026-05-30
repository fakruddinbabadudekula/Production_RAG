from fastapi import FastAPI
from app.core.db import init_db
from app.api.auth import router as auth_router
from app.api.register import router as register_router
from app.api.test import router as test_router
from contextlib import asynccontextmanager
from app.api.chat import router as chat_router
from app.api.history import router as history_router
from app.api.upload import router as file_upload
@asynccontextmanager
async def lifespan(app):
    print("server is started")
    await init_db()
    print("tables are created ")
    yield
    print("server is closed")
    
app=FastAPI(lifespan=lifespan)
app.include_router(auth_router,prefix="/api/v1/auth")
app.include_router(register_router,prefix='/api/v1')
app.include_router(test_router,prefix='/app/v1/test')
app.include_router(chat_router,prefix='/app/v1')
app.include_router(history_router,prefix='/app/v1/history')
app.include_router(file_upload,prefix="/app/v1")
@app.get("/")
def hello():
    return {
        'msg':"Hello user. How are you?"
    }