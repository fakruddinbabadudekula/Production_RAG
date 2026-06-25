
from fastapi import FastAPI
from app.core.db import init_db
from app.core.logging import setup_logging
from logging import getLogger
from app.api.exception_handlers import register_exception_handlers
from app.api.routes.auth import router as auth_router
from app.api.routes.register import router as register_router
from app.api.routes.test import router as test_router
from contextlib import asynccontextmanager
from app.api.routes.chat import router as chat_router
from app.api.routes.history import router as history_router
from app.api.routes.upload import router as file_upload
from app.api.middleware import logger_middleware
from app.core.config import settings
def create_required_dir():
    for path in [
        settings.FILE_UPLOAD_PATH,
        settings.VECTOR_FOLDER,
    ]:
        path.mkdir(
            parents=True,
            exist_ok=True
        )


logger=getLogger(__name__)
@asynccontextmanager
async def lifespan(app):
    setup_logging()
    create_required_dir()
    logger.info("Server Started")
    await init_db()
    logger.info("Tables are created ")
    yield
    logger.info("Server terminated")
    
app=FastAPI(lifespan=lifespan)
app.middleware('http')(logger_middleware)
register_exception_handlers(app=app)
app.include_router(auth_router,prefix="/api/v1/auth")
app.include_router(register_router,prefix='/api/v1')
app.include_router(test_router,prefix='/api/v1/test')
app.include_router(chat_router,prefix='/api/v1')
app.include_router(history_router,prefix='/api/v1/history')
app.include_router(file_upload,prefix="/api/v1")
@app.get("/")
def hello():
    return {
        'msg':"Hello user. How are you?"
    }