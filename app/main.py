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
        path.mkdir(parents=True, exist_ok=True)


logger = getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    setup_logging()
    create_required_dir()
    logger.info("Server Started")
    await init_db()
    logger.info("Tables are created ")
    yield
    logger.info("Server terminated")


tags_metadata = [
    {"name": "auth", "description": "Login, token refresh, and logout."},
    {
        "name": "test",
        "description": "test purpose only: have get_user which returns current user.",
    },
    {"name": "register", "description": "New user registration."},
    {
        "name": "upload",
        "description": "Upload PDF documents into a session's vector store.",
    },
    {
        "name": "chat",
        "description": "Ask questions and get RAG-powered, cited answers.",
    },
    {"name": "history", "description": "Browse past sessions and messages."},
    {"name": "health", "description": "For health checkup."},
]
app = FastAPI(
    title="Production RAG API",
    description="""
Backend for a document-grounded chat assistant. Upload PDFs into a session,
then ask questions — answers are generated from retrieved chunks of your
documents, with citations back to the source.

### Auth
Most endpoints require `Authorization: Bearer <access_token>`, obtained via
`/api/v1/auth/login`. Refresh tokens are stored in an `HttpOnly` cookie.
    """,
    version="1.0.0",
    openapi_tags=tags_metadata,
    lifespan=lifespan,
)
app.middleware("http")(logger_middleware)
register_exception_handlers(app=app)
app.include_router(auth_router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(register_router, prefix="/api/v1", tags=["register"])
app.include_router(test_router, prefix="/api/v1/test", tags=["test"])
app.include_router(chat_router, prefix="/api/v1", tags=["chat"])
app.include_router(history_router, prefix="/api/v1/history", tags=["history"])
app.include_router(file_upload, prefix="/api/v1", tags=["upload"])


@app.get("/")
def hello():
    return {"msg": "Hello user. How are you?"}


@app.get("/health", tags=["health"])
def health():
    return {"msg": "Yah, I am fine."}
