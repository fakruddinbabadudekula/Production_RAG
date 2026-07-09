# Production RAG

An async, production-oriented **Retrieval-Augmented Generation (RAG)** backend built with FastAPI. Users register, upload PDF documents into isolated per-session vector stores, and chat with an LLM that answers questions using retrieved, citation-tagged context from those documents.

Think of it as the backend for a "chat with your documents" product (NotebookLM-style), built with clean service/repository layering, JWT auth, async Postgres, and a LangGraph-orchestrated retrieval pipeline.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Environment Variables](#environment-variables)
  - [Run with Docker (recommended)](#run-with-docker-recommended)
  - [Run Locally (without Docker)](#run-locally-without-docker)
- [API Overview](#api-overview)
- [Request/Response Flow](#requestresponse-flow)
- [Data Model](#data-model)
- [Error Handling](#error-handling)
- [Logging & Observability](#logging--observability)
- [Design Notes & Known Limitations](#design-notes--known-limitations)
- [Roadmap](#roadmap)
- [Contributing](#contributing)

---

## Features

- **JWT authentication** — access + refresh token flow, refresh token stored as an `HttpOnly` cookie scoped to `/api/v1/auth/refresh`.
- **User registration & login** with Argon2 password hashing.
- **PDF ingestion pipeline** — upload → validate → chunk → embed → store, fully async, with automatic rollback/cleanup on failure.
- **Per-session isolated vector stores** — each `(user_id, session_id)` pair gets its own FAISS index on disk, so documents from one conversation never leak into another.
- **RAG chat via LangGraph** — a two-node graph (`retriever` → `chat`) orchestrates retrieval and generation, returning answers with a citation-enforced prompt (`[1]`, `[2]`, …) and the source chunks used.
- **Conversation history** — sessions and messages are persisted; retrieved sources are stored alongside assistant messages.
- **Resilient LLM calls** — retries with exponential backoff on transient errors (rate limits, timeouts, connection errors) via `tenacity`.
- **Centralized error handling** — a custom exception hierarchy (`AppException` subclasses) mapped to consistent HTTP status codes and JSON error bodies.
- **Structured, contextual logging** — every request gets a `request_id` (propagated via `X-Request-Id` header) threaded through all logs for that request.
- **Async everywhere** — SQLAlchemy 2.0 async ORM + `asyncpg`, async file I/O, async vector store operations.
- **Multi-stage Docker build** — small production image, runs as a non-root user, with a built-in healthcheck.

## Architecture

```
                              ┌──────────────────────┐
                              │        Client        │
                              └──────────┬───────────┘
                                         │ JWT (Bearer + refresh cookie)
                                         ▼
                              ┌──────────────────────┐
                              │   FastAPI (app/api)  │
                              │  routes / middleware │
                              │  exception handlers  │
                              └──────────┬───────────┘
                                         │
                 ┌───────────────────────┼────────────────────────┐
                 ▼                       ▼                        ▼
        ┌─────────────────┐      ┌────────────────┐        ┌────────────────┐
        │  Auth Service   │      │ File Ingestion │        │ Chat Service   │
        │ (login/register/│      │    Service     │        │ (RAG workflow) │
        │   refresh)      │      └───────┬────────┘        └───────┬────────┘
        └────────┬────────┘              │                         │
                 │             ┌─────────▼─────────┐       ┌───────▼────────┐
                 │             │  Document Loader  │       │  LangGraph     │
                 │             │  (PyMuPDF + split)│       │  retriever →   │
                 │             └─────────┬─────────┘       │  chat nodes    │
                 │                       ▼                 └───────┬────────┘
                 │             ┌────────────────────┐              │
                 │             │ Vector Store Svc   │◄─────────────┘
                 │             │(FAISS, per-session)│
                 │             └─────────┬──────────┘
                 │                       │
                 ▼                       ▼
        ┌──────────────────────────────────────────────┐
        │           PostgreSQL (users, sessions,       │
        │        messages, files_metadata)             │
        └──────────────────────────────────────────────┘
```

**Request lifecycle:** every HTTP request passes through `logger_middleware` (assigns/propagates a `request_id`), hits a route, flows through a service (business logic), then a repository (DB access) or the RAG layer (retrieval + LLM). Exceptions raised anywhere bubble up to centralized handlers that translate them into consistent JSON responses.

## Tech Stack

| Layer              | Technology                                                         |
|--------------------|--------------------------------------------------------------------|
| Web framework      | FastAPI, Uvicorn                                                   |
| Language           | Python 3.11+                                                       |
| Database           | PostgreSQL, SQLAlchemy 2.0 (async), `asyncpg`                      |
| Auth               | JWT (`python-jose`), `passlib[argon2]`                             |
| RAG orchestration  | LangGraph, LangChain (core / community)                            |
| Vector store       | FAISS (`faiss-cpu`)                                                |
| Embeddings         | Google Generative AI (`gemini-embedding-001`)                      |
| LLM                | OpenRouter, via `langchain-openai` / `ChatOpenAI`                  |
| Document parsing   | PyMuPDF (PDF), LangChain text splitters                            |
| Retry logic        | `tenacity`                                                         |
| Package management | `uv`                                                               |
| Containerization   | Docker (multi-stage build), Docker Compose                         |

## Project Structure

```
app/
├── main.py                    # App factory, lifespan, router registration
├── api/
│   ├── routes/                 # auth, register, chat, history, upload, test
│   ├── schemas/                 # Pydantic request/response models
│   ├── dependencies.py          # get_current_user, get_db
│   ├── middleware.py             # request-id logging middleware
│   └── exception_handlers.py      # AppException -> HTTP response mapping
├── core/
│   ├── config.py                # Settings (env-driven), paths, timeouts, retries
│   ├── security.py               # password hashing, JWT create/verify
│   ├── db.py                     # async engine/session, Base, init_db
│   ├── exceptions.py               # AppException hierarchy
│   ├── contextvar.py                # request_id / route context vars for logging
│   └── logging.py                   # logging configuration
├── models/                     # SQLAlchemy ORM models (User, Session, Message, File)
├── schemas/                     # Shared Pydantic schemas & enums
├── repositories/                 # DB access layer (users, sessions, messages, files)
├── services/                      # Business logic (auth, chat, ingestion, vector store, LLM client)
└── rag/
    ├── interface.py                # AsyncLLMClient abstract interface
    ├── document_loaders/             # PDF loading + chunking
    ├── vector/                        # FAISS store creation/load, vector store manager
    └── workflow/graph.py               # LangGraph RAG graph (retriever -> chat)
```

## Getting Started

### Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- Docker & Docker Compose (recommended path)
- PostgreSQL 16 (if running outside Docker)
- API keys: an [OpenRouter](https://openrouter.ai/) key (chat model) and a Google API key (embeddings)

### Environment Variables

Copy `.env.example` to `.env` and fill in the values:

```bash
cp .env.example .env
```

| Variable                       | Description                                              | Example / Default                                                  |
|---------------------------------|------------------------------------------------------------|----------------------------------------------------------------------|
| `OPENROUTER_BASE_URL`            | Base URL for the OpenRouter-compatible chat API              | `https://openrouter.ai/api/v1`                                        |
| `OPENROUTER_API_KEY`             | API key for the chat LLM                                     | *(required)*                                                          |
| `GOOGLE_API_KEY`                 | API key for Google Generative AI embeddings                    | *(required)*                                                          |
| `DATABASE_URL`                    | Async Postgres connection string                                | `postgresql+asyncpg://user:pass@host:5432/db`                        |
| `SECRET_KEY`                      | Secret used to sign JWTs — **generate your own, never reuse the example** | 64-char hex string                                                   |
| `ALGORITHM`                        | JWT signing algorithm                                              | `HS256`                                                               |
| `ACCESS_TOKEN_EXPIRE_MINUTES`       | Access token lifetime (minutes)                                      | `15`                                                                  |
| `REFRESH_TOKEN_EXPIRE_DAYS`          | Refresh token lifetime (days)                                          | `7`                                                                   |

Other tunables (chat model name, temperature, chunk size, timeouts, retry counts, embedding dimensions) have sensible defaults in `app/core/config.py` and can be overridden via environment variables of the same name if needed.

> ⚠️ **Security note:** `.env.example` in this repo ships with a sample `SECRET_KEY` and database credentials for local development convenience only. Always generate a fresh `SECRET_KEY` (e.g. `openssl rand -hex 32`) and use strong, unique database credentials before deploying anywhere non-local.

### Run with Docker (recommended)

```bash
docker compose up --build
```

This starts two services:
- `web` — the FastAPI app on `http://localhost:8000`, auto-reloading on code changes.
- `postgres_db` — PostgreSQL 16, with a healthcheck gating app startup.

Database tables are created automatically on startup (via `init_db()` in the app's lifespan handler).

Once running, open:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- Health check: `http://localhost:8000/health`

### Run Locally (without Docker)

```bash
# Install dependencies
uv sync

# Make sure a Postgres instance is running and DATABASE_URL points to it
# Then start the server
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
### Database Connection String

**Running with Docker**

No changes needed. Leave the `DATABASE_URL` line in `.env` as-is — the default already matches `docker-compose.yml`:

```dotenv
DATABASE_URL=postgresql+asyncpg://docker_user:secret123@postgres_db:5432/rag_docker_db
```

`postgres_db` resolves automatically via Docker's internal network, so the app connects out of the box.

**Running locally with `uv`**

Modify the line and update only `user`, `password`, `host`, and `database` to match your local Postgres setup. Keep the rest of the structure exactly as it is:

```dotenv
DATABASE_URL=postgresql+asyncpg://<user>:<password>@localhost:5432/<database>
```

**Example** — local Postgres with user `admin`, password `mypassword`, database `rag_db`:

```dotenv
DATABASE_URL=postgresql+asyncpg://admin:mypassword@localhost:5432/rag_db
```
## API Overview

All routes are versioned under `/api/v1`. Full interactive documentation (request/response schemas, auth, examples) is always available at `/docs` and `/redoc` — the table below is a quick reference, not a replacement for it.

| Method | Path                        | Auth required | Description                                              |
|--------|------------------------------|:---:|--------------------------------------------------------------|
| POST   | `/api/v1/register`             |  ❌  | Register a new user                                            |
| POST   | `/api/v1/auth/login`             |  ❌  | Log in, returns access token + sets refresh cookie                |
| POST   | `/api/v1/auth/refresh`            |  🍪  | Exchange refresh cookie for a new access token                      |
| POST   | `/api/v1/auth/logout`               |  ✅  | Clears the refresh token cookie                                       |
| POST   | `/api/v1/upload`                     |  ✅  | Upload a PDF, chunk + embed it into the session's vector store           |
| POST   | `/api/v1/chat`                          |  ✅  | Ask a question; runs the RAG graph and returns a cited answer              |
| GET    | `/api/v1/history/sessions`                |  ✅  | List the current user's conversation sessions                                |
| GET    | `/api/v1/history/messages?session_id=`      |  ✅  | List messages for a given session                                                |
| GET    | `/`                                            |  ❌  | Basic liveness message                                                              |
| GET    | `/health`                                        |  ❌  | Health check endpoint (used by the Docker healthcheck)                                |

Protected routes expect `Authorization: Bearer <access_token>`. The refresh endpoint instead reads the `refresh_token` `HttpOnly` cookie set at login.

## Request/Response Flow

**Uploading a document:**
1. Client sends `multipart/form-data` with a PDF file (and optionally an existing `session_id`).
2. If no `session_id` is given, a new session is created for the user.
3. The file is validated (extension), saved to disk under `data/upload_files/<user_id>/<session_id>/<file_id>.pdf`, loaded via PyMuPDF, and split into chunks.
4. Chunks are embedded and added to that session's FAISS index (`data/vectors/<user_id>/<session_id>/`).
5. File metadata is persisted to Postgres. If any step after file save fails, the uploaded file and any partially-written vectors are cleaned up.

**Chatting:**
1. Client sends `{ session_id, query }`.
2. The session is verified to belong to the current user.
3. Prior messages for the session are loaded and converted to LangChain messages.
4. The LangGraph workflow retrieves top-k relevant chunks from that session's vector store, builds a citation-enforced prompt, and calls the LLM (with retry/backoff on transient failures).
5. Both the user's query and the assistant's cited response (plus the source chunks used) are persisted, and the response is returned to the client.

## Data Model

| Table            | Key columns                                                              |
|-------------------|-----------------------------------------------------------------------------|
| `users`             | `user_id` (UUID, PK), `name`, `email` (unique), `hashed_password`, `created_at` |
| `sessions`           | `session_id` (UUID, PK), `user_id` (FK), `title`, `created_at`                    |
| `messages`            | `message_id` (UUID, PK), `session_id` (FK), `role` (`system`/`user`/`assistant`), `content`, `top_k_docs` (JSON), `created_at` |
| `files_metadata`       | `file_id` (UUID, PK), `session_id` (FK), `type`, `name`, `size`, `created_at`         |

A user has many sessions; a session has many messages and many uploaded files (cascade delete on both).

## Error Handling

All domain errors extend `AppException` and carry a `message` plus optional `details` dict. They're mapped centrally to HTTP status codes:

| Exception                     | HTTP Status                     |
|---------------------------------|------------------------------------|
| `ResourceNotFoundException`        | 404 Not Found                        |
| `ValidationException`                | 400 Bad Request                        |
| `UnSupportedResource`                  | 415 Unsupported Media Type               |
| `DuplicateResourceException`             | 409 Conflict                                |
| `InvalidCredentialsException`             | 401 Unauthorized                              |
| `LLMServieException`                        | 503 Service Unavailable                         |
| `InvalidFilePaths`                            | 500 Internal Server Error                         |

Pydantic validation errors are serialized into a `{ field, message }[]` list. Any unhandled exception is caught by a top-level handler and returned as a generic `500` with a safe, non-leaking message (the real exception is logged with `exc_info`).

## Logging & Observability

- Every request is assigned a `request_id` (reused from the incoming `X-Request-Id` header if present) and the current route, both stored in context variables so they can be attached to every log line for that request without threading them through every function call.
- Request start/completion are logged with method, route, status code, and duration.
- Key RAG operations (retrieval, PDF processing, graph execution, LLM calls) log structured timing and counts (`duration`, `count`, `chunks`, etc.) for performance monitoring.
- The response includes an `X-Request-Id` header, useful for correlating client-reported issues with server logs.

## Design Notes & Known Limitations

This project favors transparency over polish in a few places — worth knowing before extending it:

- **PDF only for now** — `DocumentLoader` and the upload validation only support `.pdf`. Adding a new format means implementing a loader and registering its extension.
- **FAISS on local disk** — vector stores are per-session folders on the filesystem. This is simple and fast for a single instance but doesn't horizontally scale across multiple app replicas without a shared volume or a move to a managed vector DB.
- **Refresh token rotation** — the `/auth/refresh` endpoint issues a new access + refresh token pair but doesn't yet invalidate the old refresh token server-side.
- **Message storage on failure** — if the LLM call fails, neither the user's message nor a failed-assistant placeholder is stored yet.
- **`/api/v1/test` route** — a leftover development route (`get_user`) not intended for production use.

These are called out directly in code comments throughout the project — treat them as a running list of intentional trade-offs, not bugs.

## Roadmap

- [ ] Support additional document formats (`.docx`, `.txt`, `.md`)
- [ ] Streaming chat responses (SSE/WebSocket)
- [ ] Refresh token rotation with server-side revocation
- [ ] Swap local FAISS for a managed/shared vector store for multi-instance deployments
- [ ] Rate limiting on auth and chat endpoints
- [ ] Automated test suite (unit + integration)

## Contributing

1. Fork the repo and create a feature branch.
2. Install dependencies with `uv sync`.
3. Keep the layering convention: routes call services, services call repositories/RAG components — don't access the DB or vector store directly from a route.
4. Add/adjust docstrings and OpenAPI metadata (`summary`, `response_model`, `responses`) for any new or changed endpoint so `/docs` stays accurate.
5. Open a PR describing the change and its motivation.