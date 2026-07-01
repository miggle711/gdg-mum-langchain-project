import sys
import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings  # noqa: E402 — must come after sys.path insert
from database import init_db, seed_db, init_es_index
from app.routes.chat import router as chat_router

# TODO: remove this when deploying to production and use proper database migrations instead.
init_db()
seed_db()
init_es_index()

app = FastAPI(title="LangChain Conversation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)


@app.get("/")
def root():
    return {"message": "LangChain Conversation Backend is running"}
