import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import init_db, seed_db
from app.routes.chat import router as chat_router

load_dotenv()

# TODO: remove this when deploying to production and use proper database migrations instead.
init_db()
seed_db()

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
