from __future__ import annotations
from fastapi import FastAPI
from dotenv import load_dotenv
from pathlib import Path
import os

from app.api import health
from app.api import rag as rag_api
from app.api.agents import router as agents_router
from app.api.chat import router as chat_router
from app.api import agents as agents_api


# ---- Load .env file ----
BASE_DIR = Path(__file__).resolve().parent.parent
print("BASE_DIR:", BASE_DIR)  # debug

env_path = BASE_DIR / ".env"
print("Loading .env from:", env_path)  # debug

load_dotenv(env_path, override=True)

print("RAG_INDEX_DIR from .env:", os.getenv("RAG_INDEX_DIR"))  # debug

app = FastAPI(title="ACRR FOSS Backend")

app.include_router(health.router)
app.include_router(rag_api.router)
app.include_router(agents_router)
app.include_router(chat_router)
app.include_router(agents_api.router)


@app.get("/")
def root() -> dict:
    return {"message": "ACRR FOSS backend running"}
