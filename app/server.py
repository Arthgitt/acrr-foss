# app/server.py
from __future__ import annotations
from fastapi import FastAPI
from dotenv import load_dotenv
from pathlib import Path
import os

from app.api import health
from app.api import rag as rag_api
from app.api.agents import router as agents_router


# ---- Load .env file (FORCE override) ----
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



@app.get("/")
def root() -> dict:
    return {"message": "ACRR FOSS backend running"}
