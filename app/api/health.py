# app/api/health.py

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check() -> dict:
    """
    Simple health-check endpoint.
    GET /api/health
    """
    return {"status": "ok"}
