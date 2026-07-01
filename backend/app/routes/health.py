import logging
from fastapi import APIRouter
from pydantic import BaseModel
from search import get_es
from cache import _get_redis

logger = logging.getLogger(__name__)

router = APIRouter()


class ServiceStatus(BaseModel):
    status: str
    detail: str = ""


class HealthResponse(BaseModel):
    status: str
    services: dict[str, ServiceStatus]


@router.get("/health", response_model=HealthResponse)
def health_check():
    services = {}

    # Elasticsearch
    try:
        info = get_es().cluster.health()
        es_status = info.get("status", "unknown")
        services["elasticsearch"] = ServiceStatus(status="ok", detail=es_status)
    except Exception as e:
        logger.warning("Health check: Elasticsearch unavailable: %s", e)
        services["elasticsearch"] = ServiceStatus(status="error", detail=str(e))

    # Redis
    try:
        _get_redis().ping()
        services["redis"] = ServiceStatus(status="ok")
    except Exception as e:
        logger.warning("Health check: Redis unavailable: %s", e)
        services["redis"] = ServiceStatus(status="error", detail=str(e))

    overall = "ok" if all(s.status == "ok" for s in services.values()) else "degraded"
    return HealthResponse(status=overall, services=services)
