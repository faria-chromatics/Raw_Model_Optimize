from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.config import settings
from app.schemas import HealthResponse, ModelInfoResponse, ReadyResponse

router = APIRouter(tags=["ops"])


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        service=settings.app_title,
        version=settings.app_version,
    )


@router.get("/ready")
def ready(request: Request):
    is_ready = getattr(request.app.state, "ready", False)
    backend = getattr(request.app.state, "backend", None)
    if not is_ready:
        return JSONResponse(
            status_code=503,
            content={"ready": False, "backend": None, "model_path": None},
        )
    return ReadyResponse(
        ready=True,
        backend=type(backend).__name__,
        model_path=settings.model_path if settings.model_mode == "real" else None,
    )


@router.get("/model/info", response_model=ModelInfoResponse)
def model_info(request: Request):
    backend = getattr(request.app.state, "backend", None)
    return ModelInfoResponse(
        backend_type=type(backend).__name__ if backend else "none",
        model_mode=settings.model_mode,
        model_path=settings.model_path if settings.model_mode == "real" else None,
        base_model=settings.base_model_name if settings.model_mode == "real" else None,
        max_new_tokens=settings.max_new_tokens,
        temperature=settings.temperature,
        device=str(getattr(backend, "device", None)) if hasattr(backend, "device") else None,
    )
