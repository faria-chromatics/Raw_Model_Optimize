from __future__ import annotations

import json
import logging
import logging.config
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response

from app.config import settings
from app.inference import get_backend
from app.routers import chat, health


# --------------------------------------------------------------------------- #
# Logging                                                                      #
# --------------------------------------------------------------------------- #

class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        })


def _configure_logging() -> None:
    if settings.log_format == "json":
        config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {"json": {"()": _JsonFormatter}},
            "handlers": {"default": {"class": "logging.StreamHandler", "formatter": "json"}},
            "root": {"level": settings.log_level, "handlers": ["default"]},
        }
    else:
        fmt = "%(asctime)s %(levelname)-8s %(name)s  %(message)s"
        config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {"text": {"format": fmt}},
            "handlers": {"default": {"class": "logging.StreamHandler", "formatter": "text"}},
            "root": {"level": settings.log_level, "handlers": ["default"]},
        }
    logging.config.dictConfig(config)


_configure_logging()
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Lifespan                                                                     #
# --------------------------------------------------------------------------- #

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "Starting %s v%s [mode=%s env=%s]",
        settings.app_title, settings.app_version, settings.model_mode, settings.env,
    )
    app.state.ready = False
    app.state.backend = None

    backend = get_backend(settings)
    app.state.backend = backend
    app.state.ready = True
    logger.info("Backend ready: %s", type(backend).__name__)

    yield

    logger.info("Shutting down...")
    app.state.ready = False
    backend.unload()
    logger.info("Shutdown complete")


# --------------------------------------------------------------------------- #
# App                                                                          #
# --------------------------------------------------------------------------- #

app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    lifespan=lifespan,
)


@app.middleware("http")
async def attach_request_id(request: Request, call_next) -> Response:
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


app.include_router(health.router)
app.include_router(chat.router)
