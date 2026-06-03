from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class DateRange(BaseModel):
    start_date: date | None = None
    end_date: date | None = None


class DistillOutput(BaseModel):
    intent: Literal["conversational", "knowledge", "synthesis", "sharing", "no_retrieval"]
    time_type: Literal["explicit", "soft_recency", "none"]
    date_range: DateRange
    needs_context: bool
    query: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4096)
    mode: Literal["distill", "qa"]
    context: str | None = Field(default=None, max_length=2000)
    today: date | None = None


class ChatResponse(BaseModel):
    mode: str
    raw_output: str
    # Both fields always present; one is null depending on mode.
    # This keeps the response shape uniform so callers need no branching.
    distill_result: DistillOutput | None
    answer: str | None
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class ReadyResponse(BaseModel):
    ready: bool
    backend: str | None
    model_path: str | None


class ModelInfoResponse(BaseModel):
    backend_type: str
    model_mode: str
    model_path: str | None
    base_model: str | None
    max_new_tokens: int
    temperature: float
    device: str | None
