from __future__ import annotations

import json
import logging
import time
from datetime import date

from fastapi import APIRouter, HTTPException, Request

from app.schemas import ChatRequest, ChatResponse, DistillOutput

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


def _build_prompt(req: ChatRequest) -> str:
    """
    Constructs the prompt string using the exact chat template from training.
    Distill format sourced from rag_training_data.jsonl examples.
    """
    today = req.today or date.today()
    if req.mode == "distill":
        lines = ["[DISTILL]", f"TODAY: {today}"]
        if req.context:
            lines.append(f"CONTEXT: {req.context}")
        lines.append(f'MESSAGE: "{req.message}"')
        user_content = "\n".join(lines)
    else:
        user_content = req.message

    return (
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
    backend = getattr(request.app.state, "backend", None)
    if backend is None or not backend.is_ready():
        raise HTTPException(status_code=503, detail="Service not ready")

    prompt = _build_prompt(req)
    t0 = time.perf_counter()
    try:
        raw = backend.generate(prompt, req.mode)
    except Exception as exc:
        logger.exception("Inference failed for mode=%s", req.mode)
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    latency_ms = (time.perf_counter() - t0) * 1000

    distill_result: DistillOutput | None = None
    answer: str | None = None

    if req.mode == "distill":
        try:
            distill_result = DistillOutput(**json.loads(raw))
        except (json.JSONDecodeError, ValueError):
            logger.warning("Distill output was not valid JSON; raw_output preserved")
    else:
        answer = raw

    return ChatResponse(
        mode=req.mode,
        raw_output=raw,
        distill_result=distill_result,
        answer=answer,
        latency_ms=round(latency_ms, 2),
    )
