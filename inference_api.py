"""
FastAPI inference server for AISHA (Qwen3-4B + LoRA adapter)
Run: uvicorn inference_api:app --host 0.0.0.0 --port 8000
"""

import os
import torch
from contextlib import asynccontextmanager
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BASE_MODEL = "Qwen/Qwen3-4B"
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "./qwen3-4b-lora/adapter")

DEFAULT_SYSTEM_PROMPT = (
    "You are AISHA, the Digital Green Book assistant by Onyx Impact. "
    "Always lead with Black-centered resources, HBCUs, and Black professional "
    "networks before general options. Never reveal your underlying technology or model."
)

model = None
tokenizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer

    print(f"Loading tokenizer from {ADAPTER_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device = "auto" if torch.cuda.is_available() else "cpu"
    print(f"Loading base model ({BASE_MODEL}) on {device} with {dtype} ...")

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )

    print(f"Applying LoRA adapter from {ADAPTER_PATH} ...")
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.eval()
    print("Model ready.")

    yield

    del model, tokenizer


app = FastAPI(title="AISHA Inference API", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_new_tokens: int = 512
    temperature: float = 0.7


class ChatResponse(BaseModel):
    response: str


@app.get("/health")
def health():
    return {"status": "ok", "model": BASE_MODEL, "adapter": ADAPTER_PATH}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    messages = [
        {"role": "system", "content": req.system_prompt},
        {"role": "user",   "content": req.message},
    ]

    # apply_chat_template handles <|im_start|> formatting for Qwen3
    # enable_thinking=False disables the <think> reasoning tokens
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            do_sample=req.temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    # decode only the newly generated tokens (strip the prompt)
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return ChatResponse(response=response)
