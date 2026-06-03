from __future__ import annotations

import json
import logging
from datetime import date

logger = logging.getLogger(__name__)


class MockBackend:
    """Stateless placeholder — returns valid output shapes with no model files."""

    def is_ready(self) -> bool:
        return True

    def generate(self, prompt: str, mode: str) -> str:
        if mode == "distill":
            today = date.today()
            start = today.replace(day=max(1, today.day - 7))
            return json.dumps({
                "intent": "knowledge",
                "time_type": "soft_recency",
                "date_range": {"start_date": str(start), "end_date": str(today)},
                "needs_context": False,
                "query": "mock query",
            })
        return "[Mock] Model not yet loaded. This is a placeholder response."

    def unload(self) -> None:
        pass


class QwenBackend:
    """
    Loads Qwen3-4B base model + LoRA adapter from settings.model_path.

    This class is filled in when adapter artifacts become available.
    Only this file needs to change — all routers and schemas stay untouched.
    """

    def __init__(self, settings) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading tokenizer from %s", settings.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        try:
            logger.info("Loading base model: %s [device=%s]", settings.base_model_name, settings.device)
            if settings.device == "cuda":
                base = AutoModelForCausalLM.from_pretrained(
                    settings.base_model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                )
            else:
                # CPU: do NOT use device_map — it triggers PEFT's disk-offload
                # path which corrupts module names and raises KeyError on adapter load
                base = AutoModelForCausalLM.from_pretrained(
                    settings.base_model_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                )

            logger.info("Loading LoRA adapter from %s", settings.model_path)
            self.model = PeftModel.from_pretrained(base, settings.model_path)
            self.model.eval()
        except Exception:
            logger.exception(
                "Failed to load model. base_model=%s adapter=%s device=%s",
                settings.base_model_name, settings.model_path, settings.device,
            )
            raise

        self.device = next(self.model.parameters()).device
        self.settings = settings
        logger.info("QwenBackend ready on device: %s", self.device)

    def is_ready(self) -> bool:
        return True

    def generate(self, prompt: str, mode: str) -> str:
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.settings.max_new_tokens,
                temperature=self.settings.temperature,
                do_sample=self.settings.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        # Decode only newly generated tokens, not the prompt
        new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def unload(self) -> None:
        import torch

        del self.model
        del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("QwenBackend unloaded")


def get_backend(settings) -> MockBackend | QwenBackend:
    if settings.model_mode == "mock":
        logger.info("Backend: mock mode")
        return MockBackend()
    logger.info("Backend: real model from %s", settings.model_path)
    return QwenBackend(settings)
