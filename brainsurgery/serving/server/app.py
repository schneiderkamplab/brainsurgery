from __future__ import annotations

import json
import logging
import time
import uuid
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..engine import Engine

logger = logging.getLogger("brainsurgery.serving.server")

_engine: Engine | None = None


def create_app(engine: Engine) -> FastAPI:
    global _engine
    _engine = engine
    engine.start_background_loop()

    app = FastAPI(title="brainsurgery", version="0.1.0")

    @app.on_event("shutdown")
    async def shutdown():
        engine.stop_background_loop()

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": _model_id(),
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "brainsurgery",
                }
            ],
        }

    @app.post("/v1/completions")
    async def create_completion(req: CompletionRequest):
        prompt_ids = _resolve_prompt(req.prompt)
        if req.stream:
            return StreamingResponse(
                _completion_stream(prompt_ids, req, _model_id()),
                media_type="text/event-stream",
            )
        token_ids = await _run_generation(prompt_ids, req)
        text = _decode(token_ids)
        if req.echo:
            text = _decode(prompt_ids) + text
        return {
            "id": f"cmpl-{uuid.uuid4().hex[:12]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": _model_id(),
            "choices": [
                {
                    "text": text,
                    "index": 0,
                    "finish_reason": "length",
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(token_ids),
                "total_tokens": len(prompt_ids) + len(token_ids),
            },
        }

    @app.post("/v1/chat/completions")
    async def create_chat_completion(req: ChatCompletionRequest):
        prompt = _format_chat_prompt(req.messages)
        prompt_ids = _tokenize(prompt)
        if req.stream:
            return StreamingResponse(
                _chat_completion_stream(prompt_ids, req, _model_id()),
                media_type="text/event-stream",
            )
        token_ids = await _run_generation(prompt_ids, req)
        text = _decode(token_ids)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": _model_id(),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": text,
                    },
                    "finish_reason": "length",
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(token_ids),
                "total_tokens": len(prompt_ids) + len(token_ids),
            },
        }

    @app.get("/health")
    async def health():
        eng = _get_engine()
        return {
            "status": "ok",
            "running": eng.scheduler.running_count(),
            "pending": eng.scheduler.pending_count(),
        }

    return app


# --- Pydantic schemas ---

class CompletionRequest(BaseModel):
    model: str = "default"
    prompt: str | list[int] = ""
    max_tokens: int = Field(default=16, ge=1, le=4096)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    stream: bool = False
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    n: int = Field(default=1, ge=1, le=1)
    echo: bool = False
    stop: str | list[str] | None = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "default"
    messages: list[ChatMessage] = []
    max_tokens: int = Field(default=16, ge=1, le=4096)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    stream: bool = False


# --- Internal helpers ---

def _get_engine() -> Engine:
    if _engine is None:
        raise RuntimeError("Engine not initialized")
    return _engine


def _model_id() -> str:
    eng = _get_engine()
    return eng._model.config.extra.get("_name_or_path", "axon-model")


def _tokenize(text: str) -> list[int]:
    eng = _get_engine()
    if eng._tokenizer is None:
        raise HTTPException(status_code=500, detail="No tokenizer configured")
    return eng._tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0].tolist()


def _decode(tokens: list[int]) -> str:
    eng = _get_engine()
    if eng._tokenizer is None:
        raise HTTPException(status_code=500, detail="No tokenizer configured")
    return eng._tokenizer.decode(tokens, skip_special_tokens=True)


def _resolve_prompt(prompt: str | list[int]) -> list[int]:
    if isinstance(prompt, str):
        return _tokenize(prompt)
    return prompt


def _format_chat_prompt(messages: list[ChatMessage]) -> str:
    eng = _get_engine()
    if hasattr(eng._tokenizer, "apply_chat_template") and eng._tokenizer.chat_template is not None:
        raw = [{"role": m.role, "content": m.content} for m in messages]
        return eng._tokenizer.apply_chat_template(raw, tokenize=False, add_generation_prompt=True)
    lines = "\n".join(f"{m.role}: {m.content}" for m in messages)
    return lines + "\nassistant:"


async def _generate(
    prompt_ids: list[int],
    req: CompletionRequest | ChatCompletionRequest,
) -> AsyncGenerator[int, None]:
    """Add a request and yield token IDs one by one from the background loop."""
    eng = _get_engine()
    seq_id = eng.add_request(
        prompt_ids,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=getattr(req, "top_p", 1.0),
    )
    while True:
        token_id = await eng.await_token(seq_id)
        if token_id is None:
            break
        yield token_id


async def _run_generation(
    prompt_ids: list[int],
    req: CompletionRequest | ChatCompletionRequest,
) -> list[int]:
    token_ids: list[int] = []
    async for token_id in _generate(prompt_ids, req):
        token_ids.append(token_id)
    return token_ids


async def _completion_stream(
    prompt_ids: list[int],
    req: CompletionRequest,
    model_name: str,
) -> AsyncGenerator[str, None]:
    completion_id = f"cmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    async for token_id in _generate(prompt_ids, req):
        text = _decode([token_id])
        chunk = {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "text": text,
                    "index": 0,
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    yield f"data: {json.dumps({'id': completion_id, 'object': 'text_completion', 'created': created, 'model': model_name, 'choices': [{'text': '', 'index': 0, 'finish_reason': 'length'}]})}\n\n"
    yield "data: [DONE]\n\n"


async def _chat_completion_stream(
    prompt_ids: list[int],
    req: ChatCompletionRequest,
    model_name: str,
) -> AsyncGenerator[str, None]:
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model_name, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]})}\n\n"

    async for token_id in _generate(prompt_ids, req):
        text = _decode([token_id])
        yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model_name, 'choices': [{'index': 0, 'delta': {'content': text}, 'finish_reason': None}]})}\n\n"

    yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model_name, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'length'}]})}\n\n"
    yield "data: [DONE]\n\n"
