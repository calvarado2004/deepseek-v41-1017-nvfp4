"""Official sampling and a thinking budget for DeepSeek-V4.1 (vLLM --middleware).

The 1017 model card asks for temperature 1.0 and top_p 1.0. At lower values
the reasoning loops ("Need maybe include ...", "Let me write. Go.") far more
often: on one prompt, 6 runs, 5/6 loops at 0.6/0.95 and 2/6 at 1.0/1.0
(2026-10-19). Clients override a server default (Qwen Code's DeepSeek
provider sends 0), so this ASGI middleware enforces the values on every
POST to the paths below instead.

It also caps thinking for chat requests that have thinking on and no
``thinking_token_budget`` of their own: the loop that remains at 1.0 is
bounded only by a budget (deepseek-recipe#24). The cap follows the effort.

    DSV41_FORCE_SAMPLING  1 (default) enforces; 0 passes bodies unchanged
    DSV41_TEMPERATURE     1.0
    DSV41_TOP_P           1.0
    DSV41_THINKING_BUDGET low:4096,high:8192,xhigh:16384,max:65536,default:6144
                          (empty: no budget is added)

Load with: --middleware dsv41_sampling_policy.SamplingPolicyMiddleware
"""

from __future__ import annotations

import json
import os

PATHS = {"/v1/chat/completions", "/v1/completions", "/v1/messages"}
FORCE = os.environ.get("DSV41_FORCE_SAMPLING", "1") == "1"
TEMPERATURE = float(os.environ.get("DSV41_TEMPERATURE", "1.0"))
TOP_P = float(os.environ.get("DSV41_TOP_P", "1.0"))
_BUDGET_SPEC = os.environ.get(
    "DSV41_THINKING_BUDGET", "low:4096,high:8192,xhigh:16384,max:65536,default:6144"
)
BUDGET = {k.strip(): int(v) for k, v in
          (item.split(":", 1) for item in _BUDGET_SPEC.split(",") if ":" in item)}


def _effort_name(effort) -> str | None:
    """low/high/xhigh/max, "none", or None when the request names no effort.
    Integer efforts (1-100) map to the named tier at or above them."""
    if effort is None:
        return None
    if isinstance(effort, str) and not effort.strip().lstrip("-").isdigit():
        return effort.strip().lower()
    n = int(effort)
    return "none" if n <= 0 else "low" if n <= 50 else "high" if n <= 75 else "max"


def _budget(req: dict) -> int | None:
    kwargs = req.get("chat_template_kwargs")
    kwargs = kwargs if isinstance(kwargs, dict) else {}
    if kwargs.get("thinking") is False or kwargs.get("enable_thinking") is False:
        return None
    effort = _effort_name(kwargs.get("reasoning_effort", req.get("reasoning_effort")))
    if effort in ("none", "minimal"):
        return None
    return BUDGET.get(effort or "default", BUDGET.get("default"))


def rewrite(path: str, body: bytes) -> bytes | None:
    """The rewritten body, or None to pass the request through unchanged."""
    if not FORCE:
        return None
    try:
        req = json.loads(body)
    except ValueError:
        return None  # vLLM answers the bad JSON itself
    if not isinstance(req, dict):
        return None
    req["temperature"] = TEMPERATURE
    req["top_p"] = TOP_P
    if path == "/v1/chat/completions" and req.get("thinking_token_budget") is None:
        budget = _budget(req)
        if budget:
            req["thinking_token_budget"] = budget
    return json.dumps(req).encode()


class SamplingPolicyMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        path = scope.get("path")
        if scope["type"] != "http" or scope.get("method") != "POST" or path not in PATHS:
            await self.app(scope, receive, send)
            return
        chunks, more = [], True
        while more:
            msg = await receive()
            if msg["type"] == "http.disconnect":
                return
            chunks.append(msg.get("body", b""))
            more = msg.get("more_body", False)
        body = b"".join(chunks)
        new = rewrite(path, body)
        if new is not None:
            body = new
            scope = dict(scope)
            scope["headers"] = [(k, v) for k, v in scope["headers"] if k.lower() != b"content-length"]
            scope["headers"].append((b"content-length", str(len(body)).encode()))
        replayed = False

        async def replay():
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)
