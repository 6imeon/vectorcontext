"""OpenRouter client wrapper that records tokens, cost and latency per call."""
from __future__ import annotations
import os, time, json, threading
from dotenv import load_dotenv; load_dotenv()
from openai import OpenAI

MODEL = os.environ.get("VC_MODEL", "openai/gpt-5.6-luna")
EMBED_MODEL = os.environ.get("VC_EMBED_MODEL", "openai/text-embedding-3-small")
_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"], timeout=180, max_retries=3)

class Meter:
    """Accumulates usage for one episode (one question under one condition)."""
    def __init__(self):
        self.prompt_tokens=0; self.completion_tokens=0; self.reasoning_tokens=0; self.cost=0.0
        self.llm_calls=0; self.tool_calls=0; self.latency=0.0; self.embed_tokens=0; self.embed_cost=0.0; self.peak_prompt_tokens=0; self.cached_tokens=0
    def add(self, usage, dt):
        self.prompt_tokens += usage.prompt_tokens or 0
        self.completion_tokens += usage.completion_tokens or 0
        d = getattr(usage, "completion_tokens_details", None)
        self.reasoning_tokens += (getattr(d, "reasoning_tokens", 0) or 0) if d else 0
        self.cost += getattr(usage, "cost", 0.0) or 0.0
        self.peak_prompt_tokens = max(self.peak_prompt_tokens, usage.prompt_tokens or 0)
        pd = getattr(usage, "prompt_tokens_details", None)
        self.cached_tokens += (getattr(pd, "cached_tokens", 0) or 0) if pd else 0
        self.llm_calls += 1; self.latency += dt
    def as_dict(self):
        return dict(prompt_tokens=self.prompt_tokens, completion_tokens=self.completion_tokens, reasoning_tokens=self.reasoning_tokens,
                    total_tokens=self.prompt_tokens+self.completion_tokens, cost_usd=self.cost, llm_calls=self.llm_calls,
                    tool_calls=self.tool_calls, latency_s=self.latency, peak_prompt_tokens=self.peak_prompt_tokens, cached_tokens=self.cached_tokens, embed_tokens=self.embed_tokens, embed_cost_usd=self.embed_cost)

_global_cost = 0.0; _lock = threading.Lock()
BUDGET = float(os.environ.get("VC_BUDGET_USD", "4.0"))

def chat(messages, meter: Meter, tools=None, tool_choice=None, model=None, max_tokens=None, reasoning_effort=None):
    global _global_cost
    model = model or MODEL
    if _global_cost > BUDGET:
        raise RuntimeError(f"budget exceeded: ${_global_cost:.2f}")
    kw = dict(model=model, messages=messages, extra_body={"usage": {"include": True}})
    if tools: kw["tools"] = tools
    if tool_choice: kw["tool_choice"] = tool_choice
    if max_tokens: kw["max_tokens"] = max_tokens
    if reasoning_effort: kw["extra_body"]["reasoning"] = {"effort": reasoning_effort}
    t = time.time()
    r = _client.chat.completions.create(**kw)
    dt = time.time() - t
    meter.add(r.usage, dt)
    with _lock:
        _global_cost += getattr(r.usage, "cost", 0.0) or 0.0
    return r.choices[0].message

def embed(texts: list[str], meter: Meter | None = None, model=EMBED_MODEL):
    global _global_cost
    out=[]
    for i in range(0, len(texts), 64):
        r = _client.embeddings.create(model=model, input=texts[i:i+64])
        out.extend([d.embedding for d in r.data])
        c = getattr(r.usage, "cost", 0.0) or 0.0
        with _lock: _global_cost += c
        if meter is not None:
            meter.embed_tokens += r.usage.prompt_tokens or 0; meter.embed_cost += c
    return out

def global_cost(): return _global_cost
