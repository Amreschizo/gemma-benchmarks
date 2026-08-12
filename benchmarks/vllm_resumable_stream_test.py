#!/usr/bin/env python3
"""Benchmark streaming latency: Ollama vs vLLM with prefix caching.

Measures TTFT, prefix-cached TTFT, concurrent in-flight requests, and
simulated TTS-ready chunk latency (first punctuation boundary or N chars).
"""

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
import urllib.request
from dataclasses import dataclass, asdict, field
from typing import Any, Optional

OLLAMA_HOST = "http://127.0.0.1:11435"
VLLM_HOST = "http://127.0.0.1:8000"
OLLAMA_MODEL = "batiai/gemma4-e4b:q6"
VLLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"  # APC proxy; Gemma 4 E4B blocked on vLLM 0.27
VLLM_MODEL_GEMMA4 = "google/gemma-4-E4B-it"

# Long shared prefix simulates system prompt + conversation history for TTS pipeline
SHARED_PREFIX = (
    "You are a helpful voice assistant. Respond in natural spoken language. "
    "Keep answers concise (2-4 sentences). Do not use markdown, bullet points, "
    "or code blocks. Speak as if talking to someone in person.\n\n"
    "Context: The user is testing a real-time voice pipeline where your text "
    "is streamed to a TTS engine. Latency matters. Prior conversation:\n"
    "User: What is the capital of France?\n"
    "Assistant: The capital of France is Paris, a beautiful city known for "
    "the Eiffel Tower and rich cultural history.\n"
    "User: Tell me about its weather.\n"
    "Assistant: Paris has a temperate climate with mild winters and warm summers. "
    "Spring and fall are particularly pleasant for visiting.\n\n"
)

USER_PROMPTS = [
    "What are three famous museums in Paris?",
    "How do I get from the airport to the city center?",
    "Recommend a good restaurant for traditional French cuisine.",
    "What is the best time of year to visit?",
    "Tell me about the Louvre museum briefly.",
]

TTS_CHUNK_CHARS = 20
SENTENCE_END_RE = re.compile(r"[.!?]\s")


@dataclass
class StreamMetrics:
    backend: str
    scenario: str
    run: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttft_ms: float = 0.0
    tts_ready_ms: float = 0.0  # time to first TTS-ready chunk
    tts_ready_chars: int = 0
    total_time_ms: float = 0.0
    tokens_per_sec: float = 0.0
    prefix_cached: bool = False
    error: Optional[str] = None
    extra: dict = field(default_factory=dict)


def tts_ready_offset(text: str) -> tuple[int, int]:
    """Return (char_index, chars_at_ready) for first TTS chunk boundary."""
    if len(text) >= TTS_CHUNK_CHARS:
        return TTS_CHUNK_CHARS, TTS_CHUNK_CHARS
    m = SENTENCE_END_RE.search(text)
    if m:
        return m.end(), m.end()
    return len(text), len(text)


def stream_ollama(
    model: str,
    user_prompt: str,
    prefix: str = "",
    num_predict: int = 128,
) -> StreamMetrics:
    full_prompt = prefix + f"User: {user_prompt}\nAssistant:"
    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": True,
        "think": False,
        "options": {"num_predict": num_predict, "temperature": 0.0},
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    start = time.perf_counter()
    first_token_time = None
    tts_ready_time = None
    tts_ready_chars = 0
    completion_tokens = 0
    accumulated = ""
    final_stats: dict = {}

    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            for raw_line in resp:
                line = raw_line.decode().strip()
                if not line:
                    continue
                chunk = json.loads(line)
                token_text = chunk.get("response", "")
                if token_text:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    completion_tokens += 1
                    accumulated += token_text
                    if tts_ready_time is None:
                        idx, chars = tts_ready_offset(accumulated)
                        if idx > 0 and (
                            len(accumulated) >= TTS_CHUNK_CHARS
                            or SENTENCE_END_RE.search(accumulated)
                        ):
                            tts_ready_time = time.perf_counter()
                            tts_ready_chars = chars
                if chunk.get("done"):
                    final_stats = chunk
                    break
    except Exception as e:
        return StreamMetrics(
            backend="ollama",
            scenario="",
            run=0,
            error=str(e),
        )

    end = time.perf_counter()
    ttft_ms = (first_token_time - start) * 1000 if first_token_time else 0
    if tts_ready_time is None and first_token_time:
        tts_ready_time = end
        tts_ready_chars = len(accumulated)
    tts_ready_ms = (tts_ready_time - start) * 1000 if tts_ready_time else 0
    total_ms = (end - start) * 1000
    eval_ns = final_stats.get("eval_duration", 0)
    tps = (
        completion_tokens / (eval_ns / 1e9)
        if eval_ns and completion_tokens
        else (completion_tokens / (total_ms / 1000) if total_ms else 0)
    )

    return StreamMetrics(
        backend="ollama",
        scenario="",
        run=0,
        prompt_tokens=final_stats.get("prompt_eval_count", 0),
        completion_tokens=completion_tokens,
        ttft_ms=round(ttft_ms, 2),
        tts_ready_ms=round(tts_ready_ms, 2),
        tts_ready_chars=tts_ready_chars,
        total_time_ms=round(total_ms, 2),
        tokens_per_sec=round(tps, 2),
    )


def stream_vllm_openai(
    user_prompt: str,
    prefix: str = "",
    num_predict: int = 128,
) -> StreamMetrics:
    """Stream via OpenAI-compatible /v1/completions endpoint."""
    full_prompt = prefix + f"User: {user_prompt}\nAssistant:"
    payload = {
        "model": VLLM_MODEL,
        "prompt": full_prompt,
        "max_tokens": num_predict,
        "temperature": 0.0,
        "stream": True,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{VLLM_HOST}/v1/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    start = time.perf_counter()
    first_token_time = None
    tts_ready_time = None
    tts_ready_chars = 0
    completion_tokens = 0
    accumulated = ""
    prompt_tokens = 0

    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            for raw_line in resp:
                line = raw_line.decode().strip()
                if not line or line == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    line = line[6:]
                chunk = json.loads(line)
                choices = chunk.get("choices", [])
                if not choices:
                    continue
                token_text = choices[0].get("text", "")
                usage = chunk.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                if token_text:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    completion_tokens += 1
                    accumulated += token_text
                    if tts_ready_time is None:
                        if len(accumulated) >= TTS_CHUNK_CHARS or SENTENCE_END_RE.search(
                            accumulated
                        ):
                            tts_ready_time = time.perf_counter()
                            _, chars = tts_ready_offset(accumulated)
                            tts_ready_chars = chars
    except Exception as e:
        return StreamMetrics(
            backend="vllm",
            scenario="",
            run=0,
            error=str(e),
        )

    end = time.perf_counter()
    ttft_ms = (first_token_time - start) * 1000 if first_token_time else 0
    if tts_ready_time is None and first_token_time:
        tts_ready_time = end
        tts_ready_chars = len(accumulated)
    tts_ready_ms = (tts_ready_time - start) * 1000 if tts_ready_time else 0
    total_ms = (end - start) * 1000
    decode_ms = (end - (first_token_time or start)) * 1000
    tps = completion_tokens / (decode_ms / 1000) if decode_ms and completion_tokens else 0

    return StreamMetrics(
        backend="vllm",
        scenario="",
        run=0,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        ttft_ms=round(ttft_ms, 2),
        tts_ready_ms=round(tts_ready_ms, 2),
        tts_ready_chars=tts_ready_chars,
        total_time_ms=round(total_ms, 2),
        tokens_per_sec=round(tps, 2),
    )


async def stream_vllm_async(
    user_prompt: str,
    prefix: str = "",
    num_predict: int = 128,
    client: Any = None,
) -> StreamMetrics:
    """Async streaming for concurrent in-flight benchmark."""
    import httpx

    full_prompt = prefix + f"User: {user_prompt}\nAssistant:"
    payload = {
        "model": VLLM_MODEL,
        "prompt": full_prompt,
        "max_tokens": num_predict,
        "temperature": 0.0,
        "stream": True,
    }

    start = time.perf_counter()
    first_token_time = None
    tts_ready_time = None
    tts_ready_chars = 0
    completion_tokens = 0
    accumulated = ""

    try:
        async with client.stream(
            "POST",
            f"{VLLM_HOST}/v1/completions",
            json=payload,
            timeout=600,
        ) as resp:
            async for line in resp.aiter_lines():
                if not line or line == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    line = line[6:]
                chunk = json.loads(line)
                choices = chunk.get("choices", [])
                if not choices:
                    continue
                token_text = choices[0].get("text", "")
                if token_text:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    completion_tokens += 1
                    accumulated += token_text
                    if tts_ready_time is None:
                        if len(accumulated) >= TTS_CHUNK_CHARS or SENTENCE_END_RE.search(
                            accumulated
                        ):
                            tts_ready_time = time.perf_counter()
                            _, chars = tts_ready_offset(accumulated)
                            tts_ready_chars = chars
    except Exception as e:
        return StreamMetrics(backend="vllm", scenario="", run=0, error=str(e))

    end = time.perf_counter()
    ttft_ms = (first_token_time - start) * 1000 if first_token_time else 0
    if tts_ready_time is None and first_token_time:
        tts_ready_time = end
        tts_ready_chars = len(accumulated)
    tts_ready_ms = (tts_ready_time - start) * 1000 if tts_ready_time else 0
    total_ms = (end - start) * 1000

    return StreamMetrics(
        backend="vllm",
        scenario="",
        run=0,
        completion_tokens=completion_tokens,
        ttft_ms=round(ttft_ms, 2),
        tts_ready_ms=round(tts_ready_ms, 2),
        tts_ready_chars=tts_ready_chars,
        total_time_ms=round(total_ms, 2),
    )


def check_vllm_ready() -> bool:
    try:
        req = urllib.request.Request(f"{VLLM_HOST}/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def check_ollama_ready() -> bool:
    try:
        req = urllib.request.Request(f"{OLLAMA_HOST}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def warmup_ollama(model: str):
    stream_ollama(model, "Hello", num_predict=8)


def warmup_vllm():
    stream_vllm_openai("Hello", num_predict=8)


def run_cold_warm_benchmark(
    backend: str,
    model: str,
    runs: int = 3,
) -> list[StreamMetrics]:
    results = []
    for run in range(1, runs + 1):
        # Cold: unique prefix each run (no cache benefit)
        cold_prefix = SHARED_PREFIX + f"[session-cold-{run}]\n"
        if backend == "ollama":
            m = stream_ollama(model, USER_PROMPTS[0], prefix=cold_prefix)
        else:
            m = stream_vllm_openai(USER_PROMPTS[0], prefix=cold_prefix)
        m.scenario = "cold_ttft"
        m.run = run
        results.append(m)
        print(
            f"  {backend} cold run {run}: TTFT={m.ttft_ms}ms, "
            f"TTS-ready={m.tts_ready_ms}ms, err={m.error}"
        )

        # Warm: same prefix, different user query (prefix cache hit expected for vLLM)
        warm_prefix = SHARED_PREFIX + "[session-warm]\n"
        if backend == "ollama":
            m2 = stream_ollama(model, USER_PROMPTS[1], prefix=warm_prefix)
        else:
            m2 = stream_vllm_openai(USER_PROMPTS[1], prefix=warm_prefix)
        m2.scenario = "warm_prefix_1"
        m2.run = run
        m2.prefix_cached = True
        results.append(m2)

        if backend == "vllm":
            m3 = stream_vllm_openai(USER_PROMPTS[2], prefix=warm_prefix)
        else:
            m3 = stream_ollama(model, USER_PROMPTS[2], prefix=warm_prefix)
        m3.scenario = "warm_prefix_2"
        m3.run = run
        m3.prefix_cached = True
        results.append(m3)
        print(
            f"  {backend} warm run {run}: TTFT={m3.ttft_ms}ms, "
            f"TTS-ready={m3.tts_ready_ms}ms"
        )

    return results


async def run_concurrent_ollama(n_requests: int = 5) -> list[StreamMetrics]:
    import concurrent.futures

    warm_prefix = SHARED_PREFIX + "[concurrent-ollama]\n"
    stream_ollama(OLLAMA_MODEL, USER_PROMPTS[0], prefix=warm_prefix)

    results: list[StreamMetrics] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_requests) as pool:
        futures = [
            pool.submit(
                stream_ollama,
                OLLAMA_MODEL,
                USER_PROMPTS[i % len(USER_PROMPTS)],
                warm_prefix,
            )
            for i in range(n_requests)
        ]
        for i, fut in enumerate(concurrent.futures.as_completed(futures)):
            m = fut.result()
            m.scenario = "concurrent_inflight"
            m.run = i + 1
            m.prefix_cached = True
            results.append(m)
            print(
                f"  ollama concurrent: TTFT={m.ttft_ms}ms, "
                f"TTS-ready={m.tts_ready_ms}ms, err={m.error}"
            )
    return results


async def run_concurrent_vllm(n_requests: int = 5) -> list[StreamMetrics]:
    import httpx

    warm_prefix = SHARED_PREFIX + "[concurrent-session]\n"
    # Prime prefix cache with one request
    stream_vllm_openai(USER_PROMPTS[0], prefix=warm_prefix)

    async with httpx.AsyncClient() as client:
        tasks = []
        for i in range(n_requests):
            prompt = USER_PROMPTS[i % len(USER_PROMPTS)]
            tasks.append(
                stream_vllm_async(prompt, prefix=warm_prefix, client=client)
            )
        raw = await asyncio.gather(*tasks)
    results = []
    for i, m in enumerate(raw):
        m.scenario = "concurrent_inflight"
        m.run = i + 1
        m.prefix_cached = True
        results.append(m)
        print(
            f"  vllm concurrent #{i+1}: TTFT={m.ttft_ms}ms, "
            f"TTS-ready={m.tts_ready_ms}ms, err={m.error}"
        )
    return results


def summarize(results: list[StreamMetrics]) -> dict:
    summary: dict = {}
    for backend in sorted(set(r.backend for r in results)):
        summary[backend] = {}
        for scenario in sorted(set(r.scenario for r in results if r.backend == backend)):
            subset = [
                r for r in results
                if r.backend == backend and r.scenario == scenario and not r.error
            ]
            if not subset:
                continue
            summary[backend][scenario] = {
                "count": len(subset),
                "ttft_ms_avg": round(statistics.mean(r.ttft_ms for r in subset), 2),
                "ttft_ms_min": round(min(r.ttft_ms for r in subset), 2),
                "ttft_ms_max": round(max(r.ttft_ms for r in subset), 2),
                "tts_ready_ms_avg": round(
                    statistics.mean(r.tts_ready_ms for r in subset), 2
                ),
                "tts_ready_ms_min": round(min(r.tts_ready_ms for r in subset), 2),
                "tokens_per_sec_avg": round(
                    statistics.mean(r.tokens_per_sec for r in subset), 2
                ),
                "completion_tokens_avg": round(
                    statistics.mean(r.completion_tokens for r in subset), 1
                ),
            }
    return summary


def main():
    global VLLM_HOST, VLLM_MODEL
    parser = argparse.ArgumentParser(description="vLLM resumable/prefix cache benchmark")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--skip-ollama", action="store_true")
    parser.add_argument("--skip-vllm", action="store_true")
    parser.add_argument("--vllm-host", default=VLLM_HOST)
    parser.add_argument("--vllm-model", default=VLLM_MODEL_GEMMA4)
    parser.add_argument("--output", default="/home/amreschizo/gemma test/results/vllm_resumable_results.json")
    args = parser.parse_args()
    VLLM_HOST = args.vllm_host
    VLLM_MODEL = args.vllm_model

    print("vLLM Resumable / Prefix Cache Streaming Benchmark")
    print(f"Ollama: {OLLAMA_HOST} model={OLLAMA_MODEL}")
    print(f"vLLM:   {VLLM_HOST} model={VLLM_MODEL}")
    print(f"TTS chunk simulation: {TTS_CHUNK_CHARS} chars or first sentence end")

    all_results: list[StreamMetrics] = []
    meta = {
        "vllm_ready": check_vllm_ready(),
        "ollama_ready": check_ollama_ready(),
        "vllm_version": None,
        "notes": [],
    }

    if not args.skip_ollama and meta["ollama_ready"]:
        print("\n=== Ollama baseline ===")
        warmup_ollama(OLLAMA_MODEL)
        all_results.extend(run_cold_warm_benchmark("ollama", OLLAMA_MODEL, args.runs))
        print("\n=== Ollama concurrent in-flight (shared prefix) ===")
        all_results.extend(asyncio.run(run_concurrent_ollama(5)))
    elif not args.skip_ollama:
        meta["notes"].append("Ollama not available")

    if not args.skip_vllm:
        if not meta["vllm_ready"]:
            print("Waiting for vLLM server...")
            for _ in range(120):
                if check_vllm_ready():
                    meta["vllm_ready"] = True
                    break
                time.sleep(5)
        if meta["vllm_ready"]:
            print("\n=== vLLM prefix caching ===")
            warmup_vllm()
            all_results.extend(run_cold_warm_benchmark("vllm", VLLM_MODEL, args.runs))
            print("\n=== vLLM concurrent in-flight (shared prefix) ===")
            concurrent = asyncio.run(run_concurrent_vllm(5))
            all_results.extend(concurrent)
        else:
            meta["notes"].append("vLLM server not ready after 10 min")

    summary = summarize(all_results)
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ollama_host": OLLAMA_HOST,
        "ollama_model": OLLAMA_MODEL,
        "vllm_host": VLLM_HOST,
        "vllm_model": VLLM_MODEL,
        "tts_chunk_chars": TTS_CHUNK_CHARS,
        "meta": meta,
        "raw_results": [asdict(r) for r in all_results],
        "summary": summary,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    return 0 if all_results else 1


if __name__ == "__main__":
    sys.exit(main())
