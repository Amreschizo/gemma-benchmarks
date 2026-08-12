#!/usr/bin/env python3
"""Speed benchmark for Gemma 4 E2B vs E4B via Ollama streaming API."""

import json
import statistics
import sys
import time
import urllib.request
from dataclasses import dataclass, asdict

OLLAMA_HOST = "http://127.0.0.1:11435"
MODELS = ["gemma4:e2b", "gemma4:e4b"]

PROMPTS = [
    {
        "name": "short_generation",
        "prompt": "Explain how a binary search tree works in 3 paragraphs.",
        "max_tokens": 256,
        "num_predict": 256,
    },
    {
        "name": "medium_generation",
        "prompt": (
            "Write a Python function that implements merge sort with detailed "
            "comments explaining each step. Include time complexity analysis."
        ),
        "max_tokens": 512,
        "num_predict": 512,
    },
    {
        "name": "reasoning",
        "prompt": (
            "A farmer has 17 sheep. All but 9 die. How many sheep are left? "
            "Think step by step and explain your reasoning."
        ),
        "max_tokens": 256,
        "num_predict": 256,
    },
]


@dataclass
class SpeedResult:
    model: str
    prompt_name: str
    run: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    ttft_ms: float
    total_time_ms: float
    tokens_per_sec: float
    load_duration_ms: float
    prompt_eval_ms: float
    eval_ms: float


def stream_generate(model: str, prompt: str, num_predict: int) -> SpeedResult:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "think": False,
        "options": {
            "num_predict": num_predict,
            "temperature": 0.0,
        },
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    start = time.perf_counter()
    first_token_time = None
    completion_tokens = 0
    final_stats = {}

    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw_line in resp:
            line = raw_line.decode().strip()
            if not line:
                continue
            chunk = json.loads(line)
            msg = chunk.get("message", {})
            if msg.get("content"):
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                completion_tokens += 1
            if chunk.get("done"):
                final_stats = chunk
                break

    end = time.perf_counter()
    ttft_ms = (first_token_time - start) * 1000 if first_token_time else 0
    total_ms = (end - start) * 1000
    eval_duration_ns = final_stats.get("eval_duration", 0)
    eval_ms = eval_duration_ns / 1e6 if eval_duration_ns else 0
    tokens_per_sec = (
        completion_tokens / (eval_duration_ns / 1e9)
        if eval_duration_ns and completion_tokens
        else (completion_tokens / (total_ms / 1000) if total_ms else 0)
    )

    return SpeedResult(
        model=model,
        prompt_name="",
        run=0,
        prompt_tokens=final_stats.get("prompt_eval_count", 0),
        completion_tokens=completion_tokens,
        total_tokens=final_stats.get("prompt_eval_count", 0) + completion_tokens,
        ttft_ms=round(ttft_ms, 2),
        total_time_ms=round(total_ms, 2),
        tokens_per_sec=round(tokens_per_sec, 2),
        load_duration_ms=round(final_stats.get("load_duration", 0) / 1e6, 2),
        prompt_eval_ms=round(final_stats.get("prompt_eval_duration", 0) / 1e6, 2),
        eval_ms=round(eval_ms, 2),
    )


def warmup(model: str):
    stream_generate(model, "Hello", 16)


def get_available_models() -> list[str]:
    try:
        req = urllib.request.Request(f"{OLLAMA_HOST}/api/tags")
        with urllib.request.urlopen(req, timeout=10) as resp:
            tags = json.loads(resp.read())
        names = {t["name"] for t in tags.get("models", [])}
        return [m for m in MODELS if m in names]
    except Exception:
        return MODELS[:2]  # fallback to official Q4 models


def run_benchmark(runs_per_prompt: int = 3) -> list[SpeedResult]:
    models = get_available_models()
    if not models:
        print("No models available!")
        return []
    print(f"Testing models: {', '.join(models)}")
    results = []
    for model in models:
        print(f"\n=== Warming up {model} ===")
        warmup(model)
        for prompt_cfg in PROMPTS:
            print(f"  Benchmarking: {prompt_cfg['name']}")
            for run in range(1, runs_per_prompt + 1):
                result = stream_generate(
                    model, prompt_cfg["prompt"], prompt_cfg["num_predict"]
                )
                result.prompt_name = prompt_cfg["name"]
                result.run = run
                results.append(result)
                print(
                    f"    run {run}: {result.completion_tokens} tok, "
                    f"{result.tokens_per_sec} tok/s, TTFT={result.ttft_ms}ms"
                )
    return results


def summarize(results: list[SpeedResult]) -> dict:
    summary = {}
    tested_models = sorted(set(r.model for r in results))
    for model in tested_models:
        summary[model] = {}
        for prompt_cfg in PROMPTS:
            key = prompt_cfg["name"]
            subset = [
                r for r in results
                if r.model == model and r.prompt_name == key
            ]
            if not subset:
                continue
            summary[model][key] = {
                "ttft_ms_avg": round(statistics.mean(r.ttft_ms for r in subset), 2),
                "ttft_ms_min": round(min(r.ttft_ms for r in subset), 2),
                "total_time_ms_avg": round(
                    statistics.mean(r.total_time_ms for r in subset), 2
                ),
                "tokens_per_sec_avg": round(
                    statistics.mean(r.tokens_per_sec for r in subset), 2
                ),
                "tokens_per_sec_max": round(
                    max(r.tokens_per_sec for r in subset), 2
                ),
                "completion_tokens_avg": round(
                    statistics.mean(r.completion_tokens for r in subset), 1
                ),
                "prompt_eval_ms_avg": round(
                    statistics.mean(r.prompt_eval_ms for r in subset), 2
                ),
                "eval_ms_avg": round(statistics.mean(r.eval_ms for r in subset), 2),
            }
    return summary


def main():
    print("Gemma 4 Speed Benchmark")
    print(f"Ollama host: {OLLAMA_HOST}")
    print(f"Models: {', '.join(MODELS)}")
    results = run_benchmark(runs_per_prompt=3)
    summary = summarize(results)

    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ollama_host": OLLAMA_HOST,
        "models": MODELS,
        "raw_results": [asdict(r) for r in results],
        "summary": summary,
    }
    out_path = "/home/amreschizo/gemma test/results/speed_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
