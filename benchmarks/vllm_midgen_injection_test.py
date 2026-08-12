#!/usr/bin/env python3
"""Test vLLM mid-generation context injection via AsyncLLM StreamingInput.

Uses the resumable streaming input API (vLLM 0.27+) to append context
while a generation is in flight, reusing KV cache from prior tokens.

Measures inject→first-token latency on a warm engine (post-warmup).
"""

import argparse
import asyncio
import json
import logging
import os
import statistics
import sys
import time
import uuid

from vllm import SamplingParams
from vllm.engine.protocol import StreamingInput
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.engine.arg_utils import AsyncEngineArgs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [midgen] %(message)s",
    stream=sys.stdout,
    force=True,
)
log = logging.getLogger("midgen")


def _log(msg: str) -> None:
    log.info(msg)
    sys.stdout.flush()


def _summary_stats(values: list[float]) -> dict:
    if not values:
        return {"avg": None, "min": None, "max": None, "count": 0}
    return {
        "avg": round(statistics.mean(values), 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "count": len(values),
    }


async def _warmup_generate(engine: AsyncLLM, model_label: str) -> None:
    """Run a short generation so Triton/attention kernels compile before the test."""
    _log(f"warmup: starting short generation ({model_label})")
    warmup_params = SamplingParams(max_tokens=8, temperature=0.0)
    warmup_id = f"warmup-{uuid.uuid4().hex[:8]}"
    t0 = time.perf_counter()
    async for output in engine.generate("Say hello in one word.", warmup_params, request_id=warmup_id):
        if output.finished:
            break
    _log(f"warmup: finished in {(time.perf_counter() - t0) * 1000:.1f} ms")


async def run_single_injection_iteration(
    engine: AsyncLLM,
    initial_prompt: str,
    injection_text: str,
    iteration: int,
) -> dict:
    """Run one mid-gen injection on an already-warm engine."""
    request_id = f"midgen-{iteration}-{uuid.uuid4().hex[:8]}"

    sampling_params = SamplingParams(
        max_tokens=16,
        temperature=0.0,
        output_kind=RequestOutputKind.DELTA,
    )
    stream_chunk_params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        output_kind=RequestOutputKind.DELTA,
    )

    next_input_queue: asyncio.Queue[int | None] = asyncio.Queue()
    tokens_before_injection: list[str] = []
    tokens_after_injection: list[str] = []
    injected = False
    decode_steps = 0
    api_error: str | None = None

    # Shared timing dict (mutable, written from generator + consumer).
    ts: dict[str, float | None] = {
        "t0": None,
        "t_first_token": None,
        "t_inject_queued": None,
        "t_inject_yielded": None,
        "t_first_post_inject_token": None,
    }

    async def input_generator():
        nonlocal injected
        ts["t0"] = time.perf_counter()
        _log(f"iter {iteration}: yielding initial StreamingInput")
        yield StreamingInput(prompt=initial_prompt, sampling_params=stream_chunk_params)

        step = 0
        while True:
            signal = await next_input_queue.get()
            if signal is None:
                break
            injected = True
            ts["t_inject_yielded"] = time.perf_counter()
            _log(f"iter {iteration}: yielding injection StreamingInput")
            yield StreamingInput(prompt=injection_text, sampling_params=stream_chunk_params)
            step += 1
            if step >= 1:
                break

    full_text = ""
    output_count = 0

    _log(f"iter {iteration}: starting generate loop")
    try:
        async for output in engine.generate(
            input_generator(),
            sampling_params=sampling_params,
            request_id=request_id,
        ):
            output_count += 1
            for completion in output.outputs:
                new_text = completion.text
                if new_text:
                    full_text += new_text
                    if not injected:
                        if ts["t_first_token"] is None:
                            ts["t_first_token"] = time.perf_counter()
                        tokens_before_injection.append(new_text)
                    else:
                        if ts["t_first_post_inject_token"] is None:
                            ts["t_first_post_inject_token"] = time.perf_counter()
                        tokens_after_injection.append(new_text)
            decode_steps += 1
            if not injected and len(tokens_before_injection) >= 1:
                ts["t_inject_queued"] = time.perf_counter()
                _log(f"iter {iteration}: queueing injection signal")
                await next_input_queue.put(0)
            if injected and tokens_after_injection and output.finished:
                break
            if decode_steps >= 20:
                await next_input_queue.put(None)
                break
        if injected:
            await next_input_queue.put(None)
    except Exception as exc:
        api_error = f"{type(exc).__name__}: {exc}"
        _log(f"iter {iteration}: generate failed: {api_error}")

    def _ms(a: str, b: str) -> float | None:
        if ts[a] is None or ts[b] is None:
            return None
        return round((ts[b] - ts[a]) * 1000, 2)

    inject_to_first_token_ms = _ms("t_inject_yielded", "t_first_post_inject_token")
    paused_warm_reaction_ms = _ms("t_inject_queued", "t_first_post_inject_token")
    generation_to_first_token_ms = _ms("t0", "t_first_token")
    inject_queue_to_yield_ms = _ms("t_inject_queued", "t_inject_yielded")

    success = (
        bool(full_text)
        and injected
        and len(tokens_after_injection) > 0
        and inject_to_first_token_ms is not None
    )

    return {
        "iteration": iteration,
        "request_id": request_id,
        "timestamps": {k: ts[k] for k in ts},
        "timestamps_ms_from_t0": {
            k: _ms("t0", k) for k in ts if k != "t0"
        },
        "inject_to_first_token_ms": inject_to_first_token_ms,
        "paused_warm_reaction_ms": paused_warm_reaction_ms,
        "generation_to_first_token_ms": generation_to_first_token_ms,
        "inject_queue_to_yield_ms": inject_queue_to_yield_ms,
        "tokens_before_injection": len(tokens_before_injection),
        "tokens_after_injection": len(tokens_after_injection),
        "output_chunks": output_count,
        "decode_steps": decode_steps,
        "full_output": full_text,
        "success": success,
        "api_error": api_error,
    }


async def run_midgen_injection_benchmark(
    model: str,
    gpu_id: str,
    iterations: int,
) -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

    _log(f"creating AsyncLLM model={model} gpu={gpu_id} enforce_eager=True")
    engine_args = AsyncEngineArgs(
        model=model,
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        enable_prefix_caching=True,
        enforce_eager=True,
    )
    t_engine = time.perf_counter()
    engine = AsyncLLM.from_engine_args(engine_args)
    _log(f"engine ready in {(time.perf_counter() - t_engine) * 1000:.1f} ms")

    await _warmup_generate(engine, model)

    initial_prompt = (
        "You are a helpful assistant. Continue the story naturally.\n"
        "Story: Once upon a time, a traveler entered a dark forest."
    )
    injection_text = " Suddenly, they heard a loud roar behind them."

    runs: list[dict] = []
    try:
        for i in range(1, iterations + 1):
            run = await run_single_injection_iteration(
                engine, initial_prompt, injection_text, i
            )
            runs.append(run)
            _log(
                f"iter {i}: inject_to_first_token_ms={run['inject_to_first_token_ms']} "
                f"paused_warm_reaction_ms={run['paused_warm_reaction_ms']}"
            )
    finally:
        _log("shutting down engine")
        engine.shutdown()

    successful = [r for r in runs if r["success"]]
    inject_latencies = [r["inject_to_first_token_ms"] for r in successful if r["inject_to_first_token_ms"] is not None]
    reaction_latencies = [r["paused_warm_reaction_ms"] for r in successful if r["paused_warm_reaction_ms"] is not None]
    first_token_latencies = [r["generation_to_first_token_ms"] for r in successful if r["generation_to_first_token_ms"] is not None]

    return {
        "test": "mid_generation_context_injection",
        "model": model,
        "api": "AsyncLLM.generate(StreamingInput async generator)",
        "vllm_version": "0.27.1",
        "enforce_eager": True,
        "attention_backend": "TRITON_ATTN (Gemma4 on Ampere)",
        "gpu": gpu_id,
        "iterations_requested": iterations,
        "iterations_successful": len(successful),
        "initial_prompt": initial_prompt,
        "injection_text": injection_text,
        "runs": runs,
        "summary": {
            "inject_to_first_token_ms": _summary_stats(inject_latencies),
            "paused_warm_reaction_ms": _summary_stats(reaction_latencies),
            "generation_to_first_token_ms": _summary_stats(first_token_latencies),
        },
        "http_api_available": False,
        "notes": [
            "Mid-generation injection requires AsyncLLM Python API; not exposed via OpenAI HTTP endpoint.",
            "StreamingInput with resumable async generator appends within a session (vLLM 0.27+).",
            "enforce_eager=True plus warmup avoids CUDA-graph capture; Gemma 4 still uses TRITON_ATTN on Ampere.",
            "inject_to_first_token_ms = t_first_post_inject_token - t_inject_yielded (primary metric).",
            "paused_warm_reaction_ms = t_first_post_inject_token - t_inject_queued (includes queue/yield overhead).",
            "Paused warm: engine is loaded, kernels JIT'd, and a generation is in-flight (WAITING_FOR_STREAMING_REQ) when new context is appended.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-4-E4B-it")
    parser.add_argument("--gpu", default="1", help="GPU for AsyncLLM test engine")
    parser.add_argument("--iterations", type=int, default=8, help="Warm-engine iterations")
    parser.add_argument(
        "--output",
        default="/home/amreschizo/projects/gemma-test/results/vllm_midgen_injection.json",
    )
    args = parser.parse_args()

    _log(f"midgen injection benchmark starting model={args.model} iterations={args.iterations}")
    result = asyncio.run(
        run_midgen_injection_benchmark(args.model, args.gpu, args.iterations)
    )
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result["summary"], indent=2))
    return 0 if result["iterations_successful"] == result["iterations_requested"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
