# vLLM Resumable / Prefix-Cache Streaming Benchmark

**Date:** 2026-08-11 (updated)  
**Harness:** `benchmarks/vllm_resumable_stream_test.py`  
**Raw results:** `results/vllm_gemma4_resumable_results.json`  
**Mid-gen injection:** `results/vllm_midgen_injection.json`

This benchmark measures **time to first token (TTFT)**, **simulated TTS-ready latency** (first 20 characters or sentence boundary), **warm-prefix behavior**, and **five concurrent in-flight streams** sharing the same conversation prefix—patterns typical of a voice assistant (LLM → TTS) pipeline.

---

## Executive summary

| Backend | Model | Status |
|---------|--------|--------|
| **Ollama** | `batiai/gemma4-e4b:q6` | Works (~507ms TTFT, ~104 tok/s single-stream) |
| **vLLM 0.27.1** | `google/gemma-4-E4B-it` (BF16) | **Works** — native `Gemma4ForConditionalGeneration` |
| **vLLM 0.27.1** | `Qwen/Qwen2.5-3B-Instruct` | Previous APC proxy only (not used in this run) |

**Gemma 4 E4B on vLLM with prefix caching is now working.** Two blockers from the prior report were resolved:

1. **Transformers 5.15.0 incompatibility** — `AmbiguousGlobalPerLayerAttributeError` on heterogeneous `head_dim`. Fixed by pinning `transformers==5.14.1` (upstream fix in vLLM PR #48432 not yet in 0.27.1 release).
2. **FlashInfer ninja path-with-spaces** — venv under `/home/amreschizo/gemma test/` breaks JIT. Fixed with `VLLM_USE_FLASHINFER_SAMPLER=0` (symlink `~/projects/gemma-test` alone is insufficient because venv path still has spaces).

---

## What was fixed

| Step | Action | Result |
|------|--------|--------|
| 1 | Symlink `~/projects/gemma-test` → workspace | Done |
| 2 | Upgrade vLLM 0.27.0 → **0.27.1** | Done |
| 3 | Fresh HF weight pull (`google/gemma-4-E4B-it`) | Done |
| 4 | Pin `transformers==5.14.1` | Required for config load |
| 5 | Serve with `--trust-remote-code` on GPU 1, port 8001 | **Native architecture resolved** |
| 6 | `VLLM_USE_FLASHINFER_SAMPLER=0` | Required for startup on spaced venv path |

### Server verification (from `results/vllm_gemma4_serve.log`)

```
Resolved architecture: Gemma4ForConditionalGeneration
Gemma4 model has heterogeneous head dimensions (head_dim=256, global_head_dim=512). FA4 not available, forcing TRITON_ATTN backend.
Using AttentionBackendEnum.TRITON_ATTN backend.
Prefix cache hit rate: 76.7% (during benchmark)
```

**TRITON_ATTN is forced** on RTX 3090 (Ampere, no FA4). Throughput is **~66 tok/s** — not the ~9 tok/s catastrophic regression seen in early vLLM 0.19 reports, and within acceptable range vs Ollama's ~104 tok/s.

---

## Results: Gemma 4 E4B vLLM vs Ollama Q6 baseline

TTS simulation: **20 characters** or first `.!?` boundary. Three cold/warm cycles + five concurrent requests.

### TTFT (ms, average)

| Backend | Cold TTFT | Warm prefix (2nd query) | 5× concurrent TTFT |
|---------|-----------|-------------------------|---------------------|
| Ollama Gemma4 Q6 (prior) | **507** | **504** | **2350** |
| **vLLM Gemma4 E4B BF16** | **29** | **25** | **57** |

vLLM achieves **~17× lower cold TTFT** and **~41× lower concurrent TTFT** than Ollama on the same prompt shape.

### TTS-ready latency (ms, average)

| Scenario | Ollama Q6 (prior) | vLLM Gemma4 E4B |
|----------|-------------------|-----------------|
| Cold | **530** | **89** |
| Warm prefix | **547** | **82–160** |
| 5× concurrent | **2413** | **112** |

### Throughput (decode, single stream)

| Backend | ~tokens/s |
|---------|-----------|
| Ollama Q6 (prior) | **~104** |
| **vLLM Gemma4 E4B** | **~66** |

Ollama remains faster per-stream (~1.6×), but vLLM wins decisively on **latency under concurrency** and **prefix-cached warm starts**.

### Prefix caching

APC confirmed active. Server logs during benchmark showed prefix cache hit rate climbing to **76.7%** on shared-prefix warm/concurrent scenarios.

---

## Mid-generation context injection

**API:** vLLM 0.27.1 `AsyncLLM.generate()` with an `AsyncGenerator[StreamingInput, None]`. Each `StreamingInput` chunk is processed with `resumable=True`, allowing the scheduler to pause decode (`WAITING_FOR_STREAMING_REQ`) and append new prompt tokens to the existing KV cache within the same session.

**HTTP limitation:** The OpenAI-compatible `/v1/completions` endpoint on port 8001 does **not** expose streaming input. Mid-generation injection requires embedding `AsyncLLM` directly in the voice pipeline (not a separate HTTP round-trip).

**Test status:** `benchmarks/vllm_midgen_injection_test.py` — **8/8 successful** on warm engine (2026-08-11). See `results/vllm_midgen_injection.json`.

| Metric | avg | min | max |
|--------|-----|-----|-----|
| **inject_to_first_token_ms** (primary) | **33.2** | **32.5** | **34.0** |
| paused_warm_reaction_ms | 33.2 | 32.6 | 34.0 |
| generation_to_first_token_ms | 32.8 | 32.4 | 33.3 |

**What "paused warm" means:** The AsyncLLM engine is loaded, TRITON_ATTN kernels are JIT'd (post-warmup), and a generation is in-flight with 1 decode token emitted. The scheduler enters `WAITING_FOR_STREAMING_REQ`; the client yields a new `StreamingInput` chunk with appended context. The engine prefills the injected tokens into the existing KV cache and resumes decode — no new HTTP request, no cold model load.

**vs HTTP warm TTFT (~25 ms):** HTTP warm TTFT measures time from a *new* `/v1/completions` request with a prefix-cached prompt to the first token of a fresh generation. Inject latency measures time from yielding new context into an *already-running* resumable session to the first post-inject token. Both are ~30 ms on this stack; inject is ~1.3× slower because it includes prefill of the injected sentence (~12 tokens) into the live KV cache before decode resumes, whereas warm HTTP TTFT benefits from APC on the static prefix only.

**Voice pipeline implication:** For barge-in / STT-append-while-generating, wire the voice server to `AsyncLLM` with a `StreamingInput` async generator rather than the HTTP completions API. Expect ~33 ms from context append to first new token on Gemma 4 E4B / RTX 3090.

---

## Serve command (production)

```bash
cd ~/projects/gemma-test
CUDA_VISIBLE_DEVICES=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
  .venv/bin/vllm serve google/gemma-4-E4B-it \
  --trust-remote-code \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --enable-prefix-caching \
  --port 8001
```

**Dependencies pinned for this stack:**
- `vllm==0.27.1`
- `transformers==5.14.1` (do not upgrade to 5.15.0 until vLLM ships PR #48432)

**Note:** Stopping `llama-server` was required to free GPU 1 VRAM (~18 GiB occupied). Restart Ollama separately if needed for Q6 baseline comparisons.

---

## LLM → TTS pipeline recommendations

1. **Use vLLM Gemma 4 E4B for multi-session / concurrent voice** — concurrent TTFT ~57ms vs Ollama ~2350ms.
2. **Keep Ollama for single-stream max throughput** if ~66 vs ~104 tok/s matters and concurrency is not a concern.
3. **Mid-generation context injection** — use `AsyncLLM` + `StreamingInput` Python API; not available over HTTP yet.
4. **Environment** — pin `transformers==5.14.1`; set `VLLM_USE_FLASHINFER_SAMPLER=0` until project moves to a space-free venv path.
5. **TRITON_ATTN on Ampere** — acceptable (~66 tok/s); FA4 on Hopper would improve further.

---

## Files

| File | Description |
|------|-------------|
| `VLLM_RESUMABLE_REPORT.md` | This report |
| `GEMMA4_VLLM_HANDOFF.md` | Handoff diagnosis + fix plan |
| `results/vllm_gemma4_resumable_results.json` | Gemma 4 benchmark raw + summary |
| `results/vllm_gemma4_serve.log` | Server startup + APC hit-rate logs |
| `results/vllm_midgen_injection.json` | Mid-gen injection API test status |
| `benchmarks/vllm_midgen_injection_test.py` | AsyncLLM StreamingInput injection harness |
| `results/vllm_resumable_results.json` | Prior Qwen proxy benchmark (2026-08-10) |

---

## Update (2026-08-11): Gemma 4 E4B on vLLM 0.27.1

**Status:** `google/gemma-4-E4B-it` now loads and serves on this stack (native `Gemma4ForConditionalGeneration`). Earlier 512 vs 256 weight assertion was resolved in vLLM **0.27.1** / current HF checkpoint pairing.

### Resumable streaming benchmark (HTTP)

Harness: `benchmarks/vllm_resumable_stream_test.py` against `http://127.0.0.1:8001`  
Raw: `results/vllm_gemma4_resumable_results.json` (2026-08-11)

| Metric | Gemma 4 E4B-it (vLLM) |
|--------|------------------------|
| Cold TTFT (avg) | **~29 ms** (max ~40 ms) |
| Warm prefix TTFT | **~23 ms** |
| Decode throughput | **~66 tok/s** |
| Prefix cache (serve log) | **~77%** hit rate with APC enabled |
| 5× concurrent TTFT | **~57 ms** avg |

Serve flags used: `--enable-prefix-caching`, `VLLM_USE_FLASHINFER_SAMPLER=0`, GPU 1.

### Mid-generation context injection (AsyncLLM)

Harness: `benchmarks/vllm_midgen_injection_test.py`  
Raw: `results/vllm_midgen_injection.json`

**API exists in vLLM 0.27.1** — not a missing-feature blocker:

- `from vllm.engine.protocol import StreamingInput`
- `AsyncLLM.generate(async_generator[StreamingInput], ...)` with `resumable=True` on each chunk (set inside `_add_streaming_input_request`).

**Not available over OpenAI HTTP** (`/v1/completions`); voice pipelines must embed `AsyncLLM` in-process for barge-in / context append.

**Test result (2026-08-11):** `8/8` iterations successful on warm engine. Primary metric **inject_to_first_token_ms: avg 33.2 ms, min 32.5 ms, max 34.0 ms** (see `results/vllm_midgen_injection.json`). Each run: 1 token before injection, 1 after; injected sentence appears mid-stream in combined output.

**Hang fix (harness):** Original run appeared stuck on first-request Triton `kernel_unified_attention` JIT and on a **deadlock** (`max_tokens=1` per `StreamingInput` chunk waits for the next chunk while the client waited for ≥2 tokens). Fixed with `enforce_eager=True`, explicit warmup generation, progress logging, inject after ≥1 token, and sync `engine.shutdown()`.

**Harness update:** Benchmark now timestamps `t0`, `t_first_token`, `t_inject_queued`, `t_inject_yielded`, `t_first_post_inject_token` per iteration and reports avg/min/max over N warm-engine runs.

**Operational note:** AsyncLLM midgen test needs exclusive GPU 1 (~16 GiB weights); stop `vllm serve` on the same GPU before running, or use another GPU.

