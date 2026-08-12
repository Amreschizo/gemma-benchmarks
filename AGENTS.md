# AGENTS.md — navigation for automated assistants

You are likely continuing MareBot / Gemma 4 inference work. Read this file first, then drill into linked reports. All paths are relative to the repository root.

## Mission

This repo documents **empirical benchmarks** for Gemma 4 on one RTX 3090 host:

- **Edge inference (Ollama):** compare `gemma4:e2b` vs `gemma4:e4b` speed and long-document recall.
- **Character RP (Ollama):** MLP FiM lore adherence without Equestria Girls contamination.
- **Voice stack (vLLM):** prefix-cached streaming TTFT, concurrent sessions, and **mid-generation** context injection via `AsyncLLM` (not HTTP).

## Decision tree

```
Need Ollama numbers or coherency?     → REPORT.md + results/*coherency* + results/speed_results.json
Choosing model for MLP FiM RP?        → MLP_LORE_REPORT.md + results/mlp_lore_results.json
Voice / TTS latency / barge-in?       → VLLM_RESUMABLE_REPORT.md + vllm_* results
vLLM failed to load Gemma 4?        → GEMMA4_VLLM_HANDOFF.md (then verify fixes in VLLM_RESUMABLE_REPORT.md)
Implement or re-run a harness?        → benchmarks/<name>.py (stdlib Ollama vs venv vLLM)
```

## Non-negotiable runtime facts

| Topic | Rule |
|-------|------|
| Ollama API | Set `"think": false` or responses may be empty |
| Ollama E4B | Single GPU (`CUDA_VISIBLE_DEVICES=1`) |
| Ollama host | Default in scripts: `http://127.0.0.1:11435` |
| vLLM serve | `--trust-remote-code`, log must show `Gemma4ForConditionalGeneration` |
| transformers | Pin **5.14.1** with vLLM **0.27.1** |
| Paths with spaces | Set `VLLM_USE_FLASHINFER_SAMPLER=0`; prefer `~/projects/gemma-test` symlink |
| Midgen injection | Python `AsyncLLM` only; stop HTTP server before midgen test |
| Secrets | Never commit `.env`, HF tokens, or `ollama-models/` |

## Commands (copy-paste)

**Ollama all-in-one:**

```bash
./run_benchmarks.sh
```

**MLP lore:**

```bash
python3 benchmarks/mlp_lore_test.py
```

**vLLM serve + HTTP benchmark:**

```bash
CUDA_VISIBLE_DEVICES=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
  .venv/bin/vllm serve google/gemma-4-E4B-it --trust-remote-code \
  --gpu-memory-utilization 0.90 --max-model-len 32768 \
  --enable-prefix-caching --port 8001

.venv/bin/python benchmarks/vllm_resumable_stream_test.py \
  --vllm-host http://127.0.0.1:8001 \
  --vllm-model google/gemma-4-E4B-it --runs 3
```

**Midgen (exclusive GPU):**

```bash
.venv/bin/python benchmarks/vllm_midgen_injection_test.py
```

## Artifacts map

| Path | Type |
|------|------|
| `REPORT.md` | Human summary: Ollama |
| `MLP_LORE_REPORT.md` | Human summary: MLP |
| `VLLM_RESUMABLE_REPORT.md` | Human summary: vLLM |
| `GEMMA4_VLLM_HANDOFF.md` | Runbook / diagnosis |
| `results/*.json` | Machine-readable benchmark output (source of truth for numbers) |
| `results/*.log` | Serve and run logs for debugging reproduction |
| `benchmarks/*.py` | Executable definitions of metrics and pass criteria |

## Extending benchmarks

- **Ollama scripts** use `urllib` only; edit `OLLAMA_HOST` / model lists at top of each file.
- **vLLM resumable script** accepts CLI flags (`--vllm-host`, `--vllm-model`, `--runs`); read `argparse` block in `vllm_resumable_stream_test.py`.
- **Midgen script** documents deadlock pitfalls (`max_tokens=1` per chunk); read header comments before changing.

## GitHub / MareBot context

- Published as **`Amreschizo/gemma-benchmarks`** (MareBot benchmark folder).
- GitHub user **`Marebot`** has no repos under current credentials; MareBot robot code may live in private **`Amreschizo/MareBot-`**.

## What is intentionally absent from git

- `ollama-models/` (multi-GB blobs)
- Ollama binary tarball under `bin/`
- `.venv/`
- Hugging Face weight caches

Re-download per README setup section before re-running.
