# Gemma 4 Benchmark Suite (MareBot)

Benchmarks and reports for **Google Gemma 4 Effective 2B / 4B** on a MareBot development machine: Ollama throughput and long-context coherency, **My Little Pony: FiM** in-weight lore roleplay, and **vLLM 0.27.1** resumable streaming / prefix caching / mid-generation context injection for voice pipelines (LLM → TTS).

**GitHub:** This repo lives at [`Amreschizo/gemma-benchmarks`](https://github.com/Amreschizo/gemma-benchmarks) as the dedicated **MareBot** benchmark folder. The [`Marebot`](https://github.com/Marebot) GitHub user exists but has no public repos; the authenticated account (`Amreschizo`) cannot create repositories on that user—use this repo or add this tree as a subdirectory in private [`Amreschizo/MareBot-`](https://github.com/Amreschizo/MareBot-) if you prefer a monorepo layout.

**Original workspace path:** `/home/amreschizo/gemma test` (space in path breaks FlashInfer JIT; use symlink `~/projects/gemma-test` for vLLM work).

---

## Quick orientation for AI agents

1. Read **`AGENTS.md`** for navigation, commands, and artifact map.
2. **Primary narrative reports:** [`REPORT.md`](REPORT.md) (Ollama E2B/E4B), [`MLP_LORE_REPORT.md`](MLP_LORE_REPORT.md), [`VLLM_RESUMABLE_REPORT.md`](VLLM_RESUMABLE_REPORT.md), [`GEMMA4_VLLM_HANDOFF.md`](GEMMA4_VLLM_HANDOFF.md) (diagnosis + fix checklist).
3. **Raw numbers:** `results/*.json` (committed). **Server logs:** `results/*.log` (committed).
4. **Do not expect** `ollama-models/` or Ollama binaries in git—they are gitignored (~25 GB + ~5.6 GB locally).

---

## Hardware and software baseline

| Item | Value |
|------|--------|
| GPU | NVIDIA GeForce **RTX 3090** (24 GB VRAM), typically **`CUDA_VISIBLE_DEVICES=1`** |
| CPU | AMD Ryzen 9 5900X (12c/24t) |
| RAM | 125 GB |
| Ollama | **v0.32.7** (user install under `bin/`; system Ollama too old for Gemma 4) |
| Ollama host | `127.0.0.1:11435`, models dir `./ollama-models` |
| vLLM | **0.27.1**, model `google/gemma-4-E4B-it`, port **8001** |
| Transformers | **5.14.1** only (5.15.0 breaks Gemma 4 config load until vLLM ships fix) |
| Env flag | `VLLM_USE_FLASHINFER_SAMPLER=0` if venv path contains spaces |

### Critical Ollama behavior

- Gemma 4 defaults to **thinking mode**; completion/chat with empty output unless **`think: false`**.
- **E4B** can crash on multi-GPU split; use single GPU.
- Quantization tested: **Q4_K_M** (official `gemma4:e2b` / `gemma4:e4b`); BatiAI **Q6** variants also tested for MLP/speed.

---

## Key findings (with numbers)

### Ollama speed (Q4, `think: false`) — see [REPORT.md](REPORT.md)

| Metric | E2B (`gemma4:e2b`) | E4B (`gemma4:e4b`) |
|--------|-------------------|-------------------|
| Generation tok/s (typical) | **~182** | **~121** |
| E2B advantage | **~+50%** throughput | — |

### Ollama long-context coherency — [REPORT.md](REPORT.md)

| Suite | E2B | E4B |
|-------|-----|-----|
| Standard (16 tests) | **16/16** | **16/16** |
| Extended (28 tests, up to ~21K prompt tokens) | **28/28** | **28/28** |

128K window not fully stress-tested; tests topped out ~21K prompt tokens.

### MLP FiM lore roleplay — [MLP_LORE_REPORT.md](MLP_LORE_REPORT.md)

| Model | Pass rate* | EG contamination |
|-------|------------|------------------|
| `gemma4:e4b` Q4 | **92.9%** | **0** |
| `batiai/gemma4-e4b:q6` | **92.9%** | **0** |
| `gemma4:e2b` Q4 | 64.3% | 0 |
| `batiai/gemma4-e2b:q6` | 57.1% | 0 |

\*Pass rate = (pass + 0.5×partial) / 7 characters. System prompt forbids Equestria Girls; all runs complied.

### vLLM resumable / voice latency — [VLLM_RESUMABLE_REPORT.md](VLLM_RESUMABLE_REPORT.md)

| Metric | Ollama Q6 E4B (prior) | vLLM Gemma4 E4B BF16 |
|--------|----------------------|----------------------|
| Cold TTFT | ~507 ms | **~29 ms** |
| Warm prefix TTFT | ~504 ms | **~25 ms** |
| 5× concurrent TTFT | ~2350 ms | **~57 ms** |
| Single-stream decode | ~104 tok/s | **~66 tok/s** |
| Prefix cache hit rate | — | **~77%** (APC) |

**Mid-generation injection** (`AsyncLLM` + `StreamingInput`, not HTTP): **8/8** success; **inject_to_first_token_ms avg 33.2 ms** — see `results/vllm_midgen_injection.json`.

---

## Repository layout

```
.
├── README.md                 # This file
├── AGENTS.md                 # AI agent navigation
├── REPORT.md                 # Ollama E2B/E4B speed + coherency
├── MLP_LORE_REPORT.md        # FiM roleplay benchmark
├── VLLM_RESUMABLE_REPORT.md  # vLLM streaming, APC, midgen injection
├── GEMMA4_VLLM_HANDOFF.md    # Load-failure diagnosis + fix steps
├── run_benchmarks.sh         # Ollama: speed + coherency suites
├── requirements.txt          # vLLM stack pins
├── benchmarks/               # Python harnesses
│   ├── speed_test.py
│   ├── coherency_test.py
│   ├── extended_coherency_test.py
│   ├── mlp_lore_test.py
│   ├── vllm_resumable_stream_test.py
│   └── vllm_midgen_injection_test.py
└── results/                  # JSON outputs + serve/run logs
```

### File index: `benchmarks/`

| Script | Purpose | Default output |
|--------|---------|----------------|
| `speed_test.py` | TTFT, tok/s for E2B/E4B (3 prompts × 3 runs) | `results/speed_results.json` |
| `coherency_test.py` | 16 long-doc recall tests | `results/coherency_results.json` |
| `extended_coherency_test.py` | 28 tests, 4 context sizes | `results/extended_coherency_results.json` |
| `mlp_lore_test.py` | 7 MLP FiM characters, EG guardrails | `results/mlp_lore_results.json` |
| `vllm_resumable_stream_test.py` | TTFT, warm prefix, 5× concurrent, TTS-ready latency | `results/vllm_gemma4_resumable_results.json` (Gemma) or `vllm_resumable_results.json` (Qwen proxy) |
| `vllm_midgen_injection_test.py` | In-process AsyncLLM context injection | `results/vllm_midgen_injection.json` |

### File index: `results/` (committed)

| File | Contents |
|------|----------|
| `speed_results.json` | Ollama speed + appended Q6 speed rows |
| `coherency_results.json` | Standard coherency pass/fail per test |
| `extended_coherency_results.json` | Extended suite by context size |
| `mlp_lore_results.json` | Per-character FiM/EG/trait scoring |
| `vllm_gemma4_resumable_results.json` | Gemma 4 E4B on vLLM HTTP benchmark |
| `vllm_resumable_results.json` | Earlier Qwen2.5 APC proxy run |
| `vllm_midgen_injection.json` | Mid-gen injection timings (8/8) |
| `*.log` | Ollama pull, MLP runs, vLLM serve stdout (debugging reproduction) |

---

## Setup

### Ollama (required for E2B/E4B + MLP tests)

1. Install Ollama **≥0.20** (this project used **0.32.7** tarball in `bin/`—not in git).
2. Set models directory and start server:

```bash
export CUDA_VISIBLE_DEVICES=1
export OLLAMA_HOST=127.0.0.1:11435
export OLLAMA_MODELS="$(pwd)/ollama-models"
./bin/bin/ollama serve   # after extracting/installing binary locally
```

3. Pull models:

```bash
ollama pull gemma4:e2b
ollama pull gemma4:e4b
# Optional for MLP/speed comparisons:
ollama pull batiai/gemma4-e4b:q6
ollama pull batiai/gemma4-e2b:q6
```

### Python vLLM stack (resumable + midgen tests)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Hugging Face: accept Gemma license and login if needed (use env var, do not commit token)
huggingface-cli download google/gemma-4-E4B-it
```

Use a **space-free** path or set `VLLM_USE_FLASHINFER_SAMPLER=0`.

---

## How to reproduce each benchmark

### 1. Ollama core suite (speed + coherency)

```bash
chmod +x run_benchmarks.sh && ./run_benchmarks.sh
```

Or individually:

```bash
python3 benchmarks/speed_test.py
python3 benchmarks/coherency_test.py
python3 benchmarks/extended_coherency_test.py
```

### 2. MLP FiM lore roleplay

Ensure Ollama is up and models are pulled; then:

```bash
python3 benchmarks/mlp_lore_test.py
```

Interpret scores using rules in `MLP_LORE_REPORT.md`.

### 3. vLLM HTTP resumable streaming

Start server (stop other GPU consumers on GPU 1):

```bash
cd ~/projects/gemma-test   # symlink recommended
CUDA_VISIBLE_DEVICES=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
  .venv/bin/vllm serve google/gemma-4-E4B-it \
  --trust-remote-code \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --enable-prefix-caching \
  --port 8001
```

Run harness:

```bash
.venv/bin/python benchmarks/vllm_resumable_stream_test.py \
  --vllm-host http://127.0.0.1:8001 \
  --vllm-model google/gemma-4-E4B-it --runs 3
```

### 4. Mid-generation context injection (in-process only)

**Stop** `vllm serve` on the same GPU; exclusive ~16 GiB for AsyncLLM:

```bash
.venv/bin/python benchmarks/vllm_midgen_injection_test.py
```

OpenAI `/v1/completions` does **not** support this API; embed `AsyncLLM` in the voice server.

---

## Sub-reports and handoff

| Document | When to read |
|----------|----------------|
| [REPORT.md](REPORT.md) | Ollama E2B vs E4B decisions, coherency methodology |
| [MLP_LORE_REPORT.md](MLP_LORE_REPORT.md) | MareBot / character RP model choice |
| [VLLM_RESUMABLE_REPORT.md](VLLM_RESUMABLE_REPORT.md) | Voice pipeline latency, APC, injection |
| [GEMMA4_VLLM_HANDOFF.md](GEMMA4_VLLM_HANDOFF.md) | Historical load failure, verification checklist |

---

## License and models

Model weights are subject to **Google Gemma** terms on Hugging Face and Ollama registries. This repository contains benchmark code, reports, and result JSON only—not weights.
