# Gemma 4 E2B vs E4B Benchmark Report

**Date:** August 10, 2026  
**Workspace:** `/home/amreschizo/gemma test`

---

## Setup

| Component | Details |
|-----------|---------|
| **Models** | `gemma4:e2b` (Gemma 4 Effective 2B) and `gemma4:e4b` (Gemma 4 Effective 4B) |
| **Framework** | Ollama v0.32.7 (user-space install) |
| **Quantization** | Q4_K_M (default Ollama builds) |
| **API Mode** | `/api/generate` completion (non-chat), `think: false` |
| **GPU** | NVIDIA GeForce RTX 3090 (24 GB VRAM), single GPU (`CUDA_VISIBLE_DEVICES=1`) |
| **CPU** | AMD Ryzen 9 5900X (12-core, 24 threads) |
| **RAM** | 125 GB |
| **Context Window** | 128K tokens (model spec); tested up to ~21K prompt tokens |

### Model Identification

- **E2B** = Gemma 4 "Effective 2B" — 2.3B effective parameters (5.1B with per-layer embeddings), 35 layers, edge-optimized
- **E4B** = Gemma 4 "Effective 4B" — 4.5B effective parameters (8.0B with embeddings), 42 layers, edge-optimized
- Both support text, image, and audio modalities; benchmarks used text-only completion

### Infrastructure Notes

The workspace was empty at start. Infrastructure created:
- `bin/` — Ollama v0.32.7 binary (system Ollama v0.14.2 was too old for Gemma 4)
- `ollama-models/` — Downloaded model weights (~7.2 GB E2B, ~9.6 GB E4B)
- `benchmarks/` — Speed and coherency test scripts
- `results/` — JSON result files

**Critical configuration:** Gemma 4 defaults to thinking mode. Without `"think": false`, the completion API returns empty responses (tokens go to hidden reasoning channel). E4B also crashes on multi-GPU split (`GGML_SCHED_MAX_SPLIT_INPUTS`); single-GPU mode required.

---

## Speed Results

3 runs per prompt, temperature=0, thinking disabled. Values are averages across 3 runs.

| Prompt Type | Metric | E2B (`gemma4:e2b`) | E4B (`gemma4:e4b`) | E2B Advantage |
|-------------|--------|---------------------|---------------------|---------------|
| **Short generation** (256 tok) | Tokens/sec | **182.4** | 122.3 | +49% |
| | TTFT (ms) | 1,006 | 1,878 | -46% |
| | Total time (ms) | 2,412 | 3,973 | -39% |
| **Medium generation** (512 tok) | Tokens/sec | **183.9** | 120.8 | +52% |
| | TTFT (ms) | 18,205* | 8,968* | — |
| | Total time (ms) | 20,976 | 13,209 | — |
| **Reasoning** (191–166 tok) | Tokens/sec | **182.6** | 121.5 | +50% |
| | TTFT (ms) | 17,463* | 16,398* | — |
| | Total time (ms) | 18,510 | 17,766 | — |

\*High TTFT on some runs caused by model cold-load when swapping between E2B and E4B. Steady-state TTFT after warmup is ~500–2,700 ms. Generation throughput (tok/s) is the more reliable metric.

### Speed Summary

- **E2B is ~50% faster** than E4B across all prompt types (~182 vs ~121 tokens/sec)
- Both models sustain consistent generation speed regardless of prompt complexity
- E4B takes ~1.6× longer per token due to larger parameter count
- Prompt evaluation is fast for both (< 75 ms for short prompts)

---

## Long Context Coherency Results

### Test Design

Non-chat completion prompts with accumulated document context (project knowledge base). Facts injected at document start, middle, and end; filler text simulates growing context over time.

### Standard Coherency (16 tests per model)

| Category | Tests | E2B | E4B |
|----------|-------|-----|-----|
| Basic recall | 2 | PASS | PASS |
| Cross-reference | 2 | PASS | PASS |
| Technical recall | 1 | PASS | PASS |
| Numerical recall | 2 | PASS | PASS |
| Temporal reasoning | 1 | PASS | PASS |
| Distractor resistance | 1 | PASS | PASS |
| Multi-hop reasoning | 1 | PASS | PASS |
| Distant recall (long context) | 2 | PASS | PASS |
| Long-context stress | 4 | PASS | PASS |
| **Total** | **16** | **16/16 (100%)** | **16/16 (100%)** |

### Extended Coherency (28 tests per model, 4 context sizes)

| Context Size | Prompt Tokens | E2B | E4B |
|-------------|---------------|-----|-----|
| ~8K chars | ~1,775 | 7/7 PASS | 7/7 PASS |
| ~22K chars | ~4,739 | 7/7 PASS | 7/7 PASS |
| ~50K chars | ~10,679 | 7/7 PASS | 7/7 PASS |
| ~100K chars | ~21,095 | 7/7 PASS | 7/7 PASS |
| **Total** | | **28/28 (100%)** | **28/28 (100%)** |

### Specific Examples

**What worked (both models):**
- Recalling lead engineer "Dr. Elena Vasquez" from buried project docs
- Correct budget figures ($2.4M total, $552,700 remaining, 77% spent)
- Cross-referencing milestone blockers with ticket numbers (NEB-2847, Module B thermal)
- Resisting distractor questions (correctly saying CRYSTALS-Kyber, not AES-256)
- Multi-hop: connecting Marcus Chen's Rust rewrite with Priya Sharma's anomaly module addition
- Early-position facts (secret code ALPHA-TANGO-7749) at 21K tokens
- Late-position facts (backup IP 10.42.17.93) after 100K chars of filler
- Cross-position queries requiring both early and late facts simultaneously

**What failed:**
- No failures observed in any test condition up to ~21K prompt tokens

---

## Limitations

1. **Context ceiling not fully tested** — Tests reached ~21K prompt tokens; the 128K context window was not stress-tested to its limit
2. **Single GPU only** — E4B fails on dual-GPU without `CUDA_VISIBLE_DEVICES` restriction
3. **Thinking mode** — Must explicitly disable thinking for completion API; default behavior returns empty responses
4. **Q4_K_M quantization only** — Did not test Q6, Q8, or unquantized variants
5. **Text-only** — Multimodal (image/audio) capabilities not tested
6. **Synthetic context** — Real-world RAG with noisy/ambiguous documents may show different coherency
7. **Model swap overhead** — Alternating between E2B and E4B inflates TTFT measurements

## Recommendations

| Use Case | Recommendation |
|----------|----------------|
| **Speed-critical edge deployment** | **E2B** — 50% faster, identical coherency in tests |
| **Higher reasoning quality** | **E4B** — Larger model, same coherency, more detailed responses |
| **Long document Q&A** | Both handle 21K+ tokens well; E4B responses slightly more polished |
| **Production setup** | Use Ollama ≥0.20, set `think: false`, pin to single GPU for E4B |
| **Further testing** | Push to 64K–128K tokens, test with real documents, compare quantized variants |

---

## Reproducing

```bash
# Start Ollama (if not running)
CUDA_VISIBLE_DEVICES=1 OLLAMA_HOST=127.0.0.1:11435 \
  OLLAMA_MODELS="./ollama-models" ./bin/bin/ollama serve &

# Pull models (one-time)
OLLAMA_HOST=127.0.0.1:11435 ./bin/bin/ollama pull gemma4:e2b
OLLAMA_HOST=127.0.0.1:11435 ./bin/bin/ollama pull gemma4:e4b

# Run all benchmarks
chmod +x run_benchmarks.sh && ./run_benchmarks.sh
```

Results saved to:
- `results/speed_results.json`
- `results/coherency_results.json`
- `results/extended_coherency_results.json`
