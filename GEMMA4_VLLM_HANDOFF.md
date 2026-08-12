# Gemma 4 E4B on vLLM — Diagnosis + Fix Plan for Resumable Context

**Goal:** Serve Gemma 4 E4B Q6 through vLLM with prefix-cached/resumable requests so context can be injected while the model is mid-generation (voice pipeline: STT → LLM → Kokoro TTS).

**Current status (from REPORT.md / VLLM_RESUMABLE_REPORT.md, 2026-08-10):**
- Ollama + Gemma 4 E4B Q6: **works**, ~507ms cold TTFT, ~530ms TTS-ready, ~104 tok/s single-stream, but serializes badly under concurrency (~2.4s TTFT at 5 concurrent streams).
- vLLM 0.27.0 + `google/gemma-4-E4B-it` (BF16): **fails to load** — `AssertionError: Attempted to load weight (torch.Size([512])) into parameter (torch.Size([256]))`.
- vLLM concurrency/APC mechanics were only verified against `Qwen/Qwen2.5-3B-Instruct` as a stand-in — not actually Gemma 4.

---

## 1. Root cause of the load failure (diagnosed, not guessed)

This is a known, previously-fixed class of vLLM bug, not a dead end. Gemma 4 has **heterogeneous attention head dimensions** — local (sliding-window) layers use `head_dim=256`, global layers use `head_dim=512` — plus 18 of 42 layers are **KV-sharing layers** that reuse an earlier layer's K/V projections instead of owning their own weights. When vLLM resolves the checkpoint to its native `Gemma4ForConditionalGeneration` class, this is handled correctly. When it falls back to the generic `TransformersMultiModalForCausalLM` path instead (missing `--trust-remote-code`, stale vLLM version, or a registry dispatch miss), the generic weight loader doesn't know about the local/global split or the KV-sharing layers and throws exactly the 512-vs-256 shape mismatch seen in the report.

Native Gemma 4 support landed in **vLLM v0.19.0 (2026-04-03)** via PR #38826 and has shipped in every release since. The fact that it still failed on **v0.27.0** in this environment points to one of:
- Missing `--trust-remote-code` on the serve command
- A stale/partial local vLLM install or a stale HF cache of the weights (Google shipped a **July 2026 weight/template refresh** for Gemma 4 — anything downloaded before ~July 16, 2026 is stale and should be re-pulled)
- The model resolving to the fallback Transformers path instead of the native `Gemma4ForConditionalGeneration` path

## 2. Fix-and-verify steps

Run these in order. **Do not skip the verification lines** — each step has a specific log line or command output to confirm before moving to the next step, otherwise you're debugging blind.

### Step 1 — Clean environment, no spaces in the path
The current workspace path (`/home/amreschizo/gemma test/`) contains a space, which already broke FlashInfer's ninja JIT build in the earlier Qwen benchmark. Move or symlink to a space-free path before doing anything else:
```bash
mkdir -p ~/projects
ln -s "/home/amreschizo/gemma test" ~/projects/gemma-test
cd ~/projects/gemma-test
```

### Step 2 — Upgrade vLLM to latest stable
```bash
pip install -U vllm
python -c "import vllm; print(vllm.__version__)"
```
Confirm the version is well past 0.19.0 (0.27.0 already was, so this alone likely isn't the fix — but rules it out).

### Step 3 — Force a fresh weight pull
Delete any cached `google/gemma-4-E4B-it` weights and re-download, to pick up the July 2026 refresh:
```bash
rm -rf ~/.cache/huggingface/hub/models--google--gemma-4-E4B-it
huggingface-cli download google/gemma-4-E4B-it
```

### Step 4 — Serve with `--trust-remote-code` explicitly
```bash
CUDA_VISIBLE_DEVICES=1 vllm serve google/gemma-4-E4B-it \
  --trust-remote-code \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --enable-prefix-caching \
  --port 8001
```
**Verify in the startup log** that it resolves to the native class, not the fallback:
```
INFO [model.py:549] Resolved architecture: Gemma4ForConditionalGeneration
```
If the log instead says `Using Transformers modeling backend` / `has no vLLM implementation, falling back to Transformers implementation` — the native path isn't being used and the original error will recur. Stop and report back rather than proceeding.

### Step 5 — Check the attention backend before trusting any speed number
Gemma 4's heterogeneous head dims have, in some vLLM versions, forced a fallback from FlashAttention to a much slower Triton kernel (one reported case: ~9 tok/s on an RTX 4090 vs 100+ tok/s for a comparable model). Look for this line in the log:
```
INFO [config.py:104] Gemma4 model has heterogeneous head dimensions (head_dim=256, global_head_dim=512). Forcing TRITON_ATTN backend...
```
If you see `TRITON_ATTN` being forced, **do not proceed to declare success** — benchmark actual throughput first (Step 6). A working load with crippled throughput is not a win over Ollama's ~104-122 tok/s.

### Step 6 — Re-run the existing benchmark harness against the real model
Once the server is up and resolved natively, point the existing harness at it instead of the Qwen proxy:
```bash
.venv/bin/python benchmarks/vllm_resumable_stream_test.py \
  --vllm-model google/gemma-4-E4B-it --runs 3
```
Compare directly against the Ollama Q6 numbers already in hand:
- Cold TTFT must beat ~507ms
- Steady tok/s must not regress far below ~104-122 tok/s (some loss vs Ollama's llama.cpp backend is expected/acceptable; a 10x drop, as seen in the Triton-fallback bug report, is not)
- 5-concurrent TTFT should land far below Ollama's ~2.4s (this is the actual reason to use vLLM at all)

### Step 7 — Quantization decision
Skip GGUF/Q6 on vLLM — vLLM's GGUF path for Gemma 4 isn't confirmed working anywhere in the research for this report, and the AWQ repo already tried (`majentik/gemma-4-E4B-TurboQuant-AWQ-4bit`) 404'd. E4B fits comfortably in BF16 on a 24GB 3090 (~9-16GB), so there's no memory pressure forcing quantization here — **serve BF16 directly** and only revisit quantization if Step 6 shows a real performance problem BF16 doesn't explain. Do not chase NVFP4 — that quant path needs Blackwell-class hardware and won't run on the 3090 (Ampere).

## 3. If it still doesn't load or is too slow after Steps 1-6

Don't keep patching indefinitely. Escalation order:
1. Check `github.com/vllm-project/vllm` issues filtered to `Gemma4` opened in the last ~30 days — this is an actively-patched area, so a fresh regression may already be reported/fixed upstream.
2. Try `google/gemma-4-31B-it` or the 26B MoE only if E4B specifically is broken and a larger model isn't — unlikely to help here and adds VRAM pressure, so treat as a last resort, not a real option on a single 3090 alongside Kokoro + your cognition stack.
3. **Fallback: stay on Ollama for inference.** It's proven, it's fast single-stream, and it already hits the sub-100-200ms class target once warm. What you lose is vLLM's true mid-generation context injection (async input generator / resumable session). If vLLM's Gemma 4 path can't be made to work reliably, the honest alternative is: keep single-session-at-a-time on Ollama, and handle "new input arrives while generating" at the application layer — buffer the interrupting input, cancel/discard the in-flight Ollama generation, and re-issue a new request with the updated context appended to the existing prompt (Ollama does support prompt-prefix reuse for repeated context, just not vLLM's live-append-mid-decode mechanism). This is a real capability downgrade versus the original design goal and should be flagged back to me if it's the path you end up on, rather than silently shipped.

## 4. Explicit definition of "done" for this task

Do not report this as complete until all of the following are true and logged:
- [ ] Server log shows `Resolved architecture: Gemma4ForConditionalGeneration` (native path, not Transformers fallback)
- [ ] Server log does NOT show a forced `TRITON_ATTN` fallback, OR if it does, Step 6 benchmark numbers are still acceptable (not a 5-10x regression vs Ollama)
- [ ] Benchmark harness re-run against the real Gemma 4 E4B model (not the Qwen proxy) with numbers written to `results/`
- [ ] Cold TTFT, warm TTFT, and 5-concurrent TTFT all recorded and compared against the Ollama Q6 baseline in this report
- [ ] `--enable-prefix-caching` confirmed active (check for prefix cache hit-rate in logs on a repeated-prefix test, same as the ~83% hit rate seen with the Qwen proxy)
- [ ] Resumable/async-input-generator behavior for mid-generation context injection specifically tested — a working server + APC alone is not the same as confirming context can be appended while a generation is in flight; verify this against current vLLM docs for this exact version, since the flag/API for it may have changed since 0.27.0
