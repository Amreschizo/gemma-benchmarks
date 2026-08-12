# MLP FiM Lore Roleplay Benchmark Report

**Date:** August 10, 2026  
**Workspace:** `/home/amreschizo/gemma test`  
**Ollama:** `http://127.0.0.1:11435`

---

## Purpose

Evaluate whether Gemma 4 variants can roleplay as *My Little Pony: Friendship is Magic* (main TV show) characters using in-weight lore, with an explicit system instruction to **avoid Equestria Girls** and other spin-offs.

**Script:** `benchmarks/mlp_lore_test.py`  
**Results:** `results/mlp_lore_results.json`

### Scoring (7 characters)

| Verdict | Criteria |
|---------|----------|
| **PASS** | ≥3 FiM pattern hits, ≥1 in-character trait hit, **no** EG patterns |
| **PARTIAL** | 2 FiM hits + traits, no EG |
| **FAIL** | EG contamination and/or insufficient FiM/trait evidence |

Characters: Twilight Sparkle, Rainbow Dash, Pinkie Pie, Rarity, Applejack, Fluttershy, Princess Celestia.

---

## Cross-Model Summary

| Model | Quant | Pass | Partial | Fail | Pass rate* | Avg FiM hits | EG contamination | Avg gen time |
|-------|-------|------|---------|------|------------|--------------|------------------|--------------|
| `gemma4:e4b` | Q4_K_M | 6 | 1 | 0 | **92.9%** | 3.57 | **0** | 17.3 s |
| `batiai/gemma4-e4b:q6` | Q6_K | 6 | 1 | 0 | **92.9%** | **3.86** | **0** | 12.4 s |
| `gemma4:e2b` | Q4_K_M | 3 | 3 | 1 | 64.3% | 3.14 | **0** | **1.6 s** |
| `batiai/gemma4-e2b:q6` | Q6_K | 2 | 4 | 1 | 57.1% | 2.86 | **0** | 18.0 s |

\*Pass rate = (pass + 0.5×partial) / 7 × 100, per benchmark script.

### Equestria Girls contamination

**None detected** across all 28 completed character runs (4 models × 7 characters). Every model respected the “no Equestria Girls / Canterlot High / human versions” instruction in practice.

### Per-character notes

| Character | Best performers | Weak spots |
|-----------|-----------------|------------|
| Twilight, Celestia | All models PASS except — | — |
| Rainbow, Rarity, Fluttershy | E4B variants PASS | E2B Q6 often PARTIAL |
| Pinkie Pie | E4B Q6 PASS | Official E2B/E4B often PARTIAL (party tone without enough named lore hits) |
| Applejack | Official E4B PASS | **E2B variants FAIL**; E4B Q6 **PARTIAL** (confused Sugarcube Corner with the Apple family farm) |

---

## Speed Context (Q6 BatiAI builds)

Added to `results/speed_results.json` (3 runs/prompt, `think: false`):

| Model | Short gen (tok/s) | Medium gen (tok/s) | Reasoning (tok/s) |
|-------|-------------------|--------------------|-------------------|
| `batiai/gemma4-e2b:q6` | ~160 | ~161 | ~159 |
| `batiai/gemma4-e4b:q6` | ~102 | ~102 | ~101 |

Official Q4 models (`gemma4:e2b` / `gemma4:e4b`) remain faster on this host for the same script (~182 vs ~121 tok/s in the main report), but MLP roleplay quality tracks **effective 4B** capacity more than quantization tier.

---

## Recommendations for MLP FiM Roleplay

1. **Default choice: `gemma4:e4b` (Q4)** or **`batiai/gemma4-e4b:q6`** — tied at 92.9% pass rate; E4B Q6 has slightly richer FiM pattern density (3.86 vs 3.57 avg hits) and faster per-response latency in this MLP test (~12.4 s vs ~17.3 s), at the cost of lower raw tok/s in the speed benchmark (~102 vs ~121 tok/s for official E4B Q4).

2. **Avoid `gemma4:e2b` and `batiai/gemma4-e2b:q6` for lore-heavy FiM RP** — Applejack fails on both; several mains stay PARTIAL. Higher Q6 quantization does **not** close the gap vs E4B for this task.

3. **Prompting:** Keep the existing system block that forbids EG spin-offs; all models complied. For Applejack and Pinkie, add 1–2 shot examples or a short “canon facts” bullet list in the system prompt if you need consistent PASS on secondary ponies.

4. **Latency vs quality:** Official E2B Q4 is fastest for MLP generations (~1.6 s avg) but least accurate; use only for lightweight banter, not canon-sensitive scenes.

5. **Hardware note:** BatiAI Q6 builds show very high TTFT (~15–18 s) on cold/warm cycles in speed tests; for interactive RP, keep the model loaded (Ollama `keep_alive`) and prefer E4B Q6 only when you want maximum lore fidelity without pulling the larger official E4B blob.

---

## Completion Status (this session)

| Task | Status |
|------|--------|
| `batiai/gemma4-e4b:q6` pull | Already present on Ollama :11435 |
| MLP lore test for E4B Q6 | **Done** — merged into `mlp_lore_results.json` |
| Speed test for Q6 variants | **Done** — appended to `speed_results.json` |
| `MLP_LORE_REPORT.md` | **Created** |

### `batiai/gemma4-e4b:q6` MLP scores

- **6 PASS, 1 PARTIAL (Applejack), 0 FAIL**
- **Pass rate: 92.9%** | **Avg FiM hits: 3.86** | **EG contamination: 0**
