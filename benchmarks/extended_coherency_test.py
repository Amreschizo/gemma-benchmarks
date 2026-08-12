#!/usr/bin/env python3
"""Extended long-context coherency test with progressively larger contexts."""

import json
import re
import sys
import time
import urllib.request
from dataclasses import dataclass, asdict

OLLAMA_HOST = "http://127.0.0.1:11435"
MODELS = ["gemma4:e2b", "gemma4:e4b"]

SYSTEM_PREAMBLE = """You are an AI assistant embedded in a project management system.
Use ONLY the information in the knowledge base below. Do not invent facts.

=== PROJECT KNOWLEDGE BASE ===
"""

# Critical facts buried at the START of context
EARLY_FACTS = """
[CRITICAL FACT #1] Project Codename: NEBULA-7
[CRITICAL FACT #2] Secret Access Code: ALPHA-TANGO-7749
[CRITICAL FACT #3] Lead Engineer: Dr. Elena Vasquez
[CRITICAL FACT #4] Emergency Shutdown Command: NEBULA-SCRAM-OMEGA
[CRITICAL FACT #5] Satellite Frequency: 2247.5 MHz (Ku-band)
"""

# Facts buried at the END of context (after filler)
LATE_FACTS = """
[CRITICAL FACT #6] Backup Server IP: 10.42.17.93
[CRITICAL FACT #7] Encryption Key Rotation Interval: 72 hours
[CRITICAL FACT #8] Authorized Pen Tester: CyberShield Inc. (contact: audit@cybershield.com)
[CRITICAL FACT #9] Maximum Packet Size: 4096 bytes
[CRITICAL FACT #10] Failover Datacenter: us-east-1-nebula-dr
"""

FILLER_UNIT = """
TECHNICAL DOCUMENTATION SECTION {n}
This section describes subsystem configuration parameters for the NEBULA platform.
The telemetry subsystem processes incoming satellite data packets according to CCSDS
standards. Packet headers contain version (3 bits), type (1 bit), secondary header
flag (1 bit), APID (11 bits), sequence flags (2 bits), sequence count (14 bits),
and packet data length (16 bits). The data field contains mission-specific payload.
Error correction uses Reed-Solomon coding with configurable redundancy levels.
Ground station antenna tracking uses TLE (Two-Line Element) orbital parameters
updated every 4 hours from NORAD catalog. Signal-to-noise ratio thresholds are
configured per-band: S-band minimum 8 dB, X-band minimum 12 dB, Ka-band minimum 15 dB.
Power management subsystems monitor solar panel output, battery state of charge,
and load distribution across redundant bus architectures. Thermal management
includes passive radiators, active heat pipes, and software-controlled throttling
at configurable temperature thresholds. Communication protocols support store-and-forward
for eclipse periods with configurable buffer sizes up to 2GB per satellite.
""" * 3  # ~600 chars per unit


def build_context(target_chars: int) -> str:
    context = SYSTEM_PREAMBLE + EARLY_FACTS + "\n"
    n = 0
    while len(context) < target_chars - len(LATE_FACTS) - 100:
        context += FILLER_UNIT.format(n=n) + "\n"
        n += 1
    context += LATE_FACTS + "\n=== END KNOWLEDGE BASE ===\n"
    return context


TESTS_BY_POSITION = [
    {"id": "early_fact_2", "question": "What is the secret access code for Project NEBULA-7?", "patterns": [r"ALPHA-TANGO-7749"], "position": "early"},
    {"id": "early_fact_4", "question": "What is the emergency shutdown command?", "patterns": [r"NEBULA-SCRAM-OMEGA"], "position": "early"},
    {"id": "early_fact_5", "question": "What satellite frequency does NEBULA-7 use?", "patterns": [r"2247[.,]5\s*MHz", r"ku.?band"], "position": "early"},
    {"id": "late_fact_6", "question": "What is the backup server IP address?", "patterns": [r"10\.42\.17\.93"], "position": "late"},
    {"id": "late_fact_8", "question": "Who is the authorized pen tester and what is their contact email?", "patterns": [r"cybershield", r"audit@cybershield"], "position": "late"},
    {"id": "late_fact_10", "question": "What is the failover datacenter location?", "patterns": [r"us-east-1-nebula-dr"], "position": "late"},
    {"id": "cross_position", "question": "What is the secret access code AND the backup server IP?", "patterns": [r"ALPHA-TANGO-7749", r"10\.42\.17\.93"], "position": "cross"},
]


def generate(model: str, prompt: str) -> tuple[str, int]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {"num_predict": 200, "temperature": 0.0},
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        result = json.loads(resp.read())
    return result.get("response", ""), result.get("prompt_eval_count", 0)


def check(response: str, patterns: list[str]) -> tuple[bool, list[str]]:
    matched = [p for p in patterns if re.search(p, response, re.IGNORECASE)]
    return len(matched) == len(patterns), matched


def main():
    context_sizes = [5000, 20000, 50000, 100000]
    all_results = []

    for size in context_sizes:
        context = build_context(size)
        actual_chars = len(context)
        print(f"\n{'='*60}")
        print(f"Context size: {actual_chars:,} chars (~{actual_chars//4:,} tokens est.)")
        print(f"{'='*60}")

        for model in MODELS:
            print(f"\n  Model: {model}")
            for test in TESTS_BY_POSITION:
                prompt = (
                    context
                    + f"\nBased on the knowledge base above, answer concisely:\n"
                    f"QUESTION: {test['question']}\nANSWER:"
                )
                t0 = time.perf_counter()
                response, prompt_tokens = generate(model, prompt)
                elapsed = (time.perf_counter() - t0) * 1000
                passed, matched = check(response, test["patterns"])
                status = "PASS" if passed else "FAIL"
                print(f"    [{status}] {test['id']} ({test['position']}): {response[:100].replace(chr(10),' ')}...")
                all_results.append({
                    "context_target_chars": size,
                    "context_actual_chars": actual_chars,
                    "prompt_tokens": prompt_tokens,
                    "model": model,
                    "test_id": test["id"],
                    "position": test["position"],
                    "question": test["question"],
                    "response": response.strip(),
                    "passed": passed,
                    "matched_patterns": matched,
                    "generation_time_ms": round(elapsed, 1),
                })

    # Summary
    summary = {}
    for model in MODELS:
        model_results = [r for r in all_results if r["model"] == model]
        by_size = {}
        for size in context_sizes:
            subset = [r for r in model_results if r["context_target_chars"] == size]
            passed = sum(1 for r in subset if r["passed"])
            by_size[str(size)] = {"passed": passed, "total": len(subset), "pass_rate": round(passed/len(subset)*100,1) if subset else 0}
        total_pass = sum(1 for r in model_results if r["passed"])
        summary[model] = {
            "total_passed": total_pass,
            "total_tests": len(model_results),
            "overall_pass_rate": round(total_pass/len(model_results)*100, 1),
            "by_context_size": by_size,
        }

    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "context_sizes_tested": context_sizes,
        "results": all_results,
        "summary": summary,
    }
    out_path = "/home/amreschizo/gemma test/results/extended_coherency_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
