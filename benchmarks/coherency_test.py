#!/usr/bin/env python3
"""
Long-context coherency benchmark for Gemma 4 E2B vs E4B.

Simulates a non-chat environment where facts accumulate in a single
context document (system prompt + accumulated knowledge base), then
tests whether the model recalls and uses those facts coherently.
"""

import json
import re
import sys
import time
import urllib.request
from dataclasses import dataclass, asdict, field

OLLAMA_HOST = "http://127.0.0.1:11435"
MODELS = ["gemma4:e2b", "gemma4:e4b"]

# --- Fact injection blocks (simulated document accumulation) ---

SYSTEM_PREAMBLE = """You are an AI assistant embedded in a project management system.
You have access to the following project knowledge base. Use ONLY the information
provided below to answer questions. Do not invent facts not present in the knowledge base.

=== PROJECT KNOWLEDGE BASE ===
"""

FACT_BLOCKS = [
    """
SECTION 1 - PROJECT OVERVIEW
Project Codename: NEBULA-7
Lead Engineer: Dr. Elena Vasquez
Start Date: March 15, 2024
Budget Allocation: $2.4 million
Primary Goal: Develop a quantum-resistant encryption module for satellite communications.
""",
    """
SECTION 2 - TEAM MEMBERS
- Dr. Elena Vasquez (Lead Engineer, cryptography)
- Marcus Chen (Backend, Rust specialist)
- Priya Sharma (ML Engineer, anomaly detection)
- James Okafor (DevOps, Kubernetes)
- Sarah Kim (QA Lead, security auditing)
Emergency contact for security incidents: security@nebula7.internal (ext. 4401)
""",
    """
SECTION 3 - TECHNICAL SPECIFICATIONS
Encryption Algorithm: CRYSTALS-Kyber (NIST PQC standard)
Key Exchange: CRYSTALS-Dilithium for signatures
Target Latency: < 12ms per packet on ARM Cortex-A78
Memory Budget: 48MB RAM maximum on embedded module
Supported Protocols: CCSDS Space Packet Protocol, custom NEBULA framing
Hash Function: SHA3-256 for integrity verification
""",
    """
SECTION 4 - MILESTONES AND DEADLINES
M1 (Prototype): June 30, 2024 - COMPLETED
M2 (Alpha Testing): September 15, 2024 - COMPLETED
M3 (Beta on Test Satellite): January 20, 2025 - IN PROGRESS
M4 (Production Deployment): April 30, 2025 - PENDING
M5 (Full Constellation Rollout): December 1, 2025 - PENDING
Current blocker: Thermal testing failure on Module B (ticket NEB-2847)
""",
    """
SECTION 5 - SECURITY POLICIES
- All code commits require 2 approvals from security-cleared team members
- API keys rotate every 72 hours
- Staging environment URL: https://staging.nebula7.internal (VPN required)
- Production deployment requires sign-off from Dr. Vasquez AND James Okafor
- Incident severity levels: P1 (data breach), P2 (service down), P3 (degraded), P4 (cosmetic)
- Last security audit: November 8, 2024, conducted by CyberShield Inc.
""",
    """
SECTION 6 - VENDOR AND SUPPLIER INFO
Hardware Supplier: OrbitalTech Systems (contract ORB-2024-089)
FPGA Board Model: OT-XR7800 (quantum-safe variant)
Backup Supplier: NovaChip Industries (on standby since August 2024)
Annual maintenance contract: $180,000 with OrbitalTech
Shipping address for hardware: Building 7, Gate C, Vandenberg SFB, CA 93437
""",
    """
SECTION 7 - TESTING RESULTS
Unit Test Coverage: 94.2% (target: 95%)
Integration Tests: 847 passing, 3 failing (NEB-2847, NEB-2851, NEB-2855)
Performance Benchmark (Module A): 8.7ms avg latency (PASS, target <12ms)
Performance Benchmark (Module B): 11.2ms avg latency but thermal throttling at 65°C (FAIL)
Penetration Test Score: 98/100 (CyberShield Inc., Nov 2024)
Fuzzing Campaign: 2.1M iterations, 0 crashes, 1 logic error (fixed in commit a3f8c21)
""",
    """
SECTION 8 - DEPLOYMENT CONFIGURATION
Kubernetes Cluster: nebula-prod-us-west-2 (3 nodes, t3.xlarge equivalent)
Container Registry: registry.nebula7.internal:5000
CI/CD Pipeline: GitLab CI with 4 stages (lint, test, build, deploy)
Blue-Green Deployment: enabled for M4 rollout
Rollback Window: 30 minutes after deployment
Monitoring: Prometheus + Grafana dashboard "NEBULA-OPS-MAIN"
Alert Channel: #nebula-alerts on internal Slack
""",
    """
SECTION 9 - RECENT DECISIONS LOG
2024-11-15: Switched from AES-256 to CRYSTALS-Kyber per NIST recommendation
2024-12-01: Added Priya Sharma's anomaly detection module to M3 scope
2025-01-05: Postponed M4 by 2 weeks due to Module B thermal issues
2025-01-18: Approved $45,000 emergency budget for thermal redesign (NEB-2847)
2025-02-01: Marcus Chen proposed Rust rewrite of packet parser (approved)
2025-02-10: Sarah Kim requested additional pen test before M4 (approved, scheduled March 2025)
""",
    """
SECTION 10 - FINANCIAL TRACKING
Total Budget: $2,400,000
Spent to Date: $1,847,300 (77.0%)
Remaining: $552,700
Q1 2025 Burn Rate: $198,400/month
Projected Overrun Risk: LOW (within 5% of budget)
Largest Expense: OrbitalTech hardware ($890,000)
Smallest Line Item: Grafana license ($2,400/year)
""",
]

# Filler content to simulate long context (repeated technical documentation)
FILLER_PARAGRAPHS = [
    """
APPENDIX A - CCSDS PROTOCOL REFERENCE
The Consultative Committee for Space Data Systems (CCSDS) defines standards for
space communication. The Space Packet Protocol (SPP) provides a common packet
format for space missions. Primary Header: 6 bytes (version, type, secondary header
flag, APID, sequence flags, sequence count, packet data length). Secondary Header:
optional, mission-specific. Data Field: variable length payload. NEBULA-7 uses
APID 0x07A3 for encrypted telemetry and 0x07A4 for command packets.
""",
    """
APPENDIX B - CRYSTALS-KYBER PARAMETER SETS
Kyber-512: n=256, k=2, q=3329, eta=3, du=10, dv=4. Public key: 800 bytes.
Secret key: 1632 bytes. Ciphertext: 768 bytes. Shared secret: 32 bytes.
Security level: NIST Level 1 (equivalent to AES-128). Kyber-768: n=256, k=3.
Public key: 1184 bytes. Security level: NIST Level 3. NEBULA-7 uses Kyber-768
for the balance of security and performance on embedded hardware.
""",
    """
APPENDIX C - THERMAL MANAGEMENT GUIDELINES
Module operating temperature range: -40°C to +85°C (military grade).
Thermal throttling begins at 65°C on Module B due to insufficient heat sink
design. Recommended fix: upgrade to OT-XR7800-THERMAL variant with integrated
heat spreader. Thermal simulation (ANSYS) predicts 12°C reduction with new design.
Testing scheduled for February 2025 at Vandenberg thermal chamber facility.
""",
] * 8  # Repeat to build up context length

# Coherency test questions with expected answer patterns
COHERENCY_TESTS = [
    {
        "id": "basic_recall_lead",
        "question": "Who is the lead engineer on Project NEBULA-7?",
        "expected_patterns": [r"elena\s*vasquez", r"dr\.?\s*elena"],
        "category": "basic_recall",
        "section": "SECTION 2",
    },
    {
        "id": "basic_recall_budget",
        "question": "What is the total budget for Project NEBULA-7?",
        "expected_patterns": [r"\$?2[.,]4\s*million", r"2,?400,?000"],
        "category": "basic_recall",
        "section": "SECTION 1/10",
    },
    {
        "id": "cross_reference_milestone",
        "question": "What is the current blocker for Milestone M3, and what ticket number is it?",
        "expected_patterns": [r"thermal", r"NEB-2847", r"module\s*b"],
        "category": "cross_reference",
        "section": "SECTION 4/7",
    },
    {
        "id": "cross_reference_security",
        "question": "Who must sign off on production deployment, and what is the emergency security contact?",
        "expected_patterns": [r"vasquez", r"okafor", r"security@nebula7", r"4401"],
        "category": "cross_reference",
        "section": "SECTION 2/5",
    },
    {
        "id": "technical_detail",
        "question": "What encryption algorithm and key exchange method does NEBULA-7 use?",
        "expected_patterns": [r"kyber", r"dilithium", r"crystals"],
        "category": "technical_recall",
        "section": "SECTION 3",
    },
    {
        "id": "numerical_precision",
        "question": "What is the unit test coverage percentage and how many integration tests are failing?",
        "expected_patterns": [r"94[.,]2\s*%", r"\b3\b.*fail", r"fail.*\b3\b"],
        "category": "numerical_recall",
        "section": "SECTION 7",
    },
    {
        "id": "temporal_reasoning",
        "question": "Why was Milestone M4 postponed, and when was that decision made?",
        "expected_patterns": [r"thermal", r"module\s*b", r"2025-01-05", r"january.*2025"],
        "category": "temporal_reasoning",
        "section": "SECTION 4/9",
    },
    {
        "id": "financial_calculation",
        "question": "How much budget remains for Project NEBULA-7, and what percentage has been spent?",
        "expected_patterns": [r"\$?552,?700", r"77\s*%", r"77\.0"],
        "category": "numerical_recall",
        "section": "SECTION 10",
    },
    {
        "id": "distractor_resistance",
        "question": "What AES encryption key length does NEBULA-7 use for its primary encryption?",
        "expected_patterns": [
            r"kyber", r"crystals", r"not\s+aes", r"no\s+aes",
            r"post.?quantum", r"pqc", r"switched.*aes",
            r"quantum.?resistant", r"dilithium",
        ],
        "category": "distractor_resistance",
        "section": "SECTION 3/9",
        "note": "AES was replaced; model should NOT say AES-256 is primary",
    },
    {
        "id": "multi_hop",
        "question": (
            "Marcus Chen proposed a rewrite in what language, and was the anomaly "
            "detection module added by which team member and when?"
        ),
        "expected_patterns": [r"rust", r"priya\s*sharma", r"2024-12-01", r"december.*2024"],
        "category": "multi_hop",
        "section": "SECTION 2/9",
    },
]

# Near-context questions (asked after injecting more filler to push facts further back)
DISTANT_RECALL_TESTS = [
    {
        "id": "distant_vendor",
        "question": "What is the hardware supplier for NEBULA-7 and what is their contract number?",
        "expected_patterns": [r"orbitaltech", r"ORB-2024-089"],
        "category": "distant_recall",
    },
    {
        "id": "distant_latency",
        "question": "What is the target latency per packet and what was Module A's benchmark result?",
        "expected_patterns": [r"12\s*ms", r"8[.,]7\s*ms"],
        "category": "distant_recall",
    },
]


@dataclass
class CoherencyResult:
    model: str
    test_id: str
    category: str
    question: str
    response: str
    passed: bool
    matched_patterns: list[str] = field(default_factory=list)
    missing_patterns: list[str] = field(default_factory=list)
    context_length_chars: int = 0
    prompt_eval_count: int = 0
    generation_time_ms: float = 0


def build_context(include_filler: bool = True, filler_multiplier: int = 1) -> str:
    context = SYSTEM_PREAMBLE
    for block in FACT_BLOCKS:
        context += block + "\n"
    if include_filler:
        for i, para in enumerate(FILLER_PARAGRAPHS[: len(FILLER_PARAGRAPHS) * filler_multiplier // len(FILLER_PARAGRAPHS) + len(FILLER_PARAGRAPHS)]):
            context += f"\n--- FILLER DOCUMENT {i+1} ---\n" + para
    context += "\n=== END KNOWLEDGE BASE ===\n"
    return context


def generate(model: str, prompt: str, num_predict: int = 300) -> tuple[str, dict]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
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
    with urllib.request.urlopen(req, timeout=600) as resp:
        result = json.loads(resp.read())
    elapsed_ms = (time.perf_counter() - start) * 1000
    return result.get("response", ""), {
        "prompt_eval_count": result.get("prompt_eval_count", 0),
        "eval_count": result.get("eval_count", 0),
        "generation_time_ms": round(elapsed_ms, 2),
    }


def check_patterns(response: str, patterns: list[str]) -> tuple[bool, list[str], list[str]]:
    response_lower = response.lower()
    matched = []
    missing = []
    for pattern in patterns:
        if re.search(pattern, response_lower, re.IGNORECASE):
            matched.append(pattern)
        else:
            missing.append(pattern)
    passed = len(matched) > 0
    return passed, matched, missing


def run_coherency_tests(model: str, context: str, tests: list[dict], label: str) -> list[CoherencyResult]:
    results = []
    print(f"\n  [{label}] Context: {len(context):,} chars")
    for test in tests:
        full_prompt = (
            context
            + f"\nBased on the knowledge base above, answer the following question concisely:\n"
            f"QUESTION: {test['question']}\n"
            f"ANSWER:"
        )
        response, stats = generate(model, full_prompt)
        passed, matched, missing = check_patterns(response, test["expected_patterns"])
        result = CoherencyResult(
            model=model,
            test_id=test["id"],
            category=test["category"],
            question=test["question"],
            response=response.strip(),
            passed=passed,
            matched_patterns=matched,
            missing_patterns=missing,
            context_length_chars=len(context),
            prompt_eval_count=stats["prompt_eval_count"],
            generation_time_ms=stats["generation_time_ms"],
        )
        results.append(result)
        status = "PASS" if passed else "FAIL"
        print(f"    [{status}] {test['id']}: {response[:120].replace(chr(10), ' ')}...")
    return results


def main():
    print("Gemma 4 Long Context Coherency Benchmark")
    print(f"Ollama host: {OLLAMA_HOST}")

    all_results = []

    for model in MODELS:
        print(f"\n{'='*60}")
        print(f"Testing model: {model}")
        print(f"{'='*60}")

        # Phase 1: Moderate context (facts only, no filler)
        context_moderate = build_context(include_filler=False)
        results_moderate = run_coherency_tests(
            model, context_moderate, COHERENCY_TESTS, "moderate_context"
        )
        all_results.extend(results_moderate)

        # Phase 2: Long context (facts + filler to push facts back)
        context_long = build_context(include_filler=True)
        results_distant = run_coherency_tests(
            model, context_long, DISTANT_RECALL_TESTS, "long_context"
        )
        all_results.extend(results_distant)

        # Phase 3: Re-test early facts under long context (stress test)
        early_recall_tests = [t for t in COHERENCY_TESTS if t["id"] in (
            "basic_recall_lead", "basic_recall_budget", "distractor_resistance", "multi_hop"
        )]
        results_stress = run_coherency_tests(
            model, context_long, early_recall_tests, "long_context_stress"
        )
        for r in results_stress:
            r.test_id = f"stress_{r.test_id}"
            r.category = "long_context_stress"
        all_results.extend(results_stress)

    # Summarize
    summary = {}
    for model in MODELS:
        model_results = [r for r in all_results if r.model == model]
        by_category = {}
        for r in model_results:
            by_category.setdefault(r.category, {"pass": 0, "fail": 0, "total": 0})
            by_category[r.category]["total"] += 1
            if r.passed:
                by_category[r.category]["pass"] += 1
            else:
                by_category[r.category]["fail"] += 1
        total_pass = sum(1 for r in model_results if r.passed)
        summary[model] = {
            "total_tests": len(model_results),
            "passed": total_pass,
            "failed": len(model_results) - total_pass,
            "pass_rate": round(total_pass / len(model_results) * 100, 1) if model_results else 0,
            "by_category": by_category,
        }

    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ollama_host": OLLAMA_HOST,
        "models": MODELS,
        "context_info": {
            "moderate_context_chars": len(build_context(include_filler=False)),
            "long_context_chars": len(build_context(include_filler=True)),
            "fact_blocks": len(FACT_BLOCKS),
            "filler_paragraphs": len(FILLER_PARAGRAPHS),
        },
        "results": [asdict(r) for r in all_results],
        "summary": summary,
    }

    out_path = "/home/amreschizo/gemma test/results/coherency_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
