#!/usr/bin/env python3
"""
MLP (My Little Pony: Friendship is Magic) lore coherency benchmark.

Tests whether models can roleplay as FiM characters using knowledge from
weights alone, with explicit instruction to avoid Equestria Girls lore.
"""

import json
import re
import sys
import time
import urllib.request
from dataclasses import dataclass, asdict, field

OLLAMA_HOST = "http://127.0.0.1:11435"

# All model variants to test: official Q4 + BatiAI Q6
MODELS = [
  "gemma4:e2b",
  "gemma4:e4b",
  "batiai/gemma4-e2b:q6",
  "batiai/gemma4-e4b:q6",
]

SYSTEM_INSTRUCTION = """You are roleplaying as {character} from My Little Pony: Friendship is Magic (the main TV show).
Stay in character at all times. Use only lore from the main FiM show.
IMPORTANT: Do NOT reference Equestria Girls, human versions, Canterlot High, or any spin-off media.
Speak in first person as {character}."""

CHARACTER_TESTS = [
  {
    "id": "twilight_sparkle",
    "character": "Twilight Sparkle",
    "prompt": (
      "Introduce yourself. Where do you live, who is your mentor, "
      "and who are your closest friends? Mention your special talent."
    ),
    "fim_patterns": [
      r"ponyville",
      r"canterlot",
      r"princess\s+celestia",
      r"celestia",
      r"student",
      r"library|golden oak|books|stud(y|ies|ious)",
      r"magic|unicorn|alicorn",
      r"rainbow\s+dash|applejack|pinkie\s+pie|rarity|fluttershy",
      r"element\s+of\s+magic",
    ],
    "eg_patterns": [
      r"equestria\s+girls",
      r"canterlot\s+high",
      r"human\s+version",
      r"high\s+school",
      r"sunset\s+shimmer.*human",
    ],
    "trait_patterns": [r"stud(y|ious)|book|magic|friendship|princess|alicorn|unicorn"],
  },
  {
    "id": "rainbow_dash",
    "character": "Rainbow Dash",
    "prompt": (
      "Tell me about yourself, your home in Ponyville, your pet, "
      "and which elite flying group you dream of joining."
    ),
    "fim_patterns": [
      r"ponyville",
      r"cloudsdale",
      r"wonderbolts",
      r"weather",
      r"fly|flying|pegasus|speed",
      r"applejack|twilight|pinkie|rarity|fluttershy",
      r"element\s+of\s+loyalty",
      r"tank|tortoise",
    ],
    "eg_patterns": [
      r"equestria\s+girls",
      r"canterlot\s+high",
      r"human",
      r"high\s+school",
    ],
    "trait_patterns": [r"loyal|fast|fly|pegasus|competitive|wonderbolts|weather"],
  },
  {
    "id": "pinkie_pie",
    "character": "Pinkie Pie",
    "prompt": (
      "Introduce yourself! Where do you work, who is your sister, "
      "and what makes you happiest?"
    ),
    "fim_patterns": [
      r"sugarcube|sugar\s+cube",
      r"party|parties",
      r"maude|marble|limestone|sisters",
      r"rock\s+farm|yak",
      r"ponyville",
      r"element\s+of\s+laughter",
      r"twilight|rainbow|applejack|rarity|fluttershy",
    ],
    "eg_patterns": [
      r"equestria\s+girls",
      r"canterlot\s+high",
      r"human",
    ],
    "trait_patterns": [r"party|laugh|happy|bounce|sugar|fun"],
  },
  {
    "id": "rarity",
    "character": "Rarity",
    "prompt": (
      "Describe yourself, your boutique in Ponyville, your little sister, "
      "and your creative passion."
    ),
    "fim_patterns": [
      r"carousel\s+boutique",
      r"sweetie\s+belle",
      r"fashion|dress|design",
      r"ponyville",
      r"element\s+of\s+generosity",
      r"unicorn",
      r"twilight|rainbow|applejack|pinkie|fluttershy",
    ],
    "eg_patterns": [
      r"equestria\s+girls",
      r"canterlot\s+high",
      r"human",
    ],
    "trait_patterns": [r"generous|fashion|elegant|boutique|design|dress"],
  },
  {
    "id": "applejack",
    "character": "Applejack",
    "prompt": (
      "Tell me about your family farm, your siblings, and your element of harmony."
    ),
    "fim_patterns": [
      r"sweet\s+apple\s+acres",
      r"big\s+mac|macintosh|granny\s+smith|apple\s+bloom",
      r"apple|orchard|farm",
      r"ponyville",
      r"element\s+of\s+honesty",
      r"earth\s+pony",
      r"twilight|rainbow|pinkie|rarity|fluttershy",
    ],
    "eg_patterns": [
      r"equestria\s+girls",
      r"canterlot\s+high",
      r"human",
    ],
    "trait_patterns": [r"honest|farm|apple|hard.?work|family|reliable"],
  },
  {
    "id": "fluttershy",
    "character": "Fluttershy",
    "prompt": (
      "Introduce yourself. Where is your cottage, what animals do you care for, "
      "and who is your best friend?"
    ),
    "fim_patterns": [
      r"cottage|everfree|animals",
      r"pegasus",
      r"element\s+of\s+kindness",
      r"rainbow\s+dash",
      r"ponyville",
      r"shy|quiet|gentle",
      r"twilight|applejack|pinkie|rarity",
    ],
    "eg_patterns": [
      r"equestria\s+girls",
      r"canterlot\s+high",
      r"human",
    ],
    "trait_patterns": [r"kind|shy|gentle|animal|quiet|soft"],
  },
  {
    "id": "princess_celestia",
    "character": "Princess Celestia",
    "prompt": (
      "Introduce yourself as ruler of Equestria. Who is your sister, "
      "who was your prized student, and what do you raise each day?"
    ),
    "fim_patterns": [
      r"equestria",
      r"canterlot",
      r"luna|night",
      r"twilight\s+sparkle",
      r"student",
      r"sun|day|raise",
      r"princess",
    ],
    "eg_patterns": [
      r"equestria\s+girls",
      r"canterlot\s+high",
      r"human",
      r"high\s+school",
    ],
    "trait_patterns": [r"princess|sun|wise|mentor|twilight|luna|day"],
  },
]


@dataclass
class LoreResult:
  model: str
  character_id: str
  character: str
  prompt: str
  response: str
  verdict: str  # PASS, FAIL, PARTIAL
  fim_hits: list[str] = field(default_factory=list)
  eg_hits: list[str] = field(default_factory=list)
  trait_hits: list[str] = field(default_factory=list)
  fim_score: int = 0
  generation_time_ms: float = 0
  eval_count: int = 0
  fail_reasons: list[str] = field(default_factory=list)


def generate(model: str, system: str, user_prompt: str, character: str, num_predict: int = 400) -> tuple[str, dict]:
  payload = {
    "model": model,
    "messages": [
      {"role": "system", "content": system},
      {"role": "user", "content": user_prompt},
    ],
    "stream": False,
    "think": False,
    "options": {"num_predict": num_predict, "temperature": 0.7},
  }
  data = json.dumps(payload).encode()
  req = urllib.request.Request(
    f"{OLLAMA_HOST}/api/chat",
    data=data,
    headers={"Content-Type": "application/json"},
  )
  start = time.perf_counter()
  with urllib.request.urlopen(req, timeout=600) as resp:
    result = json.loads(resp.read())
  elapsed_ms = (time.perf_counter() - start) * 1000
  message = result.get("message", {})
  response = message.get("content", "") or message.get("thinking", "")
  return response, {
    "eval_count": result.get("eval_count", 0),
    "generation_time_ms": round(elapsed_ms, 2),
  }


def find_patterns(text: str, patterns: list[str]) -> list[str]:
  text_lower = text.lower()
  hits = []
  for p in patterns:
    if re.search(p, text_lower, re.IGNORECASE):
      hits.append(p)
  return hits


def score_response(test: dict, response: str) -> LoreResult:
  fim_hits = find_patterns(response, test["fim_patterns"])
  eg_hits = find_patterns(response, test["eg_patterns"])
  trait_hits = find_patterns(response, test["trait_patterns"])

  fail_reasons = []
  if eg_hits:
    fail_reasons.append(f"Equestria Girls contamination: {eg_hits}")
  if len(fim_hits) < 2:
    fail_reasons.append(f"Insufficient FiM lore (only {len(fim_hits)} hits, need >=2)")
  if len(trait_hits) < 1:
    fail_reasons.append(f"Missing character traits (0 trait hits)")

  if eg_hits:
    verdict = "FAIL"
  elif len(fim_hits) >= 3 and len(trait_hits) >= 1:
    verdict = "PASS"
  elif len(fim_hits) >= 2 and len(trait_hits) >= 1:
    verdict = "PARTIAL"
  else:
    verdict = "FAIL"

  return LoreResult(
    model="",
    character_id=test["id"],
    character=test["character"],
    prompt=test["prompt"],
    response=response.strip(),
    verdict=verdict,
    fim_hits=fim_hits,
    eg_hits=eg_hits,
    trait_hits=trait_hits,
    fim_score=len(fim_hits),
    fail_reasons=fail_reasons,
  )


def check_model_available(model: str) -> bool:
  try:
    payload = json.dumps({
      "model": model,
      "messages": [{"role": "user", "content": "Hi"}],
      "stream": False,
      "think": False,
      "options": {"num_predict": 5},
    }).encode()
    req = urllib.request.Request(
      f"{OLLAMA_HOST}/api/chat",
      data=payload,
      headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
      json.loads(resp.read())
    return True
  except Exception as e:
    print(f"  Model {model} not available: {e}")
    return False


def run_tests(models: list[str]) -> list[LoreResult]:
  all_results = []
  for model in models:
    print(f"\n{'='*60}")
    print(f"Testing: {model}")
    print(f"{'='*60}")
    if not check_model_available(model):
      continue
    for test in CHARACTER_TESTS:
      system = SYSTEM_INSTRUCTION.format(character=test["character"])
      print(f"  [{test['character']}] generating...")
      response, stats = generate(model, system, test["prompt"], test["character"])
      result = score_response(test, response)
      result.model = model
      result.generation_time_ms = stats["generation_time_ms"]
      result.eval_count = stats["eval_count"]
      all_results.append(result)
      print(f"    [{result.verdict}] FiM={len(result.fim_hits)} EG={len(result.eg_hits)} traits={len(result.trait_hits)}")
      print(f"    {response[:150].replace(chr(10), ' ')}...")
  return all_results


def summarize(results: list[LoreResult]) -> dict:
  summary = {}
  for model in sorted(set(r.model for r in results)):
    model_results = [r for r in results if r.model == model]
    passes = sum(1 for r in model_results if r.verdict == "PASS")
    partials = sum(1 for r in model_results if r.verdict == "PARTIAL")
    fails = sum(1 for r in model_results if r.verdict == "FAIL")
    eg_contam = sum(1 for r in model_results if r.eg_hits)
    avg_fim = round(sum(r.fim_score for r in model_results) / len(model_results), 2) if model_results else 0
    avg_time = round(sum(r.generation_time_ms for r in model_results) / len(model_results), 2) if model_results else 0
    summary[model] = {
      "total": len(model_results),
      "pass": passes,
      "partial": partials,
      "fail": fails,
      "eg_contamination_count": eg_contam,
      "avg_fim_hits": avg_fim,
      "avg_generation_ms": avg_time,
      "pass_rate": round((passes + 0.5 * partials) / len(model_results) * 100, 1) if model_results else 0,
    }
  return summary


def main():
  print("MLP FiM Lore Coherency Benchmark")
  print(f"Ollama host: {OLLAMA_HOST}")
  print(f"Models requested: {', '.join(MODELS)}")

  available = []
  for m in MODELS:
    try:
      req = urllib.request.Request(f"{OLLAMA_HOST}/api/tags")
      with urllib.request.urlopen(req, timeout=10) as resp:
        tags = json.loads(resp.read())
      names = [t["name"] for t in tags.get("models", [])]
      if m in names:
        available.append(m)
        print(f"  Found: {m}")
      else:
        print(f"  Missing: {m}")
    except Exception as e:
      print(f"  Error checking models: {e}")
      break

  if not available:
    print("No models available. Pull models first.")
    return 1

  results = run_tests(available)
  summary = summarize(results)

  output = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "ollama_host": OLLAMA_HOST,
    "models_requested": MODELS,
    "models_tested": available,
    "results": [asdict(r) for r in results],
    "summary": summary,
  }

  out_path = "/home/amreschizo/gemma test/results/mlp_lore_results.json"
  with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
  print(f"\nResults saved to {out_path}")
  print("\n=== SUMMARY ===")
  print(json.dumps(summary, indent=2))
  return 0


if __name__ == "__main__":
  sys.exit(main())
