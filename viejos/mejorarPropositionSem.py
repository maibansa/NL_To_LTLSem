"""
This script implements a SNOMED CT–aligned semantic typing pipeline for a Natural Language to Temporal Logic (NL-to-LTL) system.

It performs the following steps:

1. Loads a compressed SNOMED-like ontology (SNOMEDTop.json), which defines a small set of high-level clinical semantic types:
   - ClinicalFinding
   - Procedure
   - Substance
   - ObservableEntity
   - Situation
   - Event

2. Loads a dataset of processed clinical sentences (results_output.json), where each entry contains:
   - the original sentence
   - an intermediate representation (IR)
   - extracted propositions, including "graph" type concepts requiring semantic typing

3. For each "graph" concept in the IR:
   - Sends the sentence, concept, and ontology to a local LLM (Ollama running LLaMA 3)
   - The model is constrained to select exactly ONE valid type from the ontology
   - The output is forced into JSON format: {"type": "<selected_type>"}

4. The returned type is validated against the ontology:
   - If valid → added as "snomedType"
   - If invalid → marked as "Unknown"

5. The script logs progress to the console:
   - sentence being processed
   - concept being classified
   - predicted SNOMED type

6. The enriched dataset is saved to results_output_typed.json.

Overall, this pipeline implements an ontology-constrained LLM-based semantic grounding layer that maps extracted clinical concepts to SNOMED CT top-level categories, improving downstream reasoning, SPARQL grounding, and temporal logic generation.
"""
import json
import requests
from typing import Optional

# ===============================
# CONFIG
# ===============================
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"
OLLAMA_TIMEOUT = 120

SNOMED_FILE = "SNOMEDTop.json"
INPUT_FILE = "results_output.json"
OUTPUT_FILE = "results_output_typed.json"


# ===============================
# OLLAMA CALL
# ===============================
def call_ollama(prompt: str) -> Optional[str]:
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json"
            },
            timeout=OLLAMA_TIMEOUT
        )
        response.raise_for_status()
        return response.json()["response"]
    except Exception as e:
        print("❌ OLLAMA FAILED:", e)
        return None


# ===============================
# LOAD DATA
# ===============================
print("📦 Loading SNOMED ontology...")
with open(SNOMED_FILE, "r", encoding="utf-8") as f:
    snomed = json.load(f)

print("📦 Loading dataset...")
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

ALLOWED_TYPES = [c["type"] for c in snomed["concepts"]]

print(f"✅ Ontology loaded: {len(ALLOWED_TYPES)} types")
print(f"✅ Dataset loaded: {len(data)} examples")


# ===============================
# CLASSIFIER
# ===============================
def classify(sentence: str, concept: str, ontology: dict) -> str:
    prompt = f"""
You are a clinical ontology classifier.

You MUST select EXACTLY ONE type from the ontology below.

Rules:
- Only choose from allowed types
- Do NOT invent new types
- Return ONLY JSON: {{"type": "<one_type>"}}

-------------------------
ONTOLOGY:
{json.dumps(ontology, indent=2)}

-------------------------
INPUT:
Sentence: {sentence}
Concept: {concept}
"""

    raw = call_ollama(prompt)

    if raw is None:
        return "Unknown"

    try:
        result = json.loads(raw)
        t = result.get("type", "Unknown")
    except Exception:
        # fallback matching
        for t in ALLOWED_TYPES:
            if t.lower() in raw.lower():
                return t
        return "Unknown"

    if t in ALLOWED_TYPES:
        return t

    return "Unknown"


# ===============================
# PROCESS PIPELINE
# ===============================
print("\n🚀 Starting SNOMED typing pipeline...\n")

for i, item in enumerate(data):
    sentence = item.get("sentence", "")
    props = item.get("ir", {}).get("propositions", [])

    print(f"\n🔹 [{i+1}/{len(data)}] Sentence:")
    print(f"   📝 {sentence}")

    for prop in props:
        if prop.get("type") == "graph":
            concept = prop.get("concept", "")

            print(f"   🔎 Concept: {concept}")

            snomed_type = classify(sentence, concept, snomed)

            print(f"   🧠 SNOMED type: {snomed_type}")

            prop["snomedType"] = snomed_type

    print("   ✔ Finished item")


# ===============================
# SAVE OUTPUT
# ===============================
print("\n💾 Saving enriched dataset...")
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"🎉 Done! Saved to {OUTPUT_FILE}")