# coding: utf-8

import json
import re
import requests
from typing import List, Optional, Dict
from pydantic import BaseModel, ValidationError, field_validator

# ============================================================
# TEST SENTENCES (AL PRINCIPIO)
# ============================================================

TEST_SENTENCES = [
    "If the patient has hypertension, they must receive amlodipine within 24 hours.",
    "Patients with diabetes should receive a glucose test every morning.",
    "If a patient has fever, they should eventually receive paracetamol."
]

# ============================================================
# CONFIG
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"
LOG_ONTOLOGY_FILE = "logonto.json"

# ============================================================
# MODELS
# ============================================================

class TemporalBound(BaseModel):
    value: float
    unit: str


class TemporalStructure(BaseModel):
    operator: str
    bound: Optional[TemporalBound] = None
    anchorField: str

    @field_validator("operator")
    @classmethod
    def validate_operator(cls, v):
        allowed = {"always", "eventually", "within", "until", "every"}
        if v not in allowed:
            raise ValueError("Unsupported temporal operator")
        return v


class Proposition(BaseModel):
    predicate: str
    concept: str
    role: str


class Step1IR(BaseModel):
    temporalStructure: TemporalStructure
    propositions: List[Proposition]

# ============================================================
# LOAD LOG ONTOLOGY
# ============================================================

def load_log_ontology(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ============================================================
# PROMPT
# ============================================================

def build_prompt(text: str, log_ontology: Dict) -> str:
    onto = json.dumps(log_ontology, indent=2)
    return f"""
You are a semantic interpreter for clinical rules.

There are TWO STRICTLY SEPARATED tasks:

1) TEMPORAL STRUCTURE (use ONLY the log ontology)
2) CLINICAL PROPOSITIONS (use SNOMED-like identifiers)

LOG ONTOLOGY:
{onto}

Clinical rule:
"{text}"

Return ONLY valid JSON in this exact structure:

{{
  "temporalStructure": {{
    "operator": "always|eventually|within|until|every",
    "bound": {{ "value": number, "unit": string }},
    "anchorField": "nTimestamp"
  }},
  "propositions": [
    {{
      "predicate": "hasCondition|receiveDrug|performTest",
      "concept": "SNOMED_ID",
      "role": "condition|action"
    }}
  ]
}}
"""

# ============================================================
# OLLAMA CALL
# ============================================================

def call_ollama(prompt: str) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        },
        timeout=120
    )
    response.raise_for_status()
    return response.json()["response"]

# ============================================================
# PARSE STEP 1
# ============================================================

def parse_step1_ir(llm_output: str) -> Optional[Step1IR]:
    try:
        start = llm_output.find("{")
        end = llm_output.rfind("}")
        raw = llm_output[start:end + 1]
        raw = re.sub(r",\s*([}\]])", r"\1", raw)
        data = json.loads(raw)
        return Step1IR(**data)

    except (ValidationError, json.JSONDecodeError, ValueError) as e:
        print("Step 1 parse error:", e)
        print("LLM output:")
        print(llm_output)
        return None

# ============================================================
# STEP 1
# ============================================================

def step1(text: str, log_ontology: Dict) -> Optional[Step1IR]:
    prompt = build_prompt(text, log_ontology)
    output = call_ollama(prompt)
    return parse_step1_ir(output)

# ============================================================
# STEP 1.5 - LTL
# ============================================================

def propositions_to_atoms(props: List[Proposition]) -> List[str]:
    atoms = []
    for p in props:
        atoms.append(f"{p.predicate}_{p.concept}".lower())
    return atoms


def temporal_to_ltl(t: TemporalStructure, atoms: List[str]) -> str:
    if len(atoms) < 2:
        return "G " + atoms[0]

    c = atoms[0]
    a = atoms[1]

    if t.operator == "within":
        return f"G ({c} -> F<={t.bound.value}{t.bound.unit} {a})"

    if t.operator == "always":
        return f"G ({c} -> {a})"

    if t.operator == "eventually":
        return f"G ({c} -> F {a})"

    if t.operator == "every":
        return f"G ({c} -> F {a})"

    if t.operator == "until":
        return f"({c} U {a})"

    raise ValueError("Unsupported operator")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("Running NL -> IR -> LTL with Ollama\n")

    log_ontology = load_log_ontology(LOG_ONTOLOGY_FILE)

    for i, sentence in enumerate(TEST_SENTENCES, start=1):
        print("=" * 70)
        print(f"Example {i}")
        print("Input:")
        print(sentence)

        ir = step1(sentence, log_ontology)

        if ir is None:
            print("Step 1 failed")
            continue

        print("\nStep 1 IR:")
        print(ir.model_dump_json(indent=2))

        atoms = propositions_to_atoms(ir.propositions)
        ltl = temporal_to_ltl(ir.temporalStructure, atoms)

        print("\nGenerated LTL:")
        print(ltl)

    print("\nDone.")