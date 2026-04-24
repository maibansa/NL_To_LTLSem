# coding: utf-8

import json
import re
import requests
from typing import List, Optional, Dict
from pydantic import BaseModel, ValidationError, field_validator

# ============================================================
# TEST SENTENCES
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
OLLAMA_TIMEOUT = 300   # necesario para llama3

# ============================================================
# MODELS (TU VERSIÓN BUENA)
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
    role: str   # condition | action


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
# PROMPT (SIN CAMBIOS CONCEPTUALES)
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
# OLLAMA CALL (ROBUSTO)
# ============================================================

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
        print("OLLAMA FAILED:", e)
        return None

# ============================================================
# PARSE STEP 1 (DEFENSIVO)
# ============================================================

def parse_step1_ir(llm_output: str) -> Optional[Step1IR]:
    try:
        start = llm_output.find("{")
        end = llm_output.rfind("}")
        raw = llm_output[start:end + 1]
        raw = re.sub(r",\s*([}\]])", r"\1", raw)
        data = json.loads(raw)

        if not isinstance(data.get("temporalStructure"), dict):
            raise ValueError("temporalStructure is not an object")

        if not isinstance(data.get("propositions"), list):
            raise ValueError("propositions is not a list")

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

    if output is None:
        return None

    return parse_step1_ir(output)

# ============================================================
# STEP 1.5 – LTL/DLTL SEGÚN logonto NUEVA
# ============================================================

def proposition_to_ltl(p: Proposition, var: str) -> str:
    """
    Todas las proposiciones clínicas son gSnomed (graph),
    por tanto SIEMPRE bajo variable de evento.
    """
    return f'({var})PROP.HasSnomedCode({var}[Snomed], "{p.concept}")'


def step15_to_ltl(ir: Step1IR) -> str:
    conds = [p for p in ir.propositions if p.role == "condition"]
    acts  = [p for p in ir.propositions if p.role == "action"]

    if not conds or not acts:
        raise ValueError("Need at least one condition and one action")

    x = "x"
    y = "y"

    cond_ltl = " & ".join(proposition_to_ltl(p, x) for p in conds)
    act_ltl  = " & ".join(proposition_to_ltl(p, y) for p in acts)

    t = ir.temporalStructure

    if t.operator == "within":
        seconds = t.bound.value
        if t.bound.unit == "hours":
            seconds *= 3600
        elif t.bound.unit == "minutes":
            seconds *= 60

        time_ltl = f"({y}[Timestamp] - {x}[Timestamp] <= {int(seconds)})"

        return (
            f"G {x}.("
            f"{cond_ltl} -> "
            f"X F {y}.({act_ltl} & {time_ltl})"
            f")"
        )

    if t.operator == "eventually":
        return f"G {x}.({cond_ltl} -> F {y}.({act_ltl}))"

    if t.operator == "always":
        return f"G {x}.({cond_ltl} -> {act_ltl})"

    raise ValueError(f"Unsupported temporal operator {t.operator}")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("Running NL -> IR -> quantified LTL using logonto\n")

    log_ontology = load_log_ontology(LOG_ONTOLOGY_FILE)

    for i, sentence in enumerate(TEST_SENTENCES, start=1):
        print("=" * 70)
        print(f"Example {i}")
        print("Input:")
        print(sentence)

        try:
            ir = step1(sentence, log_ontology)
        except Exception as e:
            print("LLM crashed on this sentence:", e)
            continue

        if ir is None:
            print("Step 1 failed (LLM output invalid)")
            continue

        print("\nStep 1 IR:")
        print(ir.model_dump_json(indent=2))

        try:
            ltl = step15_to_ltl(ir)
        except Exception as e:
            print("LTL generation failed:", e)
            continue

        print("\nGenerated LTL/DLTL:")
        print(ltl)

    print("\nDone.")