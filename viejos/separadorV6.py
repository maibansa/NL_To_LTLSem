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
    # 1) every – obligacion periodica (solo accion)
    "Patients with diabetes should receive a glucose test every morning.",

    # 2) eventually – condicion -> accion futura
    "If a patient has fever, they should eventually receive paracetamol.",

    # 3) within – condicion -> respuesta acotada en el tiempo
    "If the patient has hypertension, they must receive amlodipine within 24 hours.",

    # 4) within – ventana corta
    "If an allergy is detected, epinephrine must be administered within 5 minutes.",

    # 5) eventually – sin limite temporal
    "If a patient reports pain, they should eventually receive analgesia.",

    # 6) every – periodicidad con accion clinica
    "Vital signs should be recorded every 4 hours.",

    # 7) always – invariante condicionada
    "If a patient is in the ICU, monitoring must always be active.",

    # 8) within – reaccion rapida
    "If oxygen saturation drops, supplemental oxygen must be provided within 10 minutes.",

    # 9) until – mantenimiento hasta evento
    "The patient should remain fasting until surgery is performed.",

    # 10) eventually – seguimiento
    "If a laboratory result is abnormal, a follow-up test should eventually be performed."
]


# ============================================================
# CONFIG
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"
LOG_ONTOLOGY_FILE = "logonto.json"
OLLAMA_TIMEOUT = 300

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
# PROMPT STEP 1
# ============================================================

def build_prompt(text: str, log_ontology: Dict) -> str:
    onto = json.dumps(log_ontology, indent=2)
    return f"""
You are a rule interpreter.

Tasks:
1) Extract the temporal structure using ONLY the log ontology.
2) Extract propositions as identifiers (no semantics).

LOG ONTOLOGY:
{onto}

Clinical rule:
"{text}"

Return ONLY valid JSON in this structure:

{{
  "temporalStructure": {{
    "operator": "always|eventually|within|until|every",
    "bound": {{ "value": number, "unit": string }},
    "anchorField": "nTimestamp"
  }},
  "propositions": [
    {{
      "predicate": "hasCondition|receiveDrug|performTest",
      "concept": "IDENTIFIER",
      "role": "condition|action"
    }}
  ]
}}
"""

# ============================================================
# OLLAMA CALL
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
# PARSE STEP 1
# ============================================================

def parse_step1_ir(llm_output: str) -> Optional[Step1IR]:
    try:
        start = llm_output.find("{")
        end = llm_output.rfind("}")
        raw = llm_output[start:end + 1]
        raw = re.sub(r",\s*([}\]])", r"\1", raw)
        data = json.loads(raw)

        if not isinstance(data.get("temporalStructure"), dict):
            return None
        if not isinstance(data.get("propositions"), list):
            return None

        return Step1IR(**data)

    except (ValidationError, json.JSONDecodeError):
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
# STEP 1.5 – LTL / DLTL
# (UNICO CAMBIO: manejar 'every' antes del chequeo)
# ============================================================

def proposition_to_ltl_placeholder(p: Proposition, var: str) -> str:
    return f"PROP.{p.predicate}({var})"


def step15_to_ltl(ir: Step1IR) -> str:
    conds = [p for p in ir.propositions if p.role == "condition"]
    acts  = [p for p in ir.propositions if p.role == "action"]

    x = "x"
    y = "y"
    t = ir.temporalStructure

    # ✅ CASO ESPECIAL: every NO requiere condicion
    if t.operator == "every":
        if not acts:
            raise ValueError("Every requires at least one action")
        act_ltl = " & ".join(proposition_to_ltl_placeholder(p, y) for p in acts)
        return f"G F {y}.({act_ltl})"

    # ⬇️ RESTO DE OPERADORES SI requieren condicion y accion
    if not conds or not acts:
        raise ValueError("Need at least one condition and one action")

    cond_ltl = " & ".join(proposition_to_ltl_placeholder(p, x) for p in conds)
    act_ltl  = " & ".join(proposition_to_ltl_placeholder(p, y) for p in acts)

    if t.operator == "within":
        seconds = t.bound.value
        if t.bound.unit == "hours":
            seconds *= 3600
        elif t.bound.unit == "minutes":
            seconds *= 60
        elif t.bound.unit == "days":
            seconds *= 86400

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

    if t.operator == "until":
        return f"({cond_ltl} U {act_ltl})"

    raise ValueError(f"Unsupported temporal operator {t.operator}")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("Running NL -> IR -> LTL (temporal only, semantic placeholders)\n")

    log_ontology = load_log_ontology(LOG_ONTOLOGY_FILE)

    for i, sentence in enumerate(TEST_SENTENCES, start=1):
        print("=" * 70)
        print(f"Example {i}")
        print("Input:")
        print(sentence)

        ir = step1(sentence, log_ontology)

        if ir is None:
            print("Step 1 failed (no propositions or no temporal structure)")
            continue

        print("\nStep 1 IR (JSON):")
        print(ir.model_dump_json(indent=2))

        ltl = step15_to_ltl(ir)

        print("\nGenerated LTL/DLTL:")
        print(ltl)

    print("\nDone.")