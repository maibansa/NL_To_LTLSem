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
    "Vital signs should be recorded every 4 hours.",
    "Patients with diabetes should receive a glucose test every morning.",
    "If a patient has fever, they should eventually receive paracetamol.",
    "If the patient has hypertension, they must receive amlodipine within 24 hours.",
    "If an allergy is detected, epinephrine must be administered within 5 minutes.",
    "If a patient reports pain, they should eventually receive analgesia.",
    "Vital signs should be recorded every 4 hours.",
    "If a patient is in the ICU, monitoring must always be active.",
    "If oxygen saturation drops, supplemental oxygen must be provided within 10 minutes.",
    "The patient should remain fasting until surgery is performed.",
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
    role: str   # condition | action | state

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
# PROMPT STEP 1 (AJUSTADO PARA PRECISIÓN)
# ============================================================

def build_prompt(text: str, log_ontology: Dict) -> str:
    onto = json.dumps(log_ontology, indent=2)
    return f"""
You are a rule interpreter.
You MUST follow the instructions exactly.

==================================================================
TASK 1 — TEMPORAL STRUCTURE
==================================================================
- Always extract a temporal operator using ONLY the log ontology.
- Do NOT drop or simplify temporal information.
- Preserve all temporal bounds and units.

==================================================================
TASK 2 — PROPOSITIONS
==================================================================
- ALWAYS return at least one proposition.
- Propositions may describe CONDITIONS, ACTIONS, or STATES.

IMPORTANT:
- If a sentence mentions a target group (e.g. "Patients with diabetes"),
  it MUST be returned as a proposition with role = "condition".
- Never skip information introduced by an "If" clause.


==================================================================
TASK 3 — ONTOLOGY MAPPING RULES (STRICT)
==================================================================
You MUST map every extracted term to EXACTLY ONE key from the
LogSchemaOntology.
NO free interpretation is allowed.

==================================================================
STATE RULES
==================================================================
- Some rules describe STATES, not actions.
- Example: "monitoring active"
- Such propositions MUST have role = "state".

==================================================================
IMPORTANT NORMALIZATION RULES
==================================================================
- "every morning", "every X hours", "daily" MUST map to operator = "every".
- Periodic rules ("every") MUST return at least one ACTION proposition.
- Never skip the "If" part of the sentence.


Example:
"If a patient has fever, receive paracetamol"
→ 1 CONDITION (fever)
→ 1 ACTION (paracetamol)

- If the sentence mentions a specific disease or condition
  (e.g. "diabetes"), that MUST be the concept of the CONDITION
  proposition.
- DO NOT use generic predicates like "aWorkflow" if a specific
  disease or condition is mentioned.

==================================================================
LOG ONTOLOGY (REFERENCE)
==================================================================
{onto}

==================================================================
CLINICAL RULE
==================================================================
\"{text}\"

==================================================================
OUTPUT FORMAT (STRICT)
==================================================================
Return ONLY valid JSON in the following structure:

{{
  "temporalStructure": {{
    "operator": "always|eventually|within|until|every",
    "bound": {{ "value": number, "unit": string }},
    "anchorField": "nTimestamp"
  }},
  "propositions": [
    {{
      "predicate": "IDENTIFIER",
      "concept": "IDENTIFIER",
      "role": "condition|action|state"
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
# STEP 1.5 – LTL / DLTL (LOGICA REFINADA)
# ============================================================

def proposition_to_ltl_placeholder(p: Proposition, var: str) -> str:
    return f"PROP.{p.predicate}({var})"

def step15_to_ltl(ir: Step1IR) -> str:

    def norm_role(p: Proposition) -> str:
        # Todo lo que no sea condition se considera accion/estado
        return "condition" if p.role == "condition" else "action"

    conds = [p for p in ir.propositions if norm_role(p) == "condition"]
    acts  = [p for p in ir.propositions if norm_role(p) == "action"]

    x, y = "x", "y"
    t = ir.temporalStructure
    ts = t.anchorField  # p.ej. nTimestamp

    # -------------------------------
    # utilidades temporales
    # -------------------------------
    def get_seconds():
        if not t.bound:
            return None
        mult = {
            "seconds": 1,
            "minutes": 60,
            "hours": 3600,
            "days": 86400,
            "day": 86400
        }
        unit = t.bound.unit.lower()
        return int(t.bound.value * mult.get(unit, 1))

    def time_diff(x, y):
        return f"({y}[{ts}] - {x}[{ts}])"

    def cond_ltl(var):
        if not conds:
            return "TRUE"
        return " & ".join(proposition_to_ltl_placeholder(p, var) for p in conds)

    def act_ltl(var):
        if not acts:
            return "TRUE"
        return " & ".join(proposition_to_ltl_placeholder(p, var) for p in acts)

    # ======================================================
    # 1) every (PERIODICIDAD CUANTITATIVA)
    # ======================================================
    if t.operator == "every":
        seconds = get_seconds()
        base = act_ltl(y)

        # sin bound: periodicidad cualitativa
        if seconds is None or seconds == 0:
            if conds:
                return f"G {x}.({cond_ltl(x)} -> F {y}.({base}))"
            return f"G F {y}.({base})"

        # con bound: periodicidad cuantificada
        if conds:
            return (
                f"G {x}.({cond_ltl(x)} -> F {y}.("
                f"{base} & {time_diff(x, y)} <= {seconds}))"
            )
        return (
            f"G {x}.(F {y}.("
            f"{base} & {time_diff(x, y)} <= {seconds}))"
        )

    # ======================================================
    # 2) within (RESPUESTA ACOTADA)
    # ======================================================
    if t.operator == "within":
        seconds = get_seconds()
        if seconds is None:
            raise ValueError("within requires a temporal bound")

        return (
            f"G {x}.({cond_ltl(x)} -> X F {y}.("
            f"{act_ltl(y)} & {time_diff(x, y)} <= {seconds}))"
        )

    # ======================================================
    # 3) eventually (RESPUESTA NO ACOTADA)
    # ======================================================
    if t.operator == "eventually":
        return f"G {x}.({cond_ltl(x)} -> F {y}.({act_ltl(y)}))"

    # ======================================================
    # 4) always (INVARIANTE / ESTADO)
    # ======================================================
    if t.operator == "always":
        # estado/accion ocurre en el mismo instante x
        return f"G {x}.({cond_ltl(x)} -> {act_ltl(x)})"

    # ======================================================
    # 5) until (MANTENIMIENTO HASTA EVENTO)
    # ======================================================
    if t.operator == "until":
        # cond se mantiene hasta que ocurre act
        return f"({cond_ltl(x)} U {act_ltl(y)})"

    raise ValueError(f"Unsupported temporal operator {t.operator}")
# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("Running NL -> IR -> LTL\n")
    log_ontology = load_log_ontology(LOG_ONTOLOGY_FILE)

    for i, sentence in enumerate(TEST_SENTENCES, start=1):
        print("=" * 70)
        print(f"Example {i}: {sentence}")

        ir = step1(sentence, log_ontology)

        if ir is None:
            print("Step 1 failed")
            continue

        print("\nStep 1 IR:")
        print(ir.model_dump_json(indent=2))

        try:
            ltl = step15_to_ltl(ir)
            print("\nGenerated LTL/DLTL:")
            print(ltl)
        except Exception as e:
            print(f"\nLTL Generation error: {e}")

    print("\nDone.")