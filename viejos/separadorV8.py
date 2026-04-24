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
You are a rule interpreter. You MUST follow the instructions exactly.

TASK 1 — TEMPORAL STRUCTURE
- Always extract a temporal operator using ONLY the log ontology.

TASK 2 — PROPOSITIONS
- ALWAYS return at least one proposition.
- IMPORTANT: If a sentence mentions a target group (e.g. "Patients with diabetes"), it MUST be role="condition".
- Propositions may describe CONDITIONS, ACTIONS, or STATES.


TASK 3 — ONTOLOGY MAPPING RULES (STRICT):
You MUST map every extracted term to exactly one key from the LogSchemaOntology. Follow these priority rules:

1. CLINICAL DATA & DIAGNOSES (Predicate: "gSnomed")
   - Use for: Diseases, symptoms, clinical findings, or physiological measurements.
   - Examples: "diabetes", "fever", "hypertension", "allergy", "oxygen saturation".

2. MEDICATIONS & DISPENSING (Predicate: "aActionType")
   - Use for: Specific drugs, dosages, or the act of administering a substance.
   - Examples: "paracetamol", "amlodipine", "epinephrine", "insulin administration".

3. PROCEDURES & CLINICAL TASKS (Predicate: "aActivity")
   - Use for: Specific diagnostic tests, nursing tasks, or medical interventions.
   - Examples: "glucose test", "vital signs record", "surgery", "physical examination".

4. ACTORS & ROLES (Predicate: "aActor")
   - Use for: The professional or entity performing the action.
   - Examples: "nurse", "doctor", "cardiologist", "laboratory system".

5. PROCESSES & CONTEXTS (Predicate: "aWorkflow")
   - Use for: The general clinical protocol or pathway being followed.
   - Examples: "triage", "post-operative care", "chronic patient management".

6. LOCATIONS & FORMAL REFERENCES (Predicate: "sModelReference")
   - Use for: Physical locations, clinical units, or references to external formal models.
   - Examples: "ICU", "emergency room", "ward 3", "HL7_standard_ref".
   
STATE RULES:
- Some rules describe STATES, not actions (e.g., "monitoring active"). Role = "state".

IMPORTANT NORMALIZATION RULES:
- "every morning", "every X hours", "daily" MUST be operator="every".
- Periodic rules ("every") MUST return at least one ACTION proposition.
- Never skip the "If" part of the sentence. Example: "If a patient has fever, receive paracetamol" -> 1 Condition (fever) and 1 Action (paracetamol).
- If the sentence mentions a specific disease or condition (e.g. 'diabetes'), 
   that MUST be the concept of the 'condition' proposition.DO NOT use generic terms like 'aWorkflow' if a specific disease is mentioned.
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
    {{ "predicate": "IDENTIFIER", "concept": "IDENTIFIER", "role": "condition|action|state" }}
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
        return "condition" if p.role == "condition" else "action"

    conds = [p for p in ir.propositions if norm_role(p) == "condition"]
    acts  = [p for p in ir.propositions if norm_role(p) == "action"]

    x, y = "x", "y"
    t = ir.temporalStructure

    def get_seconds():
        if not t.bound: return 0
        mult = {"hours": 3600, "minutes": 60, "days": 86400, "day": 86400}
        return int(t.bound.value * mult.get(t.bound.unit.lower(), 1))

    # 1) every
    if t.operator == "every":
        if not acts: raise ValueError("Every requires an action")
        act_ltl = " & ".join(proposition_to_ltl_placeholder(p, y) for p in acts)
        if conds:
            cond_ltl = " & ".join(proposition_to_ltl_placeholder(p, x) for p in conds)
            return f"G {x}.({cond_ltl} -> F {y}.({act_ltl}))"
        return f"G F {y}.({act_ltl})"

    # 2) within
    if t.operator == "within":
        seconds = get_seconds()
        cond_ltl = " & ".join(proposition_to_ltl_placeholder(p, x) for p in conds) if conds else "TRUE"
        act_ltl = " & ".join(proposition_to_ltl_placeholder(p, y) for p in acts)
        return f"G {x}.({cond_ltl} -> X F {y}.({act_ltl} & ({y}[Timestamp] - {x}[Timestamp] <= {seconds})))"

    # 3) eventually
    if t.operator == "eventually":
        cond_ltl = " & ".join(proposition_to_ltl_placeholder(p, x) for p in conds) if conds else "TRUE"
        act_ltl = " & ".join(proposition_to_ltl_placeholder(p, y) for p in acts)
        return f"G {x}.({cond_ltl} -> F {y}.({act_ltl}))"

    # 4) until
    if t.operator == "until":
        cond_ltl = " & ".join(proposition_to_ltl_placeholder(p, x) for p in conds) if conds else "TRUE"
        act_ltl = " & ".join(proposition_to_ltl_placeholder(p, y) for p in acts) if acts else "TRUE"
        return f"({cond_ltl} U {act_ltl})"

    # 5) always
    if t.operator == "always":
        cond_ltl = " & ".join(proposition_to_ltl_placeholder(p, x) for p in conds) if conds else "TRUE"
        act_ltl = " & ".join(proposition_to_ltl_placeholder(p, x) for p in acts) # Mismo tiempo x
        return f"G {x}.({cond_ltl} -> {act_ltl})"

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