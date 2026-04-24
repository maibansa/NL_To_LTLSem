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
LTL_ONTOLOGY_FILE = "ltlonto.json" 
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
        # Se permite el mapeo dinámico a las tKeys de la ontología lógica
        return v

class Proposition(BaseModel):
    predicate: str
    type: str  # <-
    concept: str    
    role: str   # condition | action | state

class Step1IR(BaseModel):
    temporalStructure: TemporalStructure
    propositions: List[Proposition]

# ============================================================
# LOAD ONTOLOGIES
# ============================================================

def load_ontology(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ============================================================
# PROMPT STEP 1 (NUEVA ARQUITECTURA DE CAPAS)
# ============================================================

def build_prompt(text: str, log_ontology: Dict, ltl_ontology: Dict) -> str:
    onto_semantic = json.dumps(log_ontology, indent=2)
    onto_logic = json.dumps(ltl_ontology, indent=2)
    
    return f"""
Act as a Clinical Rule Compiler. 

==================================================================
CRITICAL ARCHITECTURAL REQUIREMENT: DUAL-LAYER SEPARATION
==================================================================
You MUST strictly separate the extraction into two independent layers before generating the JSON:

1. THE TEMPORAL LOGIC LAYER (The "When"): 
   Determines the formal structure of the rule. You MUST use the LOG ONTOLOGY to identify the operator, type of operator and its logic pattern.
   The 'operator' in the output MUST be one of the keys from LtlLogicOntology (e.g., tEvery, tWithin, tUntil, tAlways, tOnce, etc.).

2. THE DOMAIN SEMANTIC LAYER (The "What"): 
   Identifies clinical concepts, tasks, and locations. You MUST use the 'LogSchemaOntology' to map predicates and roles.

NEVER mix clinical concepts into the temporalStructure, and NEVER mix temporal constraints into the propositions.
==================================================================

==================================================================
ANTI-HALLUCINATION PROTOCOL (STRICT COMPLIANCE)
==================================================================
- NO CALCULATIONS: Do NOT convert time to seconds or minutes. Extract the literal value and unit from the text (e.g., "4 hours" -> value: 4, unit: "hours").
- NO INVENTED OPERATORS: If the text says "within", you MUST use 'tWithin'. If it says "until", use 'tUntil'. Never use 'tUntil' for a time deadline.
- NO INVENTED PREDICATES: You are strictly forbidden from creating new predicates. You MUST choose ONLY from the keys in LogSchemaOntology (e.g., gSnomed, aActivity, etc.).
- NULL FOR ABSENT DATA: If no temporal bound is mentioned, "bound" MUST be null. Do NOT invent default values like 3600s.
==================================================================

### REFERENCE ONTOLOGIES
1. LOG ONTOLOGY (Structure & Operators):
{onto_logic}

2. SEMANTIC ONTOLOGY (Clinical Concepts):
{onto_semantic}

### OUTPUT FORMAT (STRICTLY FOLLOW THIS STRUCTURE)
Return ONLY valid JSON in the following structure:

{{
  "temporalStructure": {{
    "operator": "tKey_from_LtlLogicOntology",
    "bound": {{ "value": number, "unit": "seconds|minutes|hours|days" }},
    "anchorField": "nTimestamp"
  }},
  "propositions": [
    {{
      "predicate": "identifier_from_LogSchemaOntology (e.g., gSnomed, aActivity, etc.)",
      "type": "the 'type' value associated with the predicate in LogSchemaOntology",(e.g., graph, atomic, string)",
      "concept": "IDENTIFIER",
      "role": "condition|action|state"
    }}
  ]
}}

### CLINICAL RULE TO ANALYZE:
"{text}"
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

def step1(text: str, log_ontology: Dict, ltl_ontology: Dict) -> Optional[Step1IR]:
    prompt = build_prompt(text, log_ontology, ltl_ontology)
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
    ts = t.anchorField

    def get_seconds():
        if not t.bound:
            return None
        mult = {
            "seconds": 1, "minutes": 60, "hours": 3600, "days": 86400, "day": 86400
        }
        unit = t.bound.unit.lower()
        return int(t.bound.value * mult.get(unit, 1))

    def time_diff(x, y):
        return f"({y}[{ts}] - {x}[{ts}])"

    def cond_ltl(var):
        if not conds: return "TRUE"
        return " & ".join(proposition_to_ltl_placeholder(p, var) for p in conds)

    def act_ltl(var):
        if not acts: return "TRUE"
        return " & ".join(proposition_to_ltl_placeholder(p, var) for p in acts)

    # Normalización del operador (mapeo tKey -> lógica)
    op = t.operator.lower()

    if "every" in op:
        seconds = get_seconds()
        base = act_ltl(y)
        if seconds is None or seconds == 0:
            return f"G {x}.({cond_ltl(x)} -> F {y}.({base}))" if conds else f"G F {y}.({base})"
        return f"G {x}.({cond_ltl(x)} -> F {y}.({base} & {time_diff(x, y)} <= {seconds}))"

    if "within" in op:
        seconds = get_seconds()
        return f"G {x}.({cond_ltl(x)} -> X F {y}.({act_ltl(y)} & {time_diff(x, y)} <= {seconds}))"

    if "eventually" in op:
        return f"G {x}.({cond_ltl(x)} -> F {y}.({act_ltl(y)}))"

    if "always" in op:
        return f"G {x}.({cond_ltl(x)} -> {act_ltl(x)})"

    if "until" in op:
        return f"({cond_ltl(x)} U {act_ltl(y)})"

    raise ValueError(f"Unsupported temporal operator {t.operator}")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("Running NL -> IR -> LTL (Multi-Ontology Edition)\n")
    log_ontology = load_ontology(LOG_ONTOLOGY_FILE)
    ltl_ontology = load_ontology(LTL_ONTOLOGY_FILE)

    for i, sentence in enumerate(TEST_SENTENCES, start=1):
        print("=" * 70)
        print(f"Example {i}: {sentence}")

        ir = step1(sentence, log_ontology, ltl_ontology)

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