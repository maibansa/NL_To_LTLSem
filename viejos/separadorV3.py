import json
import re
import warnings
from typing import List, Optional, Dict
import requests
from pydantic import BaseModel, ValidationError, field_validator

# ============================================================
# CONFIGURACIÓN
# ============================================================

warnings.filterwarnings("ignore", category=FutureWarning)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"
LOG_ONTOLOGY_FILE = "logonto.json"

# ============================================================
# MODELOS (SEPARACIÓN CLARA)
# ============================================================

# --------- BLOQUE TEMPORAL (LOG) ---------

class TemporalBound(BaseModel):
    value: float
    unit: str


class TemporalStructure(BaseModel):
    operator: str
    bound: Optional[TemporalBound] = None
    anchorField: str  # p.ej. nTimestamp

    @field_validator("operator")
    @classmethod
    def normalize_operator(cls, v):
        allowed = {"always", "eventually", "within", "until", "every"}
        if v not in allowed:
            raise ValueError(f"Unsupported temporal operator '{v}'")
        return v


# --------- BLOQUE SEMÁNTICO (SNOMED) ---------

class Proposition(BaseModel):
    predicate: str              # hasCondition, receiveDrug, ...
    concept: str                # SNOMED ID
    role: str                   # condition | action


# --------- SALIDA STEP 1 ---------

class Step1IR(BaseModel):
    temporalStructure: TemporalStructure
    propositions: List[Proposition]

# ============================================================
# CARGA ONTOLOGÍA DEL LOG (SOLO TEMPORAL / ESTRUCTURAL)
# ============================================================

def load_log_ontology(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ============================================================
# PROMPT (CLARAMENTE SEPARADO)
# ============================================================

def build_prompt(text: str, log_ontology: Dict) -> str:
    onto = json.dumps(log_ontology, indent=2)

    return f"""
You are a semantic interpreter for clinical rules.

There are TWO STRICTLY SEPARATED tasks:

1) TEMPORAL STRUCTURE
Use ONLY the log ontology below to identify:
- the temporal operator
- its bound (if any)
- the log field that acts as the temporal anchor

2) CLINICAL PROPOSITIONS
Identify clinical propositions using SNOMED CT concepts.
DO NOT include temporal information in this part.

LOG ONTOLOGY (temporal/structural only):
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

Do not mix temporal and clinical information.
Do not add explanations.
"""

# ============================================================
# LLM LOCAL (OLLAMA)
# ============================================================

def call_llm(prompt: str) -> str:
    # Añadimos una instrucción de sistema para forzar el formato
    system_instruction = (
        "You are a precise JSON generator. Output ONLY valid JSON. "
        "Do not include comments, explanations, or markdown blocks. "
        "Strictly follow the provided schema."
    )
    
    full_prompt = f"{system_instruction}\n\nInput: {prompt}"

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": full_prompt,
            "stream": False,
            "format": "json"  # <-- ESTO ES CLAVE: Ollama forzará la salida a JSON
        },
        timeout=120
    )
    response.raise_for_status()
    return response.json()["response"]

# ============================================================
# PARSEO ROBUSTO
# ============================================================

def parse_step1_ir(llm_output: str) -> Optional[Step1IR]:
    try:
        start = llm_output.find("{")
        end = llm_output.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("No JSON object found")

        raw = llm_output[start:end + 1]
        raw = re.sub(r",\s*([}\]])", r"\1", raw)  # reparar comas

        data = json.loads(raw)
        return Step1IR(**data)

    except (json.JSONDecodeError, ValidationError, ValueError) as e:
        print("❌ Error validando Step1 IR:", e)
        print("Salida del LLM:")
        print(llm_output)
        return None

# ============================================================
# STEP 1
# ============================================================

def step1(text: str, log_ontology: Dict) -> Optional[Step1IR]:
    prompt = build_prompt(text, log_ontology)
    output = call_llm(prompt)
    return parse_step1_ir(output)

# ============================================================
# EJECUCIÓN DE PRUEBA
# ============================================================

if __name__ == "__main__":

    log_ontology = load_log_ontology(LOG_ONTOLOGY_FILE)

    examples = [
        "If the patient has hypertension, they must receive amlodipine within 24 hours.",
        "Patients with diabetes should receive a glucose test every morning."
    ]

    for text in examples:
        print("\n" + "=" * 80)
        print("Input:", text)

        ir = step1(text, log_ontology)
        if not ir:
            print("❌ Falló Step 1")
            continue

        print("✅ Step 1 IR (temporal ≠ clinical):")
        print(ir.model_dump_json(indent=2))