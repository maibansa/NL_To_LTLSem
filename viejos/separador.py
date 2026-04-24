import json
import warnings
from typing import Optional, List
from pydantic import BaseModel, ValidationError, field_validator
import requests

# ============================================================
# CONFIGURACIÓN
# ============================================================

warnings.filterwarnings("ignore", category=FutureWarning)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"   # asegúrate de haber hecho: ollama pull llama3

# ============================================================
# MODELOS STEP 1
# ============================================================

class Bound(BaseModel):
    value: float
    unit: str
    type: str


class Temporal(BaseModel):
    operator: str
    bound: Optional[Bound] = None

    @field_validator("operator")
    @classmethod
    def normalize_operator(cls, v):
        normalization = {
            "always": "always",
            "eventually": "eventually",
            "within": "within",
            "until": "until",
            "every": "always",
        }
        if v not in normalization:
            raise ValueError(f"Unsupported temporal operator '{v}'")
        return normalization[v]


class EntityFragment(BaseModel):
    type: str            # condition | action
    predicate: str
    subject: str
    negated: bool = False
    modality: Optional[str] = "mandatory"

    @field_validator("type")
    @classmethod
    def check_type(cls, v):
        if v not in ["condition", "action"]:
            raise ValueError("Entity type must be 'condition' or 'action'")
        return v


class Rule(BaseModel):
    type: str = "clinical_rule"
    condition: EntityFragment
    action: EntityFragment
    temporal: Temporal


class Decomposition(BaseModel):
    T: Temporal
    E: List[EntityFragment]

# ============================================================
# PROMPT STEP 1
# ============================================================

def build_prompt(text: str) -> str:
    return f"""
You are a system that converts clinical rules into structured JSON.

Return ONLY a raw JSON object, no explanations, no markdown.

Schema:
{{
  "type": "clinical_rule",
  "condition": {{
    "predicate": "...",
    "subject": "...",
    "negated": false
  }},
  "action": {{
    "predicate": "...",
    "subject": "...",
    "modality": "mandatory|recommended|optional"
  }},
  "temporal": {{
    "operator": "always|eventually|within|until|every",
    "bound": {{
      "value": number,
      "unit": string
    }}
  }}
}}

Input:
"{text}"
"""

# ============================================================
# LLM LOCAL (OLLAMA)
# ============================================================

def call_llm(prompt: str) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )

    response.raise_for_status()
    return response.json()["response"]

# ============================================================
# PARSEO ROBUSTO
# ============================================================

def parse_ir(llm_output: str) -> Optional[Rule]:
    try:
        # Extraer el bloque JSON (por seguridad)
        start = llm_output.find("{")
        end = llm_output.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("No JSON found")

        data = json.loads(llm_output[start:end + 1])

        # Condition
        cond = data.get("condition", {})
        cond.setdefault("type", "condition")
        condition = EntityFragment(**cond)

        # Action
        act = data.get("action", {})
        act.setdefault("type", "action")
        action = EntityFragment(**act)

        # Temporal + inferencia de bound.type
        temporal_data = data.get("temporal", {})
        bound = temporal_data.get("bound")

        if bound is not None and "type" not in bound:
            op = temporal_data.get("operator")
            bound["type"] = {
                "within": "deadline",
                "always": "frequency",
                "every": "frequency",
                "until": "endpoint",
                "eventually": "existential",
            }.get(op, "unspecified")

        temporal = Temporal(**temporal_data)

        return Rule(
            condition=condition,
            action=action,
            temporal=temporal,
        )

    except (ValidationError, json.JSONDecodeError, ValueError) as e:
        print("❌ Error validando IR:", e)
        print("Salida del LLM:")
        print(llm_output)
        return None

# ============================================================
# STEP 1
# ============================================================

def generate_ir(text: str) -> Optional[Rule]:
    prompt = build_prompt(text)
    output = call_llm(prompt)
    return parse_ir(output)


def decompose(rule: Rule) -> Decomposition:
    return Decomposition(
        T=rule.temporal,
        E=[rule.condition, rule.action]
    )

# ============================================================
# EJECUCIÓN DE PRUEBA
# ============================================================

if __name__ == "__main__":

    examples = [
        "If the patient has hypertension, they must receive amlodipine within 24 hours.",
        "Patients with diabetes should receive a glucose test every morning."
    ]

    for text in examples:
        print("\n" + "=" * 60)
        print("Input:", text)

        rule = generate_ir(text)
        if not rule:
            print("❌ Falló la generación de la IR")
            continue

        decomposition = decompose(rule)

        print("✅ Step 1 Decomposition (T + E):")
        print(decomposition.model_dump_json(indent=2))