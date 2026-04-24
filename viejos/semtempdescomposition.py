import json
import warnings
from typing import Optional, List
from pydantic import BaseModel, ValidationError, field_validator
import requests

# ------------------------------
# Configuración inicial
# ------------------------------
warnings.filterwarnings("ignore", category=FutureWarning)

# >>>>> PEGA AQUÍ TU API KEY DE GEMINI <<<<<
GOOGLE_API_KEY = "AIzaSyDTB4c1M8kQ1Z50OQtumdVt2ahASGhrh4g"

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash-lite:generateContent"
)

# ------------------------------
# Modelos de datos para IR y Step 1
# ------------------------------

class Bound(BaseModel):
    """
    Representa un límite temporal opcional.
    Ejemplo: "within 24 hours" → value=24, unit="hours", type="deadline"
    """
    value: float
    unit: str
    type: str


class Temporal(BaseModel):
    """
    Representa la restricción temporal de la regla clínica.
    operator: logical temporal operator
    """
    operator: str
    bound: Optional[Bound] = None

    @field_validator("operator")
    @classmethod
    def normalize_operator(cls, v):
        """
        Normaliza operadores temporales del NL a una
        gramática lógica fija.
        """
        normalization = {
            "always": "always",
            "eventually": "eventually",
            "within": "within",
            "until": "until",
            "every": "always"
        }

        if v not in normalization:
            raise ValueError(
                f"Unsupported temporal operator '{v}'. "
                f"Allowed: {list(normalization.keys())}"
            )

        return normalization[v]


class EntityFragment(BaseModel):
    """
    Representa un fragmento semántico de la regla:
    - condition → condición clínica
    - action → acción a realizar
    """
    type: str                 # condition | action
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
    """
    Representa la IR completa de la regla clínica.
    """
    type: str = "clinical_rule"
    condition: EntityFragment
    action: EntityFragment
    temporal: Temporal


class Decomposition(BaseModel):
    """
    Step 1: Decomposition
    - T → Temporal
    - E → Entity fragments
    """
    T: Temporal
    E: List[EntityFragment]

# ------------------------------
# Prompt y llamada a Gemini (REST)
# ------------------------------

def build_prompt(text: str) -> str:
    return f"""
You are a system that converts clinical rules in natural language into structured JSON.
Return ONLY the raw JSON object. No markdown formatting, no backticks.

Schema:
- type: "clinical_rule"
- condition: predicate, subject, negated (boolean)
- action: predicate, subject, modality (mandatory/optional/recommended)
- temporal: operator (always, eventually, within, until), optional bound (value, unit, type)

Input: "{text}"
"""


def call_llm(prompt: str) -> str:
    headers = {"Content-Type": "application/json"}

    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }

    response = requests.post(
        f"{GEMINI_URL}?key={GOOGLE_API_KEY}",
        headers=headers,
        data=json.dumps(payload),
        timeout=30
    )

    response.raise_for_status()

    return response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()

# ------------------------------
# Parseo y validación de la IR
# ------------------------------

def parse_ir(llm_output: str) -> Optional[Rule]:
    try:
        # --- Extraer JSON robustamente ---
        start = llm_output.find("{")
        end = llm_output.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("No JSON object found in LLM output")

        data = json.loads(llm_output[start:end + 1])

        # Condition
        cond_data = data.get("condition", {})
        cond_data.setdefault("type", "condition")
        condition = EntityFragment(**cond_data)

        # Action
        act_data = data.get("action", {})
        act_data.setdefault("type", "action")
        action = EntityFragment(**act_data)

        # --- Temporal + bound normalization ---
        temporal_data = data.get("temporal", {})
        bound_data = temporal_data.get("bound")

        if bound_data is not None and "type" not in bound_data:
            op = temporal_data.get("operator")
            inferred_type = {
                "within": "deadline",
                "always": "frequency",
                "every": "frequency",
                "until": "endpoint",
                "eventually": "existential"
            }.get(op, "unspecified")

            bound_data["type"] = inferred_type

        temporal = Temporal(**temporal_data)

        return Rule(
            condition=condition,
            action=action,
            temporal=temporal
        )

    except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as e:
        print("❌ Error validando IR:", e)
        print("Respuesta cruda del LLM:")
        print(llm_output)
        return None

def generate_ir(text: str) -> Optional[Rule]:
    prompt = build_prompt(text)
    llm_output = call_llm(prompt)
    return parse_ir(llm_output)

# ------------------------------
# Paso 1: Decomposition
# ------------------------------

def decompose(rule: Rule) -> Optional[Decomposition]:
    try:
        return Decomposition(
            T=rule.temporal,
            E=[rule.condition, rule.action]
        )
    except Exception as e:
        print("❌ Error en decompose:", e)
        return None

# ------------------------------
# Ejecución de prueba
# ------------------------------

if __name__ == "__main__":

    examples = [
        "If the patient has hypertension, they must receive amlodipine within 24 hours.",
        "Patients with diabetes should receive a glucose test every morning."
    ]

    for text in examples:
        print("\n" + "=" * 50)
        print("Input:", text)

        rule = generate_ir(text)
        if not rule:
            print("❌ Falló la generación de la IR")
            continue

        decomposition = decompose(rule)
        if not decomposition:
            print("❌ Falló la descomposición")
            continue

        print("✅ Step 1 Decomposition (T + E):")
        print(decomposition.model_dump_json(indent=2))
