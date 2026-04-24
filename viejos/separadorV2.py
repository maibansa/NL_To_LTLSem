import json
import re
import warnings
from typing import Optional, List, Dict
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
# MODELOS STEP 1 (ONTOLOGY‑GROUNDED)
# ============================================================

class OntologyMapping(BaseModel):
    logField: str
    concept: Optional[str] = None


class Bound(BaseModel):
    value: float
    unit: str
    type: str


class Temporal(BaseModel):
    operator: str
    bound: Optional[Bound] = None
    ontologyMapping: OntologyMapping

    @field_validator("operator")
    @classmethod
    def normalize_operator(cls, v):
        mapping = {
            "always": "always",
            "eventually": "eventually",
            "within": "within",
            "until": "until",
            "every": "always",
        }
        if v not in mapping:
            raise ValueError(f"Unsupported temporal operator '{v}'")
        return mapping[v]


class EntityFragment(BaseModel):
    type: str                 # condition | action
    predicate: str
    subject: str
    negated: bool = False
    modality: Optional[str] = "mandatory"
    ontologyMapping: OntologyMapping

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
# CARGA ONTOLOGÍA DEL LOG
# ============================================================

def load_log_ontology(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ============================================================
# PROMPT GUIADO POR ONTOLOGÍA
# ============================================================

def build_prompt(text: str, log_ontology: Dict) -> str:
    onto = json.dumps(log_ontology, indent=2)

    return f"""
You are an ontology-guided semantic interpreter for clinical execution logs.

This is the ontology defining the meaning of each log field:

<LOG_ONTOLOGY>
{onto}
</LOG_ONTOLOGY>

Interpret the following clinical rule:

"{text}"

IMPORTANT CONSTRAINTS:
- Every semantic element MUST include an ontologyMapping.
- ontologyMapping.logField MUST be one of the log fields defined above.
- Use sSnomed for clinical concepts when appropriate.
- Use nTimestamp for temporal constraints.

Return ONLY valid JSON with this structure:

{{
  "condition": {{
    "type": "condition",
    "predicate": "...",
    "subject": "...",
    "negated": false,
    "ontologyMapping": {{
      "logField": "sSnomed",
      "concept": "SNOMED_CODE_OR_NULL"
    }}
  }},
  "action": {{
    "type": "action",
    "predicate": "...",
    "subject": "...",
    "modality": "mandatory|recommended|optional",
    "ontologyMapping": {{
      "logField": "sSnomed",
      "concept": "SNOMED_CODE_OR_NULL"
    }}
  }},
  "temporal": {{
    "operator": "always|eventually|within|until|every",
    "bound": {{
      "value": number,
      "unit": string
    }},
    "ontologyMapping": {{
      "logField": "nTimestamp"
    }}
  }}
}}

Do not include explanations or markdown.
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
# REPARADOR DE JSON DEL LLM
# ============================================================

def repair_json(text: str) -> str:
    text = text.strip()
    text = re.sub(r",\s*([}\]])", r"\1", text)  # comas colgantes
    return text

# ============================================================
# PARSEO ROBUSTO
# ============================================================

def parse_ir(llm_output: str) -> Optional[Rule]:
    try:
        start = llm_output.find("{")
        end = llm_output.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("No JSON found in LLM output")

        raw = llm_output[start:end + 1]

        # Reparar comas colgantes del LLM
        raw = re.sub(r",\s*([}\]])", r"\1", raw)

        data = json.loads(raw)

        condition = EntityFragment(**data["condition"])
        action = EntityFragment(**data["action"])

        temporal_data = data["temporal"]
        bound = temporal_data.get("bound")

        if bound is not None and "type" not in bound:
            bound["type"] = {
                "within": "deadline",
                "always": "frequency",
                "every": "frequency",
                "until": "endpoint",
                "eventually": "existential",
            }.get(temporal_data["operator"], "unspecified")

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

def generate_ir(text: str, log_ontology: Dict) -> Optional[Rule]:
    prompt = build_prompt(text, log_ontology)
    output = call_llm(prompt)
    return parse_ir(output)


def decompose(rule: Rule) -> Decomposition:
    return Decomposition(
        T=rule.temporal,
        E=[rule.condition, rule.action]
    )

# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":

    log_ontology = load_log_ontology(LOG_ONTOLOGY_FILE)

    examples = [
        "If the patient has hypertension, they must receive amlodipine within 24 hours.",
        "Patients with diabetes should receive a glucose test every morning."
    ]

    for text in examples:
        print("\n" + "=" * 70)
        print("Input:", text)

        rule = generate_ir(text, log_ontology)
        if not rule:
            print("❌ Falló la generación de la IR")
            continue

        decomposition = decompose(rule)

        print("✅ Step 1 Decomposition (ontology‑mapped):")
        print(decomposition.model_dump_json(indent=2))
