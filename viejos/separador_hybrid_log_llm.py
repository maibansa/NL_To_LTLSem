import json
import csv
import warnings
from typing import Optional, List
from pydantic import BaseModel, ValidationError, field_validator
import requests

# ============================================================
# CONFIGURACIÓN
# ============================================================

warnings.filterwarnings("ignore", category=FutureWarning)

# >>>>> PEGA AQUÍ TU API KEY DE GEMINI <<<<<
GOOGLE_API_KEY = "AIzaSyDTB4c1M8kQ1Z50OQtumdVt2ahASGhrh4g"

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash-lite:generateContent"
)

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
            "every": "always"
        }
        if v not in normalization:
            raise ValueError(f"Unsupported temporal operator '{v}'")
        return normalization[v]


class EntityFragment(BaseModel):
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


class Decomposition(BaseModel):
    T: Temporal
    E: List[EntityFragment]

# ============================================================
# MODELO DE EVENTO DEL LOG
# ============================================================

class TraceEvent(BaseModel):
    case_id: str
    label: str
    timestamp: int
    actor: str
    action_type: str
    snomed: Optional[str]
    event_uri: str
    workflow: str

# ============================================================
# CARGA DEL LOG (.mod / CSV)
# ============================================================

def load_log(path: str) -> List[TraceEvent]:
    events = []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            # nTimestamp viene como: Name&timestamp&actor&actionType&SNOMED&eventURI&workflow
            parts = row["nTimestamp"].split("&")

            events.append(
                TraceEvent(
                    case_id=row["aActivity"],
                    label=parts[0],
                    timestamp=int(parts[1]),
                    actor=parts[2],
                    action_type=parts[3],
                    snomed=None if "snomed.info/id/0" in parts[4] else parts[4],
                    event_uri=parts[5],
                    workflow=parts[6],
                )
            )

    return events

# ============================================================
# EXTRACCIÓN TEMPORAL (DETERMINISTA, DESDE EL LOG)
# ============================================================

def extract_temporal_from_log(events: List[TraceEvent]) -> Temporal:
    delays = [e for e in events if e.action_type.lower() == "at_delay"]

    if delays:
        return Temporal(
            operator="within",
            bound=Bound(
                value=len(delays),
                unit="step",
                type="deadline"
            )
        )

    return Temporal(operator="always")

# ============================================================
# LLM – CLASIFICADOR SEMÁNTICO AUXILIAR
# ============================================================

def call_llm(prompt: str) -> str:
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }

    response = requests.post(
        f"{GEMINI_URL}?key={GOOGLE_API_KEY}",
        headers=headers,
        data=json.dumps(payload),
        timeout=20
    )

    response.raise_for_status()
    return response.json()["candidates"][0]["content"]["parts"][0]["text"]


def classify_event_with_llm(label: str) -> Optional[dict]:
    prompt = f"""
You are a clinical semantic classifier.

Given the event name:
"{label}"

Return ONLY JSON:
{{ "type": "condition|action|ignore", "predicate": "<verb>" }}
"""

    output = call_llm(prompt)

    # extracción robusta
    start = output.find("{")
    end = output.rfind("}")
    if start == -1 or end == -1:
        return None

    return json.loads(output[start:end + 1])

# ============================================================
# EXTRACCIÓN DE ENTIDADES (LOG + LLM)
# ============================================================

def extract_entities_from_log(events: List[TraceEvent]) -> List[EntityFragment]:
    entities = []

    for e in events:
        info = classify_event_with_llm(e.label)
        if not info or info["type"] == "ignore":
            continue

        entities.append(
            EntityFragment(
                type=info["type"],
                predicate=info["predicate"],
                subject=e.label,
                modality="mandatory"
            )
        )

    return entities

# ============================================================
# STEP 1 COMPLETO (LOG + LLM)
# ============================================================

def step1_from_log_with_llm(log_path: str) -> Decomposition:
    events = load_log(log_path)

    T = extract_temporal_from_log(events)
    E = extract_entities_from_log(events)

    return Decomposition(T=T, E=E)

# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":

    LOG_FILE = "log_1000_61_80.mod"  # <-- tu fichero

    decomposition = step1_from_log_with_llm(LOG_FILE)

    print("\n✅ STEP 1 DECOMPOSITION (FROM LOG + LLM)\n")
    print(decomposition.model_dump_json(indent=2))
