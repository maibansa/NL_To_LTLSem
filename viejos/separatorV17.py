"""
SISTEMA DE TRADUCCIÓN DE REGLAS CLÍNICAS
Lenguaje Natural → Representación Intermedia (IR) → LTL/DLTL

====================================================================
DESCRIPCIÓN GENERAL
====================================================================

Este script implementa un pipeline completo, ontología‑guiado y
controlado, para traducir reglas clínicas expresadas en lenguaje
natural a fórmulas de lógica temporal (LTL/DLTL).

El sistema está diseñado para mantener una separación estricta entre:
- semántica clínica (qué ocurre),
- estructura temporal (cuándo ocurre),
- y lógica formal (cómo se expresa la regla).

El modelo de lenguaje (LLM) se utiliza únicamente como extractor
estructurado, nunca como razonador o generador libre de lógica
o semántica.

====================================================================
OBJETIVO
====================================================================

El objetivo es obtener reglas clínicas formalizadas de forma:
- trazable,
- reproducible,
- validable estructuralmente,
- y alineada con ontologías explícitas,

evitando:
- inferencias clínicas implícitas,
- creación de predicados inventados,
- conversiones temporales implícitas no controladas,
- y mezclas indebidas entre semántica y temporalidad.

NOTA: En la IR, los límites temporales se normalizan a segundos
(value en segundos, unit="seconds") para garantizar consistencia.

====================================================================
ENTRADAS PRINCIPALES
====================================================================

El sistema opera sobre:
1. Un conjunto de frases clínicas de entrada (TEST_SENTENCES).
2. Dos ontologías externas en formato JSON:

   - Log Ontology (logonto.json):
     Define los predicados permitidos, los tipos de proposición
     (graph, atomic, string, number, boolean) y los roles semánticos
     (condition, action, state).

   - LTL Ontology (ltlonto.json):
     Define los operadores temporales formales (tEvery, tWithin,
     tUntil, tAlways, tEventually, etc.) y su correspondencia con patrones lógicos.

Estas ontologías restringen explícitamente el espacio de salida
del sistema.

====================================================================
REPRESENTACIÓN INTERMEDIA (IR)
====================================================================

La Representación Intermedia (IR) se valida mediante modelos Pydantic
y consta de dos componentes principales:

1. TemporalStructure:
   - operator: operador temporal formal (de la LTL Ontology)
   - bound: límite temporal opcional (value en segundos, unit="seconds")
   - anchorField: campo temporal de referencia (ej. nTimestamp)

2. Propositions:
   Conjunto de proposiciones clínicas individuales, cada una con:
   - predicate: identificador de la Log Ontology
   - type: tipo ontológico de la proposición
   - concept: concepto clínico textual
   - role: role semántico (condition, action o state)

La IR garantiza que la semántica clínica y la estructura temporal
no se mezclen en una misma capa.

====================================================================
PASO 1 – EXTRACCIÓN DE LA IR (LLM + ONTOLOGÍAS)
====================================================================

El Paso 1 transforma una frase clínica en una IR estructurada.

====================================================================
PASO 2 – TRADUCCIÓN DE IR A LTL/DLTL
====================================================================

Este paso genera una fórmula lógica temporal a partir de la IR,
mediante plantillas deterministas.

====================================================================
PRINCIPIOS DE DISEÑO
====================================================================

- Separación estricta de capas (semántica vs temporalidad).
- Ontologías como fuente de verdad.
- Uso del LLM como extractor/selector, no como razonador libre.
- Validación estructural explícita.
- Trazabilidad completa NL → IR → LTL.
"""
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
    "If an allergy test is administered, it must always be performed by a laboratory technician",
    "Allergy tests must always be administered by a laboratory technician",
    "Always Administer_Allergy_Test implies Lab_Technician",
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
        # Permitimos validación externa (ontología / prompt). Aquí no bloqueamos.
        return v

class Proposition(BaseModel):
    predicate: str
    type: str
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
    onto_semantic = json.dumps(log_ontology, indent=2, ensure_ascii=False)
    onto_logic = json.dumps(ltl_ontology, indent=2, ensure_ascii=False)

    # IMPORTANT: restrict operators to what step15_to_ltl supports
    supported_ops = "tEvery, tWithin, tEventually, tAlways, tUntil"

    return f"""
Act as a Clinical Rule Compiler.

==================================================================
CRITICAL ARCHITECTURAL REQUIREMENT: DUAL-LAYER SEPARATION
==================================================================
You MUST strictly separate the extraction into two independent layers before generating the JSON:

1) THE TEMPORAL LOGIC LAYER (The "When"):
   The 'operator' MUST be EXACTLY one of: {supported_ops}
   Do NOT output any other operator (e.g., tObligation_release).

2) THE DOMAIN SEMANTIC LAYER (The "What"):
   You MUST use LogSchemaOntology to choose predicates and roles.

NEVER mix clinical concepts into temporalStructure, and NEVER mix temporal constraints into propositions.
==================================================================

==================================================================
ANTI-HALLUCINATION PROTOCOL (STRICT COMPLIANCE)
==================================================================
- SUPPORTED OPERATORS ONLY: operator MUST be one of {supported_ops}.
- CANONICAL TIME NORMALIZATION (REQUIRED):
  If a temporal bound is mentioned, convert it to seconds and output:
  "unit": "seconds" and "value": <number_of_seconds>.
  Examples: 4 hours -> 14400, 24 hours -> 86400, 5 minutes -> 300, 10 minutes -> 600, 1 day -> 86400.
- NO INVENTED PREDICATES: Choose ONLY from keys in LogSchemaOntology (e.g., gSnomed, aActivity, aActionType, aActor, etc.).
- NULL FOR ABSENT DATA: If no temporal bound is mentioned, bound MUST be null. Do NOT invent defaults.
- ANCHOR FIELD: anchorField MUST ALWAYS be "nTimestamp" (never "morning" or any other non-log field).
==================================================================

### REFERENCE ONTOLOGIES
1) LTL Ontology (operators):
{onto_logic}

2) Log Ontology (predicates, proposition types, roles):
{onto_semantic}

### OUTPUT FORMAT (STRICTLY FOLLOW THIS STRUCTURE)
Return ONLY valid JSON in the following structure:

{{
  "temporalStructure": {{
    "operator": "tEvery|tWithin|tEventually|tAlways|tUntil",
    "bound": {{ "value": number, "unit": "seconds" }}| null,
    "anchorField": "nTimestamp"
  }},
  "propositions": [
    {{
      "predicate": "identifier_from_LogSchemaOntology (e.g., gSnomed, aActivity, aActionType, aActor, etc.)",
      "type": "the 'type' value associated with the predicate in LogSchemaOntology (e.g., graph, atomic, string, number, boolean)",
      "concept": ,
      "role": "condition|action|state"
    }}
  ]
}}

### CLINICAL RULE TO ANALYZE:
\"{text}\"
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
        # Ollama with format=json usually returns clean JSON, but keep robust extraction
        start = llm_output.find("{")
        end = llm_output.rfind("}")
        raw = llm_output[start:end + 1]
        raw = re.sub(r",\s*([}\]])", r"\1", raw)  # remove trailing commas
        data = json.loads(raw)
        return Step1IR(**data)
    except (ValidationError, json.JSONDecodeError) as e:
        # optional debug:
        # print("PARSE/VALIDATION ERROR:", e)
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
# STEP 2 – LTL / DLTL (LOGICA REFINADA)
# ============================================================
def infer_rule_type(ir: Step1IR) -> str:
    """
    Decide el tipo de regla a partir de la IR.
    - REACTIVE: estado del paciente -> acción
    - NORMATIVE: acción -> restricción organizativa
    """

    # Hay estado clínico del paciente
    has_patient_condition = any(
        p.predicate.startswith("gSnomed")
        or p.predicate.startswith("sModelReference")
        for p in ir.propositions
    )

    # Hay acción
    has_activity = any(
        p.predicate.startswith("aActivity")
        for p in ir.propositions
    )

    # Hay actor
    has_actor = any(
        p.predicate.startswith("aActor")
        for p in ir.propositions
    )

    if has_patient_condition:
        return "REACTIVE"

    if has_activity and has_actor:
        return "NORMATIVE"

    return "REACTIVE"  # por defecto, conservador

def proposition_to_ltl_placeholder(p: Proposition, var: str) -> str:
    """
    Renderiza la proposición.
    Usa el 'concept' con guiones bajos para tipos atomic, string, number y boolean.
    """
    p_type = p.type.lower().strip()
    clean_concept = p.concept.replace(" ", "_")

    # 1) graph: PROP.<concept>(var)
    if p_type == "graph":
        concept_id = p.concept.strip().replace(" ", "_")
        return f"PROP.{concept_id}({var})"

    # 2) string: var[predicate]=='value'
    elif p_type == "string":
        return f"{var}[{p.predicate}]=='{clean_concept}'"

    # 3) number/boolean: var[predicate]==value
    elif p_type in ["number", "boolean"]:
        return f"{var}[{p.predicate}]=={clean_concept}"

    # 4) atomic: concept
    elif p_type == "atomic":
        return f"{clean_concept}"

    # fallback
    return f"{clean_concept}({var})"

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
        # Policy B: IR already normalized to seconds
        return int(t.bound.value)

    def time_diff(v_x, v_y):
        return f"({v_y}[{ts}] - {v_x}[{ts}])"

    def cond_ltl(var):
        if not conds:
            return "TRUE"
        return " & ".join(proposition_to_ltl_placeholder(p, var) for p in conds)

    def act_ltl(var):
        if not acts:
            return "TRUE"
        return " & ".join(proposition_to_ltl_placeholder(p, var) for p in acts)

    op = (t.operator or "").lower()
    seconds = get_seconds()

    # --- Lógica de Plantillas ---
    if "every" in op:
        base = act_ltl(y)
        if seconds is None or seconds == 0:
            return f"G {x}.({cond_ltl(x)} -> F {y}.({base}))" if conds else f"G F {y}.({base})"
        return f"G {x}.({cond_ltl(x)} -> F {y}.({base} & {time_diff(x, y)} <= {seconds}))"

    if "within" in op:
        return f"G {x}.({cond_ltl(x)} -> X F {y}.({act_ltl(y)} & {time_diff(x, y)} <= {seconds}))"

    if "eventually" in op:
        return f"G {x}.({cond_ltl(x)} -> F {y}.({act_ltl(y)}))"

    if "always" in op:
        rule_type = infer_rule_type(ir)

        if rule_type == "REACTIVE":
            # Regla clínica: condición -> acción
            return f"G {x}.({cond_ltl(x)} -> {act_ltl(x)})"

        if rule_type == "NORMATIVE":
            # Regla organizativa: acción -> restricción
            return f"G {x}.({act_ltl(x)} -> {cond_ltl(x)})"

    if "until" in op:
        return f"({cond_ltl(x)} U {act_ltl(y)})"

    raise ValueError(f"Unsupported temporal operator {t.operator}")

# ============================================================
# MAIN (NL -> IR -> LTL) - FORMATO VISUAL COMPLETO
# ============================================================

if __name__ == "__main__":
    print("\n🚀 INICIANDO SISTEMA DE TRADUCCIÓN CLÍNICA")
    print("=" * 70)

    # Carga de recursos
    log_ontology = load_ontology(LOG_ONTOLOGY_FILE)
    ltl_ontology = load_ontology(LTL_ONTOLOGY_FILE)
    output_file = "results_output.json"
    all_results = []

    for i, sentence in enumerate(TEST_SENTENCES, start=1):
        print(f"\n📝 PROCESANDO [{i}/{len(TEST_SENTENCES)}]: {sentence}")

        # 1. Ejecución del Modelo (IR)
        ir = step1(sentence, log_ontology, ltl_ontology)

        if ir is None:
            print("   ❌ ERROR: El Paso 1 (LLM) no devolvió datos válidos.")
            continue

        # MOSTRAR JSON POR PANTALLA
        print("\n📦 STEP 1 IR (JSON):")
        print(ir.model_dump_json(indent=2))

        # 2. Generación de LTL (Query)
        current_ltl = "ERROR"
        try:
            current_ltl = step15_to_ltl(ir)
            # MOSTRAR QUERY POR PANTALLA
            print("\n⚙️  GENERATED LTL/DLTL QUERY:")
            print(f"   {current_ltl}")
        except Exception as e:
            print(f"\n   ⚠️ LTL Generation error: {e}")

        # 3. GUARDADO EN EL FICHERO JSON
        result_item = {
            "id": i,
            "sentence": sentence,
            "ir": ir.model_dump(),
            #"ltl": current_ltl
        }
        all_results.append(result_item)

        # Guardado incremental para no perder datos
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print("-" * 70)

    print(f"\n✅ PROCESO FINALIZADO.")
    print(f"📂 Resultados exportados a: {output_file}")