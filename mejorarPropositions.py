"""
SISTEMA DE ALINEACIÓN SEMÁNTICA ENTRE REGLAS CLÍNICAS Y LOGS DE EJECUCIÓN

====================================================================
DESCRIPCIÓN GENERAL
====================================================================

Este script implementa un proceso de alineación semántica entre:
- reglas clínicas previamente extraídas (en formato JSON),
- y los valores reales observados en un log de ejecución (.mod).

El objetivo es asegurar que los conceptos semánticos utilizados en
las reglas (IR) coincidan exactamente con los actores y actividades
que aparecen en los datos reales del sistema, evitando incoherencias
entre el modelo lógico y el log.

El modelo de lenguaje (LLM, vía Ollama) se utiliza únicamente como
mecanismo de selección guiada entre alternativas existentes,
nunca como generador de nuevos conceptos.

====================================================================
OBJETIVO
====================================================================

El propósito principal del script es:
- alinear los valores de los predicados aActor y aActivity de las reglas,
- con los actores y actividades realmente presentes en el log,
- garantizando que todas las reglas puedan evaluarse sobre los datos.

El sistema evita:
- inventar actores o actividades inexistentes,
- modificar reglas que ya están correctamente alineadas,
- y realizar inferencias no justificadas por el log.

====================================================================
ENTRADAS PRINCIPALES
====================================================================

El script trabaja con tres ficheros principales:

1. LOG_MOD_FILE (.mod)
   Fichero de log de ejecución que contiene, entre otros campos:
   - identificadores de actividades
   - identificadores de actores

2. RULES_JSON
   Fichero JSON que contiene las reglas clínicas ya procesadas,
   incluyendo:
   - frase original,
   - IR con proposiciones (predicate, concept, role),
   - operadores temporales.

3. OUTPUT_JSON
   Fichero de salida donde se guardarán las reglas alineadas.

====================================================================
EXTRACCIÓN DE CONOCIMIENTO DEL LOG
====================================================================

A partir del fichero de log (.mod), el script extrae:

- ACTIVIDADES:
  Se obtienen de la segunda columna del log, tomando el primer campo
  antes del primer separador '&'.

- ACTORES:
  Se obtienen también de la segunda columna del log, tomando el tercer
  campo separado por '&'.

Los valores extraídos se consideran:
- el conjunto cerrado y autorizado de actores,
- y el conjunto cerrado y autorizado de actividades.

Estos conjuntos definen el espacio de alineación permitido.

====================================================================
USO DEL MODELO DE LENGUAJE (OLLAMA)
====================================================================

El LLM se utiliza exclusivamente como selector entre opciones válidas.

Para cada concepto NO alineado:
- se construye un prompt con:
  - la frase clínica original,
  - el concepto actual de la regla,
  - la lista cerrada de actores o actividades válidos.
- el modelo debe devolver estrictamente un JSON con UNA selección.

El script valida siempre que:
- el valor devuelto pertenezca exactamente al conjunto permitido,
- en caso contrario, el alineamiento se descarta.

====================================================================
PROCESO DE ALINEACIÓN
====================================================================

Para cada regla del fichero JSON:

1. Se recorren primero las proposiciones con predicado aActor:
   - Si el actor ya aparece en el log, no se modifica.
   - Si no aparece, se intenta alinear mediante Ollama.
   - Si la selección es válida, se actualiza el concepto.
   - Si no, se mantiene el valor original.

2. Posteriormente se recorren las proposiciones con predicado aActivity:
   - Se aplica exactamente el mismo proceso de validación y alineación.

El proceso distingue explícitamente entre actores y actividades y
nunca los mezcla.

====================================================================
SALIDA
====================================================================

El resultado final es un nuevo fichero JSON que contiene:
- las reglas originales,
- con los conceptos aActor y aActivity alineados cuando ha sido posible.

Además, el sistema imprime por pantalla:
- los actores y actividades extraídos del log,
- las decisiones de alineación realizadas,
- los casos en los que no se ha podido alinear un concepto.

====================================================================
PRINCIPIOS DE DISEÑO
====================================================================

- El log de ejecución es la fuente de verdad.
- El LLM no inventa conceptos, solo elige entre valores existentes.
- La alineación es explícita, trazable y reversible.
- Las reglas válidas no se modifican innecesariamente.
- El proceso es determinista salvo en la selección guiada.


"""
# coding: utf-8
import json
import os
import requests
from typing import Optional

# ===============================
# CONFIGURACIÓN
# ===============================
LOG_MOD_FILE = "log_1000_61_80.mod"
RULES_JSON = "resultado.json"
OUTPUT_JSON = "rules_aligned.json"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"
OLLAMA_TIMEOUT = 120

# ===============================
# OLLAMA
# ===============================
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

# ===============================
# EXTRAER ACTIVIDADES DEL LOG
# ===============================
def get_activities_from_log(file_path):
    activities = set()

    with open(file_path, "r", encoding="utf-8") as f:
        first = True
        for line in f:
            if first:
                first = False
                continue

            parts = line.strip().split(",")
            if len(parts) < 2:
                continue

            fields = parts[1].split("&")
            if len(fields) > 0:
                act = fields[0].strip()
                if act:
                    activities.add(act)

    return sorted(activities)

# ===============================
# EXTRAER ACTORES DEL LOG
# ===============================
def get_actors_from_log(file_path):
    actors = set()

    with open(file_path, "r", encoding="utf-8") as f:
        first = True
        for line in f:
            if first:
                first = False
                continue

            parts = line.strip().split(",")
            if len(parts) < 2:
                continue

            fields = parts[1].split("&")
            if len(fields) > 2:
                actor = fields[2].strip()
                if actor:
                    actors.add(actor)

    return sorted(actors)

# ===============================
# PROMPTS
# ===============================
def build_activity_prompt(concept, sentence, valid_activities):
    return f"""
Align a clinical activity concept to an existing workflow activity.

Sentence:
"{sentence}"

Concept:
"{concept}"

Available activities:
{", ".join(valid_activities)}

Return STRICT JSON:
{{ "activity": "<ONE activity from the list>" }}
"""

def build_actor_prompt(concept, sentence, valid_actors):
    return f"""
Align an actor concept to an existing workflow actor.

Sentence:
"{sentence}"

Concept:
"{concept}"

Available actors:
{", ".join(valid_actors)}

Return STRICT JSON:
{{ "actor": "<ONE actor from the list>" }}
"""

# ===============================
# CHOOSERS
# ===============================
def choose_activity_with_ollama(concept, sentence, valid_activities):
    result = call_ollama(build_activity_prompt(concept, sentence, valid_activities))
    if result is None:
        return None
    try:
        value = json.loads(result).get("activity")
    except Exception:
        return None
    return value if value in valid_activities else None

def choose_actor_with_ollama(concept, sentence, valid_actors):
    result = call_ollama(build_actor_prompt(concept, sentence, valid_actors))
    if result is None:
        return None
    try:
        value = json.loads(result).get("actor")
    except Exception:
        return None
    return value if value in valid_actors else None

# ===============================
# ALINEAR JSON
# ===============================
def align_json(json_file, valid_actors, valid_activities):
    with open(json_file, "r", encoding="utf-8") as f:
        rules = json.load(f)

    print("\n==== RECORRIENDO LOG JSON ====\n")

    for rule in rules:
        rule_id = rule.get("id")
        sentence = rule.get("sentence", "")
        propositions = rule.get("ir", {}).get("propositions", [])

        # ---- PRIMERO aActor ----
        for prop in propositions:
            if prop.get("predicate") != "aActor":
                continue

            old = prop.get("concept", "").strip()

            if old in valid_actors:
                print(f"[OK actor] Rule {rule_id}: {old}")
                continue

            new = choose_actor_with_ollama(old, sentence, valid_actors)
            if new:
                print(f"[ALIGN actor] Rule {rule_id}:")
                print(f"    viejo -> {old}")
                print(f"    nuevo -> {new}")
                prop["concept"] = new
            else:
                print(f"[SKIP actor] Rule {rule_id}: {old}")

        # ---- DESPUÉS aActivity ----
        for prop in propositions:
            if prop.get("predicate") != "aActivity":
                continue

            old = prop.get("concept", "").strip()

            if old in valid_activities:
                print(f"[OK activity] Rule {rule_id}: {old}")
                continue

            new = choose_activity_with_ollama(old, sentence, valid_activities)
            if new:
                print(f"[ALIGN activity] Rule {rule_id}:")
                print(f"    viejo -> {old}")
                print(f"    nuevo -> {new}")
                prop["concept"] = new
            else:
                print(f"[SKIP activity] Rule {rule_id}: {old}")

    return rules

# ===============================
# MAIN
# ===============================
if __name__ == "__main__":

    print("==== ACTORES EXTRAÍDOS DEL LOG ====\n")
    actors = get_actors_from_log(LOG_MOD_FILE)
    for i, a in enumerate(actors, 1):
        print(f"{i}. {a}")

    print("\n==== ACTIVIDADES EXTRAÍDAS DEL LOG ====\n")
    activities = get_activities_from_log(LOG_MOD_FILE)
    for i, a in enumerate(activities, 1):
        print(f"{i}. {a}")

    aligned_rules = align_json(RULES_JSON, actors, activities)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(aligned_rules, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Resultado guardado en {OUTPUT_JSON}")