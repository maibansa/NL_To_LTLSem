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
# OLLAMA (TU FUNCIÓN)
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
# 1. ACTIVIDADES DESDE EL LOG
# ===============================
def get_activities_from_log(file_path):
    activities = set()

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No existe el fichero {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        first = True
        for line in f:
            if first:
                first = False
                continue

            line = line.strip()
            if not line:
                continue

            parts = line.split(",")
            if len(parts) > 1:
                activity = parts[1].split("&", 1)[0].strip()
                if activity:
                    activities.add(activity)

    return sorted(activities)

# ===============================
# 2. PROMPT DE ALINEACIÓN
# ===============================
def build_alignment_prompt(concept, sentence, valid_activities):
    return f"""
Align a clinical action concept to a workflow activity.

Sentence:
"{sentence}"

Original concept:
"{concept}"

Available workflow activities:
{", ".join(valid_activities)}

Return STRICTLY a JSON object:
{{ "activity": "<ONE activity from the list>" }}

Rules:
- Choose exactly ONE activity from the list
- Do NOT invent new names
- If uncertain, choose the closest ONE
- Output ONLY valid JSON
"""

def choose_activity_with_ollama(concept, sentence, valid_activities):
    result = call_ollama(
        build_alignment_prompt(concept, sentence, valid_activities)
    )

    if result is None:
        return None

    try:
        data = json.loads(result)
        activity = data.get("activity")
    except Exception:
        return None

    # VALIDACIÓN DURA
    if activity not in valid_activities:
        print(f"[WARN] Ollama devolvió actividad inválida: {activity}")
        return None

    return activity

# ===============================
# 3. RECORRER JSON Y ALINEAR
# ===============================
def align_json_activities(json_file, valid_activities):
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"No existe el fichero {json_file}")

    with open(json_file, "r", encoding="utf-8") as f:
        rules = json.load(f)

    print("\n==== RECORRIENDO LOG JSON ====\n")

    for rule in rules:
        rule_id = rule.get("id")
        sentence = rule.get("sentence", "")
        propositions = rule.get("ir", {}).get("propositions", [])

        for prop in propositions:
            if prop.get("predicate") != "aActivity":
                continue

            old = prop.get("concept", "").strip()

            if old in valid_activities:
                print(f"[OK] Rule {rule_id}: {old}")
                continue

            new = choose_activity_with_ollama(old, sentence, valid_activities)

            if new is None:
                print(f"[SKIP] Rule {rule_id}: no alineado -> {old}")
                continue

            print(f"[ALIGN] Rule {rule_id}:")
            print(f"        viejo -> {old}")
            print(f"        nuevo -> {new}")

            prop["concept"] = new

    return rules

# ===============================
# MAIN
# ===============================
if __name__ == "__main__":

    print("==== ACTIVIDADES EXTRAÍDAS DEL LOG ====\n")
    activities = get_activities_from_log(LOG_MOD_FILE)
    for i, a in enumerate(activities, 1):
        print(f"{i}. {a}")

    aligned_rules = align_json_activities(RULES_JSON, activities)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(aligned_rules, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Resultado guardado en {OUTPUT_JSON}")
