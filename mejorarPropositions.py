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