"""
PLNtoSemanticLTL – Step 2 (STOP AT ASK)
Output of this step is ONLY the SPARQL ASK query.
Gemini is accessed via REST (Python 3.13 compatible).
"""

# ============================================================
# 0. DEPENDENCIES
# ============================================================

import json
import requests

# ============================================================
# 1. STEP 1 OUTPUT (INVENTED)
# ============================================================

E = [
    {"type": "condition", "subject": "hypertension"},
    {"type": "action", "subject": "amlodipine"}
]

# ============================================================
# 2. TASK-SPECIFIC ONTOLOGY O (SUBGRAPH, SIMPLIFIED)
# ============================================================

ONTOLOGY = {
    "38341003": "Hypertensive disorder, systemic arterial",
    "372687004": "Amlodipine"
}

# ============================================================
# 3. CACHE (L_cache)
# ============================================================

CACHE = {
    "hypertension": "38341003",
    "amlodipine": "372687004"
}

# ============================================================
# 4. GEMINI REST CONFIGURATION (L_LLM)
# ============================================================

# >>>>> PEGA AQUÍ TU API KEY (NO LA COMPARTAS) <<<<<
API_KEY = "PASTE_YOUR_API_KEY"

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash-lite:generateContent"
)

def llm_disambiguate(term):
    """
    Gemini proposes a SNOMED CT concept_id via REST.
    Gemini ONLY suggests; ontology verifies.
    """
    headers = {"Content-Type": "application/json"}

    payload = {
        "contents": [{
            "parts": [{
                "text": f"""
You are a clinical ontology assistant.

Map the clinical term below to the most appropriate
SNOMED CT concept identifier.

Term: "{term}"

Return ONLY a JSON object:
{{"concept_id":"<SNOMED_ID>"}}
"""
            }]
        }]
    }

    response = requests.post(
        f"{GEMINI_URL}?key={API_KEY}",
        headers=headers,
        data=json.dumps(payload),
        timeout=30
    )

    try:
        text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        data = json.loads(text.strip())
        return data.get("concept_id")
    except Exception:
        return None

# ============================================================
# 5. ENTITY LINKING FUNCTION L : e -> c
# ============================================================

def entity_linking(entity):
    term = entity["subject"].lower()

    # 1. Cache
    if term in CACHE:
        return CACHE[term]

    # 2. Gemini (LLM-based disambiguation)
    concept_id = llm_disambiguate(term)
    if concept_id in ONTOLOGY:
        return concept_id

    return None

# ============================================================
# 6. RELATION SELECTION
# ============================================================

def select_relation(entity):
    if entity["type"] == "action":
        return "treats"
    if entity["type"] == "condition":
        return "hasCondition"
    return None

# ============================================================
# 7. ASK QUERY GENERATION (FINAL OUTPUT)
# ============================================================

ASK_QUERIES = []

for e in E:
    c = entity_linking(e)
    r = select_relation(e)

    if c and r:
        ask = f"ASK {{ ?x {r} snomed:{c} }}"
        ASK_QUERIES.append(ask)

# ============================================================
# 8. OUTPUT (END OF STEP 2)
# ============================================================

print("=== SPARQL ASK QUERIES (Step 2 Output) ===\n")
for q in ASK_QUERIES:
    print(q)