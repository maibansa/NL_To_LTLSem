"""
DESCRIPCIÓN DEL PROCESO (Grounding Semántico - SOLO PARA GRAPH):
Este script automatiza la vinculación entre lenguaje natural y la ontología médica SNOMED CT local.

1. PROCESAMIENTO EXCLUSIVO DE GRAPH: Ignora cualquier elemento 'atomic'.
   
2. PREDICADOS DE ONTOLOGÍA: Usa los predicados específicos (gCondition, gFinding, etc.) 
   definidos en la lista de ejemplos.

3. ESTRATEGIA DE PUENTE (SINÓNIMOS):
   - Si el concepto original no está en el .ttl, pide sinónimos a Ollama.
   - Si siguen sin estar, pide sinónimos a BioPortal.
   - Cruza todos esos sinónimos contra el .ttl local. Solo devuelve el ID si existe en tu grafo.

4. GENERACIÓN DE ARTEFACTOS:
   - SPARQL ASK: Genera la consulta vinculada al ID local encontrado.
"""

import json
import requests
from rdflib import Graph, Namespace
from typing import Optional, List

# ============================================================
# 0. CONFIGURACIÓN
# ============================================================
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3" 
API_KEY = "21f105af-409c-4ea7-b978-23fdfbaf522c"
BASE_URL = "https://data.bioontology.org"
ONTOLOGY_PATH = "snomed_subgraph.ttl"

SNOMED_NS = Namespace("http://snomed.info/id/")

print(f"[INFO] Cargando subgrafo local para validación...")
local_ontology = Graph()
local_ontology.parse(ONTOLOGY_PATH, format="turtle")

# ============================================================
# 1. FUNCIONES DE APOYO (CENTRADAS EN EL GRAFO LOCAL)
# ============================================================

def search_local(terms: List[str], role: Optional[str] = None) -> Optional[str]:
    """Busca una lista de términos y retorna el ID si existe en el .ttl"""
    for term in terms:
        term_lower = term.lower().strip()
        for s, p, o in local_ontology:
            label = str(o).lower()
            if term_lower in label:
                # Evitar que una búsqueda de condición devuelva un procedimiento
                if role and "condition" in role.lower() and "(procedure)" in label:
                    continue
                return str(s).split('/')[-1]
    return None

def get_synonyms_from_ollama(term: str) -> List[str]:
    prompt = f'Provide a JSON list of 5 technical medical synonyms for: "{term}". Format: {{"synonyms": ["term1", "term2", ...]}}'
    try:
        response = requests.post(OLLAMA_URL, 
                                 json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "format": "json"}, 
                                 timeout=15)
        return json.loads(response.json()["response"]).get("synonyms", [])
    except:
        return []

def get_synonyms_from_bioportal(term: str) -> List[str]:
    syns = []
    try:
        url = f"{BASE_URL}/search"
        params = {"q": term, "ontologies": "SNOMEDCT", "apikey": API_KEY, "pagesize": 1}
        resp = requests.get(url, params=params, timeout=10).json()
        if resp.get("collection"):
            data = resp["collection"][0]
            if "prefLabel" in data: syns.append(data["prefLabel"])
            if "synonym" in data: syns.extend(data["synonym"])
    except: pass
    return syns

def entity_linking_bridge(concept_name: str, role: str) -> Optional[str]:
    # 1. Intento directo en local
    cid = search_local([concept_name], role)
    if cid: return cid

    # 2. Intento con sinónimos de Ollama en local
    print(f"    [Buscando sinónimos en Ollama para '{concept_name}'...]")
    ollama_syns = get_synonyms_from_ollama(concept_name)
    cid = search_local(ollama_syns, role)
    if cid: return cid

    # 3. Intento con sinónimos de BioPortal en local
    print(f"    [Buscando sinónimos en BioPortal para '{concept_name}'...]")
    bp_syns = get_synonyms_from_bioportal(concept_name)
    cid = search_local(bp_syns, role)
    if cid: return cid

    return None

# ============================================================
# 2. EJECUCIÓN (10 EJEMPLOS)
# ============================================================

ir_examples = [
    {"ex": 1, "propositions": [{"predicate": "gCondition", "type": "graph", "concept": "Allergic disposition", "role": "condition"}]},
    {"ex": 2, "propositions": [{"predicate": "gFinding", "type": "graph", "concept": "Clinical finding", "role": "condition"}]},
    {"ex": 3, "propositions": [{"predicate": "gProcedure", "type": "graph", "concept": "Encounter for problem", "role": "action"}]},
    {"ex": 4, "propositions": [{"predicate": "gDiagnosis", "type": "graph", "concept": "Hypersensitivity condition", "role": "condition"}]},
    {"ex": 5, "propositions": [{"predicate": "gCondition", "type": "graph", "concept": "Allergic condition", "role": "condition"}]},
    {"ex": 6, "propositions": [{"predicate": "gDisease", "type": "graph", "concept": "Diabetes Mellitus", "role": "condition"}]},
    {"ex": 7, "propositions": [{"predicate": "gObservation", "type": "graph", "concept": "Hypertension", "role": "condition"}]},
    {"ex": 8, "propositions": [{"predicate": "gDisease", "type": "graph", "concept": "Asthma", "role": "condition"}]},
    {"ex": 9, "propositions": [{"predicate": "gEmergency", "type": "graph", "concept": "Myocardial Infarction", "role": "condition"}]},
    {"ex": 10, "propositions": [{"predicate": "gInfection", "type": "graph", "concept": "Bacterial Pneumonia", "role": "condition"}]}
]

print("\n=== STEP 2: GROUNDING (PUENTE DE SINÓNIMOS A GRAFO LOCAL) ===\n")
final_output = []

for item in ir_examples:
    print(f"--- Ejemplo {item['ex']} ---")
    processed_props = []
    
    for prop in item["propositions"]:
        if prop["type"] == "graph":
            cid = entity_linking_bridge(prop["concept"], prop["role"])
            if cid:
                relation = "hasCondition" if "condition" in prop["role"].lower() else "executes"
                query = f"ASK {{ ?patient snomed:{relation} snomed:{cid} . }}"
                
                prop["snomed_id"] = cid
                prop["sparql_ask"] = query
                
                print(f"  [{prop['predicate']}] '{prop['concept']}' -> {query}")
                processed_props.append(prop)
            else:
                print(f"  [AVISO] '{prop['concept']}' no existe en el .ttl (ni por sinónimos).")
    
    final_output.append({"example": item['ex'], "data": processed_props})
    print()

with open("step2_results.json", "w", encoding="utf-8") as f:
    json.dump(final_output, f, indent=2, ensure_ascii=False)