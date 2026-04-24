"""
DESCRIPCIÓN DEL PROCESO (Grounding Semántico Integral):
Este script une la potencia de Ollama/BioPortal con la estructura real de tu .ttl.

1. EXPANSIÓN LINGÜÍSTICA (Ollama + BioPortal): 
   Si el concepto original no coincide con ninguna etiqueta en el .ttl, el script 
   genera una nube de sinónimos técnicos para intentar "cazar" el ID local para esto usa Ollama y luego BioPortal.

2. CEREBRO ADAPTATIVO (Análisis de Grafo):
   - Una vez hallado el ID, el script mira en el .ttl si ese nodo tiene hijos.
   - Si tiene jerarquía: Genera una query con Property Paths (rdfs:subClassOf*).
   - Si es un nodo hoja: Genera una query directa y rápida.

3. FILTRO DE ATÓMICOS: Se ignora cualquier proposición que no sea tipo 'graph'.
"""

import json
import requests
from rdflib import Graph, Namespace, URIRef
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
RDFS_SUBCLASS = URIRef("http://www.w3.org/2000/01/rdf-schema#subClassOf")

print(f"[INFO] Cargando subgrafo local...")
local_ontology = Graph()
local_ontology.parse(ONTOLOGY_PATH, format="turtle")

# ============================================================
# 1. MOTOR DE BÚSQUEDA Y SINÓNIMOS (EL PUENTE)
# ============================================================

def search_local(terms: List[str], role: Optional[str] = None) -> Optional[str]:
    """Busca una lista de términos y retorna el ID si existe en el .ttl"""
    for term in terms:
        term_lower = term.lower().strip()
        for s, p, o in local_ontology:
            label = str(o).lower()
            if term_lower in label:
                if role and "condition" in role.lower() and "(procedure)" in label:
                    continue
                return str(s).split('/')[-1]
    return None

def get_external_synonyms(term: str) -> List[str]:
    """Combina Ollama y BioPortal para obtener una lista de etiquetas técnicas."""
    syns = [term]
    try:
        prompt = f'Provide a JSON list of 3 clinical synonyms for "{term}". Format: {{"synonyms": ["term1", "term2", "term3"]}}'
        resp = requests.post(OLLAMA_URL, 
                             json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "format": "json"}, 
                             timeout=12)
        syns.extend(json.loads(resp.json()["response"]).get("synonyms", []))
    except Exception as e:
        print(f"    [AVISO] Ollama no disponible: {e}")

    try:
        url = f"{BASE_URL}/search"
        params = {"q": term, "ontologies": "SNOMEDCT", "apikey": API_KEY, "pagesize": 1}
        resp = requests.get(url, params=params, timeout=10).json()
        if resp.get("collection"):
            data = resp["collection"][0]
            if "prefLabel" in data: syns.append(data["prefLabel"])
            if "synonym" in data: syns.extend(data["synonym"])
    except: pass

    return list(set(syns))

# ============================================================
# 2. GENERACIÓN DE QUERIES SEGÚN EL .TTL
# ============================================================

def generate_smart_ask(concept_id: str, role: str) -> str:
    """
    Genera la query SPARQL analizando la complejidad del subgrafo local.
    """
    uri = SNOMED_NS[concept_id]
    relation = "hasCondition" if "condition" in role.lower() else "executes"
    
    # Comprobar si hay subclases (hijos) en el grafo local para este ID
    has_subclasses = (None, RDFS_SUBCLASS, uri) in local_ontology

    if has_subclasses:
        # CONSULTA JERÁRQUICA (Property Paths)
        query = f"""
        ASK {{
            ?patient snomed:{relation}/(rdfs:subClassOf|skos:broader)* snomed:{concept_id} .
        }}
        """
    else:
        # CONSULTA DIRECTA
        query = f"""
        ASK {{
            ?patient snomed:{relation} snomed:{concept_id} .
        }}
        """
    
    return " ".join(query.split())

# ============================================================
# 3. EJECUCIÓN (CON IMPRESIÓN DE QUERIES)
# ============================================================

ir_examples = [
    {"ex": 1, "propositions": [{"predicate": "gCondition", "type": "graph", "concept": "Allergic disposition", "role": "condition"}]},
    {"ex": 2, "propositions": [{"predicate": "gFinding", "type": "graph", "concept": "Clinical finding", "role": "condition"}]},
    {"ex": 3, "propositions": [{"predicate": "gProcedure", "type": "graph", "concept": "Encounter for problem", "role": "action"}]},
    {"ex": 6, "propositions": [{"predicate": "gDisease", "type": "graph", "concept": "Diabetes Mellitus", "role": "condition"}]},
    {"ex": 7, "propositions": [{"predicate": "gObservation", "type": "graph", "concept": "Hypertension", "role": "condition"}]}
]

final_output = []

for item in ir_examples:
    print(f"--- Ejemplo {item['ex']} ---")
    processed_props = []
    
    for prop in item["propositions"]:
        if prop["type"] == "graph":
            # 1. Búsqueda de sinónimos (Ollama/Bioportal)
            syns = get_external_synonyms(prop["concept"])
            
            # 2. Match contra el .ttl local
            cid = search_local(syns, prop["role"])
            
            if cid:
                # 3. Generar la query dinámica según el grafo
                query = generate_smart_ask(cid, prop["role"])
                
                # Enriquecer el objeto para el JSON
                prop["snomed_id"] = cid
                prop["sparql_ask"] = query
                
                # --- AQUÍ LAS PINTAMOS POR CONSOLA ---
                tipo_query = "JERÁRQUICA" if "*" in query else "DIRECTA"
                print(f"  Concepto: '{prop['concept']}' -> ID Local: {cid}")
                print(f"  Sintaxis {tipo_query}:")
                print(f"  >> {query}") 
                
                processed_props.append(prop)
            else:
                print(f"  [ERROR] No se pudo encontrar '{prop['concept']}' en el grafo local.")
    
    final_output.append({"example": item['ex'], "data": processed_props})
    print()

# Guardar resultados
with open("step2_results.json", "w", encoding="utf-8") as f:
    json.dump(final_output, f, indent=2, ensure_ascii=False)