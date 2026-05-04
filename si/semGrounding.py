"""
DESCRIPCIÓN DEL PROCESO (Grounding Semántico Ontología‑Guiado):

Este script implementa un proceso de grounding semántico clínico
basado en ontologías, combinando BioPortal (SNOMED CT),
un subgrafo local (.ttl) y un modelo LLM (Ollama) con roles bien definidos.

El objetivo NO es generar sinónimos lingüísticos,
sino seleccionar de forma controlada y trazable
un concepto ontológico válido para cada proposición de tipo 'graph'.

------------------------------------------------------------
FLUJO GENERAL
------------------------------------------------------------

1. RECUPERACIÓN ONTOLÓGICA (BioPortal):
   - Dado un concepto textual del IR, el sistema consulta BioPortal
     para obtener un conjunto de conceptos SNOMED CT candidatos.
   - Para cada candidato se recuperan:
       • Identificador SNOMED
       • Etiqueta preferida (prefLabel)
       • Sinónimos ontológicos declarados
       • Relaciones jerárquicas (padres / subClassOf)
   - En esta fase NO se toma ninguna decisión de grounding.

2. SELECCIÓN GUIADA (Ollama):
   - Ollama NO busca conceptos ni genera sinónimos.
   - Su única función es elegir UN concepto SNOMED entre los candidatos
     devueltos por BioPortal.
   - La elección se realiza usando:
       • El concepto original
       • El contexto completo de la frase (sentence)
       • La jerarquía ontológica de los candidatos
   - Si el concepto es vago o inespecífico,
     se fuerza la selección del concepto más general válido.

3. ANÁLISIS DEL SUBGRAFO LOCAL (.ttl):
   - Una vez seleccionado un ID SNOMED,
     el sistema verifica su estructura en el grafo local:
       • Si el nodo tiene subclases → concepto abstracto
       • Si no tiene subclases → nodo hoja
   - Esta información se utiliza para decidir la forma de la consulta SPARQL.

4. GENERACIÓN DE CONSULTAS SPARQL (ASK):
   - Conceptos abstractos → ASK con Property Paths (rdfs:subClassOf*)
   - Conceptos hoja → ASK directa y más eficiente
   - Las consultas generadas son explícitas y reproducibles.

5. FILTRADO SEMÁNTICO:
   - Solo se procesan proposiciones de tipo 'graph'
   - Las proposiciones 'atomic' se ignoran deliberadamente,
     ya que no requieren grounding ontológico.

------------------------------------------------------------
PRINCIPIOS DE DISEÑO
------------------------------------------------------------

- No se inventan conceptos ontológicos.
- No se hace inferencia clínica implícita.
- No se fuerza precisión cuando el IR es vago.
- El grounding es trazable, jerárquico y verificable.
"""




import json
import requests
from rdflib import Graph, Namespace, URIRef
from typing import Optional, List
import re   # <-- añadido solo para normalizar nombres de función

# ============================================================
# 0. CONFIGURACIÓN
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"
API_KEY = "21f105af-409c-4ea7-b978-23fdfbaf522c"
BASE_URL = "https://data.bioontology.org"
ONTOLOGY_PATH = "snomed_subgraph.ttl"
INPUT_JSON = "resultado.json"

SNOMED_NS = Namespace("http://snomed.info/id/")
RDFS_SUBCLASS = URIRef("http://www.w3.org/2000/01/rdf-schema#subClassOf")

print("[INFO] Cargando subgrafo local...")
local_ontology = Graph()
local_ontology.parse(ONTOLOGY_PATH, format="turtle")

# ============================================================
# 1. BÚSQUEDA LOCAL
# ============================================================

def search_local(terms: List[str], role: Optional[str] = None) -> Optional[str]:
    for term in terms:
        term_lower = term.lower().strip()
        for s, p, o in local_ontology:
            label = str(o).lower()
            if term_lower in label:
                if role and "condition" in role.lower() and "(procedure)" in label:
                    continue
                return str(s).split("/")[-1]
    return None

# ============================================================
# 2. BIOPORTAL: CANDIDATOS + JERARQUÍA
# ============================================================

def get_bioportal_candidates(term: str) -> List[dict]:
    candidates = []
    try:
        url = f"{BASE_URL}/search"
        params = {
            "q": term,
            "ontologies": "SNOMEDCT",
            "apikey": API_KEY,
            "pagesize": 5,
        }
        resp = requests.get(url, params=params, timeout=10).json()

        for item in resp.get("collection", []):
            iri = item.get("@id")
            cid = iri.split("/")[-1]
            parents = []
            try:
                parents_url = (
                    f"{BASE_URL}/ontologies/SNOMEDCT/classes/"
                    f"{requests.utils.quote(iri, safe='')}/parents"
                )
                parents_resp = requests.get(
                    parents_url, params={"apikey": API_KEY}, timeout=10
                ).json()
                parents = [p.get("prefLabel") for p in parents_resp if "prefLabel" in p]
            except Exception:
                pass

            candidates.append({
                "id": cid,
                "prefLabel": item.get("prefLabel"),
                "synonyms": item.get("synonym", []),
                "parents": parents,
            })
    except Exception as e:
        print(f"[ERROR] BioPortal: {e}")

    return candidates

# ============================================================
# 3. OLLAMA: SELECCIÓN FORZADA
# ============================================================

def select_best_concept_with_ollama(
    original_concept: str,
    sentence: str,
    candidates: List[dict],
) -> Optional[str]:
    if not candidates:
        return None

    options = "\n".join(
        f"- ID: {c['id']} | Label: {c['prefLabel']} | Parents: {c['parents']}"
        for c in candidates
    )

    prompt = (
        "You are performing clinical ontology grounding.\n\n"
        f"Original concept: \"{original_concept}\"\n"
        f"Sentence context: \"{sentence}\"\n\n"
        "SNOMED CT candidate concepts:\n"
        f"{options}\n\n"
        "Rules:\n"
        "- Select ONE concept ID from the list\n"
        "- If the concept is vague or unspecified, choose the MOST GENERAL clinically valid concept\n"
        "- Do NOT invent IDs\n"
        "- Respond ONLY with the Concept ID\n"
    )

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=20,
        )
        return resp.json().get("response", "").strip()
    except Exception:
        return None

# ============================================================
# 4. GENERACIÓN DE ASK
# ============================================================

def generate_smart_ask(concept_id: str, role: str) -> str:
    relation = "hasCondition" if "condition" in role.lower() else "executes"
    uri = SNOMED_NS[concept_id]
    has_subclasses = (None, RDFS_SUBCLASS, uri) in local_ontology

    if has_subclasses:
        return f"""
PREFIX snomed: http://snomed.info/id/;
PREFIX rdfs: http://www.w3.org/2000/01/rdf-schema#;
PREFIX skos: http://www.w3.org/2004/02/skos/core#;

ASK {{
    ?patient snomed:{relation}/(rdfs:subClassOf|skos:broader)* snomed:{concept_id} .
}}
"""
    else:
        return f"""
PREFIX snomed: http://snomed.info/id/;

ASK {{
    ?patient snomed:{relation} snomed:{concept_id} .
}}
"""

# ============================================================
# 5. CARGA JSON
# ============================================================

print(f"[INFO] Cargando ejemplos desde {INPUT_JSON}...")
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    json_examples = json.load(f)

final_output = []
generated_functions = []

# ============================================================
# 6. EJECUCIÓN
# ============================================================

for item in json_examples:
    sentence = item.get("sentence", "")
    propositions = item.get("ir", {}).get("propositions", [])

    for prop in propositions:
        if prop.get("type") != "graph":
            continue

        candidates = get_bioportal_candidates(prop["concept"])
        cid = select_best_concept_with_ollama(prop["concept"], sentence, candidates)

        if cid:
            query = generate_smart_ask(cid, prop["role"])

            # ✅ CAMBIO 2: imprimir ASK por pantalla
            
            print("\nFrase original:")
            print(sentence)

            print("\nASK generado:")
            print(query)
            print("-" * 60)

            # ✅ CAMBIO 1: nombre de función SIN PROP_
            raw_name = prop["concept"].lower().strip()
            raw_name = re.sub(r"\s+", "_", raw_name)
            raw_name = re.sub(r"[^a-z0-9_]", "", raw_name)
            func_name = raw_name

            generated_functions.append(f"""
def {func_name}(g1):
    q = f\"\"\"
{query}
    \"\"\"
    return ask_query(q)
""")

# ============================================================
# 7. GUARDAR RESULTADOS
# ============================================================

with open("step2_results.json", "w", encoding="utf-8") as f:
    json.dump(final_output, f, indent=2, ensure_ascii=False)

# ============================================================
# 8. GENERAR my_propositions.py
# ============================================================

with open("my_propositions.py", "w", encoding="utf-8") as f:
    f.write("""import sys
from pyoxigraph import Store
import time

DB_PATH = "test/snomed/GeneradorSynthea/auto/log_1000_61_80"
store = Store(DB_PATH)

def ask_query(q):
    return bool(store.query(q))
""")
    for fn in generated_functions:
        f.write(fn)

print("[OK] step2_results.json y my_propositions.py generados")