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
# coding: utf-8
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
INPUT_JSON = "resultado.json"

SNOMED_NS = Namespace("http://snomed.info/id/")
RDFS_SUBCLASS = URIRef("http://www.w3.org/2000/01/rdf-schema#subClassOf")

print("[INFO] Cargando subgrafo local...")
local_ontology = Graph()
local_ontology.parse(ONTOLOGY_PATH, format="turtle")

# ============================================================
# 1. BÚSQUEDA LOCAL (SIN CAMBIOS)
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
# 2. NUEVO BLOQUE: BIOPORTAL (CANDIDATOS + JERARQUÍA)
# ============================================================

def get_bioportal_candidates(term: str) -> List[dict]:
    candidates = []

    try:
        search_url = f"{BASE_URL}/search"
        params = {
            "q": term,
            "ontologies": "SNOMEDCT",
            "apikey": API_KEY,
            "pagesize": 5,
        }

        resp = requests.get(search_url, params=params, timeout=10).json()

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

            candidates.append(
                {
                    "id": cid,
                    "prefLabel": item.get("prefLabel"),
                    "synonyms": item.get("synonym", []),
                    "parents": parents,
                }
            )

    except Exception as e:
        print(f"[ERROR] BioPortal: {e}")

    return candidates

# ============================================================
# 3. NUEVO BLOQUE: OLLAMA (SELECCIÓN FORZADA)
# ============================================================

def select_best_concept_with_ollama(
    original_concept: str,
    sentence: str,
    candidates: List[dict],
) -> Optional[str]:

    if not candidates:
        return None

    options = "\n".join(
        [
            f"- ID: {c['id']} | Label: {c['prefLabel']} | Parents: {c['parents']}"
            for c in candidates
        ]
    )

    prompt = f"""
You are performing clinical ontology grounding.

Original concept: "{original_concept}"
Sentence context: "{sentence}"

SNOMED CT candidate concepts:
{options}

Rules:
- Select ONE concept ID from the list
- If the concept is vague or unspecified, choose the MOST GENERAL clinically valid concept
- Do NOT invent IDs
- Respond ONLY with the Concept ID
"""

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
            },
            timeout=20,
        )
        return resp.json().get("response", "").strip()

    except Exception as e:
        print(f"[AVISO] Ollama no disponible: {e}")
        return None

# ============================================================
# 4. GENERACIÓN DE ASK (SIN CAMBIOS)
# ============================================================

def generate_smart_ask(concept_id: str, role: str) -> str:
    uri = SNOMED_NS[concept_id]
    relation = "hasCondition" if "condition" in role.lower() else "executes"

    has_subclasses = (None, RDFS_SUBCLASS, uri) in local_ontology

    if has_subclasses:
        query = f"""
        ASK {{
            ?patient snomed:{relation}/(rdfs:subClassOf|skos:broader)* snomed:{concept_id} .
        }}
        """
    else:
        query = f"""
        ASK {{
            ?patient snomed:{relation} snomed:{concept_id} .
        }}
        """

    return " ".join(query.split())

# ============================================================
# 5. CARGA DE resultado.json
# ============================================================

print(f"[INFO] Cargando ejemplos desde {INPUT_JSON}...")
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    json_examples = json.load(f)

final_output = []

# ============================================================
# 6. EJECUCIÓN (ÚNICO CAMBIO REAL AQUÍ)
# ============================================================

for item in json_examples:
    ex_id = item.get("id")
    sentence = item.get("sentence", "")
    print(f"--- Ejemplo {ex_id} ---")

    processed_props = []
    propositions = item.get("ir", {}).get("propositions", [])

    for prop in propositions:
        if prop.get("type") != "graph":
            continue

        candidates = get_bioportal_candidates(prop["concept"])
        cid = select_best_concept_with_ollama(
            prop["concept"], sentence, candidates
        )

        if cid:
            query = generate_smart_ask(cid, prop["role"])
            prop["snomed_id"] = cid
            prop["sparql_ask"] = query

            tipo = "JERÁRQUICA" if "*" in query else "DIRECTA"
            print(f"  {prop['concept']} → {cid} ({tipo})")
            print(f"  >> {query}")

            processed_props.append(prop)
        else:
            print(f"  [ERROR] No se pudo mapear '{prop['concept']}'")

    final_output.append(
        {
            "id": ex_id,
            "graph_propositions": processed_props,
        }
    )
    print()

# ============================================================
# 7. GUARDAR RESULTADOS
# ============================================================

with open("step2_results.json", "w", encoding="utf-8") as f:
    json.dump(final_output, f, indent=2, ensure_ascii=False)

print("[OK] step2_results.json generado")