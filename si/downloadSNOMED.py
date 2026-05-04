"""
SISTEMA DE EXTRACCIÓN Y ENRIQUECIMIENTO DE ONTOLOGÍA MÉDICA (SNOMED CT)
---------------------------------------------------------------------
Este script realiza las siguientes tareas:
1. ESCANEO DE LOGS: Analiza archivos N-Quads (.nq) buscando menciones a conceptos 
   clínicos mediante expresiones regulares y parseo RDF.
2. ENRIQUECIMIENTO VÍA BIOPORTAL: Por cada ID encontrado, consulta la API oficial
   para obtener el nombre preferido (prefLabel) y sinónimos (altLabel).
3. RECURSIVIDAD JERÁRQUICA: Sube automáticamente por el árbol taxonómico de SNOMED
   hasta el nivel definido (MAX_DEPTH) para descargar los ancestros (padres).
4. GENERACIÓN DE GRAFO: Exporta un archivo en formato Turtle (.ttl) compatible con
   herramientas de Grafos y Web Semántica (GraphDB, Protégé).
5. CACHÉ: Guarda una copia local en JSON para evitar consultas repetitivas a la API.
"""
import os
import json
import requests
import re
from rdflib import Dataset, Graph, Namespace, RDF, Literal, URIRef
import warnings

# -------------------------------
# CONFIGURACIÓN
# -------------------------------
warnings.filterwarnings("ignore")
API_KEY = "21f105af-409c-4ea7-b978-23fdfbaf522c"
BASE_URL = "https://data.bioontology.org"

INPUT_NQ = "log_1000_61_80.nq"
OUTPUT_FILE = "snomed_subgraph.ttl"
CACHE_FILE = "snomed_cache.json"

MAX_DEPTH = 3 
SNOMED_BASE = "http://snomed.info/id/"
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")

# -------------------------------
# API BIOPORTAL MEJORADA
# -------------------------------
def get_bioportal_data(concept_id, endpoint=""):
    """Función genérica para consultar clases o padres"""
    # Intentamos con el URI completo que es lo estándar en SNOMED CT BioPortal
    full_uri = f"{SNOMED_BASE}{concept_id}"
    encoded_uri = requests.utils.quote(full_uri, safe='')
    
    url = f"{BASE_URL}/ontologies/SNOMEDCT/classes/{encoded_uri}{endpoint}"
    params = {"apikey": API_KEY, "display": "prefLabel,synonym,definition"}
    
    try:
        resp = requests.get(url, params=params, timeout=12)
        if resp.status_code == 200:
            return resp.json()
        
        # Si falla, reintento con el ID corto (algunas versiones de BioPortal lo prefieren)
        url_alt = f"{BASE_URL}/ontologies/SNOMEDCT/classes/{concept_id}{endpoint}"
        resp = requests.get(url_alt, params=params, timeout=12)
        if resp.status_code == 200:
            return resp.json()
    except:
        pass
    return None

def get_concept_details(concept_id):
    data = get_bioportal_data(concept_id)
    if not data:
        return {"label": concept_id, "synonyms": []}
    
    # Extraer sinónimos de múltiples campos posibles
    syns = data.get("synonym", [])
    if isinstance(syns, str): syns = [syns]
    
    # SNOMED a veces guarda términos en 'definitions'
    defs = data.get("definition", [])
    if isinstance(defs, str): defs = [defs]
    
    all_labels = list(set(syns + defs))
    return {
        "label": data.get("prefLabel", concept_id),
        "synonyms": all_labels
    }

def get_parents(concept_id):
    data = get_bioportal_data(concept_id, "/parents")
    if not data or not isinstance(data, list):
        return []
    
    parent_ids = []
    for p in data:
        p_uri = p.get("@id", "")
        p_id = p_uri.split("/")[-1]
        if p_id and p_id != concept_id:
            parent_ids.append(p_id)
    return parent_ids

# -------------------------------
# LÓGICA DE CONSTRUCCIÓN
# -------------------------------
def process_node(concept_id, g, SNOMED, cache, depth=0):
    if concept_id in cache or depth > MAX_DEPTH:
        return
    
    print(f"{'  '*depth}→ Procesando: {concept_id}")
    data = get_concept_details(concept_id)
    cache[concept_id] = data
    
    node = SNOMED[concept_id]
    g.add((node, RDF.type, SNOMED.Concept))
    g.add((node, URIRef(SNOMED_BASE + "prefLabel"), Literal(data["label"])))
    
    for s in data["synonyms"]:
        g.add((node, SKOS.altLabel, Literal(s)))
        
    # Obtener y procesar padres
    parents = get_parents(concept_id)
    for p_id in parents:
        g.add((node, URIRef(SNOMED_BASE + "is_a"), SNOMED[p_id]))
        process_node(p_id, g, SNOMED, cache, depth + 1)

# -------------------------------
# EXTRACCIÓN DEL LOG
# -------------------------------
def extract_ids(file_path):
    print(f"[INFO] Leyendo {file_path}...")
    # Usamos regex directamente sobre el archivo para máxima velocidad y alcance
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    
    # Buscamos IDs numéricos de SNOMED (normalmente 8-18 dígitos)
    ids = set(re.findall(r"id/(\d{6,18})", text))
    # También buscamos códigos que sigan a la ontología de ejemplo
    ids.update(re.findall(r"clinicalCode>\s*<http://snomed.info/id/(\d+)>", text))
    
    print(f"[INFO] IDs únicos para BioPortal: {list(ids)}")
    return list(ids)

if __name__ == "__main__":
    snomed_ids = extract_ids(INPUT_NQ)
    
    if not snomed_ids:
        print("[ERROR] No se encontraron IDs.")
    else:
        g_final = Graph()
        g_final.bind("skos", SKOS)
        SNOMED = Namespace(SNOMED_BASE)
        cache_final = {}
        
        for cid in snomed_ids:
            process_node(cid, g_final, SNOMED, cache_final)

        g_final.serialize(destination=OUTPUT_FILE, format="turtle")
        with open(CACHE_FILE, "w") as f:
            json.dump(cache_final, f, indent=2, ensure_ascii=False)
            
        print(f"\n[ÉXITO] Grafo guardado con {len(cache_final)} conceptos.")