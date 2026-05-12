import json
import requests
import re
import os
from typing import List, Dict, Set
from urllib.parse import quote
from llama_cpp import Llama

# ============================================================
# 0. CONFIGURACIÓN
# ============================================================
class Config:
    MODEL_PATH = "./gemma-2-27b-it-Q4_K_M.gguf"
    NQ_FILE = "log_1000_61_80.nq"
    API_KEY = "21f105af-409c-4ea7-b978-23fdfbaf522c"
    BASE_URL = "https://data.bioontology.org"
    INPUT_JSON = "result.json"
    OUTPUT_FILE = "my_propositions.py"

# ============================================================
# 1. UTILIDADES DE EXTRACCIÓN LOCAL Y BIOPORTAL
# ============================================================

def get_local_inventory_and_ids(file_path: str) -> Dict[str, str]:
    """Escanea el archivo .nq y extrae IDs y etiquetas locales (RDFS Label)."""
    inventory = {}
    if not os.path.exists(file_path):
        print(f"❌ ERROR: No se encuentra el archivo {file_path}")
        return {}

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        all_ids = set(re.findall(r'snomed\.info/id/(\d+)', content))
        # Buscamos: <id> <rdfs:label> "Nombre"
        labels_found = re.findall(r'snomed\.info/id/(\d+)> <http://www.w3.org/2000/01/rdf-schema#label> "([^"]+)"', content)
        
        for snomed_id, label in labels_found:
            inventory[snomed_id] = label.replace("_", " ")
            
        for idx in all_ids:
            if idx not in inventory:
                inventory[idx] = f"Concepto {idx}"
                
    print(f"🔍 Inventario local listo: {len(all_ids)} IDs detectados.")
# MOSTRAR EL DICCIONARIO EXTRAÍDO
    print("\n📖 --- DICCIONARIO EXTRAÍDO DEL LOG LOCAL ---")
    print(f"{'ID SNOMED':<20} | {'ETIQUETA (LABEL)':<40}")
    print("-" * 65)
    for idx in sorted(inventory.keys()):
        print(f"{idx:<20} | {inventory[idx]:<40}")
    print("-" * 65)
    print(f"🔍 Total: {len(all_ids)} IDs detectados.\n")
    
    return inventory

def get_info_from_bioportal(snomed_id: str) -> Dict:
    """Obtiene metadata extendida de un ID desde BioPortal."""
    try:
        concept_uri = quote(f"http://snomed.info/id/{snomed_id}", safe="")
        url = f"{Config.BASE_URL}/ontologies/SNOMEDCT/classes/{concept_uri}"
        params = {"apikey": Config.API_KEY, "display": "prefLabel,definition,synonym"}
        resp = requests.get(url, params=params, timeout=10)

        if resp.status_code == 200:
            data = resp.json()
            return {
                "label": data.get("prefLabel", ""),
                "definition": data.get("definition", [""])[0] if data.get("definition") else "",
                "synonyms": data.get("synonym", [])
            }
    except:
        pass
    return {"label": "", "definition": "", "synonyms": []}

# ============================================================
# 2. LÓGICA DE MAPPING INTELIGENTE
# ============================================================

def smart_grounding(term: str, sentence: str, candidates_bioportal: List[Dict], local_ids: Set[str], inventory: Dict[str, str], llm: Llama) -> str:
    print(f"\n   --- Analizando: '{term}' ---")
    
    # PASO 1: Match por nombre exacto local
    term_clean = term.lower().replace("_", " ")
    for idx, name in inventory.items():
        if term_clean == name.lower():
            print(f"   🏠 MATCH LOCAL DIRECTO: Encontrado en el log por nombre.")
            return idx

    # PASO 2: Match directo vía BioPortal
    for cand in candidates_bioportal:
        if cand['id'] in local_ids:
            print(f"   🎯 MATCH BIOPORTAL DIRECTO: ID {cand['id']} presente en el log.")
            return cand['id']

    # PASO 3: IA con contexto de SENTENCIA
    # Construimos las opciones para el prompt
    options = "\n".join([f"ID: {idx} (Nombre: {inventory.get(idx)})" for idx in local_ids])
    
    # Construcción del Prompt exacto
    prompt = (
        f"<start_of_turn>user\n"
        f"Context: Clinical logic mapping.\n"
        f"Natural Language Sentence: \"{sentence}\"\n"
        f"Target Concept: '{term}'\n\n"
        f"The target is not in my database. Based on the sentence, which of these local IDs is the most appropriate semantic substitute?\n"
        f"{options}\n\n"
        f"Instruction: Return ONLY the numeric ID.\n"
        f"<end_of_turn>\n<start_of_turn>model\n"
    )

    # --- MOSTRAR PROMPT POR PANTALLA ---
    print("\n   ╔═══════════════════════════════════════════════════════════════════")
    print("   ║ 🧠 PROMPT ENVIADO A GEMMA:")
    print("   ╠═══════════════════════════════════════════════════════════════════")
    # Imprimimos el prompt con una pequeña sangría para que se vea bien
    for line in prompt.split('\n'):
        print(f"   ║ {line}")
    print("   ╚═══════════════════════════════════════════════════════════════════\n")

    # Llamada a la IA
    response = llm(prompt, max_tokens=15, temperature=0.0)
    respuesta_raw = response["choices"][0]["text"].strip()
    
    match = re.search(r'\d+', respuesta_raw)
    final_id = match.group(0) if match else list(local_ids)[0]
    
    print(f"   🤖 RESPUESTA DE GEMMA: '{respuesta_raw}'")
    print(f"   ✅ ID SELECCIONADO: {final_id} ({inventory.get(final_id)})")
    
    return final_id

# ============================================================
# 3. PROCESO PRINCIPAL
# ============================================================

def main():
    print("======================================================")
    print("🚀 INICIANDO MAPPING CON CONTEXTO LOCAL Y DE SENTENCIA")
    print("======================================================")

    # Inicializar Gemma
    print("\n[1/4] Cargando Gemma 2...")
    llm = Llama(model_path=Config.MODEL_PATH, n_ctx=4096, n_threads=8, verbose=False)

    # Cargar inventario local (NQ)
    print("\n[2/4] Construyendo inventario desde el log local...")
    inventario = get_local_inventory_and_ids(Config.NQ_FILE)
    ids_en_log = set(inventario.keys())

    # Procesar JSON
    print("\n[3/4] Procesando archivo JSON...")
    with open(Config.INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    generated_functions = []
    concept_pattern = re.compile(r'PROP\.([a-zA-Z0-9_]+)')

    for item in data:
        sentence = item.get("sentence", "")
        formula = item.get("formula", "")
        found_concepts = list(set(concept_pattern.findall(formula)))
        
        for raw in found_concepts:
            term = raw.replace("_", " ")
            print(f"\n🔎 Concepto: '{term}'")

            # Buscar candidatos en BioPortal (para ver si alguno coincide con nuestro log)
            bioportal_cands = []
            try:
                params = {"q": term, "ontologies": "SNOMEDCT", "apikey": Config.API_KEY, "pagesize": 5}
                resp = requests.get(f"{Config.BASE_URL}/search", params=params, timeout=5).json()
                bioportal_cands = [{"id": c["@id"].split("/")[-1], "label": c["prefLabel"]} for c in resp.get("collection", [])]
            except:
                pass

            # Decidir ID (Local -> BioPortal -> Gemma)
            final_id = smart_grounding(term, sentence, bioportal_cands, ids_en_log, inventario, llm)
            
            # Crear la función SPARQL
            func_name = raw
            query = f"PREFIX snomed: <http://snomed.info/id/>\nPREFIX ont: <http://example.org/ontology/>\nASK {{ ?e ont:clinicalCode snomed:{final_id} . }}"
            generated_functions.append(f"def {func_name}(g1):\n    q = \"\"\"{query}\"\"\"\n    return ask_query(q)\n")

   # ... (resto del código anterior igual)

    # ============================================================
    # 4. SALIDA: GENERACIÓN DE MY_PROPOSITIONS.PY
    # ============================================================
    print("\n[4/4] Generando archivo de salida...")
    
    # Definimos la ruta de la DB para que coincida con tu configuración
    db_path_string = f"test/snomed/GeneradorSynthea/auto/{Config.NQ_FILE.replace('.nq', '')}"

    header = f"""import sys
from pyoxigraph import Store
import time

# Configuración: La ruta a tu carpeta de base de datos
path_auto = "./test/snomed/GeneradorSynthea/auto"
DB_PATH = "{db_path_string}"

# Inicializamos el store una sola vez para todas las consultas
store = Store(DB_PATH)

def ask_query(q):
    t0 = time.perf_counter()
    result = bool(store.query(q))
    t1 = time.perf_counter()
    # print(f"   ask_query: {{t1-t0:.4f}}s -> {{result}}")
    return result

"""

    with open(Config.OUTPUT_FILE, "w", encoding="utf-8") as f:
        # Escribimos el nuevo encabezado
        f.write(header)
        
        # Escribimos las funciones generadas
        for fn in generated_functions:
            f.write(fn + "\n")

    print(f"\n✨ PROCESO TERMINADO CON ÉXITO. Archivo '{Config.OUTPUT_FILE}' listo.")

if __name__ == "__main__":
    main()