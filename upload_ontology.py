import json
import os
import chromadb
from chromadb.utils import embedding_functions

# --- CONFIGURACIÓN ---
DB_PATH = "./clinica_knowledge_db"
COLLECTION_NAME = "ontology_layer"
JSON_FILE = "list.json"

def upload_to_rag():
    # 1. Inicializar el cliente de ChromaDB
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # 2. Usar la función de embeddings por defecto (corre en local)
    default_ef = embedding_functions.DefaultEmbeddingFunction()
    
    # 3. Si la colección ya existe, la borramos para actualizarla de cero
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"🗑️ Colección antigua '{COLLECTION_NAME}' eliminada.")
    except:
        pass
    
    collection = client.create_collection(
        name=COLLECTION_NAME, 
        embedding_function=default_ef
    )

    # 4. Leer el archivo JSON
    if not os.path.exists(JSON_FILE):
        print(f"❌ Error: No se encuentra el archivo {JSON_FILE}")
        return

    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        ontology = json.load(f)

    # 5. Preparar los datos para la carga
    documents = []
    metadatas = []
    ids = []

    print(f"📦 Procesando conceptos...")
    for category, items in ontology.items():
        for item in items:
            # Combinamos concepto y descripción para dar contexto rico al vector
            full_text = f"{category} {item['concept']}: {item['desc']}"
            
            documents.append(full_text)
            metadatas.append({
                "concept": item['concept'], 
                "type": category,
                "original_desc": item['desc']
            })
            ids.append(f"id_{category}_{item['concept']}")

    # 6. Insertar en la Base de Datos de Vectores
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    print(f"✅ Éxito: Se han indexado {len(documents)} conceptos clínicos.")
    print(f"📂 Base de datos guardada en: {DB_PATH}")

if __name__ == "__main__":
    upload_to_rag()