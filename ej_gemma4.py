# coding: utf-8

import json
import chromadb
import re  # Necesario para separar las palabras de la regla
from typing import List, Optional, Dict
from pydantic import BaseModel
from llama_cpp import Llama, LlamaGrammar

# ============================================================
# CONFIGURACIÓN DEL SISTEMA (ADAPTADA AL MODELO DE 27B)
# ============================================================

# Cargamos el modelo que descargaste. n_ctx=4096 es para el contexto largo.
print("🚀 Cargando Gemma 2 27B en RAM... (Aprovechando tus 32GB)")
llm = Llama(
    model_path="./gemma-2-27b-it-Q4_K_M.gguf", 
    n_ctx=4096, 
    n_threads=8, 
    verbose=False
)

# --- INTEGRACIÓN DE GRAMÁTICA (Directa en String para evitar fallos de archivo) ---
# --- INTEGRACIÓN DE GRAMÁTICA (Versión de Máxima Compatibilidad) ---
# --- INTEGRACIÓN DE GRAMÁTICA (Versión Final Blindada) ---
dltl_grammar_text = r"""
root  ::= out
out   ::= "{" space "\"analysis\":" space str "," space "\"propositions\":" space list "," space "\"formula\":" space "\"" form "\"" space "}"

# --- FÓRMULA LTL + FREEZE (Refinada para evitar confusiones F/X) ---
form        ::= space expr space

# Separamos explícitamente para que el modelo no se "vuelva vago"
expr        ::= binary | unary | primary

# Prioridad: Las operaciones binarias (&, |, ->) conectan expresiones
binary      ::= primary space bop space expr

# Los unarios (G, F, X, O, etc.) deben preceder a una expresión
unary       ::= uop space expr

primary     ::= atom | freeze | "(" space expr space ")"

# Operadores divididos por intención (ayuda al motor de búsqueda de tokens)
uop         ::= [GHF] | [OXY] | "!"
bop         ::= "&" | "|" | "->" | "<->" | "U" | "S"

# Freeze Logic (Métrica obligatoria si hay condición)
freeze      ::= [a-z] ".(" space fcont space ")"
fcont       ::= expr (space "&" space metric)? | metric
metric      ::= "\"" "(" vlist ")" space cond "\""
vlist       ::= [a-z] ("," space [a-z])*
cond        ::= [^"]+

# --- ESTRUCTURA JSON (La que ya sabemos que no peta) ---
atom        ::= "p" [1-9] [0-9]*
list        ::= "[" space (obj (space "," space obj)*)? space "]"
obj         ::= "{" space "\"id\":" space str "," space "\"predicate\":" space str "," space "\"type\":" space str "," space "\"concept\":" space str space "}"

str         ::= "\"" ([^"\\\x00-\x1F] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4}))* "\""
space       ::= [ \t\n\r]*
"""
try:
    dltl_grammar = LlamaGrammar.from_string(dltl_grammar_text)
except Exception as e:
    print(f"❌ Error crítico en la gramática: {e}")
    dltl_grammar = None

# Conexión a la base de datos de vectores (RAG) - Sin cambios
DB_PATH = "./clinica_knowledge_db"
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(name="ontology_layer")

# ============================================================
# MODELOS DE DATOS (Sin cambios)
# ============================================================

class Proposition(BaseModel):
    id: str
    predicate: str
    type: str
    concept: str

class ClinicalRuleOutput(BaseModel):
    analysis: str  
    propositions: List[Proposition]
    formula: str

# ============================================================
# MOTOR RAG (Sin cambios)
# ============================================================

def get_relevant_context(text: str, n_results: int = 7) -> str:
    """Consulta la DB de vectores y devuelve los conceptos relevantes formateados."""
    try:
        results = collection.query(
            query_texts=[text],
            n_results=n_results
        )
        
        context_lines = []
        for meta in results['metadatas'][0]:
            context_lines.append(f"- {meta['concept']} (Type: {meta['type']})")
        
        return "\n".join(context_lines)
    except Exception as e:
        print(f"⚠️ Error consultando RAG: {e}")
        return "No specific context found."

# ============================================================
# CONSTRUCTOR DEL PROMPT DINÁMICO (Sin cambios)
# ============================================================

def build_compiler_prompt(text: str, context: str, ltl_onto: Dict) -> str:
    return f"""
Act as a Clinical Rule Compiler expert in LTL and Freeze Logic.

==================================================================
STRICT ONTOLOGY CONTEXT (ONLY USE THESE CONCEPTS):
{context}
==================================================================

CRITICAL ARCHITECTURAL REQUIREMENT:
1. THE DOMAIN SEMANTIC LAYER:
   Extract clinical concepts into 'propositions'. 
   Use the 'Type' from the context above as the 'predicate' (e.g., tActivity, aWorkflow).

2. LOGICAL HIERARCHY RULES:
   - Temporal operators (G, F, H, O) cover the widest scope.
   - The biconditional (<->) MUST be enclosed in parentheses if inside an F or G operator.
   - Explain the nesting in the "analysis" field before writing the formula.
   - STRICT REQUIREMENT: Do NOT use any temporal operators (G, F, X, O, H) unless the input text explicitly contains temporal keywords like 'always', 'future', or 'next'. If no time keyword is present, the formula MUST ONLY contain the atomic proposition ID (e.g., 'p1').
   - MANDATORY LOGICAL FIDELITY: You MUST translate 'in the future' as 'F' and 'next' as 'X'. It is STRICTLY FORBIDDEN to use 'X' when the text says 'future'. If the rule starts with 'Always', the entire formula must be wrapped in 'G(...)'. Do not simplify or alter the temporal structure; if the text says 'future Practitioner', you must write 'F p2'.
THE FORMULA LAYER (The "When" and "How"):
    
    You MUST represent the logical and temporal structure recursively in the formula field using ONLY these patterns:

        Atomic: Use the proposition ID string (e.g., "p1").

        LTL Ontology:

        Logig Operators: 
        ! f     NOT f
        f | g   f OR g
        f & g   f AND g
        f -> g  If f then g
        f <-> g f if and only if g

        LTL Operators:
        G f     Always 'f' (Globally, future)
        H f     'f' happens for every past state (Historical Always)
        F f     Eventually 'f' will happen (Future)
        O f     'f' happened in the past (Once)
        X f     'f' happens at the Next event (False for the last event)
        Y f     'f' happened at the previous event (False for the first event)
        f U g   'f' happens Until 'g' is met
        f S g   'g' happens Since 'f' happened

FREEZE OPERATOR
The freeze operator binds the value of a non-atomic attribute at a specific trace position to a variable (z) for use in a sub-formula.

Syntax: z.(<formula>)



NEVER mix clinical concepts into temporalStructure, and NEVER mix temporal constraints into propositions.
==================================================================

- CANONICAL TIME NORMALIZATION (REQUIRED):
 If a temporal bound is mentioned, convert it to seconds
==================================================================
 - ABOUT DOMAIN SEMANTICS

gSnomed propositions always are of type graph and in the formula you must include PROP.concept_name(x,y,..) where x,y ... are freeze variables and different worlds of concept_name must include _ in the spaces 

==================================================================

Return ONLY valid JSON:
{{
"analysis": "Explanation of the nested temporal structure",
"propositions": [
{{
"id": "p1",
"predicate": "gSnomed | aActivity | aActionType | aWorkflow | aActor",
"type": "graph | atomic | string | number | boolean",
"concept": "Name_From_List"
}}
],
"formula": "F x.(p1 & ...)"
}}
### CLINICAL RULE TO TRANSLATE:
"{text}"
"""

# ============================================================
# LLAMADA AL MODELO (Directa con Gramática integrada)
# ============================================================

import unicodedata

def call_gemma(prompt: str) -> Optional[ClinicalRuleOutput]:
    try:
        full_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        
        response = llm(
            full_prompt, 
            max_tokens=1024, 
            temperature=0.0, 
            grammar=dltl_grammar,
            stop=["<end_of_turn>"]
        )
        
        raw_output = response["choices"][0]["text"].strip()
        
        # --- LA LIMPIEZA ANTIPETE ---
        # 1. Eliminamos caracteres de control invisibles (categoría "C" de Unicode)
        clean_output = "".join(ch for ch in raw_output if unicodedata.category(ch)[0] != "C" or ch in "\t")
        
        # 2. Reemplazamos saltos de línea reales por espacios (el JSON no los permite dentro de strings)
        clean_output = clean_output.replace('\n', ' ').replace('\r', ' ')

        # 3. Localizamos el inicio y fin del JSON por si hay basura fuera de las llaves
        start = clean_output.find('{')
        end = clean_output.rfind('}') + 1
        if start == -1 or end == 0:
            raise ValueError("No se detectó un objeto JSON en la salida.")
            
        json_str = clean_output[start:end]

        # Intentamos cargar. Si Gemma puso comillas dobles sin escapar dentro de un texto, 
        # esto fallará, pero la gramática (si usas la que tiene la regla 'str' corregida) 
        # debería haber evitado eso.
        data = json.loads(json_str)
        
        return ClinicalRuleOutput(**data)

    except json.JSONDecodeError as je:
        print(f"❌ Error de formato JSON (pete): {je}")
        # Te muestra exactamente dónde se rompió para que sepas qué carácter fue
        print(f"📍 Cerca de: {json_str[max(0, je.pos-40):je.pos+40]}")
        return None
    except Exception as e:
        print(f"❌ Error en la compilación: {e}")
        return None

# ============================================================
# EJECUCIÓN PRINCIPAL (Sin cambios)
# ============================================================

if __name__ == "__main__":
    ltl_ontology = {"operators": ["G", "F", "X", "U", "!", "&", "|", "->", "freeze"]}

    TEST_RULES = [
        "Allergy_Panel",
        "Allergy_Panel & Practitioner or Not_Atopic ", 
        "Always Allergy_Panel implies that in the future Practitioner if and only if next  Delay_For_Atopy",
        "Always Allergy_Panel implies that in the future Practitioner if and only if next Delay_For_Atopy and in the past Food_Allergy_Incidence_Submodule and the time between the first and last instances is less that two hours",
        "Always if an event x verifies Allergy_Panel this implies that in the future there is a point y verifies Practitioner if and only if next Delay_For_Atopy and in the past Food_Allergy_Incidence_Submodule and the time between x and y is less that 2 hours"

        
       
      ]

    print(f"🚀 RAG-COMPILER INICIADO (MODELO LOCAL: Gemma 2 27B)")
    print(f"📂 USANDO DB EN: {DB_PATH}")
    print("=" * 80)

    for i, rule in enumerate(TEST_RULES, 1):
        print(f"\n[{i}] REGLA: {rule}")
        
        words = re.findall(r'\b\w{4,}\b', rule) 
        all_segments = []
        for word in words:
            seg = get_relevant_context(word, n_results=3)
            if seg: 
                all_segments.append(seg)
        
        relevant_context = "\n".join(list(set("\n".join(all_segments).split("\n"))))
        
        print("\n🔍 [DEBUG] CONTEXTO RECUPERADO DEL RAG:")
        if relevant_context:
            print(relevant_context)
        else:
            print("⚠️ ADVERTENCIA: El RAG no devolvió ningún concepto relevante.")
        print("-" * 40)
        
        prompt = build_compiler_prompt(rule, relevant_context, ltl_ontology)
        result = call_gemma(prompt)
        
        if result:
            print(f"\n🧠 ANÁLISIS LÓGICO: {result.analysis}")
            print("\n📦 PROPOSICIONES FINALES:")
            for p in result.propositions:
                print(f"   • {p.id}: [{p.predicate}] {p.concept} ({p.type})")
            
            print("\n⚙️ FÓRMULA RESULTANTE:")
            print(f"   {result.formula}")
        else:
            print("\n⚠️ Error: Gemma devolvió un formato inválido o vacío.")
        
        print("=" * 80)