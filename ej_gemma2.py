# coding: utf-8

import json
import chromadb
import re  # Necesario para separar las palabras de la regla
from typing import List, Optional, Dict
from pydantic import BaseModel
from llama_cpp import Llama 

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

# Conexión a la base de datos de vectores (RAG) - Sin cambios
DB_PATH = "./clinica_knowledge_db"
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(name="ontology_layer")

# ============================================================
# MODELOS DE DATOS (Modificado: Añadido analysis)
# ============================================================

class Proposition(BaseModel):
    id: str
    predicate: str
    type: str
    concept: str

class ClinicalRuleOutput(BaseModel):
    analysis: str  # Campo añadido para mejorar razonamiento lógico
    propositions: List[Proposition]
    formula: str

# ============================================================
# MOTOR RAG (Modificado: Sin descripciones largas)
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
            # Cambio: Se elimina original_desc para evitar distracciones al modelo
            context_lines.append(f"- {meta['concept']} (Type: {meta['type']})")
        
        return "\n".join(context_lines)
    except Exception as e:
        print(f"⚠️ Error consultando RAG: {e}")
        return "No specific context found."

# ============================================================
# CONSTRUCTOR DEL PROMPT DINÁMICO (Modificado: Instrucciones de Jerarquía)
# ============================================================

def build_compiler_prompt(text: str, context: str, ltl_onto: Dict) -> str:
    # Mantenemos tu prompt con los añadidos de jerarquía lógica
    return f"""
Act as a Clinical Rule Compiler.

==================================================================
CRITICAL ARCHITECTURAL REQUIREMENT: THREE-LAYER SEPARATION
==================================================================
You MUST strictly separate the extraction into two independent layers before generating the JSON:

    THE DOMAIN SEMANTIC LAYER (The "What"):
    Extract clinical concepts into a flat list of 'propositions'.
   
    List of tActivity: Administer_Allergy_Test, Age_Guard, Allergist_Guard, Allergist_Initial_Visit, Allergy_Panel, Allergy_Unspecified, Atopic, Delay_For_Allergist_Initial_Visit, Delay_For_Atopy, Delay_For_Smoking_History, Drug_Allergy_Incidence_Submodule, End_Allergist_Initial_Visit, Environmental_Allergy_Incidence_Submodule, Female, Food_Allergy_Incidence_Submodule, General_Allergy_CarePlan, Immunotherapy_Submodule, Initial, Living_With_Allergies, Male, No_Infection, Not_Atopic, Potential_COPD_Nonsmoker, Potential_COPD_Smoker, Prescribe_Epipen, Prescribe_OTC_Antihistamine, Terminal.
    List of tActor: Care_Manager, Clinical_System, Lab_Technician, Pharmacist, Physician, Practitioner, Radiologist.

    List of aActionType: at_AllergyOnset, at_CallSubmodule,at_CarePlanStart,at_Delay,at_Encounter,at_EncounterEnd,at_Guard,at_Initial,at_MedicationOrder,at_Procedure,at_Simple,at_Terminal

    List of aWorflow: allergies, appendicitis, asthma, breast_cancer,colorectal_cancer,copd,dermatitis,ear_infections,epilepsy,fibromyalgia

    List of labels corresponding to atomic propositions: Activity | ActionType | Workflow 
    
    List of labels corresponding to non-atomic propositions: Timestamp | Actor

    List of labels corresponding to graph propositions: Snomed
    
    THE FORMULA LAYER (The "When" and "How"):
    
    You MUST represent the logical and temporal structure recursively in the formula field using ONLY these patterns:
    This is a grammar for the correct formulas:
    Precedence (low → high):
    ->, <->
    &
    |
    U, S
    F, O
    G, H
    X, Y
    !
    
    Associativity:
    ->, <-> : right-associative
    &, | : left-associative
    U, S : left-associative
    
    ! : associative
    F, O : right-associative
    G, H : right-associative
    X, Y : right-associative
    
    Terminals:
    CHAR      ::= [a-z]
    ID        ::= [a-zA-Z_][a-zA-Z_0-9]*
    
    Productions:
    list_of_vars ::=
          '('')'
        | '(' CHAR (',' CHAR)* ')'
    
    body ::= [^\"]+                         #Comment: the "body" is a one line python code
    code_form ::= '"' list_of_vars body '"'
    freeze_form ::= CHAR '.' '(' dltl_form ')'
    
    dltl_form ::= 
          dltl_form '->' dltl_form
        | dltl_form '<->' dltl_form
        | dltl_form '&' dltl_form
        | dltl_form '|' dltl_form
        | dltl_form 'U' dltl_form
        | dltl_form 'S' dltl_form
        | 'F' dltl_form
        | 'G' dltl_form
        | 'H' dltl_form
        | 'X' dltl_form
        | 'Y' dltl_form
        | 'O' dltl_form
        | '!' dltl_form
        | '(' dltl_form ')'
        | freeze_form
        | code_form
        | ID
        | 'true'
        | 'false'
        


FREEZE OPERATOR
The freeze operator binds the value of a non-atomic attribute at a specific trace position to a variable (x,y,z...) for use in a sub-formula.

Syntax: x.(<formula>)
Variable: Any value in the range a-z can be used as the identifier.
- Examples:
      F a & x.("(x)x[V]<=34+7") where x is a freeze variable, a is one of the labels corresponding to atomic propositions and V is one of the labels of attributes that are non-atomic propositions. "(x)x[V]<=34+7"  is a kind of lambda definition containing the list of variables (x) and python code that has to be translated as it is
      a | b & x.("(x)x[H]==4 or x[H]==2") where x is a freeze variable, a and b are labels corresponding to atomic propositions and H is one of the labels of attributes that are non-atomic propositions. 
      F b & x.(F y.(a & "(x,y)x[V]==y[V]"))
      F ((a | b) & x.((X false) & "(x)x[#] == 2"))  being x a freeze variable, x[#] evaluates to the position of the event in the trace, starting at position 1
      F x.(X F  y.("(x,y)x[#] + 8 == y[#]"))
 

Note 1: when using freeze variables, for instance 'x', to refer to specific 
        events the way to get access to the attributes will by means of expressions
        of the form "x[<attribute_name>]" or "x[#]".   

Note 2: By default, "my_propositions.py" is imported at booting time as "PROP". 
        This file is the place to write the Python functions used in the non-atomic propositions.
        
Note 3: In the domain of a freeze variable, a string of the form "(list_of_freeze_names)python_code" is a kind of lambda definition, that has to be translated as it is
 
==================================================================

NEVER mix clinical concepts into temporalStructure, and NEVER mix temporal constraints into propositions.
==================================================================

- CANONICAL TIME NORMALIZATION (REQUIRED):
 If a temporal bound is mentioned, convert it to seconds
==================================================================
 - ABOUT DOMAIN SEMANTICS

Snomed propositions always are of type graph and in the formula you must include "PROP.concept_name(x,y,..)" where x,y ... are freeze variables and different worlds of concept_name must include _ in the spaces  This must be a string of the form "(list_of_freeze_names)python_code" 

==================================================================




Return ONLY valid JSON based on this template:

{{
"propositions": [
{{
"id": "p1",
"predicate":  
"type": "graph",
"concept": "concept_name"
}}
],
"formula":
{{

}}
"formula2":
{{
formula where propositions p1, p2... (only propositions) has been replaced by the corresponding concepts.  PROP.concept_name  must not be replaced.
}}
### CLINICAL RULES:  
Always, if 'at_AllergyOnset' happens, the patient must be 'Atopic' within 7200 seconds or the care plan is void.

"{text}"
"""

# ============================================================
# LLAMADA AL MODELO (AHORA LOCAL DIRECTA)
# ============================================================

def call_gemma(prompt: str) -> Optional[ClinicalRuleOutput]:
    try:
        # Formateamos el prompt con los tags que Gemma 2 espera (igual que en la web)
        full_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        
        response = llm(
            full_prompt, 
            max_tokens=1024, 
            temperature=0.0, 
            stop=["<end_of_turn>"]
        )
        
        raw_output = response["choices"][0]["text"]
        
        # Limpieza por si hay texto extra alrededor del JSON
        start = raw_output.find('{')
        end = raw_output.rfind('}') + 1
        data = json.loads(raw_output[start:end])
        
        return ClinicalRuleOutput(**data)
    except Exception as e:
        print(f"❌ Error en la compilación: {e}")
        return None

# ============================================================
# EJECUCIÓN PRINCIPAL (Cambio en la lógica del RAG aplicado aquí)
# ============================================================

if __name__ == "__main__":
    ltl_ontology = {"operators": ["G", "F", "X", "U", "!", "&", "|", "->", "freeze"]}

    TEST_RULES = [
        "Allergy_Panel & Practitioner or Not_Atopic ",
        "Always Allergy_Panel implies that in the future Practitioner if and only if next  Delay_For_Atopy",
        "Always Allergy_Panel implies that in the future Practitioner if and only if next Delay_For_Atopy and in the past Food_Allergy_Incidence_Submodule and the time between the first and last instances is less that two hours"]

    print(f"🚀 RAG-COMPILER INICIADO (MODELO LOCAL: Gemma 2 27B)")
    print(f"📂 USANDO DB EN: {DB_PATH}")
    print("=" * 80)

    for i, rule in enumerate(TEST_RULES, 1):
        print(f"\n[{i}] REGLA: {rule}")
        
        # --- PARCHE DE BÚSQUEDA POR TÉRMINOS PARA NO PERDER CONCEPTOS ---
        words = re.findall(r'\b\w{4,}\b', rule) 
        all_segments = []
        for word in words:
            seg = get_relevant_context(word, n_results=3)
            if seg: 
                all_segments.append(seg)
        
        relevant_context = "\n".join(list(set("\n".join(all_segments).split("\n"))))
        # ---------------------------------------------------------------
        
        print("\n🔍 [DEBUG] CONTEXTO RECUPERADO DEL RAG:")
        if relevant_context:
            print(relevant_context)
        else:
            print("⚠️ ADVERTENCIA: El RAG no devolvió ningún concepto relevante.")
        print("-" * 40)
        
        prompt = build_compiler_prompt(rule, relevant_context, ltl_ontology)
        result = call_gemma(prompt)
        
        if result:
            # Añadido print del análisis para depuración en el paper
            print(f"\n🧠 ANÁLISIS LÓGICO: {result.analysis}")
            print("\n📦 PROPOSICIONES FINALES:")
            for p in result.propositions:
                print(f"   • {p.id}: [{p.predicate}] {p.concept} ({p.type})")
            
            print("\n⚙️ FÓRMULA RESULTANTE:")
            print(f"   {result.formula}")
        else:
            print("\n⚠️ Error: Gemma devolvió un formato inválido o vacío.")
        
        print("=" * 80)