# coding: utf-8

import json
import re
import requests
from typing import List, Optional, Dict
from pydantic import BaseModel, ValidationError

# ============================================================
# CONFIGURACIÓN DEL SISTEMA
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma:7b"  # Se recomienda 7b para mayor precisión lógica
OLLAMA_TIMEOUT = 300

# ============================================================
# MODELOS DE DATOS (REPRESENTACIÓN FINAL)
# ============================================================

class Proposition(BaseModel):
    id: str
    predicate: str
    type: str
    concept: str

class ClinicalRuleOutput(BaseModel):
    propositions: List[Proposition]
    formula: str

# ============================================================
# CONSTRUCTOR DEL PROMPT (TU NUEVA ARQUITECTURA)
# ============================================================

def build_compiler_prompt(text: str, log_ontology: Dict, ltl_ontology: Dict) -> str:
    onto_semantic = json.dumps(log_ontology, indent=2, ensure_ascii=False)
    onto_logic = json.dumps(ltl_ontology, indent=2, ensure_ascii=False)

    return f"""
Act as a Clinical Rule Compiler.

==================================================================
CRITICAL ARCHITECTURAL REQUIREMENT: THREE-LAYER SEPARATION
==================================================================
You MUST strictly separate the extraction into two independent layers before generating the JSON:

    THE DOMAIN SEMANTIC LAYER (The "What"):
    Extract clinical concepts into a flat list of 'propositions'.
    You MUST use LogSchemaOntology to choose predicates.
    List of tActivity: Administer_Allergy_Test, Age_Guard, Allergist_Guard, Allergist_Initial_Visit, Allergy_Panel, Allergy_Unspecified, Atopic, Delay_For_Allergist_Initial_Visit, Delay_For_Atopy, Delay_For_Smoking_History, Drug_Allergy_Incidence_Submodule, End_Allergist_Initial_Visit, Environmental_Allergy_Incidence_Submodule, Female, Food_Allergy_Incidence_Submodule, General_Allergy_CarePlan, Immunotherapy_Submodule, Initial, Living_With_Allergies, Male, No_Infection, Not_Atopic, Potential_COPD_Nonsmoker, Potential_COPD_Smoker, Prescribe_Epipen, Prescribe_OTC_Antihistamine, Terminal.
    List of tActor: Care_Manager, Clinical_System, Lab_Technician, Pharmacist, Physician, Practitioner, Radiologist.

    List of aActionType: at_AllergyOnset, at_CallSubmodule,at_CarePlanStart,at_Delay,at_Encounter,at_EncounterEnd,at_Guard,at_Initial,at_MedicationOrder,at_Procedure,at_Simple,at_Terminal

    List of aWorflow: allergies, appendicitis, asthma, breast_cancer,colorectal_cancer,copd,dermatitis,ear_infections,epilepsy,fibromyalgia

    THE FORMULA LAYER (The "When" and "How"):
    
    You MUST represent the logical and temporal structure recursively in the formula field using ONLY these patterns:

        Atomic: Use the proposition ID string (e.g., "p1").

        LTL Operators:
        ! f (NOT), f | g (OR), f & g (AND), f -> g (Implies), f <-> g (IFF)
        G f (Always), H f (Hist. Always), F f (Eventually), O f (Once), X f (Next), Y f (Previous), f U g (Until), f S g (Since)

FREEZE OPERATOR
The freeze operator binds the value of a non-atomic attribute at a specific trace position to a variable (z) for use in a sub-formula.
Syntax: z.(<formula>)
Example: F x.(b & "(x)x[p]['a']>9")

Note 1: Access attributes via "x[<attribute_name>]".
Note 2: Use PROP.concept_name(x,y) for graph/SNOMED types.
Note 3: p in x[p] must be a predicate (gSnomed | aActivity | aActionType | aWorkflow | aActor).
==================================================================

NEVER mix clinical concepts into temporalStructure, and NEVER mix temporal constraints into propositions.
==================================================================
- CANONICAL TIME NORMALIZATION (REQUIRED): Convert all durations to seconds.
==================================================================

### REFERENCE ONTOLOGIES
#1) LTL Ontology:
{onto_logic}

#2) Log Ontology:
{onto_semantic}

Return ONLY valid JSON based on this template:
{{
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

### CLINICAL RULE:
"{text}"
"""

# ============================================================
# LLAMADA A OLLAMA (GEMMA)
# ============================================================

def call_gemma(prompt: str) -> Optional[ClinicalRuleOutput]:
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0.1  # Baja temperatura para consistencia lógica
                }
            },
            timeout=OLLAMA_TIMEOUT
        )
        response.raise_for_status()
        raw_output = response.json()["response"]
        
        # Limpieza y parsing
        data = json.loads(raw_output)
        return ClinicalRuleOutput(**data)
    except Exception as e:
        print(f"❌ Error en la compilación: {e}")
        return None

# ============================================================
# EJECUCIÓN PRINCIPAL
# ============================================================

if __name__ == "__main__":
    # Estas ontologías deberían cargarse de tus archivos .json
    # Aquí pongo un ejemplo vacío para que el script sea autoejecutable
    mock_log_onto = {"predicates": ["aActivity", "aActor", "aActionType", "gSnomed"]}
    mock_ltl_onto = {"operators": ["G", "F", "X", "U", "freeze"]}

    TEST_RULES = [
        "If an allergy is detected, epinephrine must be administered within 5 minutes.",
        "Every at_Simple recording must be followed by another at_Simple within 14400 seconds.",
        "If the workflow is dermatitis, the activity Prescribe_Epipen is strictly forbidden."
    ]

    print(f"🚀 COMPILADOR CLÍNICO INICIADO (MODELO: {OLLAMA_MODEL})")
    print("=" * 80)

    for i, rule in enumerate(TEST_RULES, 1):
        print(f"\n[{i}] PROCESANDO: {rule}")
        
        result = call_gemma(build_compiler_prompt(rule, mock_log_onto, mock_ltl_onto))
        
        if result:
            print("\n📦 CAPA SEMÁNTICA (Proposiciones):")
            for p in result.propositions:
                print(f"   • {p.id}: [{p.predicate}] {p.concept} ({p.type})")
            
            print("\n⚙️ CAPA DE FÓRMULA (LTL + Freeze):")
            print(f"   {result.formula}")
        else:
            print("   ⚠️ No se pudo generar la regla formal.")
        
        print("-" * 80)