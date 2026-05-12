import sys
from pyoxigraph import Store
import time

# Configuración: La ruta a tu carpeta de base de datos
path_auto = "./test/snomed/GeneradorSynthea/auto"
DB_PATH = "test/snomed/GeneradorSynthea/auto/log_1000_61_80"

# Inicializamos el store una sola vez para todas las consultas
store = Store(DB_PATH)

def ask_query(q):
    t0 = time.perf_counter()
    result = bool(store.query(q))
    t1 = time.perf_counter()
    # print(f"   ask_query: {t1-t0:.4f}s -> {result}")
    return result

def Asthma(g1):
    q = """PREFIX snomed: <http://snomed.info/id/>
PREFIX ont: <http://example.org/ontology/>
ASK { ?e ont:clinicalCode snomed:195967001 . }"""
    return ask_query(q)

def Inhaled_steroid_therapy(g1):
    q = """PREFIX snomed: <http://snomed.info/id/>
PREFIX ont: <http://example.org/ontology/>
ASK { ?e ont:clinicalCode snomed:710818004 . }"""
    return ask_query(q)

def Gynecology_service(g1):
    q = """PREFIX snomed: <http://snomed.info/id/>
PREFIX ont: <http://example.org/ontology/>
ASK { ?e ont:clinicalCode snomed:310061009 . }"""
    return ask_query(q)

def Smoking_cessation_therapy(g1):
    q = """PREFIX snomed: <http://snomed.info/id/>
PREFIX ont: <http://example.org/ontology/>
ASK { ?e ont:clinicalCode snomed:710081004 . }"""
    return ask_query(q)

def Atopic_dermatitis(g1):
    q = """PREFIX snomed: <http://snomed.info/id/>
PREFIX ont: <http://example.org/ontology/>
ASK { ?e ont:clinicalCode snomed:24079001 . }"""
    return ask_query(q)

def Administration_of_intravenous_fluids(g1):
    q = """PREFIX snomed: <http://snomed.info/id/>
PREFIX ont: <http://example.org/ontology/>
ASK { ?e ont:clinicalCode snomed:103744005 . }"""
    return ask_query(q)

def Malignant_neoplasm_of_breast(g1):
    q = """PREFIX snomed: <http://snomed.info/id/>
PREFIX ont: <http://example.org/ontology/>
ASK { ?e ont:clinicalCode snomed:254837009 . }"""
    return ask_query(q)

def Allergic_disposition(g1):
    q = """PREFIX snomed: <http://snomed.info/id/>
PREFIX ont: <http://example.org/ontology/>
ASK { ?e ont:clinicalCode snomed:609328004 . }"""
    return ask_query(q)

def Allergy_screening_test(g1):
    q = """PREFIX snomed: <http://snomed.info/id/>
PREFIX ont: <http://example.org/ontology/>
ASK { ?e ont:clinicalCode snomed:395142003 . }"""
    return ask_query(q)

def Asthma_screening(g1):
    q = """PREFIX snomed: <http://snomed.info/id/>
PREFIX ont: <http://example.org/ontology/>
ASK { ?e ont:clinicalCode snomed:171231001 . }"""
    return ask_query(q)

def Contact_dermatitis(g1):
    q = """PREFIX snomed: <http://snomed.info/id/>
PREFIX ont: <http://example.org/ontology/>
ASK { ?e ont:clinicalCode snomed:40275004 . }"""
    return ask_query(q)

def Chronic_obstructive_bronchitis(g1):
    q = """PREFIX snomed: <http://snomed.info/id/>
PREFIX ont: <http://example.org/ontology/>
ASK { ?e ont:clinicalCode snomed:185086009 . }"""
    return ask_query(q)

def Malignant_neoplasm_of_colon(g1):
    q = """PREFIX snomed: <http://snomed.info/id/>
PREFIX ont: <http://example.org/ontology/>
ASK { ?e ont:clinicalCode snomed:93761005 . }"""
    return ask_query(q)

def Surgical_procedure(g1):
    q = """PREFIX snomed: <http://snomed.info/id/>
PREFIX ont: <http://example.org/ontology/>
ASK { ?e ont:clinicalCode snomed:73761001 . }"""
    return ask_query(q)

def Therapeutic_procedure(g1):
    q = """PREFIX snomed: <http://snomed.info/id/>
PREFIX ont: <http://example.org/ontology/>
ASK { ?e ont:clinicalCode snomed:737567002 . }"""
    return ask_query(q)

