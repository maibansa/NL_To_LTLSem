import sys
from pyoxigraph import Store
import time

DB_PATH = "test/snomed/GeneradorSynthea/auto/log_1000_61_80"
store = Store(DB_PATH)

def ask_query(q):
    return bool(store.query(q))

def atopic_condition(g1):
    q = f"""

PREFIX snomed: &lt;http://snomed.info/id/&gt;

ASK {
    ?patient snomed:hasCondition snomed:24079001 .
}

    """
    return ask_query(q)

def allergy_unspecified(g1):
    q = f"""

PREFIX snomed: &lt;http://snomed.info/id/&gt;

ASK {
    ?patient snomed:hasCondition snomed:426232007 .
}

    """
    return ask_query(q)

def respiratory_reaction(g1):
    q = f"""

PREFIX snomed: &lt;http://snomed.info/id/&gt;

ASK {
    ?patient snomed:hasCondition snomed:293851003 .
}

    """
    return ask_query(q)

def environmental_allergy(g1):
    q = f"""

PREFIX snomed: &lt;http://snomed.info/id/&gt;

ASK {
    ?patient snomed:hasCondition snomed:426232007 .
}

    """
    return ask_query(q)
