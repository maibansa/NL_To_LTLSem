import sys
from pyoxigraph import Store
import time

DB_PATH = "test/snomed/GeneradorSynthea/auto/log_1000_61_80"
store = Store(DB_PATH)

def ask_query(q):
    return bool(store.query(q))
