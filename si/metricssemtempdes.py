# metrics.py
from typing import Dict, Set

# Mapa de unidades comunes a normalizar
UNIT_MAP = {
    "h": "hours",
    "hr": "hours",
    "hrs": "hours",
    "hour": "hours",
    "hours": "hours",
    "d": "days",
    "day": "days",
    "days": "days"
}

# =========================
# UTILIDADES
# =========================
def normalize_text(text: str) -> str:
    if not text:
        return ""
    return text.strip().lower()

def normalize_bound(bound):
    if bound is None:
        return None
    unit = UNIT_MAP.get(bound.unit.lower(), bound.unit.lower())
    return type(bound)(value=float(bound.value), unit=unit, type=bound.type)

def _extract_entities(rule):
    entities: Set = set()
    for e in [rule.condition, rule.action]:
        entities.add((
            normalize_text(e.type),
            normalize_text(e.predicate),
            normalize_text(e.subject),
            e.negated
        ))
    return entities

# =========================
# EXACT MATCH
# =========================
def exact_match(pred, ref) -> int:
    """
    Match estricto de toda la estructura. Solo útil si los campos ya están normalizados.
    """
    return int(pred.model_dump() == ref.model_dump())

# =========================
# F1 ENTITIES
# =========================
def f1_entities(pred, ref) -> Dict:
    pred_set = _extract_entities(pred)
    ref_set = _extract_entities(ref)

    tp = len(pred_set & ref_set)
    fp = len(pred_set - ref_set)
    fn = len(ref_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {"precision": precision, "recall": recall, "f1": f1}

# =========================
# TEMPORAL ACCURACY
# =========================
def temporal_accuracy(pred, ref) -> float:
    score = 0
    total = 3  # operator + value + unit

    # Operator
    if normalize_text(pred.temporal.operator) == normalize_text(ref.temporal.operator):
        score += 1

    pred_bound = normalize_bound(pred.temporal.bound)
    ref_bound = normalize_bound(ref.temporal.bound)

    if pred_bound and ref_bound:
        if float(pred_bound.value) == float(ref_bound.value):
            score += 1
        if normalize_text(pred_bound.unit) == normalize_text(ref_bound.unit):
            score += 1
    elif pred_bound is None and ref_bound is None:
        score += 2  # si ambos no tienen bound

    return score / total

# =========================
# EVALUACIÓN GLOBAL
# =========================
def evaluate(pred, ref) -> Dict:
    return {
        "exact_match": exact_match(pred, ref),
        "f1_entities": f1_entities(pred, ref),
        "temporal_accuracy": temporal_accuracy(pred, ref)
    }