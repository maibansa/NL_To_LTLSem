#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, ValidationError
import argparse


# ============================================================
# FILE NAMES (as requested)
# ============================================================

INPUT_JSON = "results_output.json"
OUTPUT_JSON = "results_output_with_ltl.json"


# ============================================================
# Pydantic models (minimal, to parse item["ir"])
# ============================================================

class TemporalBound(BaseModel):
    value: float
    unit: str

class TemporalStructure(BaseModel):
    operator: str
    bound: Optional[TemporalBound] = None
    anchorField: str

class Proposition(BaseModel):
    predicate: str
    type: str
    concept: str
    role: str

class Step1IR(BaseModel):
    temporalStructure: TemporalStructure
    propositions: List[Proposition]


# ============================================================
# LTL generation (your functions, fixed to plain ASCII operators)
# ============================================================

def proposition_to_ltl_placeholder(p: Proposition, var: str) -> str:
    """
    Renderiza la proposición.
    Usa el 'concept' con guiones bajos para tipos atomic, string, number y boolean.
    """
    p_type = p.type.lower().strip()
    clean_concept = p.concept.replace(" ", "_")

    # 1) graph: PROP.<concept>(var)
    if p_type == "graph":
        concept_id = p.concept.strip().replace(" ", "_")
        return f"PROP.{concept_id}({var})"

    # 2) string: var[predicate]=='value'
    elif p_type == "string":
        return f"{var}[{p.predicate}]=='{clean_concept}'"

    # 3) number/boolean: var[predicate]==value
    elif p_type in ["number", "boolean"]:
        return f"{var}[{p.predicate}]=={clean_concept}"

    # 4) atomic: concept
    elif p_type == "atomic":
        return f"{clean_concept}"

    # fallback
    return f"{clean_concept}({var})"


def step15_to_ltl(ir: Step1IR) -> str:
    def norm_role(p: Proposition) -> str:
        return "condition" if p.role == "condition" else "action"

    conds = [p for p in ir.propositions if norm_role(p) == "condition"]
    acts  = [p for p in ir.propositions if norm_role(p) == "action"]

    x, y = "x", "y"
    t = ir.temporalStructure
    ts = t.anchorField

    def get_seconds():
        if not t.bound:
            return None
        # Policy B: IR already normalized to seconds
        return int(t.bound.value)

    def time_diff(v_x, v_y):
        return f"({v_y}[{ts}] - {v_x}[{ts}])"

    def cond_ltl(var):
        if not conds:
            return "TRUE"
        return " & ".join(proposition_to_ltl_placeholder(p, var) for p in conds)

    def act_ltl(var):
        if not acts:
            return "TRUE"
        return " & ".join(proposition_to_ltl_placeholder(p, var) for p in acts)

    op = (t.operator or "").lower()
    seconds = get_seconds()

    # --- Lógica de Plantillas ---
    if "every" in op:
        base = act_ltl(y)
        if seconds is None or seconds == 0:
            return f"G {x}.({cond_ltl(x)} -> F {y}.({base}))" if conds else f"G F {y}.({base})"
        return f"G {x}.({cond_ltl(x)} -> F {y}.({base} & {time_diff(x, y)} <= {seconds}))"

    if "within" in op:
        return f"G {x}.({cond_ltl(x)} -> X F {y}.({act_ltl(y)} & {time_diff(x, y)} <= {seconds}))"

    if "eventually" in op:
        return f"G {x}.({cond_ltl(x)} -> F {y}.({act_ltl(y)}))"

    if "always" in op:
        return f"G {x}.({cond_ltl(x)} -> {act_ltl(x)})"

    if "until" in op:
        return f"({cond_ltl(x)} U {act_ltl(y)})"

    raise ValueError(f"Unsupported temporal operator {t.operator}")


# ============================================================
# Main: read JSON, generate LTL, write JSON
# ============================================================

def generate_file(input_path: str, output_path: str, overwrite_ltl: bool = False, only_missing: bool = False) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of items (the results_output.json format).")

    n_total = 0
    n_ok = 0
    n_err = 0

    for item in data:
        n_total += 1

        if only_missing:
            existing = item.get("ltl", "")
            if isinstance(existing, str) and existing.strip() not in ("", "ERROR", "SKIPPED"):
                continue

        try:
            ir_obj = Step1IR(**item["ir"])
            ltl = step15_to_ltl(ir_obj)
            n_ok += 1
            if overwrite_ltl:
                item["ltl"] = ltl
            else:
                item["ltl_generated"] = ltl
        except (ValidationError, KeyError, TypeError, ValueError) as e:
            n_err += 1
            if overwrite_ltl:
                item["ltl"] = "ERROR"
            else:
                item["ltl_generated"] = "ERROR"
            item["ltl_generation_error"] = str(e)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Done. total={n_total}, ok={n_ok}, errors={n_err}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LTL from results_output.json using deterministic templates.")
    parser.add_argument("--input", default=INPUT_JSON, help=f"Input JSON (default: {INPUT_JSON})")
    parser.add_argument("--output", default=OUTPUT_JSON, help=f"Output JSON (default: {OUTPUT_JSON})")
    parser.add_argument("--overwrite-ltl", action="store_true", help="Overwrite field 'ltl' instead of writing 'ltl_generated'")
    parser.add_argument("--only-missing", action="store_true", help="Generate only if ltl is missing/empty/ERROR/SKIPPED")
    args = parser.parse_args()

    generate_file(args.input, args.output, overwrite_ltl=args.overwrite_ltl, only_missing=args.only_missing)