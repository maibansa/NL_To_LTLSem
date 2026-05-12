import json

INPUT_FILE = "result.json"
OUTPUT_FILE = "formulas.txt"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    for item in data:
        formula = item["formula"].replace("\n", " ").strip()
        out.write(formula + "\n")
