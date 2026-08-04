import argparse
import json
import re
from pathlib import Path


FIELDS = [
    "Current City",
    "Transportation",
    "Breakfast",
    "Attraction",
    "Lunch",
    "Dinner",
    "Accommodation",
]


def get_result_key(model_name, mode):
    return f"{model_name}_{mode}_results"

def get_parsed_key(model_name, mode):
    return f"{model_name}_{mode}_parsed_results"


def clean_value(x):
    if x is None:
        return "-"
    x = str(x).strip()
    return x if x else "-"


def fallback_parse_plan(text):
    """
    Parse natural-language TravelPlanner output directly.

    Expected pattern:
    Day 1:
    Current City: ...
    Transportation: ...
    Breakfast: ...
    Attraction: ...
    Lunch: ...
    Dinner: ...
    Accommodation: ...
    """

    if not text or not isinstance(text, str):
        return None

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"^\s*Travel Plan:\s*", "", text.strip(), flags=re.IGNORECASE)

    # Split while keeping day numbers.
    day_matches = list(re.finditer(r"(?im)^\s*Day\s*(\d+)\s*:\s*", text))
    if not day_matches:
        return None

    parsed_days = []

    for i, match in enumerate(day_matches):
        day_num = int(match.group(1))
        start = match.end()
        end = day_matches[i + 1].start() if i + 1 < len(day_matches) else len(text)
        block = text[start:end].strip()

        item = {"days": day_num}

        for idx, field in enumerate(FIELDS):
            # Capture from "Field:" until the next known field or end of block.
            next_fields = "|".join([re.escape(f) for f in FIELDS])
            pattern = rf"(?is){re.escape(field)}\s*:\s*(.*?)(?=\n\s*(?:{next_fields})\s*:|\Z)"
            m = re.search(pattern, block)
            key = field.lower().replace(" ", "_")
            item[key] = clean_value(m.group(1)) if m else "-"

        parsed_days.append(item)

    return parsed_days if parsed_days else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--set_type", default="validation")
    parser.add_argument("--model_name", default="gpt-3.5-turbo-0125")
    parser.add_argument("--mode", default="two-stage")
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()

    folder = Path(args.output_dir) / args.set_type

    result_key = get_result_key(args.model_name, args.mode)
    parsed_key = get_parsed_key(args.model_name, args.mode)

    repaired = 0
    already_ok = 0
    failed = 0

    for idx in range(1, args.num_samples + 1):
        path = folder / f"generated_plan_{idx}.json"

        if not path.exists():
            print(f"[missing] {idx}: {path}")
            failed += 1
            continue

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list) or not data:
            print(f"[bad-json-shape] {idx}")
            failed += 1
            continue

        obj = data[0]

        if obj.get(parsed_key) not in [None, "", []]:
            already_ok += 1
            continue

        raw_plan = obj.get(result_key, "")
        parsed = fallback_parse_plan(raw_plan)

        if parsed:
            obj[parsed_key] = parsed

            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)

            print(f"[repaired] {idx}: {len(parsed)} days")
            repaired += 1
        else:
            print(f"[failed] {idx}: could not parse raw plan")
            failed += 1

    print("\nSummary")
    print("Already OK:", already_ok)
    print("Repaired:", repaired)
    print("Failed:", failed)


if __name__ == "__main__":
    main()