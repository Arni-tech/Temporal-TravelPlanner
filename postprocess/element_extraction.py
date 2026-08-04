import argparse
from datasets import load_dataset
from tqdm import tqdm
import json
import ast


def extract_json_like_text(raw: str) -> str:
    raw = raw.strip()

    # Remove leading index written by openai_request.py, e.g. "1\t[ {...} ]"
    if "\t" in raw:
        raw = raw.split("\t", 1)[1].strip()

    # Case 1: ```json ... ```
    if "```json" in raw:
        return raw.split("```json", 1)[1].split("```", 1)[0].strip()

    # Case 2: ``` ... ```
    if "```" in raw:
        return raw.split("```", 1)[1].split("```", 1)[0].strip()

    # Case 3: plain JSON list
    start = raw.find("[")
    end = raw.rfind("]")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not find JSON list brackets.")

    return raw[start:end + 1].strip()


def parse_json_like(result: str):
    try:
        return json.loads(result)
    except json.JSONDecodeError:
        return ast.literal_eval(result)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--set_type", type=str, default="validation")
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo-0125")
    parser.add_argument("--mode", type=str, default="two-stage")
    parser.add_argument("--strategy", type=str, default="direct")
    parser.add_argument("--output_dir", type=str, default="./")
    parser.add_argument("--tmp_dir", type=str, default="./")
    parser.add_argument("--num_samples", type=int, default=5)

    args = parser.parse_args()

    if args.mode == "two-stage":
        suffix = ""
    elif args.mode == "sole-planning":
        suffix = f"_{args.strategy}"
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    result_path = f"{args.tmp_dir}/{args.set_type}_{args.model_name}{suffix}_{args.mode}.txt"

    with open(result_path, "r", encoding="utf-8") as f:
        results = f.read().strip().split("\n")

    if args.set_type == "train":
        query_data_list = load_dataset("osunlp/TravelPlanner", "train")["train"]
    elif args.set_type == "validation":
        query_data_list = load_dataset("osunlp/TravelPlanner", "validation")["validation"]
    elif args.set_type == "test":
        query_data_list = load_dataset("osunlp/TravelPlanner", "test")["test"]
    else:
        raise ValueError(f"Unknown set_type: {args.set_type}")

    max_available = min(args.num_samples, len(results), len(query_data_list))
    idx_number_list = list(range(1, max_available + 1))

    parsed_key = f"{args.model_name}{suffix}_{args.mode}_parsed_results"
    result_key = f"{args.model_name}{suffix}_{args.mode}_results"

    for idx in tqdm(idx_number_list):
        generated_plan_path = f"{args.output_dir}/{args.set_type}/generated_plan_{idx}.json"

        with open(generated_plan_path, "r", encoding="utf-8") as f:
            generated_plan = json.load(f)

        original_result = generated_plan[-1].get(result_key)

        if original_result not in ["", "Max Token Length Exceeded.", None]:
            raw = results[idx - 1]

            try:
                result_text = extract_json_like_text(raw)
                parsed = parse_json_like(result_text)
                generated_plan[-1][parsed_key] = parsed

            except Exception as e:
                print(f"\nFailed to parse plan {idx}")
                print("Raw output:")
                print(raw)
                print("\nError:")
                print(e)
                generated_plan[-1][parsed_key] = None

        else:
            generated_plan[-1][parsed_key] = None

        with open(generated_plan_path, "w", encoding="utf-8") as f:
            json.dump(generated_plan, f, indent=4)