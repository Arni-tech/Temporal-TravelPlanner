import os, sys
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from commonsense_constraint import evaluation as commonsense_eval
from hard_constraint import evaluation as hard_eval

import json
from tqdm import tqdm
from datasets import load_dataset
import argparse


def load_line_json_data(filename):
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            return data
        for line in content.split("\n"):
            data.append(json.loads(line))
    return data


def count_true_false(data):
    return data.count(True), data.count(False)


def statistics(constraint_statistic):
    result = {
        level: {day: {} for day in constraint_statistic[level]}
        for level in constraint_statistic
    }

    for level, days in constraint_statistic.items():
        for day, dicts in days.items():
            for dct in dicts:
                if dct:
                    for key, data in dct.items():
                        true_count, false_count = count_true_false(data)
                        if key not in result[level][day]:
                            result[level][day][key] = {"true": 0, "false": 0}
                        result[level][day][key]["true"] += true_count
                        result[level][day][key]["false"] += false_count

    return result


def paper_term_mapping(commonsense_constraint_record, hard_constraint_record):
    mapping_dict = {
        "is_valid_information_in_current_city": "Within Current City",
        "is_valid_information_in_sandbox": "Within Sandbox",
        "is_reasonable_visiting_city": "Reasonable City Route",
        "is_valid_restaurants": "Diverse Restaurants",
        "is_valid_transportation": "Non-conf. Transportation",
        "is_valid_attractions": "Diverse Attractions",
        "is_valid_accommodation": "Minimum Nights Stay",
        "is_not_absent": "Complete Information",
        "valid_cost": "Budget",
        "valid_room_rule": "Room Rule",
        "valid_cuisine": "Cuisine",
        "valid_room_type": "Room Type",
        "valid_transportation": "Transportation",
    }

    remap_commonsense = {level: {day: {} for day in [3, 5, 7]} for level in ["easy", "medium", "hard"]}
    remap_hard = {level: {day: {} for day in [3, 5, 7]} for level in ["easy", "medium", "hard"]}

    for level in commonsense_constraint_record:
        for day in commonsense_constraint_record[level]:
            remap_commonsense[level][day] = {
                mapping_dict.get(key, key): val
                for key, val in commonsense_constraint_record[level][day].items()
            }
            remap_hard[level][day] = {
                mapping_dict.get(key, key): val
                for key, val in hard_constraint_record[level][day].items()
            }

    return remap_commonsense, remap_hard


def load_queries(set_type, query_file=None):
    if query_file is not None:
        return load_line_json_data(query_file)

    project_root = Path(__file__).resolve().parents[1]
    local_query_file = project_root / "database" / f"{set_type}_ref_info.jsonl"

    if local_query_file.exists():
        return load_line_json_data(local_query_file)

    if set_type == "train":
        return load_dataset("osunlp/TravelPlanner", "train")["train"]
    if set_type == "validation":
        return load_dataset("osunlp/TravelPlanner", "validation")["validation"]
    if set_type == "test":
        return load_dataset("osunlp/TravelPlanner", "test")["test"]

    raise ValueError(f"Unknown set_type: {set_type}")

def eval_score(set_type: str, file_path: str, num_samples: int = None, query_file: str = None):
    query_data_list = [x for x in load_queries(set_type, query_file=query_file)]

    if num_samples is not None:
        query_data_list = query_data_list[:num_samples]

    tested_plans = load_line_json_data(file_path)

    if len(tested_plans) < len(query_data_list):
        raise ValueError(
            f"Evaluation file has {len(tested_plans)} plans, but {len(query_data_list)} queries are being evaluated. "
            f"Use --num_samples {len(tested_plans)} or generate more plans."
        )

    tested_plans = tested_plans[:len(query_data_list)]

    hardConstraint_statistic = {
        level: {day: [] for day in [3, 5, 7]}
        for level in ["easy", "medium", "hard"]
    }
    commonsenseConstraint_statistic = {
        level: {day: [] for day in [3, 5, 7]}
        for level in ["easy", "medium", "hard"]
    }

    delivery_cnt = 0
    plan_constraint_store = []

    for idx in tqdm(range(len(query_data_list))):
        query_data = query_data_list[idx]
        tested_plan = tested_plans[idx]

        if isinstance(query_data, str):
            query_data = eval(query_data)

        if isinstance(tested_plan, str):
            tested_plan = eval(tested_plan)

        if isinstance(query_data["local_constraint"], str):
            query_data["local_constraint"] = eval(query_data["local_constraint"])

        if tested_plan.get("plan"):
            delivery_cnt += 1
            commonsense_info_box = commonsense_eval(query_data, tested_plan["plan"])
        else:
            commonsense_info_box = None

        if (
            commonsense_info_box
            and commonsense_info_box["is_not_absent"][0]
            and commonsense_info_box["is_valid_information_in_sandbox"][0]
        ):
            hard_info_box = hard_eval(query_data, tested_plan["plan"])
        else:
            hard_info_box = None

        plan_constraint_store.append({
            "commonsense_constraint": commonsense_info_box,
            "hard_constraint": hard_info_box,
        })

        commonsenseConstraint_statistic[query_data["level"]][query_data["days"]].append(commonsense_info_box)
        hardConstraint_statistic[query_data["level"]][query_data["days"]].append(hard_info_box)

    constraint_record = {
        key: {day: {"house rule": 0, "cuisine": 0, "room type": 0, "transportation": 0} for day in [3, 5, 7]}
        for key in ["medium", "hard"]
    }

    constraint_mapping = {
        "house rule": "valid_room_rule",
        "cuisine": "valid_cuisine",
        "room type": "valid_room_type",
        "transportation": "valid_transportation",
    }

    mapping_constraint_record = {
        key: {day: {"valid_room_rule": 0, "valid_cuisine": 0, "valid_room_type": 0, "valid_transportation": 0} for day in [3, 5, 7]}
        for key in ["medium", "hard"]
    }

    count_record = {
        key: {day: 0 for day in [3, 5, 7]}
        for key in ["easy", "medium", "hard"]
    }

    for unit in query_data_list:
        if isinstance(unit["local_constraint"], str):
            unit["local_constraint"] = eval(unit["local_constraint"])

        count_record[unit["level"]][unit["days"]] += 1

        for key in constraint_record["medium"][3]:
            if unit["local_constraint"].get(key) is not None:
                constraint_record[unit["level"]][unit["days"]][key] += 1
                mapping_constraint_record[unit["level"]][unit["days"]][constraint_mapping[key]] += 1

    commonsense_processed = statistics(commonsenseConstraint_statistic)
    hard_processed = statistics(hardConstraint_statistic)

    data_record = {
        key: {day: [] for day in [3, 5, 7]}
        for key in ["easy", "medium", "hard"]
    }

    constraint_dis_record = {
        "commonsense": {"pass": 0, "total": 0},
        "hard": {"pass": 0, "total": 0},
    }

    constraint_count = {
        key: {day: {} for day in [3, 5, 7]}
        for key in ["easy", "medium", "hard"]
    }

    key_dict = {
        "commonsense": [
            "is_valid_information_in_current_city",
            "is_valid_information_in_sandbox",
            "is_reasonable_visiting_city",
            "is_valid_restaurants",
            "is_valid_transportation",
            "is_valid_attractions",
            "is_valid_accommodation",
            "is_not_absent",
        ],
        "hard": [
            "valid_cost",
            "valid_room_rule",
            "valid_cuisine",
            "valid_room_type",
            "valid_transportation",
        ],
    }

    for constraint in ["commonsense", "hard"]:
        constraint_statistic = commonsense_processed if constraint == "commonsense" else hard_processed

        for level in constraint_statistic:
            for day in constraint_statistic[level]:
                for key in key_dict[constraint]:
                    data_record[level][day].append("0/0")

                    if key not in constraint_statistic[level][day]:
                        continue

                    true_count = constraint_statistic[level][day][key]["true"]
                    constraint_dis_record[constraint]["pass"] += true_count

                    if constraint == "hard":
                        if level == "hard" and key in ["valid_room_rule", "valid_cuisine", "valid_room_type", "valid_transportation"]:
                            total = mapping_constraint_record[level][day][key]
                        elif level == "medium" and key in ["valid_room_rule", "valid_cuisine", "valid_room_type"]:
                            total = mapping_constraint_record[level][day][key]
                        else:
                            total = count_record[level][day] if key in ["valid_cost", "valid_visitng_city_number", "valid_days"] else 0

                        data_record[level][day][-1] = f"{true_count}/{total}"
                        constraint_dis_record[constraint]["total"] += total
                        hard_processed[level][day][key]["total"] = total

                    else:
                        total = count_record[level][day]
                        data_record[level][day][-1] = f"{true_count}/{total}"
                        constraint_dis_record[constraint]["total"] += total
                        constraint_count[level][day][key] = total
                        commonsense_processed[level][day][key]["total"] = total

    final_all_cnt = 0
    final_commonsense_cnt = 0
    final_hardConstraint_cnt = 0
    final_all_cnt_map = {level: 0 for level in ["easy", "medium", "hard"]}

    for idx in range(len(query_data_list)):
        if plan_constraint_store[idx]["commonsense_constraint"]:
            final_commonsense_pass = True
            final_hardConstraint_pass = True

            for item in plan_constraint_store[idx]["commonsense_constraint"]:
                value = plan_constraint_store[idx]["commonsense_constraint"][item][0]
                if value is not None and not value:
                    final_commonsense_pass = False
                    break

            if plan_constraint_store[idx]["hard_constraint"] is None:
                continue

            for item in plan_constraint_store[idx]["hard_constraint"]:
                value = plan_constraint_store[idx]["hard_constraint"][item][0]
                if value is not None and value is False:
                    final_hardConstraint_pass = False
                    break

            if final_commonsense_pass:
                final_commonsense_cnt += 1

            if final_hardConstraint_pass:
                final_hardConstraint_cnt += 1

            if final_commonsense_pass and final_hardConstraint_pass:
                final_all_cnt += 1
                final_all_cnt_map[query_data_list[idx]["level"]] += 1

    n = len(query_data_list)
    commonsense_total = constraint_dis_record["commonsense"]["total"]
    hard_total = constraint_dis_record["hard"]["total"]

    result = {
        "Delivery Rate": delivery_cnt / n if n else 0,
        "Commonsense Constraint Micro Pass Rate": constraint_dis_record["commonsense"]["pass"] / commonsense_total if commonsense_total else 0,
        "Commonsense Constraint Macro Pass Rate": final_commonsense_cnt / n if n else 0,
        "Hard Constraint Micro Pass Rate": constraint_dis_record["hard"]["pass"] / hard_total if hard_total else 0,
        "Hard Constraint Macro Pass Rate": final_hardConstraint_cnt / n if n else 0,
        "Final Pass Rate": final_all_cnt / n if n else 0,
    }

    remap_commonsense, remap_hard = paper_term_mapping(commonsense_processed, hard_processed)

    return result, {
        "Commonsense Constraint": remap_commonsense,
        "Hard Constraint": remap_hard,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--set_type", type=str, default="validation")
    parser.add_argument("--evaluation_file_path", type=str, default="./")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--temporal_eval", action="store_true")
    parser.add_argument("--temporal_output_dir", type=str, default=None)
    parser.add_argument("--temporal_system_name", type=str, default="current_system")
    parser.add_argument("--temporal_compare_output_dir", type=str, default=None)
    parser.add_argument("--temporal_compare_name", type=str, default="baseline")
    parser.add_argument(
        "--dwell_model_dir",
        type=str,
        default="dwell_model_exports",
    )
    parser.add_argument("--temporal_model_name", type=str, default="gpt-3.5-turbo-0125")
    parser.add_argument("--temporal_mode", type=str, default="two-stage")
    parser.add_argument("--temporal_save_dir", type=str, default=None)
    parser.add_argument("--query_file", type=str, default=None)

    args = parser.parse_args()

    scores, detailed_scores = eval_score(
        args.set_type,
        file_path=args.evaluation_file_path,
        num_samples=args.num_samples,
    )

    for key, value in scores.items():
        print(f"{key}: {value * 100}%")

    print("------------------")
    print(detailed_scores)
    print("------------------")
    if args.temporal_eval:
        if args.temporal_output_dir is None:
            raise ValueError("--temporal_output_dir is required when --temporal_eval is used.")

        from temporal_feasibility import run_temporal_eval_block

        run_temporal_eval_block(
            output_dir=args.temporal_output_dir,
            system_name=args.temporal_system_name,
            set_type=args.set_type,
            model_name=args.temporal_model_name,
            mode=args.temporal_mode,
            num_samples=args.num_samples,
            dwell_model_dir=args.dwell_model_dir,
            compare_output_dir=args.temporal_compare_output_dir,
            compare_name=args.temporal_compare_name,
            save_dir=args.temporal_save_dir,
        )
