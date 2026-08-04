import argparse
import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset


MEAL_MINUTES = {
    "breakfast": 30.0,
    "lunch": 45.0,
    "dinner": 60.0,
}


# -----------------------------
# Basic helpers
# -----------------------------

def safe_literal_eval(x):
    if x is None:
        return None

    if isinstance(x, (list, dict)):
        return x

    try:
        return ast.literal_eval(x)
    except Exception:
        return None


def normalize_text(x):
    if x is None:
        return ""

    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass

    x = str(x).lower().strip()
    x = re.sub(r"[^a-z0-9]+", " ", x)
    x = re.sub(r"\s+", " ", x).strip()

    return x


def is_missing_value(x):
    if x is None:
        return True

    x = str(x).strip()

    return x == "" or x == "-"


def split_semicolon_items(x):
    if is_missing_value(x):
        return []

    parts = [p.strip() for p in str(x).split(";")]

    return [p for p in parts if p and p != "-"]


def extract_name_city(item):
    """
    Example:
    'SkyWheel Myrtle Beach, Myrtle Beach'
    ->
    ('SkyWheel Myrtle Beach', 'Myrtle Beach')
    """
    item = str(item).strip()

    if "," in item:
        name, city = item.rsplit(",", 1)
        return name.strip(), city.strip()

    return item.strip(), ""


def parse_duration_minutes(text):
    """
    Extracts transport duration if explicitly present.

    Handles examples:
    - 'Duration: 6 hours 47 mins'
    - '1 hours 40 minutes'
    - '47 minutes'
    """
    if is_missing_value(text):
        return 0.0

    s = str(text).lower()

    m = re.search(
        r"(\d+(?:\.\d+)?)\s*hours?\s*(\d+(?:\.\d+)?)?\s*(?:mins?|minutes?)?",
        s,
    )

    if m:
        hours = float(m.group(1))
        mins = float(m.group(2)) if m.group(2) else 0.0
        return hours * 60.0 + mins

    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:mins?|minutes?)", s)

    if m:
        return float(m.group(1))

    return 0.0


# -----------------------------
# Attraction lookup
# -----------------------------

def load_attraction_lookup(attraction_path):
    df = pd.read_csv(attraction_path)

    required = ["Name", "City", "predicted_dwell_minutes"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns in attraction file: {missing}")

    df["name_norm"] = df["Name"].apply(normalize_text)
    df["city_norm"] = df["City"].apply(normalize_text)

    name_city_lookup = (
        df.groupby(["name_norm", "city_norm"], as_index=False)["predicted_dwell_minutes"]
        .median()
    )

    name_only_lookup = (
        df.groupby("name_norm", as_index=False)["predicted_dwell_minutes"]
        .median()
    )

    global_fallback_dwell = float(df["predicted_dwell_minutes"].median())

    return df, name_city_lookup, name_only_lookup, global_fallback_dwell


def lookup_dwell(name, city, name_city_lookup, name_only_lookup):
    name_norm = normalize_text(name)
    city_norm = normalize_text(city)

    if not name_norm:
        return np.nan, "empty"

    # First: exact name + city.
    if city_norm:
        match = name_city_lookup[
            (name_city_lookup["name_norm"] == name_norm)
            & (name_city_lookup["city_norm"] == city_norm)
        ]

        if len(match) > 0:
            return float(match["predicted_dwell_minutes"].iloc[0]), "matched_name_city"

    # Second: name-only fallback.
    match = name_only_lookup[name_only_lookup["name_norm"] == name_norm]

    if len(match) > 0:
        return float(match["predicted_dwell_minutes"].iloc[0]), "matched_name_only"

    return np.nan, "fallback_unmatched"


def collect_train_matched_dwell_values(train_plans, name_city_lookup, name_only_lookup):
    """
    Uses only attractions that appear in human-annotated TravelPlanner train plans.

    This produces a train-calibrated fallback dwell value, which is more consistent
    with the human-plan threshold calibration than using the full attraction database.
    """
    dwell_values = []
    missing_items = []

    for plan_obj in train_plans:
        for day in plan_obj["plan"]:
            attraction_items = split_semicolon_items(day.get("attraction", "-"))

            for item in attraction_items:
                name, city = extract_name_city(item)
                dwell, source = lookup_dwell(name, city, name_city_lookup, name_only_lookup)

                if pd.isna(dwell):
                    missing_items.append(item)
                else:
                    dwell_values.append(float(dwell))

    if not dwell_values:
        raise ValueError("Could not compute train-calibrated fallback: no matched train attractions.")

    train_fallback_dwell = float(np.median(dwell_values))

    return train_fallback_dwell, dwell_values, missing_items


# -----------------------------
# Plan extraction
# -----------------------------

def extract_plan_from_annotated_plan(raw):
    """
    Extracts real day dictionaries from TravelPlanner train annotated_plan.
    Removes padding {} entries.
    """
    obj = safe_literal_eval(raw)

    if obj is None:
        return None

    candidate = None

    if isinstance(obj, list):
        for part in obj:
            if isinstance(part, list) and part and all(isinstance(x, dict) for x in part):
                if any("current_city" in x or "attraction" in x for x in part):
                    candidate = part
                    break

    if candidate is None:
        return None

    cleaned = []

    for d in candidate:
        if not isinstance(d, dict):
            continue

        if not d:
            continue

        if "days" not in d and "day" not in d:
            continue

        if any(
            k in d
            for k in [
                "current_city",
                "transportation",
                "breakfast",
                "attraction",
                "lunch",
                "dinner",
                "accommodation",
            ]
        ):
            cleaned.append(d)

    return cleaned if cleaned else None


def load_train_plans(set_type="train"):
    train = load_dataset("osunlp/TravelPlanner", "train")["train"]

    train_plans = []
    failed = 0

    for i, row in enumerate(train):
        plan = extract_plan_from_annotated_plan(row["annotated_plan"])

        if plan:
            train_plans.append(
                {
                    "idx": i,
                    "query": row["query"],
                    "level": row["level"],
                    "days_expected": row["days"],
                    "plan": plan,
                }
            )
        else:
            failed += 1

    return train_plans, failed, len(train)


def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


# -----------------------------
# Day-load logic
# -----------------------------

def meal_load_minutes(day):
    total = 0.0

    for meal, minutes in MEAL_MINUTES.items():
        if not is_missing_value(day.get(meal, "-")):
            total += minutes

    return total


def attraction_load_minutes(day, name_city_lookup, name_only_lookup, fallback_dwell):
    items = split_semicolon_items(day.get("attraction", "-"))

    total = 0.0
    attraction_count = 0
    matched_count = 0
    fallback_count = 0
    missing_items = []
    match_sources = []

    for item in items:
        name, city = extract_name_city(item)

        if is_missing_value(name):
            continue

        attraction_count += 1

        dwell, source = lookup_dwell(
            name,
            city,
            name_city_lookup,
            name_only_lookup,
        )

        if pd.isna(dwell):
            total += fallback_dwell
            fallback_count += 1
            missing_items.append(item)
            match_sources.append("fallback_unmatched")
        else:
            total += float(dwell)
            matched_count += 1
            match_sources.append(source)

    return {
        "attraction_minutes": total,
        "attraction_count": attraction_count,
        "matched_attraction_count": matched_count,
        "fallback_attraction_count": fallback_count,
        "missing_attraction_count": fallback_count,
        "missing_attractions": missing_items,
        "match_sources": match_sources,
    }


def compute_day_load(day, name_city_lookup, name_only_lookup, fallback_dwell):
    attraction_info = attraction_load_minutes(
        day,
        name_city_lookup,
        name_only_lookup,
        fallback_dwell,
    )

    meal_minutes = meal_load_minutes(day)
    transport_minutes = parse_duration_minutes(day.get("transportation", "-"))

    activity_load_minutes = attraction_info["attraction_minutes"] + meal_minutes
    full_day_load_minutes = activity_load_minutes + transport_minutes

    return {
        "day": day.get("days", day.get("day", None)),
        "current_city": day.get("current_city", ""),
        "attraction_minutes": attraction_info["attraction_minutes"],
        "meal_minutes": meal_minutes,
        "transport_minutes": transport_minutes,
        "activity_load_minutes": activity_load_minutes,
        "full_day_load_minutes": full_day_load_minutes,
        "attraction_count": attraction_info["attraction_count"],
        "matched_attraction_count": attraction_info["matched_attraction_count"],
        "fallback_attraction_count": attraction_info["fallback_attraction_count"],
        "missing_attraction_count": attraction_info["missing_attraction_count"],
        "missing_attractions": attraction_info["missing_attractions"],
        "match_sources": "|".join(attraction_info["match_sources"]),
    }


# -----------------------------
# Train threshold calibration
# -----------------------------

def compute_train_day_loads(train_plans, name_city_lookup, name_only_lookup, train_fallback_dwell):
    rows = []

    for plan_obj in train_plans:
        for day in plan_obj["plan"]:
            day_row = compute_day_load(
                day,
                name_city_lookup,
                name_only_lookup,
                train_fallback_dwell,
            )

            day_row["plan_idx"] = plan_obj["idx"]
            day_row["level"] = plan_obj["level"]
            day_row["days_expected"] = plan_obj["days_expected"]
            day_row["query"] = plan_obj["query"]

            rows.append(day_row)

    return pd.DataFrame(rows)


def derive_thresholds(train_day_df):
    thresholds = {
        "activity_p75_train_human": float(train_day_df["activity_load_minutes"].quantile(0.75)),
        "activity_p90_train_human": float(train_day_df["activity_load_minutes"].quantile(0.90)),
        "activity_p95_train_human": float(train_day_df["activity_load_minutes"].quantile(0.95)),
        "full_day_p75_train_human": float(train_day_df["full_day_load_minutes"].quantile(0.75)),
        "full_day_p90_train_human": float(train_day_df["full_day_load_minutes"].quantile(0.90)),
        "full_day_p95_train_human": float(train_day_df["full_day_load_minutes"].quantile(0.95)),
    }

    return thresholds


# -----------------------------
# Submission evaluation
# -----------------------------

def evaluate_submission(
    submission_path,
    system_name,
    thresholds,
    name_city_lookup,
    name_only_lookup,
    fallback_dwell,
    num_samples=None,
):
    submissions = load_jsonl(submission_path)

    if num_samples is not None:
        submissions = submissions[:num_samples]

    attempted_plans = len(submissions)

    day_rows = []
    plan_rows = []

    for item in submissions:
        idx = item.get("idx")
        plan = item.get("plan")

        if not plan:
            plan_summary = {
                "system": system_name,
                "idx": idx,
                "valid_plan": False,
                "num_days": 0,
                "attempted": True,
            }

            for t_name in thresholds:
                plan_summary[f"plan_pass_{t_name}"] = False
                plan_summary[f"overloaded_days_{t_name}"] = None

            plan_rows.append(plan_summary)
            continue

        plan_day_rows = []

        for day in plan:
            row = compute_day_load(
                day,
                name_city_lookup,
                name_only_lookup,
                fallback_dwell,
            )

            row["system"] = system_name
            row["idx"] = idx

            day_rows.append(row)
            plan_day_rows.append(row)

        plan_summary = {
            "system": system_name,
            "idx": idx,
            "valid_plan": True,
            "num_days": len(plan_day_rows),
            "attempted": True,
        }

        for t_name, t_value in thresholds.items():
            load_col = "activity_load_minutes" if t_name.startswith("activity_") else "full_day_load_minutes"

            day_passes = [r[load_col] <= t_value for r in plan_day_rows]

            plan_summary[f"plan_pass_{t_name}"] = all(day_passes)
            plan_summary[f"overloaded_days_{t_name}"] = int(sum(not x for x in day_passes))

        plan_rows.append(plan_summary)

    day_df = pd.DataFrame(day_rows)
    plan_df = pd.DataFrame(plan_rows)

    if len(day_df) > 0:
        for t_name, t_value in thresholds.items():
            load_col = "activity_load_minutes" if t_name.startswith("activity_") else "full_day_load_minutes"

            day_df[f"day_pass_{t_name}"] = day_df[load_col] <= t_value
            day_df[f"overload_minutes_{t_name}"] = np.maximum(0, day_df[load_col] - t_value)

    return day_df, plan_df, attempted_plans


def summarize(day_df, plan_df, thresholds, attempted_plans):
    rows = []

    valid_plans = plan_df[plan_df["valid_plan"]].copy()
    evaluated_plans = int(valid_plans.shape[0])
    evaluated_days = int(day_df.shape[0])

    total_attractions = int(day_df["attraction_count"].sum()) if len(day_df) else 0
    total_matched = int(day_df["matched_attraction_count"].sum()) if len(day_df) else 0
    total_fallback = int(day_df["fallback_attraction_count"].sum()) if len(day_df) else 0

    fallback_rate = total_fallback / total_attractions if total_attractions else 0.0
    matched_rate = total_matched / total_attractions if total_attractions else 0.0

    delivery_rate = evaluated_plans / attempted_plans if attempted_plans else 0.0

    for t_name, t_value in thresholds.items():
        load_col = "activity_load_minutes" if t_name.startswith("activity_") else "full_day_load_minutes"

        if evaluated_plans > 0:
            evaluated_plan_feasibility = float(valid_plans[f"plan_pass_{t_name}"].mean())
            feasible_plan_count = int(valid_plans[f"plan_pass_{t_name}"].sum())
        else:
            evaluated_plan_feasibility = 0.0
            feasible_plan_count = 0

        delivery_adjusted_plan_success = feasible_plan_count / attempted_plans if attempted_plans else 0.0

        if evaluated_days > 0:
            day_feasibility = float(day_df[f"day_pass_{t_name}"].mean())
            overloaded_day_rate = float((~day_df[f"day_pass_{t_name}"]).mean())
            mean_load = float(day_df[load_col].mean())
            median_load = float(day_df[load_col].median())
            mean_overload = float(day_df[f"overload_minutes_{t_name}"].mean())
            mean_attraction_minutes = float(day_df["attraction_minutes"].mean())
            mean_meal_minutes = float(day_df["meal_minutes"].mean())
            mean_transport_minutes = float(day_df["transport_minutes"].mean())
            mean_attraction_count = float(day_df["attraction_count"].mean())
        else:
            day_feasibility = 0.0
            overloaded_day_rate = 0.0
            mean_load = 0.0
            median_load = 0.0
            mean_overload = 0.0
            mean_attraction_minutes = 0.0
            mean_meal_minutes = 0.0
            mean_transport_minutes = 0.0
            mean_attraction_count = 0.0

        rows.append(
            {
                "threshold": t_name,
                "threshold_minutes": float(t_value),
                "attempted_plans": int(attempted_plans),
                "evaluated_plans": int(evaluated_plans),
                "invalid_or_missing_plans": int(attempted_plans - evaluated_plans),
                "delivery_rate": float(delivery_rate),
                "evaluated_days": int(evaluated_days),
                "day_temporal_feasibility_rate": day_feasibility,
                "plan_temporal_feasibility_rate_evaluated_only": evaluated_plan_feasibility,
                "delivery_adjusted_plan_temporal_success": float(delivery_adjusted_plan_success),
                "feasible_plan_count": int(feasible_plan_count),
                "overloaded_day_rate": overloaded_day_rate,
                "mean_load_minutes": mean_load,
                "median_load_minutes": median_load,
                "mean_overload_minutes": mean_overload,
                "mean_attraction_minutes_per_day": mean_attraction_minutes,
                "mean_meal_minutes_per_day": mean_meal_minutes,
                "mean_transport_minutes_per_day": mean_transport_minutes,
                "mean_attraction_count_per_day": mean_attraction_count,
                "total_attractions": int(total_attractions),
                "total_matched_attractions": int(total_matched),
                "total_fallback_attractions": int(total_fallback),
                "matched_attraction_rate": float(matched_rate),
                "fallback_attraction_rate": float(fallback_rate),
            }
        )

    return pd.DataFrame(rows)


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--attraction_features", required=True)
    parser.add_argument("--baseline_submission", required=True)
    parser.add_argument("--dwell_submission", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=None)

    parser.add_argument("--baseline_name", type=str, default="baseline")
    parser.add_argument("--dwell_name", type=str, default="dwell_aware")

    args = parser.parse_args()

    attraction_features = Path(args.attraction_features)
    baseline_submission = Path(args.baseline_submission)
    dwell_submission = Path(args.dwell_submission)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    attractions, name_city_lookup, name_only_lookup, global_fallback_dwell = load_attraction_lookup(
        attraction_features
    )

    train_plans, failed_train_plans, total_train_plans = load_train_plans()

    train_fallback_dwell, train_dwell_values, train_missing_items = collect_train_matched_dwell_values(
        train_plans,
        name_city_lookup,
        name_only_lookup,
    )

    train_day_df = compute_train_day_loads(
        train_plans,
        name_city_lookup,
        name_only_lookup,
        train_fallback_dwell,
    )

    thresholds = derive_thresholds(train_day_df)

    print("Attraction rows:", len(attractions))
    print("Global fallback dwell from all attraction_features.csv:", round(global_fallback_dwell, 3))
    print("Train-calibrated fallback dwell:", round(train_fallback_dwell, 3))
    print("Train fallback source attractions:", len(train_dwell_values))

    print("Train annotated plans:", total_train_plans - failed_train_plans)
    print("Failed train plans:", failed_train_plans)
    print("Train evaluated days:", len(train_day_df))
    print("Train total attractions:", int(train_day_df["attraction_count"].sum()))
    print("Train fallback/missing attractions:", int(train_day_df["fallback_attraction_count"].sum()))

    if train_missing_items:
        print("Warning: some train attractions were unmatched.")
        print("First 10 unmatched train attractions:", train_missing_items[:10])

    print("\nDerived thresholds:")
    for k, v in thresholds.items():
        print(f"{k}: {v:.2f}")

    baseline_day_df, baseline_plan_df, baseline_attempted = evaluate_submission(
        baseline_submission,
        args.baseline_name,
        thresholds,
        name_city_lookup,
        name_only_lookup,
        train_fallback_dwell,
        num_samples=args.num_samples,
    )

    dwell_day_df, dwell_plan_df, dwell_attempted = evaluate_submission(
        dwell_submission,
        args.dwell_name,
        thresholds,
        name_city_lookup,
        name_only_lookup,
        train_fallback_dwell,
        num_samples=args.num_samples,
    )

    baseline_summary = summarize(
        baseline_day_df,
        baseline_plan_df,
        thresholds,
        attempted_plans=baseline_attempted,
    )

    dwell_summary = summarize(
        dwell_day_df,
        dwell_plan_df,
        thresholds,
        attempted_plans=dwell_attempted,
    )

    comparison = pd.concat(
        [
            baseline_summary.assign(system=args.baseline_name),
            dwell_summary.assign(system=args.dwell_name),
        ],
        ignore_index=True,
    )

    # Save outputs.
    train_day_df.to_csv(output_dir / "train_human_day_loads.csv", index=False)
    baseline_day_df.to_csv(output_dir / f"{args.baseline_name}_temporal_day_level.csv", index=False)
    dwell_day_df.to_csv(output_dir / f"{args.dwell_name}_temporal_day_level.csv", index=False)
    baseline_plan_df.to_csv(output_dir / f"{args.baseline_name}_temporal_plan_level.csv", index=False)
    dwell_plan_df.to_csv(output_dir / f"{args.dwell_name}_temporal_plan_level.csv", index=False)
    comparison.to_csv(output_dir / "temporal_feasibility_comparison.csv", index=False)

    metadata = {
        "global_fallback_dwell": global_fallback_dwell,
        "train_calibrated_fallback_dwell": train_fallback_dwell,
        "train_fallback_source_attractions": len(train_dwell_values),
        "meal_minutes": MEAL_MINUTES,
        "thresholds": thresholds,
        "num_samples": args.num_samples,
    }

    with open(output_dir / "temporal_eval_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nMain activity-load comparison:")
    main_rows = comparison[
        comparison["threshold"].isin(
            [
                "activity_p75_train_human",
                "activity_p90_train_human",
                "activity_p95_train_human",
            ]
        )
    ][
        [
            "system",
            "threshold",
            "threshold_minutes",
            "attempted_plans",
            "evaluated_plans",
            "invalid_or_missing_plans",
            "delivery_rate",
            "evaluated_days",
            "day_temporal_feasibility_rate",
            "plan_temporal_feasibility_rate_evaluated_only",
            "delivery_adjusted_plan_temporal_success",
            "overloaded_day_rate",
            "mean_load_minutes",
            "mean_overload_minutes",
            "mean_attraction_count_per_day",
            "total_attractions",
            "total_fallback_attractions",
            "fallback_attraction_rate",
        ]
    ]

    print(main_rows.to_string(index=False))

    print("\nSaved outputs to:", output_dir)


if __name__ == "__main__":
    main()