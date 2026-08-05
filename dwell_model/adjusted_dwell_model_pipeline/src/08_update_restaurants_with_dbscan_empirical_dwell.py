from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


EPISODE_PATH = Path(
    r"C:\Users\negia\trip_plan\dwell_model\adjusted_dwell_model_pipeline\data\best_eps_100m_min_5\clean_episode_dwell_observations.csv"
)

RESTAURANTS_PATH = Path(
    r"C:\Users\negia\trip_plan\database\restaurants\clean_restaurant_2022.csv"
)

OUTPUT_PATH = Path(
    r"C:\Users\negia\trip_plan\database\restaurants\restaurants_dbscan_empirical_dwell.csv"
)

SUMMARY_PATH = Path(
    r"C:\Users\negia\trip_plan\database\restaurants\restaurants_dbscan_empirical_dwell_summary.json"
)

BACKUP_PATH = Path(
    r"C:\Users\negia\trip_plan\database\restaurants\restaurants_before_dbscan_empirical_dwell.csv"
)


def assign_meal_period(hour: int) -> str:
    """
    Broad meal windows for empirical food/leisure duration estimation.
    These are not claims about true meals; they are temporal buckets over
    food/leisure activity episodes.
    """
    if 5 <= hour < 11:
        return "breakfast"
    if 11 <= hour < 16:
        return "lunch"
    if 17 <= hour < 22:
        return "dinner"
    return "off_meal"


def summarize_duration(series: pd.Series) -> dict:
    series = pd.to_numeric(series, errors="coerce").dropna()

    if series.empty:
        return {
            "obs_count": 0,
            "p25": np.nan,
            "median": np.nan,
            "p75": np.nan,
            "mean": np.nan,
            "std": np.nan,
        }

    return {
        "obs_count": int(len(series)),
        "p25": float(series.quantile(0.25)),
        "median": float(series.median()),
        "p75": float(series.quantile(0.75)),
        "mean": float(series.mean()),
        "std": float(series.std()),
    }


def choose_conservative_meal_defaults(summary: dict) -> dict:
    """
    Convert empirical food/leisure activity episode estimates into practical
    restaurant defaults.

    We use conservative values because food/leisure episodes include cafes,
    bars, nightlife, social stops, and mixed leisure behaviour, not just
    restaurant service time.
    """
    general = summary["general_food_leisure"]

    # Fallbacks from overall food/leisure distribution.
    general_p25 = general["p25"]
    general_median = general["median"]

    defaults = {}

    for meal in ["breakfast", "lunch", "dinner"]:
        s = summary.get(meal, {})

        if s.get("obs_count", 0) >= 100:
            p25 = s["p25"]
            median = s["median"]
        else:
            p25 = general_p25
            median = general_median

        # Conservative meal defaults:
        # breakfast closer to p25, lunch between p25/median, dinner closer to median.
        if meal == "breakfast":
            value = p25
        elif meal == "lunch":
            value = (p25 + median) / 2
        else:
            value = median

        # Guardrails for TravelPlanner restaurant use.
        value = float(np.clip(value, 30, 90))
        defaults[meal] = round(value, 1)

    # General restaurant default: conservative midpoint between p25 and median.
    default_value = (general_p25 + general_median) / 2
    defaults["default"] = round(float(np.clip(default_value, 35, 85)), 1)

    return defaults


def main():
    if not EPISODE_PATH.exists():
        raise FileNotFoundError(f"Episode file not found: {EPISODE_PATH}")

    if not RESTAURANTS_PATH.exists():
        raise FileNotFoundError(f"Restaurants file not found: {RESTAURANTS_PATH}")

    episodes = pd.read_csv(EPISODE_PATH)
    restaurants = pd.read_csv(RESTAURANTS_PATH)

    print("Loaded clean episodes:", len(episodes))
    print("Loaded restaurants:", len(restaurants))

    if not BACKUP_PATH.exists():
        restaurants.to_csv(BACKUP_PATH, index=False)
        print("Saved backup:", BACKUP_PATH)

    required_episode_cols = [
        "dominant_category_group",
        "episode_start",
        "episode_duration_proxy_minutes",
    ]

    missing = [c for c in required_episode_cols if c not in episodes.columns]
    if missing:
        raise ValueError(f"Missing required columns in episode file: {missing}")

    food = episodes[
        episodes["dominant_category_group"] == "food_leisure"
    ].copy()

    food["episode_start"] = pd.to_datetime(food["episode_start"], errors="coerce")
    food = food.dropna(subset=["episode_start", "episode_duration_proxy_minutes"]).copy()

    food["hour"] = food["episode_start"].dt.hour
    food["meal_period"] = food["hour"].apply(assign_meal_period)

    summary = {
        "general_food_leisure": summarize_duration(food["episode_duration_proxy_minutes"])
    }

    for meal in ["breakfast", "lunch", "dinner", "off_meal"]:
        meal_df = food[food["meal_period"] == meal]
        summary[meal] = summarize_duration(meal_df["episode_duration_proxy_minutes"])

    defaults = choose_conservative_meal_defaults(summary)

    print("\nEmpirical food/leisure summary:")
    for key, val in summary.items():
        print(key, val)

    print("\nChosen restaurant defaults:")
    print(defaults)

    restaurants = restaurants.copy()

    restaurants["predicted_breakfast_dwell_minutes"] = defaults["breakfast"]
    restaurants["predicted_lunch_dwell_minutes"] = defaults["lunch"]
    restaurants["predicted_dinner_dwell_minutes"] = defaults["dinner"]

    # Generic value used if the restaurant tool/evaluator does not know meal period.
    restaurants["predicted_dwell_minutes"] = defaults["default"]

    restaurants["dwell_prediction_source"] = "dbscan_food_leisure_empirical"

    restaurants.to_csv(OUTPUT_PATH, index=False)

    output_summary = {
        "episode_path": str(EPISODE_PATH),
        "restaurants_path": str(RESTAURANTS_PATH),
        "output_path": str(OUTPUT_PATH),
        "rows_restaurants": int(len(restaurants)),
        "food_leisure_summary": summary,
        "chosen_defaults": defaults,
        "notes": [
            "Restaurant dwell values are empirical defaults derived from DBSCAN food/leisure activity episodes.",
            "They are not model predictions from the Random Forest attraction dwell model.",
            "Conservative defaults are used because food/leisure episodes include restaurants, cafes, bars, nightlife, and social leisure stops.",
        ],
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(output_summary, f, indent=4)

    print("\nSaved updated restaurants file:")
    print(OUTPUT_PATH)

    print("\nSaved summary:")
    print(SUMMARY_PATH)


if __name__ == "__main__":
    main()
