import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


ITINERARY_GROUPS = {
    "food_leisure",
    "shopping",
    "tourism_attraction",
}

CATEGORY_CAPS = {
    "food_leisure": (15, 180),
    "shopping": (10, 240),
    "tourism_attraction": (15, 240),
}


FOOD_KEYWORDS = [
    "restaurant", "cafe", "coffee", "bar", "pub", "bakery", "pizza",
    "burger", "diner", "bistro", "food", "sandwich", "bbq", "kebab",
    "noodle", "sushi", "steak", "seafood", "ice cream", "dessert",
    "donut", "tea", "breakfast", "lunch", "deli", "buffet",
    "nightclub", "club", "lounge", "karaoke", "beer garden",
]

SHOPPING_KEYWORDS = [
    "shop", "store", "mall", "market", "supermarket", "grocery",
    "bookstore", "book store", "clothing", "boutique", "department",
    "electronics", "jewelry", "shoe", "flea market", "shopping",
    "retail", "hardware", "furniture", "gift", "toy", "cosmetics",
]

TOURISM_KEYWORDS = [
    "museum", "park", "theater", "theatre", "cinema", "movie theater",
    "historic", "monument", "landmark", "gallery", "art", "zoo",
    "aquarium", "stadium", "arena", "church", "mosque", "temple",
    "cathedral", "scenic", "lookout", "waterfront", "beach", "plaza",
    "bridge", "garden", "castle", "palace", "attraction", "theme park",
    "concert hall", "music venue", "cultural center", "outdoors",
]


def normalize_text(x):
    x = "" if pd.isna(x) else str(x).lower()
    x = re.sub(r"[^a-z0-9\s/]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def assign_category_group(category):
    text = normalize_text(category)

    if any(k in text for k in FOOD_KEYWORDS):
        return "food_leisure"

    if any(k in text for k in SHOPPING_KEYWORDS):
        return "shopping"

    if any(k in text for k in TOURISM_KEYWORDS):
        return "tourism_attraction"

    return "exclude"


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0

    lat1 = np.radians(lat1.astype(float))
    lon1 = np.radians(lon1.astype(float))
    lat2 = np.radians(lat2.astype(float))
    lon2 = np.radians(lon2.astype(float))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    return 2 * r * np.arcsin(np.sqrt(a))


def infer_dataset_city(path):
    stem = Path(path).stem.lower()
    stem = stem.replace("_checkins", "")
    stem = stem.replace("checkins_", "")
    stem = stem.replace("-checkins", "")
    return stem


def normalize_city_name(x):
    return str(x).strip().lower().replace(" ", "_")


def load_all_csvs(input_dir):
    input_dir = Path(input_dir)
    files = sorted(input_dir.glob("*.csv"))

    if not files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    frames = []

    for path in files:
        print("Loading:", path)
        df = pd.read_csv(path)

        if "dataset_city" not in df.columns:
            df["dataset_city"] = infer_dataset_city(path)

        df["dataset_city"] = df["dataset_city"].apply(normalize_city_name)

        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def build_adjusted_dwell(df, min_obs=5):
    required = [
        "dataset_city",
        "trail_id",
        "user_id",
        "venue_id",
        "latitude",
        "longitude",
        "name",
        "venue_category",
        "venue_city",
        "venue_country",
        "timestamp",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()

    if "category_group" not in df.columns:
        print("category_group not found. Creating category_group from venue_category keywords.")
        df["category_group"] = df["venue_category"].apply(assign_category_group)

    df = df[df["category_group"].isin(ITINERARY_GROUPS)].copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    for col in ["latitude", "longitude"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=[
            "timestamp",
            "latitude",
            "longitude",
            "venue_id",
            "trail_id",
            "user_id",
        ]
    ).copy()

    df["global_venue_id"] = (
        df["dataset_city"].astype(str)
        + "::"
        + df["venue_id"].astype(str)
    )

    df = df.sort_values(
        ["dataset_city", "user_id", "trail_id", "timestamp"]
    ).copy()

    group_cols = ["dataset_city", "user_id", "trail_id"]

    df["next_timestamp"] = df.groupby(group_cols)["timestamp"].shift(-1)
    df["next_venue_id"] = df.groupby(group_cols)["venue_id"].shift(-1)
    df["next_latitude"] = df.groupby(group_cols)["latitude"].shift(-1)
    df["next_longitude"] = df.groupby(group_cols)["longitude"].shift(-1)
    df["next_category_group"] = df.groupby(group_cols)["category_group"].shift(-1)

    df["gap_minutes"] = (
        df["next_timestamp"] - df["timestamp"]
    ).dt.total_seconds() / 60

    gap_df = df[
        df["gap_minutes"].notna()
        & (df["gap_minutes"] > 0)
    ].copy()

    gap_df = gap_df[
        (gap_df["gap_minutes"] >= 5)
        & (gap_df["gap_minutes"] <= 240)
    ].copy()

    gap_df = gap_df.dropna(
        subset=["next_latitude", "next_longitude"]
    ).copy()

    gap_df["distance_to_next_poi_km"] = haversine_km(
        gap_df["latitude"],
        gap_df["longitude"],
        gap_df["next_latitude"],
        gap_df["next_longitude"],
    )

    gap_df["transition_speed_kmh"] = (
        gap_df["distance_to_next_poi_km"] / (gap_df["gap_minutes"] / 60)
    )

    gap_df["assumed_speed_kmh"] = np.where(
        gap_df["distance_to_next_poi_km"] <= 1.0,
        4.5,
        18.0,
    )

    gap_df["estimated_travel_time_min"] = (
        gap_df["distance_to_next_poi_km"]
        / gap_df["assumed_speed_kmh"]
        * 60
    )

    gap_df["adjusted_dwell_minutes"] = (
        gap_df["gap_minutes"] - gap_df["estimated_travel_time_min"]
    )

    gap_df["near_duplicate_transition"] = (
        gap_df["distance_to_next_poi_km"] <= 0.05
    ).astype(int)

    gap_df["date_changed"] = (
        gap_df["timestamp"].dt.date != gap_df["next_timestamp"].dt.date
    ).astype(int)

    gap_df["next_hour"] = gap_df["next_timestamp"].dt.hour

    gap_df["overnight_transition"] = (
        (gap_df["date_changed"] == 1)
        | (gap_df["next_hour"].between(3, 6))
    ).astype(int)

    gap_df["category_min_dwell"] = gap_df["category_group"].map(
        lambda g: CATEGORY_CAPS.get(g, (5, 240))[0]
    )

    gap_df["category_max_dwell"] = gap_df["category_group"].map(
        lambda g: CATEGORY_CAPS.get(g, (5, 240))[1]
    )

    clean_gap = gap_df[
        gap_df["adjusted_dwell_minutes"].notna()
        & (gap_df["adjusted_dwell_minutes"] >= gap_df["category_min_dwell"])
        & (gap_df["adjusted_dwell_minutes"] <= gap_df["category_max_dwell"])
        & (gap_df["distance_to_next_poi_km"] <= 10)
        & (gap_df["transition_speed_kmh"] <= 80)
        & (gap_df["overnight_transition"] == 0)
    ].copy()

    clean_gap["gap_observation_id"] = np.arange(len(clean_gap))

    # This POI table is only for diagnostics and Google-merge count checking.
    # Final deduplicated target is recomputed later from clean_gap after Google place_id matching.
    poi_debug = (
        clean_gap
        .groupby("global_venue_id")
        .agg(
            poi_name=("name", "first"),
            venue_category=("venue_category", "first"),
            category_group=("category_group", "first"),
            dataset_city=("dataset_city", "first"),
            venue_city=("venue_city", "first"),
            venue_country=("venue_country", "first"),
            latitude=("latitude", "first"),
            longitude=("longitude", "first"),

            obs_count=("adjusted_dwell_minutes", "count"),
            median_adjusted_dwell_minutes=("adjusted_dwell_minutes", "median"),
            mean_adjusted_dwell_minutes=("adjusted_dwell_minutes", "mean"),
            std_adjusted_dwell_minutes=("adjusted_dwell_minutes", "std"),

            median_raw_gap_minutes=("gap_minutes", "median"),
            median_estimated_travel_time_min=("estimated_travel_time_min", "median"),
            median_distance_to_next_poi_km=("distance_to_next_poi_km", "median"),
            near_duplicate_transition_rate=("near_duplicate_transition", "mean"),
        )
        .reset_index()
    )

    poi_debug = poi_debug[poi_debug["obs_count"] >= min_obs].copy()
    poi_debug["log_obs_count"] = np.log1p(poi_debug["obs_count"])

    # Keep only clean gap observations belonging to POIs that meet min_obs.
    valid_global_ids = set(poi_debug["global_venue_id"])
    clean_gap_min_obs = clean_gap[
        clean_gap["global_venue_id"].isin(valid_global_ids)
    ].copy()

    return gap_df, clean_gap, clean_gap_min_obs, poi_debug


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-obs", type=int, default=5)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_all_csvs(args.input_dir)

    gap_df, clean_gap, clean_gap_min_obs, poi_debug = build_adjusted_dwell(
        df,
        min_obs=args.min_obs,
    )

    gap_df.to_csv(output_dir / "debug_all_candidate_gaps.csv", index=False)
    clean_gap.to_csv(output_dir / "debug_clean_adjusted_gaps_before_min_obs.csv", index=False)
    clean_gap_min_obs.to_csv(output_dir / "clean_adjusted_gap_observations.csv", index=False)
    poi_debug.to_csv(output_dir / "adjusted_dwell_pois_debug.csv", index=False)

    print("All candidate gaps:", len(gap_df))
    print("Clean adjusted gaps before min_obs:", len(clean_gap))
    print("Clean adjusted gaps after min_obs:", len(clean_gap_min_obs))
    print("POIs meeting min_obs:", len(poi_debug))
    print("Saved clean gap observations:", output_dir / "clean_adjusted_gap_observations.csv")
    print("Saved POI debug table:", output_dir / "adjusted_dwell_pois_debug.csv")


if __name__ == "__main__":
    main()