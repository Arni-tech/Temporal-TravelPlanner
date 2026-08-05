import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FINAL_COLUMNS = [
    # identifiers / matching
    "poi_id",
    "poi_name",
    "google_place_id",
    "google_name",
    "dataset_city",
    "venue_city",
    "venue_country",
    "latitude",
    "longitude",

    # target
    "median_adjusted_dwell_minutes",

    # reliability
    "obs_count",
    "log_obs_count",

    # main paper-style features
    "category_group",
    "google_rating_filled",
    "google_user_rating_count",
    "log_google_user_rating_count_capped",
    "has_google_rating",
    "has_google_user_rating_count",
    "parking_availability",
    "public_transport_availability",
    "osm_safety_proxy",
    "osm_security_proxy",

    # detailed ablation features
    "venue_category_clean",
    "google_primary_type_clean",
    "google_price_level_str",
    "google_business_status",

    # OSM diagnostic / extended features
    "parking_count_400m",
    "parking_count_600m",
    "parking_nearest_m",

    "public_transport_count_400m",
    "public_transport_count_600m",
    "public_transport_nearest_m",

    "osm_safety_proxy_count_400m",
    "osm_safety_proxy_count_600m",
    "osm_safety_proxy_nearest_m",

    "osm_security_proxy_count_400m",
    "osm_security_proxy_count_600m",
    "osm_security_proxy_nearest_m",

    # target diagnostics
    "median_raw_gap_minutes",
    "median_estimated_travel_time_min",
    "median_distance_to_next_poi_km",
    "near_duplicate_transition_rate",
]


def unique_preserve_order(items):
    out = []

    for x in items:
        if x not in out:
            out.append(x)

    return out


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    df = pd.read_csv(args.input)

    cols = unique_preserve_order(FINAL_COLUMNS)

    for col in cols:
        if col not in df.columns:
            if col in [
                "parking_availability",
                "public_transport_availability",
                "osm_safety_proxy",
                "osm_security_proxy",
            ]:
                df[col] = "unknown"

            elif (
                col.endswith("_count_400m")
                or col.endswith("_count_600m")
            ):
                df[col] = 0

            elif col.endswith("_nearest_m"):
                df[col] = np.nan

            else:
                df[col] = np.nan

    out = df[cols].copy()

    required = [
        "median_adjusted_dwell_minutes",
        "latitude",
        "longitude",
        "category_group",
        "dataset_city",
    ]

    out = out.dropna(subset=required).copy()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    out.to_csv(output_path, index=False)

    print("Saved:", output_path)
    print("Rows:", len(out))
    print("Columns:")
    print(out.columns.tolist())

    print("\nTarget summary:")
    print(out["median_adjusted_dwell_minutes"].describe())

    print("\nCategory group counts:")
    print(out["category_group"].value_counts(dropna=False))

    for col in [
        "parking_availability",
        "public_transport_availability",
        "osm_safety_proxy",
        "osm_security_proxy",
    ]:
        if col in out.columns:
            print(f"\n{col}:")
            print(out[col].value_counts(dropna=False))


if __name__ == "__main__":
    main()




 
