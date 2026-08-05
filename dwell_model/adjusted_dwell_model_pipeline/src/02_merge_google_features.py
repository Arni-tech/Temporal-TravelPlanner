import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def collapse_rare_categories(series, min_count=20, other_label="OTHER_RARE"):
    counts = series.value_counts()
    keep = counts[counts >= min_count].index
    return series.where(series.isin(keep), other_label)


def first_non_null(series):
    s = series.dropna()
    return s.iloc[0] if len(s) else np.nan


def mode_or_first(series):
    s = series.dropna()

    if len(s) == 0:
        return np.nan

    mode = s.mode()
    return mode.iloc[0] if len(mode) else s.iloc[0]


def weighted_mean(values, weights):
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce").fillna(0)

    mask = values.notna() & (weights > 0)

    if mask.sum() == 0:
        return values.median()

    return np.average(values[mask], weights=weights[mask])


def aggregate_true_google_place_median(gap_google):
    rows = []

    for google_place_id, group in gap_google.groupby("google_place_id"):
        # One row per adjusted dwell observation.
        # This is now a true median across all underlying adjusted gaps mapped to the same Google place.
        obs_count = len(group)

        row = {
            "poi_id": google_place_id,
            "google_place_id": google_place_id,
            "source_global_venue_ids": "|".join(
                group["global_venue_id"].astype(str).dropna().unique()
            ),

            "poi_name": mode_or_first(group["name"]),
            "google_name": mode_or_first(group["google_name"]),

            "dataset_city": mode_or_first(group["dataset_city"]),
            "venue_city": mode_or_first(group["venue_city"]),
            "venue_country": mode_or_first(group["venue_country"]),

            "latitude": group["latitude"].median(),
            "longitude": group["longitude"].median(),

            "category_group": mode_or_first(group["category_group"]),
            "venue_category": mode_or_first(group["venue_category"]),

            "obs_count": obs_count,
            "median_adjusted_dwell_minutes": group["adjusted_dwell_minutes"].median(),
            "mean_adjusted_dwell_minutes": group["adjusted_dwell_minutes"].mean(),
            "std_adjusted_dwell_minutes": group["adjusted_dwell_minutes"].std(),

            "median_raw_gap_minutes": group["gap_minutes"].median(),
            "median_estimated_travel_time_min": group["estimated_travel_time_min"].median(),
            "median_distance_to_next_poi_km": group["distance_to_next_poi_km"].median(),
            "near_duplicate_transition_rate": group["near_duplicate_transition"].mean(),

            "google_primary_type": mode_or_first(group["google_primary_type"]),
            "google_types": mode_or_first(group["google_types"]) if "google_types" in group.columns else np.nan,
            "google_rating": first_non_null(group["google_rating"]),
            "google_user_rating_count": first_non_null(group["google_user_rating_count"]),
            "google_price_level": mode_or_first(group["google_price_level"]),
            "google_business_status": mode_or_first(group["google_business_status"]),

            "google_distance_m_median": group["google_distance_m"].median(),
            "name_similarity_median": group["name_similarity"].median(),
            "match_score_median": group["match_score"].median(),
            "type_relevance_score_median": group["type_relevance_score"].median(),
        }

        rows.append(row)

    return pd.DataFrame(rows)


def add_clean_features(out):
    out = out.copy()

    out["log_obs_count"] = np.log1p(out["obs_count"])

    out["venue_category_clean"] = collapse_rare_categories(
        out["venue_category"].fillna("UNKNOWN").astype(str),
        min_count=20,
    )

    out["google_primary_type_clean"] = collapse_rare_categories(
        out["google_primary_type"].fillna("UNKNOWN").astype(str),
        min_count=20,
    )

    out["google_price_level_str"] = out["google_price_level"].fillna("UNKNOWN").astype(str)
    out["google_business_status"] = out["google_business_status"].fillna("UNKNOWN").astype(str)

    out["google_rating"] = pd.to_numeric(out["google_rating"], errors="coerce")
    out["google_user_rating_count"] = pd.to_numeric(
        out["google_user_rating_count"],
        errors="coerce",
    )

    global_rating_median = out["google_rating"].median()

    rating_by_google_type = out.groupby("google_primary_type_clean")["google_rating"].transform("median")
    rating_by_venue_cat = out.groupby("venue_category_clean")["google_rating"].transform("median")
    rating_by_group = out.groupby("category_group")["google_rating"].transform("median")

    out["google_rating_filled"] = (
        out["google_rating"]
        .fillna(rating_by_google_type)
        .fillna(rating_by_venue_cat)
        .fillna(rating_by_group)
        .fillna(global_rating_median)
    )

    out["log_google_user_rating_count_raw"] = np.log1p(out["google_user_rating_count"])

    global_log_reviews_median = out["log_google_user_rating_count_raw"].median()

    log_reviews_by_google_type = out.groupby("google_primary_type_clean")[
        "log_google_user_rating_count_raw"
    ].transform("median")

    log_reviews_by_venue_cat = out.groupby("venue_category_clean")[
        "log_google_user_rating_count_raw"
    ].transform("median")

    log_reviews_by_group = out.groupby("category_group")[
        "log_google_user_rating_count_raw"
    ].transform("median")

    out["log_google_user_rating_count_filled"] = (
        out["log_google_user_rating_count_raw"]
        .fillna(log_reviews_by_google_type)
        .fillna(log_reviews_by_venue_cat)
        .fillna(log_reviews_by_group)
        .fillna(global_log_reviews_median)
    )

    cap_value = out["log_google_user_rating_count_filled"].quantile(0.99)

    out["log_google_user_rating_count_capped"] = (
        out["log_google_user_rating_count_filled"].clip(upper=cap_value)
    )

    out["has_google_rating"] = out["google_rating"].notna().astype(int)
    out["has_google_user_rating_count"] = out["google_user_rating_count"].notna().astype(int)

    return out


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--clean-gaps", required=True)
    parser.add_argument("--google-enriched", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-google-place-obs", type=int, default=5)

    args = parser.parse_args()

    clean_gaps = pd.read_csv(args.clean_gaps)
    google = pd.read_csv(args.google_enriched)

    google_cols = [
        "global_venue_id",
        "status",
        "match_confidence",
        "google_place_id",
        "google_name",
        "google_distance_m",
        "name_similarity",
        "match_score",
        "type_relevance_score",
        "google_primary_type",
        "google_types",
        "google_rating",
        "google_user_rating_count",
        "google_price_level",
        "google_business_status",
    ]

    google_cols = [c for c in google_cols if c in google.columns]

    merged = clean_gaps.merge(
        google[google_cols],
        on="global_venue_id",
        how="left",
    )

    numeric_cols = [
        "google_distance_m",
        "name_similarity",
        "match_score",
        "type_relevance_score",
        "google_rating",
        "google_user_rating_count",
        "adjusted_dwell_minutes",
        "gap_minutes",
        "estimated_travel_time_min",
        "distance_to_next_poi_km",
        "near_duplicate_transition",
        "latitude",
        "longitude",
    ]

    for col in numeric_cols:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    trusted_gap_google = merged[
        (merged["status"] == "matched")
        & (merged["match_confidence"].isin(["high", "medium"]))
        & (merged["google_place_id"].notna())
        & (merged["google_distance_m"].notna())
        & (merged["google_distance_m"] <= 500)
        & (merged["adjusted_dwell_minutes"].notna())
    ].copy()

    out = aggregate_true_google_place_median(trusted_gap_google)

    out = out[out["obs_count"] >= args.min_google_place_obs].copy()

    out = add_clean_features(out)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    out.to_csv(output_path, index=False)

    print("Clean adjusted gap rows:", len(clean_gaps))
    print("Trusted Google-matched gap rows:", len(trusted_gap_google))
    print("Final Google-place POIs after true median aggregation:", len(out))
    print("Saved:", output_path)

    print("\nTarget summary:")
    print(out["median_adjusted_dwell_minutes"].describe())

    print("\nCategory group counts:")
    print(out["category_group"].value_counts())


if __name__ == "__main__":
    main()

