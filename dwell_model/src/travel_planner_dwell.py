import argparse
import math
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


MODEL_FEATURES = [
    "category_group",
    "google_rating_filled",
    "log_google_user_rating_count_capped",
    "parking_availability",
    "public_transport_availability",
    "osm_safety_proxy",
    "osm_security_proxy",
    "dataset_city",
]


def normalize_text(x):
    if pd.isna(x):
        return ""

    x = str(x).lower().strip()
    x = x.replace("_", " ")
    x = re.sub(r"[^a-z0-9]+", " ", x)
    x = re.sub(r"\s+", " ", x).strip()

    return x


def normalize_city(x):
    return normalize_text(x).replace(" ", "_")


def safe_numeric(series, default=np.nan):
    if series is None:
        return pd.Series(default)

    return pd.to_numeric(series, errors="coerce")


def infer_category_group(row):
    text = " ".join(
        [
            str(row.get("google_primary_type", "")),
            str(row.get("google_types", "")),
            str(row.get("Name", "")),
        ]
    ).lower()

    shopping_terms = [
        "store",
        "shopping",
        "mall",
        "market",
        "supermarket",
        "book_store",
        "clothing",
        "department_store",
        "electronics_store",
        "jewelry",
        "shoe",
        "retail",
    ]

    food_terms = [
        "restaurant",
        "cafe",
        "bar",
        "bakery",
        "meal",
        "food",
        "coffee",
        "fast_food",
        "pub",
        "brewery",
    ]

    tourism_terms = [
        "tourist",
        "museum",
        "park",
        "monument",
        "landmark",
        "church",
        "mosque",
        "temple",
        "zoo",
        "aquarium",
        "art_gallery",
        "stadium",
        "amusement",
        "point_of_interest",
        "national",
        "historic",
        "historical",
        "gallery",
        "theater",
        "theatre",
    ]

    if any(t in text for t in shopping_terms):
        return "shopping"

    if any(t in text for t in food_terms):
        return "food_leisure"

    if any(t in text for t in tourism_terms):
        return "tourism_attraction"

    return "tourism_attraction"


def clip_duration(x, default=90.0, min_minutes=15.0, max_minutes=240.0):
    try:
        x = float(x)

        if math.isnan(x):
            x = default

    except Exception:
        x = default

    return round(max(min_minutes, min(max_minutes, x)), 1)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True, help="Input attractions_google_osm_features.csv")
    parser.add_argument("--output", required=True, help="Output attractions_model_features.csv")
    parser.add_argument("--model", required=True, help="Path to robust_dwell_model.joblib")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    model_path = Path(args.model)

    df = pd.read_csv(input_path)

    model = joblib.load(model_path)

    required_original_cols = [
        "Name",
        "Latitude",
        "Longitude",
        "Address",
        "Phone",
        "Website",
        "City",
    ]

    missing_original = [c for c in required_original_cols if c not in df.columns]
    if missing_original:
        raise ValueError(f"Missing original TravelPlanner columns: {missing_original}")

    # Treat bad Google matches as missing Google metadata.
    if "google_match_status" in df.columns:
        bad_google = df["google_match_status"].ne("matched")

        for col in [
            "google_rating",
            "google_user_rating_count",
            "google_primary_type",
            "google_types",
        ]:
            if col in df.columns:
                df.loc[bad_google, col] = np.nan

    # Build / clean category_group.
    inferred_categories = df.apply(infer_category_group, axis=1)

    if "category_group" not in df.columns:
        df["category_group"] = inferred_categories
    else:
        df["category_group"] = df["category_group"].replace("", np.nan)
        df["category_group"] = df["category_group"].fillna(inferred_categories)

    # Build dataset_city in the same format as training.
    df["dataset_city"] = df["City"].apply(normalize_city)

    # Google numeric features.
    if "google_rating" not in df.columns:
        df["google_rating"] = np.nan

    if "google_user_rating_count" not in df.columns:
        df["google_user_rating_count"] = np.nan

    df["google_rating"] = pd.to_numeric(df["google_rating"], errors="coerce")
    df["google_user_rating_count"] = pd.to_numeric(
        df["google_user_rating_count"],
        errors="coerce",
    )

    # Impute Google rating.
    rating_median = df["google_rating"].median()
    if pd.isna(rating_median):
        rating_median = 4.3

    df["google_rating_filled"] = df["google_rating"].fillna(rating_median)

    # Impute and log-transform review count.
    review_median = df["google_user_rating_count"].median()
    if pd.isna(review_median):
        review_median = 0.0

    df["log_google_user_rating_count_capped"] = np.log1p(
        df["google_user_rating_count"].fillna(review_median)
    )

    cap = df["log_google_user_rating_count_capped"].quantile(0.99)
    if not pd.isna(cap):
        df["log_google_user_rating_count_capped"] = (
            df["log_google_user_rating_count_capped"].clip(upper=cap)
        )

    # Ensure OSM categorical feature columns exist.
    for col in [
        "parking_availability",
        "public_transport_availability",
        "osm_safety_proxy",
        "osm_security_proxy",
    ]:
        if col not in df.columns:
            df[col] = "unknown"

        df[col] = df[col].fillna("unknown").astype(str)

    # Build final model input.
    X = df[MODEL_FEATURES].copy()

    predictions = model.predict(X)

    df["predicted_dwell_minutes"] = [
        clip_duration(x) for x in predictions
    ]

    df["dwell_prediction_source"] = "robust_regression_model_google_osm"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print("Saved:", output_path)
    print("Rows:", len(df))

    print("\nPredicted dwell summary:")
    print(df["predicted_dwell_minutes"].describe())

    print("\nCategory counts:")
    print(df["category_group"].value_counts(dropna=False))

    if "google_match_status" in df.columns:
        print("\nGoogle match status:")
        print(df["google_match_status"].value_counts(dropna=False))

    for col in [
        "parking_availability",
        "public_transport_availability",
        "osm_safety_proxy",
        "osm_security_proxy",
    ]:
        print(f"\n{col}:")
        print(df[col].value_counts(dropna=False))


if __name__ == "__main__":
    main()

