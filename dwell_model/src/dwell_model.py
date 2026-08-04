from pathlib import Path
import json
import re
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Ridge


DATA_PATH = Path(
    r"C:\Users\negia\trip_plan\dwell_model\int\processed_massive_steps\google_places_all16k_clean_model_data.csv"
)

MODEL_DIR = Path(
    r"C:\Users\negia\trip_plan\dwell_model_exports"
)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def clean_category_value(x):
    if pd.isna(x):
        return "UNKNOWN"
    x = str(x).strip().lower()
    x = re.sub(r"[^a-z0-9]+", "_", x)
    x = re.sub(r"_+", "_", x).strip("_")
    return x if x else "UNKNOWN"


def main():
    df = pd.read_csv(DATA_PATH)
    print("Loaded:", df.shape)

    # Recreate clean feature columns.
    df["venue_category_clean"] = df["venue_category"].apply(clean_category_value)
    df["google_primary_type_clean"] = df["google_primary_type"].apply(clean_category_value)

    if "log_google_user_rating_count_capped" not in df.columns:
        df["log_google_user_rating_count_capped"] = df[
            "log_google_user_rating_count"
        ].clip(
            upper=df["log_google_user_rating_count"].quantile(0.99)
        )

    df["google_price_level_str"] = (
        df["google_price_level_str"]
        .fillna("UNKNOWN")
        .astype(str)
    )

    df["google_business_status"] = (
        df["google_business_status"]
        .fillna("UNKNOWN")
        .astype(str)
    )

    df["has_google_rating"] = (
        pd.to_numeric(df["has_google_rating"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    df["has_google_user_rating_count"] = (
        pd.to_numeric(df["has_google_user_rating_count"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    # Deduplicate to unique Google POIs.
    df = df.sort_values(
        by=["obs_count", "match_score", "name_similarity"],
        ascending=[False, False, False],
    )

    df_dedup = df.drop_duplicates(
        subset=["google_place_id"],
        keep="first",
    ).copy()

    print("Before dedup:", len(df))
    print("After dedup:", len(df_dedup))
    print("Unique google_place_id:", df_dedup["google_place_id"].nunique())

    target = "median_duration_proxy"

    native_features = [
        "category_group",
        "venue_category_clean",
        "dataset_city",
        "latitude",
        "longitude",
        "distance_to_city_center_km",
        "log_obs_count",
    ]

    google_features = native_features + [
        "google_primary_type_clean",
        "google_rating_filled",
        "log_google_user_rating_count_capped",
        "has_google_rating",
        "has_google_user_rating_count",
        "google_price_level_str",
        "google_business_status",
    ]

    missing = [c for c in google_features + [target] if c not in df_dedup.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    numeric_features = [
        "latitude",
        "longitude",
        "distance_to_city_center_km",
        "log_obs_count",
        "google_rating_filled",
        "log_google_user_rating_count_capped",
        "has_google_rating",
        "has_google_user_rating_count",
    ]

    categorical_features = [c for c in google_features if c not in numeric_features]

    model_data = df_dedup[google_features + [target]].copy()

    for col in numeric_features + [target]:
        model_data[col] = pd.to_numeric(model_data[col], errors="coerce")

    model_data = model_data.dropna(subset=[target]).copy()

    X = model_data[google_features]
    y = model_data[target].astype(float)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    huber_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", HuberRegressor(max_iter=5000,tol=1e-4)),
        ]
    )

    ridge_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", Ridge(alpha=1.0)),
        ]
    )

    print("Training Huber...")
    huber_model.fit(X, y)

    print("Training Ridge...")
    ridge_model.fit(X, y)

    joblib.dump(huber_model, MODEL_DIR / "huber_native_google.joblib")
    joblib.dump(ridge_model, MODEL_DIR / "ridge_native_google.joblib")

    with open(MODEL_DIR / "feature_columns.json", "w") as f:
        json.dump(google_features, f, indent=2)

    metadata = {
        "source_file": str(DATA_PATH),
        "target": target,
        "rows_before_dedup": int(len(df)),
        "rows_after_dedup": int(len(df_dedup)),
        "unique_google_place_ids": int(df_dedup["google_place_id"].nunique()),
        "primary_model": "HuberRegressor native + Google",
        "huber_max_iter": 5000,
        "huber_tol": 1e-4,
        "secondary_model": "Ridge native + Google",
        "features": google_features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "deduplication": (
            "Sorted by obs_count, match_score, name_similarity descending; "
            "then kept first row per google_place_id."
        ),
        "notes": (
            "Recovered/exported from google_places_all16k_clean_model_data.csv. "
            "Target is POI-level median trajectory-derived dwell/activity-duration proxy."
        ),
    }

    with open(MODEL_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Export category lookup.
    category_lookup = (
        df_dedup.groupby(["category_group", "venue_category_clean"])["median_duration_proxy"]
        .median()
        .reset_index()
        .rename(columns={"median_duration_proxy": "estimated_duration_minutes"})
        .sort_values(["category_group", "estimated_duration_minutes"], ascending=[True, False])
    )

    category_lookup.to_csv(
        MODEL_DIR / "duration_lookup_by_venue_category_clean.csv",
        index=False,
    )

    # Export Google type lookup.
    google_type_lookup = (
        df_dedup.groupby(["category_group", "google_primary_type_clean"])["median_duration_proxy"]
        .median()
        .reset_index()
        .rename(columns={"median_duration_proxy": "estimated_duration_minutes"})
        .sort_values(["category_group", "estimated_duration_minutes"], ascending=[True, False])
    )

    google_type_lookup.to_csv(
        MODEL_DIR / "duration_lookup_by_google_primary_type_clean.csv",
        index=False,
    )

    # Export POI feature table for TravelPlanner matching.
    poi_feature_cols = [
        "google_enrichment_row_id",
        "global_venue_id",
        "google_place_id",
        "name",
        "google_name",
        "dataset_city",
        "venue_city",
        "venue_country",
        "latitude",
        "longitude",
        "category_group",
        "venue_category",
        "venue_category_clean",
        "distance_to_city_center_km",
        "obs_count",
        "log_obs_count",
        "google_primary_type",
        "google_primary_type_clean",
        "google_rating",
        "google_rating_filled",
        "google_user_rating_count",
        "log_google_user_rating_count",
        "log_google_user_rating_count_capped",
        "has_google_rating",
        "has_google_user_rating_count",
        "google_price_level_str",
        "google_business_status",
        "median_duration_proxy",
    ]

    poi_feature_cols = [c for c in poi_feature_cols if c in df_dedup.columns]
    poi_features = df_dedup[poi_feature_cols].copy()

    poi_features.to_csv(
        MODEL_DIR / "poi_dwell_feature_table.csv",
        index=False,
    )

    print("\nSaved exports:")
    for p in sorted(MODEL_DIR.iterdir()):
        print(" -", p.name)


if __name__ == "__main__":
    main()