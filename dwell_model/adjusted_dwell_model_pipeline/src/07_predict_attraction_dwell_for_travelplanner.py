from __future__ import annotations

"""
Predict TravelPlanner attraction dwell/activity duration using the selected
DBSCAN episode-based dwell model.

Input:
    C:\\Users\\negia\\trip_plan\\database\\attractions\\attractions_google_osm_features.csv

Model:
    C:\\Users\\negia\\trip_plan\\dwell_model\\adjusted_dwell_model_pipeline\\data\\dbscan_model_outputs\\dbscan_robust_final_model.joblib

Output:
    C:\\Users\\negia\\trip_plan\\database\\attractions\\attractions_google_osm_features_dbscan_dwell.csv

Important:
    This does NOT rerun Google Places or OSM.
    It reuses the already enriched attraction feature table and replaces
    predicted_dwell_minutes with predictions from the DBSCAN episode-trained model.

The old prediction is preserved as:
    old_predicted_dwell_minutes
    old_dwell_prediction_source
"""

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


MODEL_PATH = Path(
    r"C:\Users\negia\trip_plan\dwell_model\adjusted_dwell_model_pipeline\data\dbscan_model_outputs\dbscan_robust_final_model.joblib"
)

METADATA_PATH = Path(
    r"C:\Users\negia\trip_plan\dwell_model\adjusted_dwell_model_pipeline\data\dbscan_model_outputs\dbscan_robust_final_model_metadata.json"
)

INPUT_PATH = Path(
    r"C:\Users\negia\trip_plan\database\attractions\attractions_google_osm_features.csv"
)

OUTPUT_PATH = Path(
    r"C:\Users\negia\trip_plan\database\attractions\attractions_google_osm_features_dbscan_dwell.csv"
)

DIAGNOSTICS_PATH = Path(
    r"C:\Users\negia\trip_plan\database\attractions\attractions_google_osm_features_dbscan_dwell_diagnostics.csv"
)

SUMMARY_PATH = Path(
    r"C:\Users\negia\trip_plan\database\attractions\attractions_google_osm_features_dbscan_dwell_summary.json"
)

BACKUP_PATH = Path(
    r"C:\Users\negia\trip_plan\database\attractions\attractions_google_osm_features_before_dbscan_dwell.csv"
)


# These are the fair predictor columns used in the robust training script.
# The saved pipeline will select what it needs, but we create them explicitly
# so prediction does not fail due to missing columns.
EXPECTED_NUMERIC_FEATURES = [
    "google_rating_filled",
    "log_google_user_rating_count_capped",
    "log_google_user_rating_count_filled",
    "log_google_user_rating_count_raw",
    "has_google_rating",
    "has_google_user_rating_count",

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
]

EXPECTED_CATEGORICAL_FEATURES = [
    "category_group",
    "dataset_city",
    "venue_category_clean",
    "google_primary_type_clean",
    "google_price_level_str",
    "google_business_status",

    "parking_availability",
    "public_transport_availability",
    "osm_safety_proxy",
    "osm_security_proxy",
]


def clean_text(value) -> str:
    if pd.isna(value):
        return "unknown"

    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9\s/_-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text if text else "unknown"


def normalize_city(value) -> str:
    if pd.isna(value):
        return "unknown"

    text = str(value).strip().lower().replace(" ", "_")
    return text if text else "unknown"


def ensure_numeric(df: pd.DataFrame, col: str, default=np.nan) -> None:
    if col not in df.columns:
        df[col] = default

    df[col] = pd.to_numeric(df[col], errors="coerce")


def ensure_categorical(df: pd.DataFrame, col: str, default="unknown") -> None:
    if col not in df.columns:
        df[col] = default

    df[col] = df[col].fillna(default).astype(str)


def get_model_expected_columns(model) -> tuple[list[str], list[str]]:
    """
    Try to read the actual feature columns from the saved sklearn pipeline.
    If this fails, fall back to EXPECTED_* lists.
    """
    try:
        preprocessor = model.named_steps["preprocess"]

        numeric_cols = []
        categorical_cols = []

        for name, transformer, cols in preprocessor.transformers_:
            if name == "numeric":
                numeric_cols = list(cols)
            elif name == "categorical":
                categorical_cols = list(cols)

        if numeric_cols or categorical_cols:
            return numeric_cols, categorical_cols

    except Exception:
        pass

    return EXPECTED_NUMERIC_FEATURES, EXPECTED_CATEGORICAL_FEATURES


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ------------------------------------------------------------
    # Broad category and city fields
    # ------------------------------------------------------------
    if "category_group" not in df.columns:
        df["category_group"] = "tourism_attraction"

    df["category_group"] = (
        df["category_group"]
        .fillna("tourism_attraction")
        .astype(str)
        .replace({"": "tourism_attraction"})
    )

    # For this attractions file, all rows should normally be tourism_attraction.
    # Do not forcibly overwrite existing values, but repair blanks.
    df.loc[df["category_group"].str.strip() == "", "category_group"] = "tourism_attraction"

    if "dataset_city" not in df.columns:
        if "City" in df.columns:
            df["dataset_city"] = df["City"].apply(normalize_city)
        else:
            df["dataset_city"] = "unknown"

    df["dataset_city"] = df["dataset_city"].apply(normalize_city)

    # ------------------------------------------------------------
    # Cleaned venue / Google type fields
    # ------------------------------------------------------------
    if "venue_category_clean" not in df.columns:
        if "venue_category" in df.columns:
            df["venue_category_clean"] = df["venue_category"].apply(clean_text)
        elif "google_primary_type" in df.columns:
            df["venue_category_clean"] = df["google_primary_type"].apply(clean_text)
        else:
            df["venue_category_clean"] = "attraction"

    df["venue_category_clean"] = df["venue_category_clean"].apply(clean_text)

    if "google_primary_type_clean" not in df.columns:
        if "google_primary_type" in df.columns:
            df["google_primary_type_clean"] = df["google_primary_type"].apply(clean_text)
        else:
            df["google_primary_type_clean"] = "unknown"

    df["google_primary_type_clean"] = df["google_primary_type_clean"].apply(clean_text)

    if "google_price_level_str" not in df.columns:
        if "google_price_level" in df.columns:
            df["google_price_level_str"] = df["google_price_level"].fillna("unknown").astype(str)
        else:
            df["google_price_level_str"] = "unknown"

    df["google_price_level_str"] = df["google_price_level_str"].fillna("unknown").astype(str)

    if "google_business_status" not in df.columns:
        df["google_business_status"] = "unknown"

    df["google_business_status"] = df["google_business_status"].fillna("unknown").astype(str)

    # ------------------------------------------------------------
    # Google rating / review features
    # ------------------------------------------------------------
    ensure_numeric(df, "google_rating", np.nan)
    ensure_numeric(df, "google_user_rating_count", np.nan)

    df["has_google_rating"] = df["google_rating"].notna().astype(int)
    df["has_google_user_rating_count"] = df["google_user_rating_count"].notna().astype(int)

    rating_median = df["google_rating"].median(skipna=True)
    if pd.isna(rating_median):
        rating_median = 0.0

    df["google_rating_filled"] = df["google_rating"].fillna(rating_median)

    review_count = df["google_user_rating_count"].clip(lower=0)

    df["log_google_user_rating_count_raw"] = np.log1p(review_count)

    review_median = df["google_user_rating_count"].median(skipna=True)
    if pd.isna(review_median):
        review_median = 0.0

    df["log_google_user_rating_count_filled"] = np.log1p(
        df["google_user_rating_count"].fillna(review_median).clip(lower=0)
    )

    cap_value = df["log_google_user_rating_count_filled"].quantile(0.99)
    if pd.isna(cap_value):
        cap_value = df["log_google_user_rating_count_filled"].max()

    if pd.isna(cap_value):
        cap_value = 0.0

    df["log_google_user_rating_count_capped"] = df[
        "log_google_user_rating_count_filled"
    ].clip(upper=cap_value)

    # ------------------------------------------------------------
    # OSM numeric columns
    # ------------------------------------------------------------
    osm_numeric_cols = [
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
    ]

    for col in osm_numeric_cols:
        ensure_numeric(df, col, np.nan)

    # ------------------------------------------------------------
    # OSM categorical columns
    # ------------------------------------------------------------
    osm_categorical_cols = [
        "parking_availability",
        "public_transport_availability",
        "osm_safety_proxy",
        "osm_security_proxy",
    ]

    for col in osm_categorical_cols:
        ensure_categorical(df, col, "unknown")

    return df


def save_backup_once(df: pd.DataFrame) -> None:
    if not BACKUP_PATH.exists():
        df.to_csv(BACKUP_PATH, index=False)
        print("Saved backup:", BACKUP_PATH)
    else:
        print("Backup already exists:", BACKUP_PATH)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input attraction feature file not found: {INPUT_PATH}")

    model = joblib.load(MODEL_PATH)
    original_df = pd.read_csv(INPUT_PATH)

    print("Loaded attractions:", len(original_df))
    print("Input file:", INPUT_PATH)
    print("Model:", MODEL_PATH)

    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        print("Model metadata selected_model:", metadata.get("selected_model"))
        print("Model metadata target:", metadata.get("target"))

    save_backup_once(original_df)

    df = prepare_features(original_df)

    numeric_cols, categorical_cols = get_model_expected_columns(model)

    required_cols = list(dict.fromkeys(numeric_cols + categorical_cols))
    missing_cols = [c for c in required_cols if c not in df.columns]

    if missing_cols:
        raise ValueError(
            "The following model-required columns are still missing after feature preparation: "
            f"{missing_cols}"
        )

    # Coerce exactly the model-expected numeric/categorical columns.
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in categorical_cols:
        df[col] = df[col].fillna("unknown").astype(str)

    print("\nModel expects numeric columns:")
    print(numeric_cols)

    print("\nModel expects categorical columns:")
    print(categorical_cols)

    predictions_raw = model.predict(df)

    # Guardrail for itinerary use. The model was trained on activity-duration
    # proxy values, but extreme predictions should not break TravelPlanner.
    predictions_clipped = np.clip(predictions_raw, 10, 240)

    # Preserve old predictions.
    if "predicted_dwell_minutes" in df.columns:
        df["old_predicted_dwell_minutes"] = df["predicted_dwell_minutes"]
    else:
        df["old_predicted_dwell_minutes"] = np.nan

    if "dwell_prediction_source" in df.columns:
        df["old_dwell_prediction_source"] = df["dwell_prediction_source"]
    else:
        df["old_dwell_prediction_source"] = "none"

    df["predicted_dwell_minutes_raw_dbscan"] = predictions_raw
    df["predicted_dwell_minutes"] = predictions_clipped
    df["dwell_prediction_source"] = "dbscan_episode_random_forest"

    # Difference against old prediction, if available.
    df["prediction_delta_minutes"] = (
        pd.to_numeric(df["predicted_dwell_minutes"], errors="coerce")
        - pd.to_numeric(df["old_predicted_dwell_minutes"], errors="coerce")
    )

    df.to_csv(OUTPUT_PATH, index=False)

    # Diagnostics file with useful inspection columns.
    diagnostic_cols = [
        "Name",
        "City",
        "category_group",
        "google_primary_type",
        "google_primary_type_clean",
        "google_rating",
        "google_user_rating_count",
        "parking_availability",
        "public_transport_availability",
        "osm_safety_proxy",
        "osm_security_proxy",
        "old_predicted_dwell_minutes",
        "predicted_dwell_minutes_raw_dbscan",
        "predicted_dwell_minutes",
        "prediction_delta_minutes",
        "old_dwell_prediction_source",
        "dwell_prediction_source",
    ]

    diagnostic_cols = [c for c in diagnostic_cols if c in df.columns]
    df[diagnostic_cols].to_csv(DIAGNOSTICS_PATH, index=False)

    summary = {
        "input_path": str(INPUT_PATH),
        "model_path": str(MODEL_PATH),
        "output_path": str(OUTPUT_PATH),
        "diagnostics_path": str(DIAGNOSTICS_PATH),
        "rows": int(len(df)),
        "prediction_source": "dbscan_episode_random_forest",
        "raw_prediction_summary": {
            k: float(v)
            for k, v in pd.Series(predictions_raw).describe().to_dict().items()
        },
        "clipped_prediction_summary": {
            k: float(v)
            for k, v in df["predicted_dwell_minutes"].describe().to_dict().items()
        },
        "old_prediction_summary": (
            {
                k: float(v)
                for k, v in pd.to_numeric(
                    df["old_predicted_dwell_minutes"],
                    errors="coerce",
                ).describe().to_dict().items()
            }
            if "old_predicted_dwell_minutes" in df.columns
            else {}
        ),
        "delta_summary": {
            k: float(v)
            for k, v in df["prediction_delta_minutes"].describe().to_dict().items()
            if pd.notna(v)
        },
        "model_numeric_columns": numeric_cols,
        "model_categorical_columns": categorical_cols,
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("\nSaved updated attraction feature file:")
    print(OUTPUT_PATH)

    print("\nSaved diagnostics:")
    print(DIAGNOSTICS_PATH)

    print("\nSaved summary:")
    print(SUMMARY_PATH)

    print("\nPrediction summary:")
    print(df["predicted_dwell_minutes"].describe())

    print("\nOld prediction summary:")
    print(pd.to_numeric(df["old_predicted_dwell_minutes"], errors="coerce").describe())

    print("\nPrediction delta summary:")
    print(df["prediction_delta_minutes"].describe())

    print("\nTop 10 highest predicted dwell:")
    cols_to_show = ["Name", "City", "google_primary_type", "predicted_dwell_minutes"]
    cols_to_show = [c for c in cols_to_show if c in df.columns]
    print(df.sort_values("predicted_dwell_minutes", ascending=False)[cols_to_show].head(10).to_string(index=False))

    print("\nTop 10 lowest predicted dwell:")
    print(df.sort_values("predicted_dwell_minutes", ascending=True)[cols_to_show].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
