import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "median_adjusted_dwell_minutes"

FEATURES = [
    "category_group",
    "google_rating_filled",
    "log_google_user_rating_count_capped",
    "parking_availability",
    "public_transport_availability",
    "osm_safety_proxy",
    "osm_security_proxy",
    "dataset_city",
]

CATEGORICAL_FEATURES = [
    "category_group",
    "parking_availability",
    "public_transport_availability",
    "osm_safety_proxy",
    "osm_security_proxy",
    "dataset_city",
]

NUMERIC_FEATURES = [
    "google_rating_filled",
    "log_google_user_rating_count_capped",
]


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def fit_pipeline(model, X_train, y_train, sample_weight=None):
    if sample_weight is not None:
        model.fit(X_train, y_train, regressor__sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)
    return model


def evaluate(name, model, X_test, y_test):
    pred = model.predict(X_test)

    return {
        "model": name,
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": rmse(y_test, pred),
        "R2": r2_score(y_test, pred),
    }


def get_feature_names(fitted_pipeline):
    preprocessor = fitted_pipeline.named_steps["preprocess"]

    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES)

    return list(cat_names) + NUMERIC_FEATURES


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    required = FEATURES + [TARGET, "obs_count"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=[TARGET, "category_group", "dataset_city"]).copy()

    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("unknown").astype(str)

    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["obs_count"] = pd.to_numeric(df["obs_count"], errors="coerce").fillna(1)
    df["sample_weight"] = np.sqrt(df["obs_count"].clip(lower=1))

    X = df[FEATURES]
    y = df[TARGET]
    weights = df["sample_weight"]

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X,
        y,
        weights,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df["category_group"],
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
            ("num", numeric_pipeline, NUMERIC_FEATURES),
        ]
    )

    dummy_model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("regressor", DummyRegressor(strategy="median")),
        ]
    )

    linear_model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    robust_model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "regressor",
                HuberRegressor(
                    epsilon=1.35,
                    alpha=0.0001,
                    max_iter=1000,
                ),
            ),
        ]
    )

    results = []

    fit_pipeline(dummy_model, X_train, y_train)
    results.append(evaluate("Dummy median baseline", dummy_model, X_test, y_test))

    fit_pipeline(linear_model, X_train, y_train, sample_weight=w_train)
    results.append(evaluate("Weighted multiple linear regression", linear_model, X_test, y_test))

    fit_pipeline(robust_model, X_train, y_train, sample_weight=w_train)
    results.append(evaluate("Robust weighted multiple linear regression", robust_model, X_test, y_test))

    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(output_dir / "model_metrics.csv", index=False)

    print("\nModel metrics:")
    print(metrics_df)

    # Final model trained on all data
    final_model = robust_model
    fit_pipeline(final_model, X, y, sample_weight=weights)

    joblib.dump(final_model, output_dir / "robust_dwell_model.joblib")

    # Predictions
    df["predicted_dwell_minutes"] = final_model.predict(X)
    df["residual_minutes"] = df[TARGET] - df["predicted_dwell_minutes"]

    pred_cols = [
        "poi_id",
        "poi_name",
        "dataset_city",
        "category_group",
        TARGET,
        "predicted_dwell_minutes",
        "residual_minutes",
        "obs_count",
    ]
    pred_cols = [c for c in pred_cols if c in df.columns]

    df[pred_cols].to_csv(output_dir / "model_predictions.csv", index=False)

    # Coefficients
    feature_names = get_feature_names(final_model)
    coefficients = final_model.named_steps["regressor"].coef_

    coef_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "abs_coefficient": np.abs(coefficients),
        }
    ).sort_values("abs_coefficient", ascending=False)

    coef_df.to_csv(output_dir / "model_coefficients.csv", index=False)

    metadata = {
        "target": TARGET,
        "features": FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "model": "HuberRegressor robust weighted multiple linear regression",
        "sample_weight": "sqrt(obs_count)",
        "test_size": args.test_size,
        "random_state": args.random_state,
        "rows_used": int(len(df)),
        "notes": [
            "Target is median adjusted dwell-time proxy.",
            "Categorical features are one-hot encoded with drop='first'.",
            "Numeric features are median-imputed and standardized.",
            "obs_count is used as sample weight, not as a predictor.",
            "OSM safety/security are proxy variables, not direct crime or accident measures.",
        ],
    }

    with open(output_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved:")
    print(output_dir / "robust_dwell_model.joblib")
    print(output_dir / "model_metrics.csv")
    print(output_dir / "model_coefficients.csv")
    print(output_dir / "model_predictions.csv")
    print(output_dir / "model_metadata.json")


if __name__ == "__main__":
    main()


 