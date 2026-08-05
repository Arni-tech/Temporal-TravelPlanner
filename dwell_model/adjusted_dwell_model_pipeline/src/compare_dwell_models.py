"""
Train and compare dwell-time prediction models.

Input:
    Feature-enriched POI training table containing DBSCAN-derived
    median_adjusted_dwell_minutes labels and Google/OSM features.

Output:
    - dwell_model_comparison.csv
    - model comparison plots
    - selected_dwell_model.joblib
    - selected_model_metadata.json

This script is the final modelling stage after DBSCAN dwell-label construction
and Google/OSM feature enrichment.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "median_adjusted_dwell_minutes"


# These are inference-available features.
# Do NOT use target-derived columns such as raw gap, estimated next-travel time, etc.
NUMERIC_FEATURES = [
    "google_rating_filled",
    "log_google_user_rating_count_capped",
    "has_google_rating",
    "has_google_user_rating_count",
]

CATEGORICAL_FEATURES = [
    "category_group",
    "dataset_city",
    "parking_availability",
    "public_transport_availability",
    "osm_safety_proxy",
    "osm_security_proxy",
]


def rmse(y_true, y_pred) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def make_onehot_encoder():
    """
    sklearn changed sparse -> sparse_output in newer versions.
    This keeps the script compatible with older/newer sklearn.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(numeric_features, categorical_features) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_onehot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


class CategoryMedianBaseline:
    """
    Simple category median baseline:
    predict the training median dwell for each category_group.
    Falls back to global median when category is unseen.
    """

    def __init__(self, category_col="category_group"):
        self.category_col = category_col
        self.global_median_ = None
        self.category_medians_ = None

    def fit(self, X, y, sample_weight=None):
        tmp = X.copy()
        tmp["_target"] = y.values if hasattr(y, "values") else y

        if sample_weight is None:
            self.category_medians_ = tmp.groupby(self.category_col)["_target"].median().to_dict()
            self.global_median_ = float(tmp["_target"].median())
        else:
            # Weighted median is more complicated; for baseline clarity, use unweighted category median.
            self.category_medians_ = tmp.groupby(self.category_col)["_target"].median().to_dict()
            self.global_median_ = float(tmp["_target"].median())

        return self

    def predict(self, X):
        preds = []
        for value in X[self.category_col].fillna("UNKNOWN").astype(str):
            preds.append(self.category_medians_.get(value, self.global_median_))
        return np.array(preds, dtype=float)


def evaluate_model(name, model, X_test, y_test) -> dict:
    pred = model.predict(X_test)

    return {
        "model": name,
        "MAE": float(mean_absolute_error(y_test, pred)),
        "RMSE": rmse(y_test, pred),
        "MedianAE": float(median_absolute_error(y_test, pred)),
        "R2": float(r2_score(y_test, pred)),
    }


def fit_with_optional_weights(model, X_train, y_train, sample_weight):
    """
    Most sklearn estimators support sample_weight.
    Some, like older MLPRegressor versions, may not.
    """
    try:
        model.fit(X_train, y_train, model__sample_weight=sample_weight)
    except Exception:
        try:
            model.fit(X_train, y_train, sample_weight=sample_weight)
        except Exception:
            model.fit(X_train, y_train)

    return model


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return []


def save_linear_coefficients(model, output_path: Path):
    """
    Saves coefficients for linear-style pipelines.
    Works for LinearRegression, Ridge, HuberRegressor inside a Pipeline.
    """
    if not isinstance(model, Pipeline):
        return

    if "preprocess" not in model.named_steps or "model" not in model.named_steps:
        return

    estimator = model.named_steps["model"]

    if not hasattr(estimator, "coef_"):
        return

    feature_names = get_feature_names(model.named_steps["preprocess"])

    if not feature_names:
        feature_names = [f"feature_{i}" for i in range(len(estimator.coef_))]

    coef_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": estimator.coef_,
        }
    )

    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coefficient", ascending=False)

    coef_df.to_csv(output_path, index=False)


def plot_model_comparison(results_df: pd.DataFrame, output_dir: Path):
    plot_df = results_df.sort_values("MAE")

    plt.figure(figsize=(10, 5))
    plt.bar(plot_df["model"], plot_df["MAE"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("MAE minutes")
    plt.title("Dwell-Time Model Comparison by MAE")
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison_mae.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(plot_df["model"], plot_df["RMSE"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("RMSE minutes")
    plt.title("Dwell-Time Model Comparison by RMSE")
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison_rmse.png", dpi=200)
    plt.close()


def plot_prediction_scatter(model, X_test, y_test, output_dir: Path, model_name: str):
    pred = model.predict(X_test)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, pred, alpha=0.5)
    min_v = min(float(np.min(y_test)), float(np.min(pred)))
    max_v = max(float(np.max(y_test)), float(np.max(pred)))
    plt.plot([min_v, max_v], [min_v, max_v])
    plt.xlabel("Actual dwell minutes")
    plt.ylabel("Predicted dwell minutes")
    plt.title(f"Predicted vs Actual: {model_name}")
    plt.tight_layout()
    plt.savefig(output_dir / "selected_model_predicted_vs_actual.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Final training CSV containing median_adjusted_dwell_minutes and engineered features.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where model comparison outputs will be saved.",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--selected-model",
        type=str,
        default="Robust weighted Huber regression",
        help="Model to export as selected_dwell_model.joblib.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    if TARGET not in df.columns:
        raise ValueError(f"Target column missing: {TARGET}")

    available_numeric = [c for c in NUMERIC_FEATURES if c in df.columns]
    available_categorical = [c for c in CATEGORICAL_FEATURES if c in df.columns]

    if not available_numeric and not available_categorical:
        raise ValueError("No usable model features found in the input CSV.")

    needed = [TARGET] + available_numeric + available_categorical
    if "obs_count" in df.columns:
        needed.append("obs_count")

    data = df[needed].copy()
    data = data.dropna(subset=[TARGET])

    for col in available_categorical:
        data[col] = data[col].fillna("UNKNOWN").astype(str)

    for col in available_numeric:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    X = data[available_numeric + available_categorical].copy()
    y = data[TARGET].astype(float)

    if "obs_count" in data.columns:
        obs = pd.to_numeric(data["obs_count"], errors="coerce").fillna(1.0).clip(lower=1.0)
        sample_weight = np.sqrt(obs)
    else:
        sample_weight = np.ones(len(data), dtype=float)

    stratify_col = X["category_group"] if "category_group" in X.columns else None

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X,
        y,
        sample_weight,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=stratify_col,
    )

    preprocessor = build_preprocessor(available_numeric, available_categorical)

    models = {}

    models["Dummy median baseline"] = DummyRegressor(strategy="median")

    if "category_group" in X.columns:
        models["Category median baseline"] = CategoryMedianBaseline(category_col="category_group")

    models["Weighted multiple linear regression"] = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LinearRegression()),
        ]
    )

    models["Weighted Ridge regression"] = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(available_numeric, available_categorical)),
            ("model", Ridge(alpha=1.0, random_state=args.random_state)),
        ]
    )

    models["Robust weighted Huber regression"] = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(available_numeric, available_categorical)),
            ("model", HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=1000)),
        ]
    )

    models["Random Forest regression"] = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(available_numeric, available_categorical)),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_leaf=3,
                    random_state=args.random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    models["Gradient Boosting regression"] = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(available_numeric, available_categorical)),
            (
                "model",
                GradientBoostingRegressor(
                    n_estimators=300,
                    learning_rate=0.03,
                    max_depth=3,
                    random_state=args.random_state,
                ),
            ),
        ]
    )

    models["MLP regression"] = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(available_numeric, available_categorical)),
            (
                "model",
                MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    alpha=0.001,
                    learning_rate_init=0.001,
                    max_iter=1000,
                    early_stopping=True,
                    random_state=args.random_state,
                ),
            ),
        ]
    )

    results = []
    fitted_models = {}

    for name, model in models.items():
        print(f"\nTraining: {name}")

        if name in ["Dummy median baseline", "Category median baseline"]:
            model.fit(X_train, y_train)
        else:
            fit_with_optional_weights(model, X_train, y_train, w_train)

        fitted_models[name] = model
        metrics = evaluate_model(name, model, X_test, y_test)
        results.append(metrics)

        print(
            f"MAE={metrics['MAE']:.3f}, "
            f"RMSE={metrics['RMSE']:.3f}, "
            f"MedianAE={metrics['MedianAE']:.3f}, "
            f"R2={metrics['R2']:.3f}"
        )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("MAE").reset_index(drop=True)

    results_path = output_dir / "dwell_model_comparison.csv"
    results_df.to_csv(results_path, index=False)

    plot_model_comparison(results_df, output_dir)

    selected_name = args.selected_model

    if selected_name not in fitted_models:
        print(f"\nRequested selected model not found: {selected_name}")
        print("Selecting lowest-MAE model instead.")
        selected_name = results_df.iloc[0]["model"]

    selected_model = fitted_models[selected_name]

    joblib.dump(selected_model, output_dir / "selected_dwell_model.joblib")

    selected_metadata = {
        "selected_model": selected_name,
        "target": TARGET,
        "numeric_features": available_numeric,
        "categorical_features": available_categorical,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "sample_weight": "sqrt(obs_count)" if "obs_count" in data.columns else "uniform",
        "metrics": results_df.to_dict(orient="records"),
    }

    with open(output_dir / "selected_model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(selected_metadata, f, indent=2)

    plot_prediction_scatter(selected_model, X_test, y_test, output_dir, selected_name)

    # Save coefficients for interpretable linear models.
    for name in [
        "Weighted multiple linear regression",
        "Weighted Ridge regression",
        "Robust weighted Huber regression",
    ]:
        if name in fitted_models:
            safe_name = (
                name.lower()
                .replace(" ", "_")
                .replace("/", "_")
                .replace("-", "_")
            )
            save_linear_coefficients(
                fitted_models[name],
                output_dir / f"{safe_name}_coefficients.csv",
            )

    print("\n===== Model comparison =====")
    print(results_df.to_string(index=False))

    print("\nSaved outputs to:")
    print(output_dir)

    print("\nSelected model:")
    print(selected_name)


if __name__ == "__main__":
    main()
