from __future__ import annotations

"""
Robust model comparison for DBSCAN episode-based dwell labels.

Input:
    dbscan_adjusted_google_osm_pois.csv

Target:
    median_adjusted_dwell_minutes

Important:
    In the DBSCAN pipeline, median_adjusted_dwell_minutes is a compatibility
    alias for median_episode_dwell_minutes.

Purpose:
    Compare supervised regression models for predicting DBSCAN episode-based
    activity-duration proxy labels from fair inference-time POI features.

Models:
    - Dummy median baseline
    - Ridge regression
    - Huber regression
    - Random Forest
    - HistGradientBoosting
    - MLP Regressor

Selection logic:
    Final model is selected by lowest cross-validated MAE on the training set.
    Holdout test metrics are reported only for evaluation.

Important leakage control:
    The model does NOT use trajectory-derived DBSCAN diagnostic columns as
    predictors, because those would not be available for TravelPlanner
    attractions at inference time.

Outputs:
    dbscan_robust_model_metrics.csv
    dbscan_robust_gridsearch_results.csv
    dbscan_robust_holdout_predictions.csv
    dbscan_robust_final_model.joblib
    dbscan_robust_final_model_metadata.json
    dbscan_robust_selection_summary.txt
"""

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


DEFAULT_INPUT_PATH = Path(
    r"C:\Users\negia\trip_plan\dwell_model\adjusted_dwell_model_pipeline\data\dbscan_adjusted_google_osm_pois.csv"
)

DEFAULT_OUTPUT_DIR = Path(
    r"C:\Users\negia\trip_plan\dwell_model\adjusted_dwell_model_pipeline\data\dbscan_model_outputs"
)

DEFAULT_TARGET = "median_adjusted_dwell_minutes"


# ---------------------------------------------------------------------
# Fair predictor set:
# These features describe the POI / Google match / OSM context.
# They should be available when predicting TravelPlanner attraction dwell.
# ---------------------------------------------------------------------

CANDIDATE_NUMERIC_FEATURES = [
    # Google quality / popularity features
    "google_rating_filled",
    "log_google_user_rating_count_capped",
    "log_google_user_rating_count_filled",
    "log_google_user_rating_count_raw",
    "has_google_rating",
    "has_google_user_rating_count",

    # OSM parking context
    "parking_count_400m",
    "parking_count_600m",
    "parking_nearest_m",

    # OSM public transport context
    "public_transport_count_400m",
    "public_transport_count_600m",
    "public_transport_nearest_m",

    # OSM safety/security proxy context
    "osm_safety_proxy_count_400m",
    "osm_safety_proxy_count_600m",
    "osm_safety_proxy_nearest_m",
    "osm_security_proxy_count_400m",
    "osm_security_proxy_count_600m",
    "osm_security_proxy_nearest_m",
]


CANDIDATE_CATEGORICAL_FEATURES = [
    "category_group",
    "dataset_city",
    "venue_category_clean",
    "google_primary_type_clean",
    "google_price_level_str",
    "google_business_status",

    # These are categorical/text labels in your CSV.
    "parking_availability",
    "public_transport_availability",
    "osm_safety_proxy",
    "osm_security_proxy",
]


# ---------------------------------------------------------------------
# Columns that must NOT be used as predictors.
# They are targets, old targets, trajectory-derived diagnostics, weights,
# IDs, or merge/match bookkeeping fields.
# ---------------------------------------------------------------------

FORBIDDEN_PREDICTOR_COLUMNS = {
    # New DBSCAN labels / target columns
    "p25_episode_dwell_minutes",
    "median_episode_dwell_minutes",
    "p75_episode_dwell_minutes",
    "mean_episode_dwell_minutes",
    "std_episode_dwell_minutes",
    "median_adjusted_dwell_minutes",
    "mean_adjusted_dwell_minutes",
    "std_adjusted_dwell_minutes",

    # Old direct-TTNE labels
    "old_direct_ttne_obs_count",
    "old_direct_ttne_median_adjusted_dwell_minutes",
    "old_direct_ttne_mean_adjusted_dwell_minutes",
    "old_direct_ttne_std_adjusted_dwell_minutes",
    "old_direct_ttne_log_obs_count",

    # Trajectory-derived diagnostics
    "median_raw_gap_minutes",
    "median_estimated_travel_time_min",
    "median_distance_to_next_poi_km",
    "near_duplicate_transition_rate",
    "median_distance_to_next_episode_km",
    "median_estimated_travel_time_to_next_episode_min",
    "multi_poi_episode_rate",

    # Weights / observation counts
    "obs_count",
    "log_obs_count",
    "dbscan_episode_obs_count",
    "dbscan_log_obs_count",
    "dbscan_matched_source_count",

    # IDs / text identifiers
    "poi_id",
    "google_place_id",
    "source_global_venue_ids",
    "dbscan_source_global_venue_ids",
    "poi_name",
    "google_name",
    "latitude",
    "longitude",
}


def make_onehot_encoder():
    """
    sklearn changed 'sparse' to 'sparse_output' in newer versions.
    This keeps the script compatible across versions.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_predictions(y_true, y_pred, sample_weight=None) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
        "weighted_mae": (
            float(mean_absolute_error(y_true, y_pred, sample_weight=sample_weight))
            if sample_weight is not None
            else np.nan
        ),
    }


def existing_columns(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    return [c for c in candidates if c in df.columns]


def remove_forbidden_features(features: list[str]) -> list[str]:
    return [c for c in features if c not in FORBIDDEN_PREDICTOR_COLUMNS]


def clean_feature_lists(
    df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Make feature typing robust.

    If a supposed numeric feature contains strings, move it to categorical
    instead of letting the median imputer crash.
    """
    df = df.copy()

    numeric_features = remove_forbidden_features(numeric_features)
    categorical_features = remove_forbidden_features(categorical_features)

    safe_numeric = []
    final_categorical = list(categorical_features)

    for col in numeric_features:
        converted = pd.to_numeric(df[col], errors="coerce")
        non_null_original = df[col].notna().sum()
        non_null_converted = converted.notna().sum()

        if non_null_original == 0:
            df[col] = converted
            safe_numeric.append(col)
            continue

        convertible_ratio = non_null_converted / non_null_original

        if convertible_ratio >= 0.95:
            df[col] = converted
            safe_numeric.append(col)
        else:
            print(f"Moving non-numeric numeric candidate to categorical: {col}")
            final_categorical.append(col)

    final_categorical = remove_forbidden_features(final_categorical)
    final_categorical = list(dict.fromkeys(final_categorical))

    for col in final_categorical:
        df[col] = df[col].fillna("unknown").astype(str)

    return df, safe_numeric, final_categorical


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("onehot", make_onehot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def build_model_grids(
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int,
) -> dict:
    return {
        "ridge": {
            "pipeline": Pipeline(
                steps=[
                    ("preprocess", build_preprocessor(numeric_features, categorical_features)),
                    ("model", Ridge(random_state=random_state)),
                ]
            ),
            "param_grid": {
                "model__alpha": [0.01, 0.1, 1.0, 10.0, 50.0, 100.0],
            },
            "supports_sample_weight": True,
        },

        "huber": {
            "pipeline": Pipeline(
                steps=[
                    ("preprocess", build_preprocessor(numeric_features, categorical_features)),
                    (
                        "model",
                        HuberRegressor(
                            max_iter=5000,
                            tol=1e-4,
                        ),
                    ),
                ]
            ),
            "param_grid": {
                "model__epsilon": [1.35, 1.75, 2.0],
                "model__alpha": [0.0001, 0.001, 0.01],
            },
            "supports_sample_weight": True,
        },

        "random_forest": {
            "pipeline": Pipeline(
                steps=[
                    ("preprocess", build_preprocessor(numeric_features, categorical_features)),
                    (
                        "model",
                        RandomForestRegressor(
                            random_state=random_state,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            "param_grid": {
                "model__n_estimators": [200, 400],
                "model__max_depth": [4, 8, 12, None],
                "model__min_samples_leaf": [2, 5, 10],
                "model__max_features": ["sqrt", 0.7, 1.0],
            },
            "supports_sample_weight": True,
        },

        "hist_gradient_boosting": {
            "pipeline": Pipeline(
                steps=[
                    ("preprocess", build_preprocessor(numeric_features, categorical_features)),
                    (
                        "model",
                        HistGradientBoostingRegressor(
                            random_state=random_state,
                            early_stopping=True,
                        ),
                    ),
                ]
            ),
            "param_grid": {
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__max_iter": [150, 300, 500],
                "model__max_leaf_nodes": [7, 15, 31],
                "model__l2_regularization": [0.0, 0.1, 1.0],
                "model__min_samples_leaf": [10, 20, 40],
            },
            "supports_sample_weight": True,
        },

        "mlp": {
            "pipeline": Pipeline(
                steps=[
                    ("preprocess", build_preprocessor(numeric_features, categorical_features)),
                    (
                        "model",
                        MLPRegressor(
                            hidden_layer_sizes=(64,),
                            activation="relu",
                            solver="adam",
                            early_stopping=True,
                            validation_fraction=0.15,
                            max_iter=1500,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
            "param_grid": {
                "model__hidden_layer_sizes": [(32,), (64,), (64, 32)],
                "model__alpha": [0.001, 0.01],
                "model__learning_rate_init": [0.0005, 0.001],
            },
            # Your sklearn version does not support sample_weight for MLPRegressor.
            "supports_sample_weight": False,
        },
    }


def extract_feature_names(
    model: Pipeline,
    numeric_features: list[str],
    categorical_features: list[str],
) -> list[str]:
    preprocessor = model.named_steps["preprocess"]

    names = []

    if numeric_features:
        names.extend(numeric_features)

    if categorical_features:
        cat_pipe = preprocessor.named_transformers_["categorical"]
        onehot = cat_pipe.named_steps["onehot"]
        names.extend(onehot.get_feature_names_out(categorical_features).tolist())

    return names


def save_coefficients_if_available(
    model_name: str,
    model: Pipeline,
    output_dir: Path,
    numeric_features: list[str],
    categorical_features: list[str],
):
    estimator = model.named_steps.get("model")

    if estimator is None or not hasattr(estimator, "coef_"):
        return

    feature_names = extract_feature_names(model, numeric_features, categorical_features)

    coef_df = pd.DataFrame(
        {
            "model": model_name,
            "feature": feature_names,
            "coefficient": estimator.coef_,
            "abs_coefficient": np.abs(estimator.coef_),
        }
    ).sort_values("abs_coefficient", ascending=False)

    coef_df.to_csv(
        output_dir / f"dbscan_robust_{model_name}_coefficients.csv",
        index=False,
    )


def fit_grid_search(grid, X_train, y_train, w_train, supports_sample_weight: bool):
    if supports_sample_weight:
        grid.fit(X_train, y_train, model__sample_weight=w_train)
    else:
        grid.fit(X_train, y_train)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-path",
        default=str(DEFAULT_INPUT_PATH),
        help="Path to dbscan_adjusted_google_osm_pois.csv",
    )

    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output folder for robust DBSCAN model comparison.",
    )

    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=5)

    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional debugging row limit. Leave empty for full training.",
    )

    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    if args.max_rows is not None:
        df = df.sample(n=min(args.max_rows, len(df)), random_state=args.random_state)

    print("Loaded:", input_path)
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    if args.target not in df.columns:
        raise ValueError(f"Target column not found: {args.target}")

    if "median_episode_dwell_minutes" in df.columns:
        mismatch = (
            pd.to_numeric(df[args.target], errors="coerce")
            - pd.to_numeric(df["median_episode_dwell_minutes"], errors="coerce")
        ).abs().max()

        print(
            "Max absolute difference between target and median_episode_dwell_minutes:",
            mismatch,
        )

    df = df[df[args.target].notna()].copy()

    df = df[
        (pd.to_numeric(df[args.target], errors="coerce") >= 5)
        & (pd.to_numeric(df[args.target], errors="coerce") <= 300)
    ].copy()

    df[args.target] = pd.to_numeric(df[args.target], errors="coerce")

    print("Rows after target filter:", len(df))

    numeric_features = existing_columns(df, CANDIDATE_NUMERIC_FEATURES)
    categorical_features = existing_columns(df, CANDIDATE_CATEGORICAL_FEATURES)

    df, numeric_features, categorical_features = clean_feature_lists(
        df,
        numeric_features,
        categorical_features,
    )

    if not numeric_features and not categorical_features:
        raise ValueError("No usable features found.")

    print("\nNumeric features used:")
    print(numeric_features)

    print("\nCategorical features used:")
    print(categorical_features)

    forbidden_used = sorted(
        set(numeric_features + categorical_features) & FORBIDDEN_PREDICTOR_COLUMNS
    )

    if forbidden_used:
        raise ValueError(f"Forbidden leakage columns used as predictors: {forbidden_used}")

    feature_cols = numeric_features + categorical_features

    X = df[feature_cols].copy()
    y = df[args.target].astype(float).copy()

    if "obs_count" in df.columns:
        weights = np.sqrt(
            pd.to_numeric(df["obs_count"], errors="coerce")
            .fillna(1)
            .clip(lower=1)
        )
        weight_source = "sqrt(obs_count)"
    elif "dbscan_episode_obs_count" in df.columns:
        weights = np.sqrt(
            pd.to_numeric(df["dbscan_episode_obs_count"], errors="coerce")
            .fillna(1)
            .clip(lower=1)
        )
        weight_source = "sqrt(dbscan_episode_obs_count)"
    else:
        weights = pd.Series(np.ones(len(df)), index=df.index)
        weight_source = "uniform"

    stratify_col = df["category_group"] if "category_group" in df.columns else None

    X_train, X_test, y_train, y_test, w_train, w_test, train_idx, test_idx = train_test_split(
        X,
        y,
        weights,
        df.index,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=stratify_col,
    )

    print("\nTrain rows:", len(X_train))
    print("Test rows:", len(X_test))
    print("Weight source:", weight_source)

    cv = KFold(
        n_splits=args.cv_folds,
        shuffle=True,
        random_state=args.random_state,
    )

    all_metrics = []
    all_grid_rows = []
    trained_models = {}

    # ------------------------------------------------------------------
    # Dummy baseline
    # ------------------------------------------------------------------
    dummy = DummyRegressor(strategy="median")
    dummy.fit(X_train, y_train)

    dummy_train_pred = dummy.predict(X_train)
    dummy_test_pred = dummy.predict(X_test)

    dummy_train = evaluate_predictions(
        y_train,
        dummy_train_pred,
        sample_weight=w_train,
    )
    dummy_test = evaluate_predictions(
        y_test,
        dummy_test_pred,
        sample_weight=w_test,
    )

    all_metrics.append(
        {
            "model": "dummy_median",
            "is_baseline": True,
            "supports_sample_weight": False,
            "best_params": "{}",
            "cv_mean_mae": np.nan,
            "cv_std_mae": np.nan,
            "train_mae": dummy_train["mae"],
            "test_mae": dummy_test["mae"],
            "train_rmse": dummy_train["rmse"],
            "test_rmse": dummy_test["rmse"],
            "train_r2": dummy_train["r2"],
            "test_r2": dummy_test["r2"],
            "train_weighted_mae": dummy_train["weighted_mae"],
            "test_weighted_mae": dummy_test["weighted_mae"],
            "generalization_gap_mae": dummy_test["mae"] - dummy_train["mae"],
        }
    )

    trained_models["dummy_median"] = dummy

    # ------------------------------------------------------------------
    # Grid-search models
    # ------------------------------------------------------------------
    model_grids = build_model_grids(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        random_state=args.random_state,
    )

    for model_name, spec in model_grids.items():
        print(f"\nRunning GridSearchCV for: {model_name}")

        grid = GridSearchCV(
            estimator=spec["pipeline"],
            param_grid=spec["param_grid"],
            scoring="neg_mean_absolute_error",
            cv=cv,
            n_jobs=-1,
            refit=True,
            return_train_score=True,
            verbose=1,
            error_score=np.nan,
        )

        try:
            fit_grid_search(
                grid,
                X_train,
                y_train,
                w_train,
                supports_sample_weight=spec["supports_sample_weight"],
            )
        except Exception as exc:
            print(f"Model failed completely: {model_name}")
            print(str(exc))

            all_metrics.append(
                {
                    "model": model_name,
                    "is_baseline": False,
                    "supports_sample_weight": spec["supports_sample_weight"],
                    "best_params": "{}",
                    "cv_mean_mae": np.nan,
                    "cv_std_mae": np.nan,
                    "train_mae": np.nan,
                    "test_mae": np.nan,
                    "train_rmse": np.nan,
                    "test_rmse": np.nan,
                    "train_r2": np.nan,
                    "test_r2": np.nan,
                    "train_weighted_mae": np.nan,
                    "test_weighted_mae": np.nan,
                    "generalization_gap_mae": np.nan,
                    "failure_reason": str(exc),
                }
            )
            continue

        if np.isnan(grid.cv_results_["mean_test_score"]).all():
            print(f"All CV scores are NaN for model: {model_name}")
            continue

        best_model = grid.best_estimator_
        trained_models[model_name] = best_model

        train_pred = best_model.predict(X_train)
        test_pred = best_model.predict(X_test)

        train_scores = evaluate_predictions(
            y_train,
            train_pred,
            sample_weight=w_train,
        )
        test_scores = evaluate_predictions(
            y_test,
            test_pred,
            sample_weight=w_test,
        )

        cv_mean_mae = float(-grid.best_score_)
        best_idx = int(grid.best_index_)
        cv_std_mae = float(grid.cv_results_["std_test_score"][best_idx])

        all_metrics.append(
            {
                "model": model_name,
                "is_baseline": False,
                "supports_sample_weight": spec["supports_sample_weight"],
                "best_params": json.dumps(grid.best_params_),
                "cv_mean_mae": cv_mean_mae,
                "cv_std_mae": cv_std_mae,
                "train_mae": train_scores["mae"],
                "test_mae": test_scores["mae"],
                "train_rmse": train_scores["rmse"],
                "test_rmse": test_scores["rmse"],
                "train_r2": train_scores["r2"],
                "test_r2": test_scores["r2"],
                "train_weighted_mae": train_scores["weighted_mae"],
                "test_weighted_mae": test_scores["weighted_mae"],
                "generalization_gap_mae": test_scores["mae"] - train_scores["mae"],
                "failure_reason": "",
            }
        )

        cv_results = pd.DataFrame(grid.cv_results_)
        cv_results["model"] = model_name
        cv_results["supports_sample_weight"] = spec["supports_sample_weight"]
        all_grid_rows.append(cv_results)

        save_coefficients_if_available(
            model_name,
            best_model,
            output_dir,
            numeric_features,
            categorical_features,
        )

        joblib.dump(
            best_model,
            output_dir / f"dbscan_robust_{model_name}_best_model.joblib",
        )

    metrics_df = pd.DataFrame(all_metrics)

    # ------------------------------------------------------------------
    # Final model selection
    # ------------------------------------------------------------------
    candidate_metrics = metrics_df[
        (metrics_df["is_baseline"] == False)
        & metrics_df["cv_mean_mae"].notna()
    ].copy()

    if candidate_metrics.empty:
        raise ValueError(
            "No non-baseline model completed successfully. "
            "Check grid-search failures and feature preprocessing."
        )

    candidate_metrics = candidate_metrics.sort_values(
        ["cv_mean_mae", "cv_std_mae", "generalization_gap_mae"],
        ascending=[True, True, True],
    )

    final_row = candidate_metrics.iloc[0]
    final_model_name = final_row["model"]
    final_model = trained_models[final_model_name]

    metrics_df["selected_final_model"] = metrics_df["model"] == final_model_name

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    metrics_path = output_dir / "dbscan_robust_model_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    grid_results_path = output_dir / "dbscan_robust_gridsearch_results.csv"
    if all_grid_rows:
        grid_df = pd.concat(all_grid_rows, ignore_index=True)
        grid_df.to_csv(grid_results_path, index=False)
    else:
        pd.DataFrame().to_csv(grid_results_path, index=False)

    pred_df = df.loc[test_idx].copy()
    pred_df["y_true"] = y_test.values
    pred_df["sample_weight"] = w_test.values

    for model_name, model in trained_models.items():
        pred_df[f"pred_{model_name}"] = model.predict(X_test)

    predictions_path = output_dir / "dbscan_robust_holdout_predictions.csv"
    pred_df.to_csv(predictions_path, index=False)

    final_model_path = output_dir / "dbscan_robust_final_model.joblib"
    joblib.dump(final_model, final_model_path)

    metadata = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "target": args.target,
        "target_note": (
            "median_adjusted_dwell_minutes is used as a compatibility alias. "
            "For DBSCAN episode labels it should equal median_episode_dwell_minutes."
        ),
        "rows_used": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_size": args.test_size,
        "random_state": args.random_state,
        "cv_folds": args.cv_folds,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "forbidden_predictor_columns": sorted(FORBIDDEN_PREDICTOR_COLUMNS),
        "weighting": weight_source,
        "mlp_weighting_note": (
            "MLPRegressor was trained without sample weights because this "
            "sklearn version does not support sample_weight for MLPRegressor."
        ),
        "selection_rule": (
            "Final model selected by lowest cross-validated MAE on training data. "
            "Holdout test metrics are reported only for final evaluation, not tuning."
        ),
        "selected_model": final_model_name,
        "selected_model_cv_mean_mae": float(final_row["cv_mean_mae"]),
        "selected_model_test_mae": float(final_row["test_mae"]),
        "selected_model_test_rmse": float(final_row["test_rmse"]),
        "selected_model_test_r2": float(final_row["test_r2"]),
        "selected_model_best_params": final_row["best_params"],
    }

    metadata_path = output_dir / "dbscan_robust_final_model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    summary_path = output_dir / "dbscan_robust_selection_summary.txt"

    baseline_row = metrics_df[metrics_df["model"] == "dummy_median"].iloc[0]

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("DBSCAN Episode-Based Dwell Model Selection Summary\n")
        f.write("=" * 60 + "\n\n")

        f.write("Dataset:\n")
        f.write(f"Rows used: {len(df)}\n")
        f.write(f"Train rows: {len(X_train)}\n")
        f.write(f"Test rows: {len(X_test)}\n")
        f.write(f"Target: {args.target}\n")
        f.write(f"Weighting: {weight_source}\n\n")

        f.write("Leakage control:\n")
        f.write(
            "Only Google/OSM/POI descriptor features were used as predictors. "
            "DBSCAN trajectory diagnostics and dwell-label columns were excluded "
            "from the feature set.\n\n"
        )

        f.write("Selection rule:\n")
        f.write(
            "The final model was selected using the lowest cross-validated MAE "
            "on the training set. The holdout test set was used only for final "
            "evaluation, not for selecting hyperparameters.\n\n"
        )

        f.write(f"Selected model: {final_model_name}\n")
        f.write(f"Best parameters: {final_row['best_params']}\n")
        f.write(f"CV mean MAE: {final_row['cv_mean_mae']:.4f}\n")
        f.write(f"CV std MAE: {final_row['cv_std_mae']:.4f}\n")
        f.write(f"Test MAE: {final_row['test_mae']:.4f}\n")
        f.write(f"Test RMSE: {final_row['test_rmse']:.4f}\n")
        f.write(f"Test R2: {final_row['test_r2']:.4f}\n")
        f.write(f"Test weighted MAE: {final_row['test_weighted_mae']:.4f}\n\n")

        f.write("Baseline comparison:\n")
        f.write(f"Dummy median test MAE: {baseline_row['test_mae']:.4f}\n")
        f.write(f"Selected model test MAE: {final_row['test_mae']:.4f}\n")
        improvement = baseline_row["test_mae"] - final_row["test_mae"]
        f.write(f"MAE improvement over dummy: {improvement:.4f} minutes\n\n")

        f.write("All model metrics:\n")
        f.write(metrics_df.to_string(index=False))

    print("\nMetrics:")
    print(metrics_df.sort_values("cv_mean_mae", na_position="last").to_string(index=False))

    print("\nSelected final model:")
    print(final_model_name)
    print("Best params:", final_row["best_params"])

    print("\nSaved outputs:")
    print(metrics_path)
    print(grid_results_path)
    print(predictions_path)
    print(final_model_path)
    print(metadata_path)
    print(summary_path)


if __name__ == "__main__":
    main()