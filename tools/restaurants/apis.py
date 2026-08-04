import pandas as pd
from pandas import DataFrame
from utils.func import extract_before_parenthesis


class Restaurants:
    def __init__(
        self,
        path="../database/restaurants/clean_restaurant_2022.csv",
        default_breakfast_minutes=49.3,
        default_lunch_minutes=57.9,
        default_dinner_minutes=63.9,
        default_meal_minutes=56.9,
        top_k=15,
    ):
        """
        Dwell-aware RestaurantSearch tool.

        Restaurant dwell values are empirical DBSCAN food/leisure duration
        estimates, not restaurant-specific ML predictions.

        For generation, the LLM only needs compact restaurant options.
        The evaluator and planner repair logic handle formal temporal loading.
        """

        self.path = path
        self.default_breakfast_minutes = default_breakfast_minutes
        self.default_lunch_minutes = default_lunch_minutes
        self.default_dinner_minutes = default_dinner_minutes
        self.default_meal_minutes = default_meal_minutes
        self.top_k = top_k

        self.data = pd.read_csv(self.path)
        self._prepare_data()

        print(
            f"Restaurants loaded from {self.path} "
            f"with compact empirical meal dwell defaults."
        )

    def _prepare_data(self):
        keep_cols = [
            "Name",
            "Average Cost",
            "Cuisines",
            "Aggregate Rating",
            "City",
            "predicted_breakfast_dwell_minutes",
            "predicted_lunch_dwell_minutes",
            "predicted_dinner_dwell_minutes",
            "predicted_dwell_minutes",
            "dwell_prediction_source",
        ]

        keep_cols = [c for c in keep_cols if c in self.data.columns]
        self.data = self.data[keep_cols].copy()

        required = ["Name", "City"]
        missing = [c for c in required if c not in self.data.columns]

        if missing:
            raise ValueError(f"Restaurant data missing required columns: {missing}")

        self.data["Name"] = self.data["Name"].astype(str).str.strip()
        self.data["City"] = self.data["City"].astype(str).str.strip()

        self.data = self.data[
            (self.data["Name"] != "")
            & (self.data["Name"].str.lower() != "nan")
            & (self.data["City"] != "")
            & (self.data["City"].str.lower() != "nan")
        ].copy()

        if "Average Cost" in self.data.columns:
            self.data["Average Cost"] = pd.to_numeric(
                self.data["Average Cost"],
                errors="coerce",
            )
        else:
            self.data["Average Cost"] = 0.0

        if "Aggregate Rating" in self.data.columns:
            self.data["Aggregate Rating"] = pd.to_numeric(
                self.data["Aggregate Rating"],
                errors="coerce",
            ).fillna(0)
        else:
            self.data["Aggregate Rating"] = 0.0

        if "Cuisines" not in self.data.columns:
            self.data["Cuisines"] = "unknown"
        else:
            self.data["Cuisines"] = self.data["Cuisines"].fillna("unknown")

        defaults = {
            "predicted_breakfast_dwell_minutes": self.default_breakfast_minutes,
            "predicted_lunch_dwell_minutes": self.default_lunch_minutes,
            "predicted_dinner_dwell_minutes": self.default_dinner_minutes,
            "predicted_dwell_minutes": self.default_meal_minutes,
        }

        for col, default in defaults.items():
            if col not in self.data.columns:
                self.data[col] = default

            self.data[col] = pd.to_numeric(
                self.data[col],
                errors="coerce",
            ).fillna(default)

            self.data[col] = self.data[col].clip(lower=15, upper=180).round(1)

        if "dwell_prediction_source" not in self.data.columns:
            self.data["dwell_prediction_source"] = "dbscan_food_leisure_empirical"
        else:
            self.data["dwell_prediction_source"] = self.data[
                "dwell_prediction_source"
            ].fillna("dbscan_food_leisure_empirical")

        cost = self.data["Average Cost"].fillna(self.data["Average Cost"].median())

        if cost.isna().all():
            cost = pd.Series([0] * len(self.data), index=self.data.index)

        cost_penalty = cost.clip(lower=0, upper=200) / 100.0

        self.data["planner_rank_score"] = (
            self.data["Aggregate Rating"] * 2.0
            - cost_penalty
        ).round(3)

        # Compact field for GPT-3.5. The exact slot-specific values are still
        # kept internally and returned for evaluation/debugging if needed.
        self.data["meal_dwell_note"] = (
            "empirical_meal_dwell:"
            + " breakfast="
            + self.data["predicted_breakfast_dwell_minutes"].astype(str)
            + ", lunch="
            + self.data["predicted_lunch_dwell_minutes"].astype(str)
            + ", dinner="
            + self.data["predicted_dinner_dwell_minutes"].astype(str)
        )

        self.data = self.data.sort_values(
            by=["City", "planner_rank_score"],
            ascending=[True, False],
        ).reset_index(drop=True)

    def load_db(self):
        self.data = pd.read_csv(self.path)
        self._prepare_data()

    def run(self, city: str) -> DataFrame:
        """
        Search for restaurants by city.

        Returns compact restaurant rows with empirical meal dwell signals.
        """

        city = extract_before_parenthesis(city)

        results = self.data[self.data["City"] == city].copy()
        results = results.reset_index(drop=True)

        if len(results) == 0:
            return "There is no restaurant in this city."

        display_cols = [
            "Name",
            "City",
            "Cuisines",
            "Average Cost",
            "Aggregate Rating",
            "meal_dwell_note",
            "planner_rank_score",
        ]

        display_cols = [c for c in display_cols if c in results.columns]

        return results[display_cols].head(self.top_k).reset_index(drop=True)

    def run_for_annotation(self, city: str) -> DataFrame:
        """
        Search for restaurants by city for annotation/evaluation.

        Keeps all city restaurant rows and all dwell columns.
        """

        city = extract_before_parenthesis(city)

        results = self.data[self.data["City"] == city].copy()
        results = results.reset_index(drop=True)

        return results