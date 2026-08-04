import pandas as pd
from pandas import DataFrame
from utils.func import extract_before_parenthesis


class Attractions:
    def __init__(
        self,
        path="../database/attractions/attractions.csv",
        default_dwell_minutes=90,
        top_k=12,
    ):
        """
        Dwell-aware AttractionSearch tool.

        This tool keeps TravelPlanner's original search structure but manually
        enriches attraction rows with compact dwell-time planning signals.

        The LLM should not calculate feasibility. It only sees:
        - attraction name/city
        - predicted dwell
        - dwell bucket
        - short planning guidance
        - simple travel/full-day safety flags

        Formal temporal feasibility is handled by planner repair/evaluation.
        """

        self.path = path
        self.default_dwell_minutes = default_dwell_minutes
        self.top_k = top_k

        self.data = pd.read_csv(self.path)
        self._prepare_data()

        print(
            f"Attractions loaded from {self.path} "
            f"with compact DBSCAN dwell guidance."
        )

    def _prepare_data(self):
        keep_cols = [
            "Name",
            "City",
            "Latitude",
            "Longitude",
            "predicted_dwell_minutes",
            "dwell_prediction_source",
            "category_group",
            "google_rating",
            "google_user_rating_count",
            "google_primary_type",
            "google_business_status",
        ]

        keep_cols = [c for c in keep_cols if c in self.data.columns]
        self.data = self.data[keep_cols].copy()

        if "Name" not in self.data.columns:
            raise ValueError("Attraction data must contain a 'Name' column.")

        if "City" not in self.data.columns:
            raise ValueError("Attraction data must contain a 'City' column.")

        self.data["Name"] = self.data["Name"].astype(str).str.strip()
        self.data["City"] = self.data["City"].astype(str).str.strip()

        self.data = self.data[
            (self.data["Name"] != "")
            & (self.data["Name"].str.lower() != "nan")
            & (self.data["City"] != "")
            & (self.data["City"].str.lower() != "nan")
        ].copy()

        if "predicted_dwell_minutes" not in self.data.columns:
            self.data["predicted_dwell_minutes"] = self.default_dwell_minutes

        self.data["predicted_dwell_minutes"] = pd.to_numeric(
            self.data["predicted_dwell_minutes"],
            errors="coerce",
        ).fillna(self.default_dwell_minutes)

        self.data["predicted_dwell_minutes"] = self.data[
            "predicted_dwell_minutes"
        ].clip(lower=15, upper=240).round(1)

        if "google_rating" in self.data.columns:
            self.data["google_rating"] = pd.to_numeric(
                self.data["google_rating"],
                errors="coerce",
            ).fillna(0)
        else:
            self.data["google_rating"] = 0.0

        if "google_user_rating_count" in self.data.columns:
            self.data["google_user_rating_count"] = pd.to_numeric(
                self.data["google_user_rating_count"],
                errors="coerce",
            ).fillna(0)
        else:
            self.data["google_user_rating_count"] = 0.0

        if "category_group" not in self.data.columns:
            self.data["category_group"] = "unknown"
        else:
            self.data["category_group"] = self.data["category_group"].fillna("unknown")

        if "google_primary_type" not in self.data.columns:
            self.data["google_primary_type"] = "unknown"
        else:
            self.data["google_primary_type"] = self.data[
                "google_primary_type"
            ].fillna("unknown")

        if "dwell_prediction_source" not in self.data.columns:
            self.data["dwell_prediction_source"] = "dbscan_episode_random_forest"
        else:
            self.data["dwell_prediction_source"] = self.data[
                "dwell_prediction_source"
            ].fillna("dbscan_episode_random_forest")

        self.data["dwell_bucket"] = self.data["predicted_dwell_minutes"].apply(
            self._assign_dwell_bucket
        )

        self.data["planning_note"] = self.data["predicted_dwell_minutes"].apply(
            self._assign_compact_planning_note
        )

        self.data["safe_for_travel_day"] = self.data[
            "predicted_dwell_minutes"
        ].apply(lambda x: bool(float(x) <= 100))

        self.data["safe_for_full_day_pairing"] = self.data[
            "predicted_dwell_minutes"
        ].apply(lambda x: bool(float(x) <= 100))

        # Simple ranking signal for display order only.
        self.data["planner_rank_score"] = (
            self.data["google_rating"] * 2.0
            + self.data["google_user_rating_count"].clip(upper=5000) / 1000.0
            - self.data["predicted_dwell_minutes"] / 240.0
        ).round(3)

        self.data = self.data.sort_values(
            by=["City", "planner_rank_score"],
            ascending=[True, False],
        ).reset_index(drop=True)

    def _assign_dwell_bucket(self, minutes):
        try:
            minutes = float(minutes)
        except Exception:
            minutes = float(self.default_dwell_minutes)

        if minutes <= 60:
            return "short_visit"

        if minutes <= 120:
            return "medium_visit"

        return "long_visit"

    def _assign_compact_planning_note(self, minutes):
        """
        Compact notes are easier for GPT-3.5 than long repeated sentences.
        """
        try:
            minutes = float(minutes)
        except Exception:
            minutes = float(self.default_dwell_minutes)

        if minutes <= 60:
            return "short; pair_ok"

        if minutes <= 100:
            return "medium; travel_day_1; full_day_1_or_2"

        if minutes <= 120:
            return "medium_long; avoid_many"

        return "long; main_visit"

    def load_db(self):
        self.data = pd.read_csv(self.path)
        self._prepare_data()

    def run(self, city: str) -> DataFrame:
        """
        Search for attractions by city.

        Returns compact dwell-aware rows for GPT-3.5.
        """

        city = extract_before_parenthesis(city)

        results = self.data[self.data["City"] == city].copy()
        results = results.reset_index(drop=True)

        if len(results) == 0:
            return "There is no attraction in this city."

        display_cols = [
            "Name",
            "City",
            "predicted_dwell_minutes",
            "dwell_bucket",
            "planning_note",
            "safe_for_travel_day",
            "safe_for_full_day_pairing",
            "planner_rank_score",
            "google_rating",
            "google_user_rating_count",
            "category_group",
            "google_primary_type",
            "dwell_prediction_source",
        ]

        display_cols = [c for c in display_cols if c in results.columns]

        return results[display_cols].head(self.top_k).reset_index(drop=True)

    def run_for_annotation(self, city: str) -> DataFrame:
        """
        Search for attractions by city for annotation/evaluation.

        Keeps all city rows rather than top-k only.
        """

        city = extract_before_parenthesis(city)

        results = self.data[self.data["City"] == city].copy()
        results = results.reset_index(drop=True)

        return results