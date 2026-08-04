from pathlib import Path
import json
import re
import math
import joblib
import numpy as np
import pandas as pd


MODEL_DIR = Path(r"C:\Users\negia\trip_plan\dwell_model_exports")


def normalize_name(x):
    if pd.isna(x):
        return ""
    x = str(x).lower().strip()
    x = re.sub(r"[^a-z0-9]+", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def haversine_m(lat1, lon1, lat2, lon2):
    """
    Distance in metres between two latitude/longitude points.
    """
    r = 6371000
    lat1, lon1, lat2, lon2 = map(
        np.radians,
        [lat1, lon1, lat2, lon2],
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    return 2 * r * np.arcsin(np.sqrt(a))


class DwellDurationPredictor:
    def __init__(
        self,
        model_dir=MODEL_DIR,
        mode="huber",
        default_duration=90.0,
        min_minutes=15.0,
        max_minutes=240.0,
        max_match_distance_m=300.0,
    ):
        self.model_dir = Path(model_dir)
        self.mode = mode
        self.default_duration = float(default_duration)
        self.min_minutes = float(min_minutes)
        self.max_minutes = float(max_minutes)
        self.max_match_distance_m = float(max_match_distance_m)

        self.poi = pd.read_csv(self.model_dir / "poi_dwell_feature_table.csv")

        self.poi["name_norm"] = self.poi["name"].apply(normalize_name)
        self.poi["google_name_norm"] = self.poi["google_name"].apply(normalize_name)
        self.poi["dataset_city_norm"] = self.poi["dataset_city"].apply(normalize_name)

        self.model = None
        self.feature_columns = None

        if self.mode == "huber":
            self.model = joblib.load(self.model_dir / "huber_native_google.joblib")
            with open(self.model_dir / "feature_columns.json", "r") as f:
                self.feature_columns = json.load(f)

    def clip_duration(self, value):
        if value is None or math.isnan(float(value)):
            value = self.default_duration

        value = float(value)
        return round(max(self.min_minutes, min(self.max_minutes, value)), 1)

    def find_best_match(self, name, city, latitude, longitude):
        name_norm = normalize_name(name)
        city_norm = normalize_name(city)

        candidates = self.poi.copy()

        # Prefer same city if possible.
        same_city = candidates[candidates["dataset_city_norm"] == city_norm]
        if len(same_city) > 0:
            candidates = same_city

        # Coordinate match.
        candidates = candidates.copy()
        candidates["distance_m"] = haversine_m(
            float(latitude),
            float(longitude),
            candidates["latitude"].astype(float),
            candidates["longitude"].astype(float),
        )

        candidates = candidates.sort_values("distance_m")

        # Keep nearby candidates first.
        nearby = candidates[candidates["distance_m"] <= self.max_match_distance_m]
        if len(nearby) > 0:
            candidates = nearby

        # Simple name score.
        def name_score(row):
            n1 = row.get("name_norm", "")
            n2 = row.get("google_name_norm", "")

            if name_norm == n1 or name_norm == n2:
                return 3
            if name_norm in n1 or n1 in name_norm:
                return 2
            if name_norm in n2 or n2 in name_norm:
                return 2

            # token overlap fallback
            a = set(name_norm.split())
            b = set(str(n1).split()) | set(str(n2).split())
            if not a or not b:
                return 0
            return len(a & b) / len(a | b)

        candidates["name_score_local"] = candidates.apply(name_score, axis=1)

        candidates = candidates.sort_values(
            ["name_score_local", "distance_m"],
            ascending=[False, True],
        )

        best = candidates.iloc[0]

        return best

    def predict_from_row(self, row):
        try:
            name = row["Name"]
            city = row["City"]
            lat = row["Latitude"]
            lon = row["Longitude"]

            matched = self.find_best_match(
                name=name,
                city=city,
                latitude=lat,
                longitude=lon,
            )

            if self.mode == "median":
                return self.clip_duration(matched["median_duration_proxy"])

            if self.mode == "huber":
                X = pd.DataFrame([matched[self.feature_columns].to_dict()])
                pred = float(self.model.predict(X)[0])
                return self.clip_duration(pred)

            return self.default_duration

        except Exception:
            return self.default_duration