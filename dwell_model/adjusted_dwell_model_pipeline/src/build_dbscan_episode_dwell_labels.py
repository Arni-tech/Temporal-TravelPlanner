from __future__ import annotations

"""
DBSCAN-assisted episode-based proxy dwell construction for Massive-STEPS.

Purpose
-------
Build cleaner proxy activity-duration labels from sparse POI check-ins.

Main idea
---------
Instead of:
    POI dwell = current POI -> next POI gap

we do:
    1. Cluster POIs spatially within each city using DBSCAN.
    2. Convert each user trail into a sequence of spatial clusters.
    3. Merge consecutive check-ins in the same DBSCAN cluster into one episode.
    4. Estimate duration from current episode start to next episode start,
       minus estimated travel time between episode centroids.
    5. Aggregate clean episode durations into POI-level labels.

This script sweeps multiple DBSCAN eps values and min_samples values.
It saves cluster validation and dwell summary metrics for each setting.

Notes
-----
- This does not recover true dwell time, because Massive-STEPS has check-in
  timestamps but no departure timestamps.
- It constructs an episode-based adjusted activity-duration proxy.
- No Google Places or OSM calls are needed here.
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score


EARTH_RADIUS_KM = 6371.0088


ITINERARY_GROUPS = {
    "food_leisure",
    "shopping",
    "tourism_attraction",
}


CATEGORY_CAPS = {
    "food_leisure": (10, 150),
    "shopping": (10, 210),
    "tourism_attraction": (15, 240),
}


FOOD_KEYWORDS = [
    "restaurant", "cafe", "coffee", "bar", "pub", "bakery", "pizza",
    "burger", "diner", "bistro", "food", "sandwich", "bbq", "barbecue",
    "kebab", "noodle", "sushi", "steak", "seafood", "ice cream",
    "dessert", "donut", "doughnut", "tea", "breakfast", "lunch",
    "deli", "buffet", "nightclub", "club", "lounge", "karaoke",
    "beer garden", "brewery", "wine bar", "fast food", "taco",
    "ramen", "japanese restaurant", "chinese restaurant", "asian restaurant",
    "indian restaurant", "italian restaurant", "mexican restaurant",
    "thai restaurant", "american restaurant", "food court", "food truck",
]

SHOPPING_KEYWORDS = [
    "shop", "store", "mall", "market", "supermarket", "grocery",
    "bookstore", "book store", "clothing", "boutique", "department",
    "electronics", "jewelry", "shoe", "flea market", "shopping",
    "retail", "hardware", "furniture", "gift", "toy", "cosmetics",
]

TOURISM_KEYWORDS = [
    "museum", "park", "theater", "theatre", "cinema", "movie theater",
    "historic", "monument", "landmark", "gallery", "art", "zoo",
    "aquarium", "stadium", "arena", "church", "mosque", "temple",
    "cathedral", "scenic", "lookout", "waterfront", "beach", "plaza",
    "bridge", "garden", "castle", "palace", "attraction", "theme park",
    "concert hall", "music venue", "cultural center", "outdoors",
    "other great outdoors", "soccer stadium", "city hall",
]


def normalize_text(x) -> str:
    x = "" if pd.isna(x) else str(x).lower()
    x = re.sub(r"[^a-z0-9\s/&+-]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def assign_category_group(category: str) -> str:
    text = normalize_text(category)

    if any(k in text for k in FOOD_KEYWORDS):
        return "food_leisure"
    if any(k in text for k in SHOPPING_KEYWORDS):
        return "shopping"
    if any(k in text for k in TOURISM_KEYWORDS):
        return "tourism_attraction"

    return "exclude"


def infer_dataset_city(path: Path) -> str:
    stem = path.stem.lower()
    for token in ["_checkins", "checkins_", "-checkins", "_pois", "pois_"]:
        stem = stem.replace(token, "")
    return stem


def normalize_city_name(x) -> str:
    return str(x).strip().lower().replace(" ", "_")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    aliases = {
        "lat": "latitude",
        "lon": "longitude",
        "lng": "longitude",
        "venue_name": "name",
        "poi_name": "name",
        "category": "venue_category",
        "poi_category": "venue_category",
        "poi_id": "venue_id",
        "userid": "user_id",
        "userID": "user_id",
        "trailID": "trail_id",
        "venueID": "venue_id",
    }

    for old, new in aliases.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    return df


def parse_timestamp_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        numeric = pd.to_numeric(s, errors="coerce")
        median_val = numeric.dropna().median()

        if pd.isna(median_val):
            return pd.to_datetime(s, errors="coerce")
        if median_val > 1e11:
            return pd.to_datetime(s, unit="ms", errors="coerce")
        if median_val > 1e8:
            return pd.to_datetime(s, unit="s", errors="coerce")

    return pd.to_datetime(s, errors="coerce")


def haversine_km(lat1, lon1, lat2, lon2):
    r = EARTH_RADIUS_KM

    lat1 = np.radians(lat1.astype(float))
    lon1 = np.radians(lon1.astype(float))
    lat2 = np.radians(lat2.astype(float))
    lon2 = np.radians(lon2.astype(float))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    return 2 * r * np.arcsin(np.sqrt(a))


def estimate_travel_time_min(distance_km: pd.Series) -> pd.Series:
    assumed_speed_kmh = np.where(distance_km <= 1.0, 4.5, 18.0)
    return distance_km / assumed_speed_kmh * 60


def parse_float_list(x: str) -> List[float]:
    return [float(v.strip()) for v in x.split(",") if v.strip()]


def parse_int_list(x: str) -> List[int]:
    return [int(v.strip()) for v in x.split(",") if v.strip()]


def dominant_value(values: pd.Series) -> Optional[str]:
    values = values.dropna()
    if values.empty:
        return None
    return values.value_counts().index[0]


def load_all_csvs(input_dir: str) -> pd.DataFrame:
    input_dir = Path(input_dir)
    files = sorted(input_dir.glob("*.csv"))

    if not files:
        raise FileNotFoundError(f"No CSV files found in: {input_dir}")

    frames = []

    for path in files:
        print("Loading:", path)
        df = pd.read_csv(path)
        df = standardize_columns(df)

        if "dataset_city" not in df.columns:
            df["dataset_city"] = infer_dataset_city(path)

        df["dataset_city"] = df["dataset_city"].apply(normalize_city_name)
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def prepare_checkins(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "dataset_city",
        "trail_id",
        "user_id",
        "venue_id",
        "latitude",
        "longitude",
        "name",
        "venue_category",
        "timestamp",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()

    if "venue_city" not in df.columns:
        df["venue_city"] = ""

    if "venue_country" not in df.columns:
        df["venue_country"] = ""

    df["timestamp"] = parse_timestamp_series(df["timestamp"])

    for col in ["latitude", "longitude"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "category_group" not in df.columns:
        print("category_group not found. Creating category_group from venue_category keywords.")
        df["category_group"] = df["venue_category"].apply(assign_category_group)

    df = df[df["category_group"].isin(ITINERARY_GROUPS)].copy()

    df = df.dropna(
        subset=[
            "timestamp",
            "latitude",
            "longitude",
            "venue_id",
            "trail_id",
            "user_id",
        ]
    ).copy()

    df["dataset_city"] = df["dataset_city"].apply(normalize_city_name)

    df["global_user_id"] = (
        df["dataset_city"].astype(str) + "::" + df["user_id"].astype(str)
    )

    df["global_trail_id"] = (
        df["dataset_city"].astype(str) + "::" + df["trail_id"].astype(str)
    )

    df["global_venue_id"] = (
        df["dataset_city"].astype(str) + "::" + df["venue_id"].astype(str)
    )

    df["venue_category_norm"] = df["venue_category"].apply(normalize_text)

    return df.sort_values(["global_user_id", "global_trail_id", "timestamp"]).copy()


def build_unique_pois(checkins: pd.DataFrame) -> pd.DataFrame:
    poi = (
        checkins
        .groupby("global_venue_id", dropna=False)
        .agg(
            dataset_city=("dataset_city", "first"),
            venue_id=("venue_id", "first"),
            name=("name", "first"),
            venue_category=("venue_category", "first"),
            category_group=("category_group", "first"),
            latitude=("latitude", "first"),
            longitude=("longitude", "first"),
            venue_city=("venue_city", "first"),
            venue_country=("venue_country", "first"),
            checkin_count=("timestamp", "count"),
        )
        .reset_index()
    )

    return poi


def approx_dunn_index(coords_rad: np.ndarray, labels: np.ndarray) -> float:
    """
    Approximate Dunn Validity Index:
        min centroid-to-centroid distance / max within-cluster radius*2

    This is not exact Dunn, but it is fast and useful for comparing eps values.
    """
    valid = labels != -1
    coords = coords_rad[valid]
    labs = labels[valid]

    unique_labels = np.unique(labs)

    if len(unique_labels) < 2:
        return np.nan

    centroids = []
    max_diameter = 0.0

    for lab in unique_labels:
        points = coords[labs == lab]
        centroid = points.mean(axis=0)
        centroids.append(centroid)

        if len(points) > 1:
            d = haversine_distance_matrix_km(points, centroid.reshape(1, -1)).max()
            max_diameter = max(max_diameter, float(2 * d))

    if max_diameter <= 0:
        return np.nan

    centroids = np.vstack(centroids)
    inter = haversine_distance_matrix_km(centroids, centroids)
    inter[inter == 0] = np.inf
    min_inter = np.min(inter)

    return float(min_inter / max_diameter)


def haversine_distance_matrix_km(a_rad: np.ndarray, b_rad: np.ndarray) -> np.ndarray:
    """
    Pairwise haversine distance for arrays already in radians.
    Shape:
        a_rad: n x 2
        b_rad: m x 2
    """
    lat1 = a_rad[:, [0]]
    lon1 = a_rad[:, [1]]
    lat2 = b_rad[:, 0][None, :]
    lon2 = b_rad[:, 1][None, :]

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    x = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(x))


def cluster_city_pois_dbscan(
    city_pois: pd.DataFrame,
    eps_m: float,
    min_samples: int,
    silhouette_sample_size: int = 5000,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict]:
    city_pois = city_pois.copy()

    coords_rad = np.radians(city_pois[["latitude", "longitude"]].to_numpy(dtype=float))
    eps_rad = (eps_m / 1000.0) / EARTH_RADIUS_KM

    model = DBSCAN(
        eps=eps_rad,
        min_samples=min_samples,
        metric="haversine",
        algorithm="ball_tree",
    )

    labels = model.fit_predict(coords_rad)
    city_pois["dbscan_label"] = labels

    # Noise points become singleton clusters so they do not disappear.
    city_pois["spatial_cluster_id"] = np.where(
        city_pois["dbscan_label"] == -1,
        city_pois["dataset_city"].astype(str) + "::noise::" + city_pois["global_venue_id"].astype(str),
        city_pois["dataset_city"].astype(str) + "::cluster::" + city_pois["dbscan_label"].astype(str),
    )

    n_pois = len(city_pois)
    n_noise = int((labels == -1).sum())
    n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
    noise_rate = float(n_noise / n_pois) if n_pois else np.nan

    non_noise_labels = labels[labels != -1]
    cluster_sizes = pd.Series(non_noise_labels).value_counts() if len(non_noise_labels) else pd.Series(dtype=int)

    approx_dunn = approx_dunn_index(coords_rad, labels)

    sil = np.nan
    if n_clusters >= 2 and len(non_noise_labels) >= 10:
        valid_idx = np.where(labels != -1)[0]
        if len(valid_idx) > silhouette_sample_size:
            rng = np.random.default_rng(random_state)
            valid_idx = rng.choice(valid_idx, size=silhouette_sample_size, replace=False)

        try:
            sil = float(
                silhouette_score(
                    coords_rad[valid_idx],
                    labels[valid_idx],
                    metric="haversine",
                )
            )
        except Exception:
            sil = np.nan

    metrics = {
        "dataset_city": str(city_pois["dataset_city"].iloc[0]),
        "eps_m": float(eps_m),
        "min_samples": int(min_samples),
        "n_pois": int(n_pois),
        "n_clusters_excluding_noise": int(n_clusters),
        "n_noise_pois": int(n_noise),
        "noise_rate": float(noise_rate),
        "mean_cluster_size": float(cluster_sizes.mean()) if len(cluster_sizes) else np.nan,
        "median_cluster_size": float(cluster_sizes.median()) if len(cluster_sizes) else np.nan,
        "max_cluster_size": int(cluster_sizes.max()) if len(cluster_sizes) else 0,
        "approx_dunn_index": approx_dunn,
        "silhouette_haversine": sil,
    }

    return city_pois, metrics


def cluster_all_pois_dbscan(
    pois: pd.DataFrame,
    eps_m: float,
    min_samples: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    clustered_parts = []
    metrics = []

    for city, city_pois in pois.groupby("dataset_city", sort=False):
        clustered_city, city_metrics = cluster_city_pois_dbscan(
            city_pois,
            eps_m=eps_m,
            min_samples=min_samples,
        )
        clustered_parts.append(clustered_city)
        metrics.append(city_metrics)

    return pd.concat(clustered_parts, ignore_index=True), pd.DataFrame(metrics)


def attach_clusters_to_checkins(checkins: pd.DataFrame, clustered_pois: pd.DataFrame) -> pd.DataFrame:
    cluster_cols = [
        "global_venue_id",
        "dbscan_label",
        "spatial_cluster_id",
    ]

    out = checkins.merge(
        clustered_pois[cluster_cols],
        on="global_venue_id",
        how="left",
    )

    missing = out["spatial_cluster_id"].isna().sum()
    if missing:
        raise ValueError(f"Missing spatial cluster assignment for {missing} check-ins.")

    return out


def assign_cluster_episodes(checkins: pd.DataFrame) -> pd.DataFrame:
    """
    Episode changes whenever spatial_cluster_id changes within the same user trail.

    This is the core DBSCAN-based episode construction step.
    No manual distance threshold is used for episode splitting here;
    DBSCAN cluster membership defines the spatial activity region.
    """
    checkins = checkins.sort_values(
        ["global_user_id", "global_trail_id", "timestamp"]
    ).copy()

    group_cols = ["global_user_id", "global_trail_id"]

    checkins["prev_spatial_cluster_id"] = (
        checkins.groupby(group_cols)["spatial_cluster_id"].shift(1)
    )

    checkins["new_episode_flag"] = (
        checkins["prev_spatial_cluster_id"].isna()
        | (checkins["spatial_cluster_id"] != checkins["prev_spatial_cluster_id"])
    ).astype(int)

    checkins["episode_id_within_trail"] = (
        checkins.groupby(group_cols)["new_episode_flag"].cumsum() - 1
    )

    checkins["global_episode_id"] = (
        checkins["global_trail_id"].astype(str)
        + "::ep"
        + checkins["episode_id_within_trail"].astype(str)
    )

    return checkins


def build_episode_table(checkins: pd.DataFrame) -> pd.DataFrame:
    grouped = checkins.groupby("global_episode_id", dropna=False)

    episode = grouped.agg(
        global_user_id=("global_user_id", "first"),
        global_trail_id=("global_trail_id", "first"),
        dataset_city=("dataset_city", "first"),
        trail_id=("trail_id", "first"),
        user_id=("user_id", "first"),

        spatial_cluster_id=("spatial_cluster_id", "first"),
        dbscan_label=("dbscan_label", "first"),

        episode_start=("timestamp", "min"),
        episode_end=("timestamp", "max"),
        checkin_count=("timestamp", "count"),
        unique_venue_count=("global_venue_id", "nunique"),

        first_global_venue_id=("global_venue_id", "first"),
        first_venue_id=("venue_id", "first"),
        first_poi_name=("name", "first"),
        first_venue_category=("venue_category", "first"),
        first_category_group=("category_group", "first"),

        dominant_category_group=("category_group", dominant_value),
        dominant_venue_category=("venue_category", dominant_value),

        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),

        venue_city=("venue_city", "first"),
        venue_country=("venue_country", "first"),
    ).reset_index()

    episode["observed_episode_span_minutes"] = (
        episode["episode_end"] - episode["episode_start"]
    ).dt.total_seconds() / 60

    episode = episode.sort_values(
        ["global_user_id", "global_trail_id", "episode_start"]
    ).copy()

    group_cols = ["global_user_id", "global_trail_id"]

    episode["next_episode_id"] = episode.groupby(group_cols)["global_episode_id"].shift(-1)
    episode["next_episode_start"] = episode.groupby(group_cols)["episode_start"].shift(-1)
    episode["next_episode_latitude"] = episode.groupby(group_cols)["latitude"].shift(-1)
    episode["next_episode_longitude"] = episode.groupby(group_cols)["longitude"].shift(-1)
    episode["next_spatial_cluster_id"] = episode.groupby(group_cols)["spatial_cluster_id"].shift(-1)
    episode["next_dominant_category_group"] = episode.groupby(group_cols)["dominant_category_group"].shift(-1)

    episode["gap_to_next_episode_minutes"] = (
        episode["next_episode_start"] - episode["episode_start"]
    ).dt.total_seconds() / 60

    has_next = (
        episode["gap_to_next_episode_minutes"].notna()
        & (episode["gap_to_next_episode_minutes"] > 0)
    )

    episode["distance_to_next_episode_km"] = np.nan
    episode.loc[has_next, "distance_to_next_episode_km"] = haversine_km(
        episode.loc[has_next, "latitude"],
        episode.loc[has_next, "longitude"],
        episode.loc[has_next, "next_episode_latitude"],
        episode.loc[has_next, "next_episode_longitude"],
    )

    episode["episode_transition_speed_kmh"] = (
        episode["distance_to_next_episode_km"]
        / (episode["gap_to_next_episode_minutes"] / 60)
    )

    episode["estimated_travel_time_to_next_episode_min"] = estimate_travel_time_min(
        episode["distance_to_next_episode_km"]
    )

    episode["raw_episode_duration_proxy_minutes"] = (
        episode["gap_to_next_episode_minutes"]
        - episode["estimated_travel_time_to_next_episode_min"]
    )

    episode["episode_date_changed"] = (
        episode["episode_start"].dt.date != episode["next_episode_start"].dt.date
    ).astype(int)

    episode["next_episode_hour"] = episode["next_episode_start"].dt.hour

    episode["overnight_to_next_episode"] = (
        (episode["episode_date_changed"] == 1)
        | (episode["next_episode_hour"].between(3, 6))
    ).astype(int)

    episode["category_min_dwell"] = episode["dominant_category_group"].map(
        lambda g: CATEGORY_CAPS.get(g, (5, 240))[0]
    )

    episode["category_max_dwell"] = episode["dominant_category_group"].map(
        lambda g: CATEGORY_CAPS.get(g, (5, 240))[1]
    )

    return episode


def filter_clean_episodes(
    episode: pd.DataFrame,
    max_episode_distance_km: float,
    max_episode_speed_kmh: float,
) -> pd.DataFrame:
    episode = episode.copy()

    duration_ok = (
        episode["raw_episode_duration_proxy_minutes"].notna()
        & (episode["raw_episode_duration_proxy_minutes"] >= episode["category_min_dwell"])
        & (episode["raw_episode_duration_proxy_minutes"] <= episode["category_max_dwell"])
    )

    movement_ok = (
        episode["distance_to_next_episode_km"].notna()
        & (episode["distance_to_next_episode_km"] <= max_episode_distance_km)
        & (episode["episode_transition_speed_kmh"] <= max_episode_speed_kmh)
        & (episode["overnight_to_next_episode"] == 0)
    )

    clean = episode[duration_ok & movement_ok].copy()

    clean["episode_duration_proxy_minutes"] = clean["raw_episode_duration_proxy_minutes"]

    return clean


def aggregate_poi_labels(
    clean_episode: pd.DataFrame,
    min_obs: int,
    poi_label_mode: str,
) -> pd.DataFrame:
    """
    poi_label_mode:
        single_only:
            Use only episodes with exactly one unique POI.
            Most conservative for POI-level training labels.
        first:
            Assign each episode to its first POI.
            More data, but weaker when cluster episode has multiple POIs.
    """
    if poi_label_mode == "single_only":
        label_source = clean_episode[clean_episode["unique_venue_count"] == 1].copy()
    elif poi_label_mode == "first":
        label_source = clean_episode.copy()
    else:
        raise ValueError("poi_label_mode must be 'single_only' or 'first'.")

    poi = (
        label_source
        .groupby("first_global_venue_id")
        .agg(
            global_venue_id=("first_global_venue_id", "first"),
            poi_name=("first_poi_name", "first"),
            venue_category=("first_venue_category", "first"),
            category_group=("dominant_category_group", "first"),
            dataset_city=("dataset_city", "first"),
            venue_city=("venue_city", "first"),
            venue_country=("venue_country", "first"),
            latitude=("latitude", "first"),
            longitude=("longitude", "first"),

            obs_count=("episode_duration_proxy_minutes", "count"),
            p25_episode_dwell_minutes=(
                "episode_duration_proxy_minutes",
                lambda x: np.percentile(x, 25),
            ),
            median_episode_dwell_minutes=("episode_duration_proxy_minutes", "median"),
            p75_episode_dwell_minutes=(
                "episode_duration_proxy_minutes",
                lambda x: np.percentile(x, 75),
            ),
            mean_episode_dwell_minutes=("episode_duration_proxy_minutes", "mean"),
            std_episode_dwell_minutes=("episode_duration_proxy_minutes", "std"),

            multi_poi_episode_rate=(
                "unique_venue_count",
                lambda x: (x > 1).mean(),
            ),
            median_distance_to_next_episode_km=("distance_to_next_episode_km", "median"),
            median_estimated_travel_time_to_next_episode_min=(
                "estimated_travel_time_to_next_episode_min",
                "median",
            ),
        )
        .reset_index(drop=True)
    )

    poi = poi[poi["obs_count"] >= min_obs].copy()
    poi["log_obs_count"] = np.log1p(poi["obs_count"])

    return poi.sort_values(["category_group", "dataset_city", "poi_name"])


def make_category_summary(clean_episode: pd.DataFrame) -> pd.DataFrame:
    return (
        clean_episode
        .groupby(["dominant_category_group"], dropna=False)
        .agg(
            obs_count=("episode_duration_proxy_minutes", "count"),
            p25_episode_dwell_minutes=(
                "episode_duration_proxy_minutes",
                lambda x: np.percentile(x, 25),
            ),
            median_episode_dwell_minutes=("episode_duration_proxy_minutes", "median"),
            p75_episode_dwell_minutes=(
                "episode_duration_proxy_minutes",
                lambda x: np.percentile(x, 75),
            ),
            mean_episode_dwell_minutes=("episode_duration_proxy_minutes", "mean"),
            std_episode_dwell_minutes=("episode_duration_proxy_minutes", "std"),
            multi_poi_episode_rate=(
                "unique_venue_count",
                lambda x: (x > 1).mean(),
            ),
        )
        .reset_index()
        .sort_values("dominant_category_group")
    )


def run_for_setting(
    checkins: pd.DataFrame,
    pois: pd.DataFrame,
    eps_m: float,
    min_samples: int,
    args,
    write_outputs: bool = False,
    setting_output_dir: Optional[Path] = None,
) -> Dict:
    clustered_pois, cluster_metrics = cluster_all_pois_dbscan(
        pois,
        eps_m=eps_m,
        min_samples=min_samples,
    )

    clustered_checkins = attach_clusters_to_checkins(checkins, clustered_pois)
    clustered_checkins = assign_cluster_episodes(clustered_checkins)

    episode = build_episode_table(clustered_checkins)

    clean_episode = filter_clean_episodes(
        episode,
        max_episode_distance_km=args.max_episode_distance_km,
        max_episode_speed_kmh=args.max_episode_speed_kmh,
    )

    poi_labels = aggregate_poi_labels(
        clean_episode,
        min_obs=args.min_obs,
        poi_label_mode=args.poi_label_mode,
    )

    category_summary = make_category_summary(clean_episode)

    metrics = {
        "eps_m": float(eps_m),
        "min_samples": int(min_samples),
        "checkins": int(len(checkins)),
        "unique_pois": int(len(pois)),
        "all_episodes": int(len(episode)),
        "clean_episodes": int(len(clean_episode)),
        "poi_labels_min_obs": int(len(poi_labels)),
        "median_city_approx_dunn": float(cluster_metrics["approx_dunn_index"].median(skipna=True)),
        "mean_city_approx_dunn": float(cluster_metrics["approx_dunn_index"].mean(skipna=True)),
        "median_city_silhouette": float(cluster_metrics["silhouette_haversine"].median(skipna=True)),
        "mean_noise_rate": float(cluster_metrics["noise_rate"].mean(skipna=True)),
        "mean_clusters_per_city": float(cluster_metrics["n_clusters_excluding_noise"].mean(skipna=True)),
    }

    for _, row in category_summary.iterrows():
        cat = row["dominant_category_group"]
        metrics[f"{cat}_obs"] = int(row["obs_count"])
        metrics[f"{cat}_median"] = float(row["median_episode_dwell_minutes"])
        metrics[f"{cat}_p25"] = float(row["p25_episode_dwell_minutes"])
        metrics[f"{cat}_p75"] = float(row["p75_episode_dwell_minutes"])

    if write_outputs:
        if setting_output_dir is None:
            raise ValueError("setting_output_dir is required when write_outputs=True.")

        setting_output_dir.mkdir(parents=True, exist_ok=True)

        clustered_pois.to_csv(setting_output_dir / "dbscan_clustered_pois.csv", index=False)
        cluster_metrics.to_csv(setting_output_dir / "dbscan_city_cluster_metrics.csv", index=False)
        clustered_checkins.to_csv(setting_output_dir / "debug_checkins_with_dbscan_episodes.csv", index=False)
        episode.to_csv(setting_output_dir / "episode_level_observations.csv", index=False)
        clean_episode.to_csv(setting_output_dir / "clean_episode_dwell_observations.csv", index=False)
        poi_labels.to_csv(setting_output_dir / "episode_dwell_pois_debug.csv", index=False)
        category_summary.to_csv(setting_output_dir / "category_dwell_summary.csv", index=False)

        with open(setting_output_dir / "setting_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)

    return metrics


def choose_best_setting(sweep_df: pd.DataFrame) -> pd.Series:
    """
    Choose the DBSCAN setting for POI-level episode dwell-label construction.

    Selection standard
    ------------------
    The aim of DBSCAN here is not broad neighbourhood discovery. The aim is to
    group nearby POIs into local activity regions suitable for POI-level
    activity-duration proxy labels.

    Therefore, parameter selection follows a validity-constrained approach:

    1. Spatial scale constraint:
       eps must be within a local POI-complex scale. We use eps <= 200m because
       the target unit is a local activity region, not a neighbourhood.

    2. Data sufficiency constraint:
       The setting must retain enough clean episode observations and POI-level
       labels for downstream supervised regression.

    3. Cluster validity:
       Among valid settings, choose the highest median city-level silhouette
       score. Dunn index is retained as a diagnostic but not used as the primary
       selector because it can favour overly broad clusters.

    4. Tie-break:
       Prefer more labelled POIs, then more clean episodes, then smaller eps.
    """

    df = sweep_df.copy()

    numeric_cols = [
        "eps_m",
        "min_samples",
        "clean_episodes",
        "poi_labels_min_obs",
        "mean_clusters_per_city",
        "mean_noise_rate",
        "median_city_silhouette",
        "median_city_approx_dunn",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    valid = df[
        (df["eps_m"] <= 200)
        & (df["clean_episodes"] >= 40000)
        & (df["poi_labels_min_obs"] >= 1000)
        & (df["mean_clusters_per_city"] >= 100)
        & (df["mean_noise_rate"] <= 0.45)
    ].copy()

    if valid.empty:
        print(
            "Warning: no setting passed the predefined validity constraints. "
            "Falling back to settings with eps <= 200m."
        )
        valid = df[df["eps_m"] <= 200].copy()

    if valid.empty:
        print(
            "Warning: no setting with eps <= 200m found. "
            "Falling back to all settings."
        )
        valid = df.copy()

    valid["selection_score"] = valid["median_city_silhouette"]

    missing = valid["selection_score"].isna()
    valid.loc[missing, "selection_score"] = valid.loc[
        missing, "median_city_approx_dunn"
    ]

    valid = valid.sort_values(
        [
            "selection_score",
            "poi_labels_min_obs",
            "clean_episodes",
            "eps_m",
        ],
        ascending=[
            False,
            False,
            False,
            True,
        ],
    )

    return valid.iloc[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "--select-only",
    action="store_true",
    help=(
        "Only select the best DBSCAN setting from an existing "
        "dbscan_sweep_metrics.csv file. Does not rerun DBSCAN."
        ),
    )

    parser.add_argument(
        "--sweep-metrics-path",
        default=None,
        help=(
            "Path to an existing dbscan_sweep_metrics.csv file. "
            "Used with --select-only. If omitted, uses output-dir/dbscan_sweep_metrics.csv."
        ),
    )

    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument(
        "--eps-grid-m",
        default="100,150,200,300,500",
        help="Comma-separated DBSCAN eps values in metres.",
    )

    parser.add_argument(
        "--min-samples-grid",
        default="3,5",
        help="Comma-separated DBSCAN min_samples values.",
    )

    parser.add_argument("--min-obs", type=int, default=5)

    parser.add_argument(
        "--poi-label-mode",
        choices=["single_only", "first"],
        default="single_only",
        help=(
            "single_only is safer for POI-level labels. "
            "first gives more data but assigns multi-POI episodes to first POI."
        ),
    )

    parser.add_argument("--max-episode-distance-km", type=float, default=10.0)
    parser.add_argument("--max-episode-speed-kmh", type=float, default=80.0)

    parser.add_argument(
        "--write-all-settings",
        action="store_true",
        help="Write full detailed outputs for every eps/min_samples setting.",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.select_only:
        if args.sweep_metrics_path is not None:
            sweep_path = Path(args.sweep_metrics_path)
        else:
            sweep_path = output_dir / "dbscan_sweep_metrics.csv"

        if not sweep_path.exists():
            raise FileNotFoundError(f"Sweep metrics file not found: {sweep_path}")

        sweep_df = pd.read_csv(sweep_path)
        best = choose_best_setting(sweep_df)

        ranked = sweep_df.copy()
        ranked["selected"] = False

        match = (
            (ranked["eps_m"].astype(float) == float(best["eps_m"]))
            & (ranked["min_samples"].astype(int) == int(best["min_samples"]))
        )
        ranked.loc[match, "selected"] = True

        selected_path = output_dir / "selected_dbscan_setting.json"
        ranked_path = output_dir / "dbscan_sweep_metrics_with_selection.csv"

        with open(selected_path, "w", encoding="utf-8") as f:
            json.dump(best.to_dict(), f, indent=4)

        ranked.to_csv(ranked_path, index=False)

        print("\nLoaded sweep metrics from:")
        print(sweep_path)

        print("\nSelected DBSCAN setting:")
        print(best.to_string())

        print("\nSaved selected setting to:")
        print(selected_path)

        print("\nSaved sweep table with selected flag to:")
        print(ranked_path)

        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eps_values = parse_float_list(args.eps_grid_m)
    min_samples_values = parse_int_list(args.min_samples_grid)

    raw = load_all_csvs(args.input_dir)
    checkins = prepare_checkins(raw)
    pois = build_unique_pois(checkins)

    print("\nPrepared data:")
    print("Check-ins:", len(checkins))
    print("Unique POIs:", len(pois))
    print("Cities:", checkins["dataset_city"].nunique())

    sweep_metrics = []

    for eps_m in eps_values:
        for min_samples in min_samples_values:
            print(f"\nRunning DBSCAN setting: eps={eps_m}m, min_samples={min_samples}")

            setting_name = f"eps_{int(eps_m)}m_min_{min_samples}"
            setting_dir = output_dir / setting_name

            metrics = run_for_setting(
                checkins=checkins,
                pois=pois,
                eps_m=eps_m,
                min_samples=min_samples,
                args=args,
                write_outputs=args.write_all_settings,
                setting_output_dir=setting_dir,
            )

            sweep_metrics.append(metrics)

            print(
                "clean_episodes=",
                metrics["clean_episodes"],
                "poi_labels=",
                metrics["poi_labels_min_obs"],
                "median_dunn=",
                metrics["median_city_approx_dunn"],
            )

    sweep_df = pd.DataFrame(sweep_metrics)
    sweep_path = output_dir / "dbscan_sweep_metrics.csv"
    sweep_df.to_csv(sweep_path, index=False)

    best = choose_best_setting(sweep_df)

    best_eps = float(best["eps_m"])
    best_min_samples = int(best["min_samples"])
    best_name = f"best_eps_{int(best_eps)}m_min_{best_min_samples}"

    print("\nBest setting selected:")
    print(best.to_string())

    best_dir = output_dir / best_name

    _ = run_for_setting(
        checkins=checkins,
        pois=pois,
        eps_m=best_eps,
        min_samples=best_min_samples,
        args=args,
        write_outputs=True,
        setting_output_dir=best_dir,
    )

    with open(output_dir / "best_setting.json", "w", encoding="utf-8") as f:
        json.dump(best.to_dict(), f, indent=4)

    print("\nSaved sweep metrics:")
    print(sweep_path)

    print("\nSaved best setting outputs to:")
    print(best_dir)

    print("\nUse this POI-level label file for retraining:")
    print(best_dir / "episode_dwell_pois_debug.csv")

    print("\nInspect this file to compare eps values:")
    print(output_dir / "dbscan_sweep_metrics.csv")


if __name__ == "__main__":
    main()