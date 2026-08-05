from __future__ import annotations

"""
02_merge_google_features.py

Fetch and attach Google Places features to DBSCAN dwell-labelled POIs.

This script is part of the adjusted dwell model pipeline.

Input:
    A POI-level dwell-label table produced by:
        01_build_dbscan_episode_dwell_labels.py

Process:
    For each POI, call the Google Places Text Search API using:
        - POI name
        - city name
        - optional latitude/longitude location bias

    Candidate Google Places results are scored using:
        - name similarity
        - city/address match
        - geographic distance
        - availability of rating/review metadata

Output:
    1. Full input table with Google Places features appended.
    2. A standalone Google features table.
    3. A JSONL cache of API results.
    4. A JSONL error log.

Environment:
    Requires one of:
        GOOGLE_MAPS_API_KEY
        GOOGLE_API_KEY

Important:
    Do not commit API keys, .env files, or large cache files to GitHub.
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.parse import unquote

import pandas as pd
import requests


SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"

FIELD_MASK = ",".join(
    [
        "places.id",
        "places.displayName",
        "places.formattedAddress",
        "places.location",
        "places.rating",
        "places.userRatingCount",
        "places.primaryType",
        "places.types",
        "places.businessStatus",
        "places.regularOpeningHours",
    ]
)

DEFAULT_TIMEOUT = 30
MAX_RETRIES = 5
DEFAULT_SLEEP_BETWEEN_CALLS = 0.2


# ---------------------------------------------------------------------
# Column helpers
# ---------------------------------------------------------------------


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def require_column(df: pd.DataFrame, candidates: list[str], logical_name: str) -> str:
    col = first_existing_column(df, candidates)

    if col is None:
        raise ValueError(
            f"Could not find a column for {logical_name}. "
            f"Tried: {candidates}. "
            f"Available columns: {list(df.columns)}"
        )

    return col


def standardise_input_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert different possible DBSCAN / Massive-STEPS column names into
    a standard set used by this script.

    Standard columns created:
        poi_key
        poi_name_for_google
        city_name_for_google
        lat_for_google
        lon_for_google
    """

    df = df.copy()

    name_col = require_column(
        df,
        candidates=[
            "poi_name",
            "name",
            "venue_name",
            "poiName",
            "google_name",
            "Name",
        ],
        logical_name="POI name",
    )

    lat_col = require_column(
        df,
        candidates=[
            "latitude",
            "lat",
            "venue_latitude",
            "matched_lat",
            "google_latitude",
        ],
        logical_name="latitude",
    )

    lon_col = require_column(
        df,
        candidates=[
            "longitude",
            "lon",
            "lng",
            "venue_longitude",
            "matched_lon",
            "google_longitude",
        ],
        logical_name="longitude",
    )

    city_col = first_existing_column(
        df,
        [
            "venue_city",
            "city_name",
            "dataset_city",
            "city",
            "City",
        ],
    )

    id_col = first_existing_column(
        df,
        [
            "global_venue_id",
            "poi_id",
            "venue_id",
            "poiID",
            "id",
        ],
    )

    df["poi_name_for_google"] = df[name_col].fillna("").astype(str)

    if city_col is not None:
        df["city_name_for_google"] = df[city_col].fillna("").astype(str)
    else:
        df["city_name_for_google"] = ""

    df["lat_for_google"] = pd.to_numeric(df[lat_col], errors="coerce")
    df["lon_for_google"] = pd.to_numeric(df[lon_col], errors="coerce")

    if id_col is not None:
        df["poi_key"] = df[id_col].fillna("").astype(str)
    else:
        df["poi_key"] = ""

    # Make a stable key even when no single ID column exists.
    missing_or_blank_key = df["poi_key"].str.strip() == ""
    df.loc[missing_or_blank_key, "poi_key"] = (
        df.loc[missing_or_blank_key, "city_name_for_google"].astype(str)
        + "::"
        + df.loc[missing_or_blank_key, "poi_name_for_google"].astype(str)
        + "::"
        + df.loc[missing_or_blank_key, "lat_for_google"].round(6).astype(str)
        + "::"
        + df.loc[missing_or_blank_key, "lon_for_google"].round(6).astype(str)
    )

    return df


# ---------------------------------------------------------------------
# Text / distance helpers
# ---------------------------------------------------------------------


def safe_float(x: Any) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def normalize_name(x: Any) -> str:
    x = "" if pd.isna(x) else str(x)
    x = unquote(x)
    x = x.replace("_", " ")
    x = x.replace("&", "and")
    x = " ".join(x.strip().lower().split())
    return x


def token_jaccard(a: str, b: str) -> float:
    a_set = set(normalize_name(a).split())
    b_set = set(normalize_name(b).split())

    if not a_set or not b_set:
        return 0.0

    return len(a_set & b_set) / len(a_set | b_set)


def haversine_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    r = 6371.0

    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    )

    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------


def load_cached_records(cache_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load completed Google features from a JSONL cache.

    The cache makes the script safe to resume without repeating API calls.
    """

    records: Dict[str, Dict[str, Any]] = {}

    if not cache_path.exists():
        return records

    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except Exception:
                continue

            key = record.get("poi_key")
            status = record.get("google_match_status")

            if key and status in {"matched", "no_match", "ambiguous", "flagged"}:
                records[key] = record

    return records


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv_row(csv_path: Path, row_dict: Dict[str, Any], header: List[str]) -> None:
    file_exists = csv_path.exists()

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row_dict)


# ---------------------------------------------------------------------
# Google Places API helpers
# ---------------------------------------------------------------------


def build_payload(row: pd.Series, radius_m: float) -> Dict[str, Any]:
    poi_name = normalize_name(row["poi_name_for_google"])
    city_name = str(row["city_name_for_google"]).strip()

    text_query = poi_name

    if city_name:
        text_query = f"{poi_name}, {city_name}"

    payload: Dict[str, Any] = {
        "textQuery": text_query,
        "pageSize": 5,
        "languageCode": "en",
    }

    lat = safe_float(row.get("lat_for_google"))
    lon = safe_float(row.get("lon_for_google"))

    if lat is not None and lon is not None:
        payload["locationBias"] = {
            "circle": {
                "center": {
                    "latitude": lat,
                    "longitude": lon,
                },
                "radius": float(radius_m),
            }
        }

    return payload


def call_text_search(api_key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": FIELD_MASK,
    }

    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                SEARCH_URL,
                headers=headers,
                json=payload,
                timeout=DEFAULT_TIMEOUT,
            )

            if response.status_code == 200:
                return response.json()

            if response.status_code in {429, 500, 502, 503, 504}:
                last_error = RuntimeError(
                    f"HTTP {response.status_code}: {response.text[:500]}"
                )
                time.sleep(min(2**attempt, 30))
                continue

            raise RuntimeError(
                f"HTTP {response.status_code}: {response.text[:1000]}"
            )

        except requests.RequestException as exc:
            last_error = exc
            time.sleep(min(2**attempt, 30))

    raise RuntimeError(f"Text Search failed after retries: {last_error}")


# ---------------------------------------------------------------------
# Matching logic
# ---------------------------------------------------------------------


def score_candidate(
    candidate: Dict[str, Any],
    poi_name: str,
    city_name: str,
    lat: Optional[float],
    lon: Optional[float],
) -> Dict[str, Any]:
    score = 0.0

    candidate_name = ((candidate.get("displayName") or {}).get("text")) or ""
    candidate_address = candidate.get("formattedAddress", "") or ""

    poi_name_normalized = normalize_name(poi_name)
    candidate_name_normalized = normalize_name(candidate_name)

    city_name_lower = city_name.lower().strip()
    candidate_address_lower = candidate_address.lower().strip()

    name_similarity = token_jaccard(poi_name_normalized, candidate_name_normalized)

    if candidate_name_normalized == poi_name_normalized:
        score += 5.0
    elif (
        poi_name_normalized in candidate_name_normalized
        or candidate_name_normalized in poi_name_normalized
    ):
        score += 3.0
    else:
        score += 2.0 * name_similarity

    city_in_address = bool(city_name_lower and city_name_lower in candidate_address_lower)

    if city_in_address:
        score += 2.0

    distance_km = None
    candidate_location = candidate.get("location") or {}
    candidate_lat = candidate_location.get("latitude")
    candidate_lon = candidate_location.get("longitude")

    if (
        lat is not None
        and lon is not None
        and candidate_lat is not None
        and candidate_lon is not None
    ):
        distance_km = haversine_km(lat, lon, candidate_lat, candidate_lon)

        if distance_km <= 0.2:
            score += 3.0
        elif distance_km <= 1.0:
            score += 2.0
        elif distance_km <= 5.0:
            score += 1.0

    if candidate.get("rating") is not None:
        score += 0.2

    if candidate.get("userRatingCount") is not None:
        score += min(float(candidate["userRatingCount"]) / 1000.0, 1.0)

    return {
        "score": score,
        "distance_km": distance_km,
        "name_similarity": round(name_similarity, 4),
        "city_in_address": city_in_address,
    }


def derive_match_quality(
    score: float,
    distance_km: Optional[float],
    name_similarity: float,
    city_in_address: bool,
) -> str:
    if (
        score >= 5.0
        and (distance_km is None or distance_km <= 1.0)
        and name_similarity >= 0.5
        and city_in_address
    ):
        return "high"

    if (
        score >= 4.0
        and (distance_km is None or distance_km <= 5.0)
        and name_similarity >= 0.3
    ):
        return "medium"

    return "low"


def no_match_record(row: pd.Series) -> Dict[str, Any]:
    return {
        "poi_key": row["poi_key"],
        "google_match_status": "no_match",
        "google_match_score": None,
        "google_match_quality": "low",
        "google_review_needed": True,
        "google_match_distance_km": None,
        "google_name_similarity": None,
        "google_city_in_address": None,
        "google_place_id": None,
        "google_name": None,
        "google_formatted_address": None,
        "google_latitude": None,
        "google_longitude": None,
        "google_rating": None,
        "google_user_rating_count": None,
        "google_primary_type": None,
        "google_types": None,
        "google_business_status": None,
        "google_opening_hours_present": None,
    }


def choose_best_match(response: Dict[str, Any], row: pd.Series) -> Dict[str, Any]:
    places = response.get("places", []) or []

    if not places:
        return no_match_record(row)

    poi_name = str(row["poi_name_for_google"])
    city_name = str(row["city_name_for_google"])
    lat = safe_float(row.get("lat_for_google"))
    lon = safe_float(row.get("lon_for_google"))

    scored: List[tuple[float, Dict[str, Any], Dict[str, Any]]] = []

    for candidate in places:
        meta = score_candidate(
            candidate=candidate,
            poi_name=poi_name,
            city_name=city_name,
            lat=lat,
            lon=lon,
        )
        scored.append((meta["score"], candidate, meta))

    scored.sort(key=lambda x: x[0], reverse=True)

    best_score, best_candidate, best_meta = scored[0]

    match_status = "matched"

    if len(scored) > 1 and abs(best_score - scored[1][0]) < 0.5:
        match_status = "ambiguous"

    match_quality = derive_match_quality(
        score=best_score,
        distance_km=best_meta["distance_km"],
        name_similarity=best_meta["name_similarity"],
        city_in_address=best_meta["city_in_address"],
    )

    review_needed = match_status != "matched" or match_quality != "high"

    if match_status == "matched" and match_quality == "low":
        match_status = "flagged"
        review_needed = True

    display_name = (best_candidate.get("displayName") or {}).get("text")
    location = best_candidate.get("location") or {}

    return {
        "poi_key": row["poi_key"],
        "google_match_status": match_status,
        "google_match_score": round(float(best_score), 3),
        "google_match_quality": match_quality,
        "google_review_needed": review_needed,
        "google_match_distance_km": (
            None
            if best_meta["distance_km"] is None
            else round(float(best_meta["distance_km"]), 4)
        ),
        "google_name_similarity": best_meta["name_similarity"],
        "google_city_in_address": best_meta["city_in_address"],
        "google_place_id": best_candidate.get("id"),
        "google_name": display_name,
        "google_formatted_address": best_candidate.get("formattedAddress"),
        "google_latitude": location.get("latitude"),
        "google_longitude": location.get("longitude"),
        "google_rating": best_candidate.get("rating"),
        "google_user_rating_count": best_candidate.get("userRatingCount"),
        "google_primary_type": best_candidate.get("primaryType"),
        "google_types": "|".join(best_candidate.get("types", []) or []),
        "google_business_status": best_candidate.get("businessStatus"),
        "google_opening_hours_present": bool(best_candidate.get("regularOpeningHours")),
    }


# ---------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------


GOOGLE_FEATURE_COLUMNS = [
    "poi_key",
    "google_match_status",
    "google_match_score",
    "google_match_quality",
    "google_review_needed",
    "google_match_distance_km",
    "google_name_similarity",
    "google_city_in_address",
    "google_place_id",
    "google_name",
    "google_formatted_address",
    "google_latitude",
    "google_longitude",
    "google_rating",
    "google_user_rating_count",
    "google_primary_type",
    "google_types",
    "google_business_status",
    "google_opening_hours_present",
]


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Input DBSCAN dwell-labelled POI CSV.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV containing original input rows plus Google Places features.",
    )

    parser.add_argument(
        "--outdir",
        required=True,
        help="Directory for cache, error log, and standalone Google feature table.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for testing API calls.",
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=DEFAULT_SLEEP_BETWEEN_CALLS,
        help="Delay between API calls to reduce quota/rate-limit pressure.",
    )

    parser.add_argument(
        "--location-bias-radius-m",
        type=float,
        default=5000.0,
        help="Radius in metres for Google Places location bias.",
    )

    parser.add_argument(
        "--city",
        type=str,
        default=None,
        help="Optional city filter for testing/auditing.",
    )

    parser.add_argument(
        "--skip-api",
        action="store_true",
        help=(
            "Do not call Google Places. Only merge existing cache records into "
            "the input table. Useful for reproducible GitHub/demo runs."
        ),
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    outdir = Path(args.outdir)

    outdir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cache_jsonl = outdir / "google_places_cache.jsonl"
    errors_jsonl = outdir / "google_places_errors.jsonl"
    google_features_csv = outdir / "poi_google_features.csv"

    df_raw = pd.read_csv(input_path)
    df = standardise_input_columns(df_raw)

    if args.city is not None:
        city_filter = args.city.strip().lower()
        df = df[
            df["city_name_for_google"].fillna("").astype(str).str.lower() == city_filter
        ].copy()

    cached_records = load_cached_records(cache_jsonl)

    print("Input rows:", len(df))
    print("Cached Google records:", len(cached_records))

    if args.skip_api:
        api_key = None
        print("Running in --skip-api mode. No Google Places calls will be made.")
    else:
        api_key = os.environ.get("GOOGLE_MAPS_API_KEY") or os.environ.get("GOOGLE_API_KEY")

        if not api_key:
            print("ERROR: set GOOGLE_MAPS_API_KEY or GOOGLE_API_KEY, or use --skip-api.")
            return 1

    # Rows still needing API calls.
    todo = df[~df["poi_key"].isin(cached_records.keys())].copy()

    if args.limit is not None:
        todo = todo.head(args.limit)

    print("Rows to process now:", len(todo))

    consecutive_failures = 0

    if not args.skip_api:
        for i, (_, row) in enumerate(todo.iterrows(), start=1):
            try:
                payload = build_payload(
                    row=row,
                    radius_m=args.location_bias_radius_m,
                )

                response = call_text_search(
                    api_key=api_key,
                    payload=payload,
                )

                google_record = choose_best_match(
                    response=response,
                    row=row,
                )

                append_jsonl(cache_jsonl, google_record)
                write_csv_row(
                    google_features_csv,
                    google_record,
                    GOOGLE_FEATURE_COLUMNS,
                )

                cached_records[google_record["poi_key"]] = google_record

                consecutive_failures = 0

                if i % 20 == 0:
                    print(f"Processed {i}/{len(todo)}")

                time.sleep(args.sleep)

            except Exception as exc:
                consecutive_failures += 1

                error_record = {
                    "poi_key": row.get("poi_key"),
                    "poi_name": row.get("poi_name_for_google"),
                    "city_name": row.get("city_name_for_google"),
                    "error": str(exc),
                }

                append_jsonl(errors_jsonl, error_record)
                print(f"[ERROR] {row.get('poi_key')}: {exc}")

                if consecutive_failures >= 10:
                    print("Stopping after 10 consecutive failures to avoid wasting quota.")
                    break

    # Merge cached/new Google features back into full input table.
    google_df = pd.DataFrame(cached_records.values())

    if google_df.empty:
        print("WARNING: no Google feature records were available.")
        google_df = pd.DataFrame(columns=GOOGLE_FEATURE_COLUMNS)

    # Use the standardised dataframe so poi_key exists.
    enriched = df.merge(
        google_df.drop_duplicates(subset=["poi_key"]),
        on="poi_key",
        how="left",
    )

    # Drop internal helper columns from final output.
    helper_cols = [
        "poi_name_for_google",
        "city_name_for_google",
        "lat_for_google",
        "lon_for_google",
    ]

    enriched = enriched.drop(columns=[c for c in helper_cols if c in enriched.columns])

    enriched.to_csv(output_path, index=False)

    # Also rewrite a clean standalone Google feature table from cache.
    google_df = google_df.reindex(columns=GOOGLE_FEATURE_COLUMNS)
    google_df.to_csv(google_features_csv, index=False)

    print("\nDone.")
    print("Saved enriched table:", output_path)
    print("Saved Google feature table:", google_features_csv)
    print("Saved cache:", cache_jsonl)
    print("Saved errors:", errors_jsonl)

    print("\nGoogle match status counts:")
    if "google_match_status" in enriched.columns:
        print(enriched["google_match_status"].value_counts(dropna=False))

    print("\nGoogle match quality counts:")
    if "google_match_quality" in enriched.columns:
        print(enriched["google_match_quality"].value_counts(dropna=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
