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
SLEEP_BETWEEN_CALLS = 0.2


def make_poi_key(row: pd.Series) -> str:
    return f"{row['city_code']}::{row['poiID']}::{row['poiName']}"


def safe_float(x: Any) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def normalize_name(x: str) -> str:
    x = str(x)
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


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
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


def load_done_keys(cache_path: Path) -> Set[str]:
    done = set()
    if not cache_path.exists():
        return done

    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                key = obj.get("poi_key")
                status = obj.get("match_status")
                if key and status in {"matched", "no_match", "ambiguous", "flagged"}:
                    done.add(key)
            except Exception:
                continue
    return done


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_payload(row: pd.Series) -> Dict[str, Any]:
    poi_name = normalize_name(row["poiName"])
    city_name = str(row["city_name"]).strip()

    payload: Dict[str, Any] = {
        "textQuery": f"{poi_name}, {city_name}",
        "pageSize": 5,
        "languageCode": "en",
    }

    lat = safe_float(row.get("lat"))
    lon = safe_float(row.get("lon"))
    if lat is not None and lon is not None:
        payload["locationBias"] = {
            "circle": {
                "center": {"latitude": lat, "longitude": lon},
                "radius": 5000.0,
            }
        }

    return payload


def call_text_search(api_key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": FIELD_MASK,
    }

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                SEARCH_URL,
                headers=headers,
                json=payload,
                timeout=DEFAULT_TIMEOUT,
            )
            if resp.status_code == 200:
                return resp.json()

            if resp.status_code in {429, 500, 502, 503, 504}:
                last_err = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
                time.sleep(min(2 ** attempt, 30))
                continue

            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:1000]}")
        except requests.RequestException as e:
            last_err = e
            time.sleep(min(2 ** attempt, 30))

    raise RuntimeError(f"Text Search failed after retries: {last_err}")


def score_candidate(
    candidate: Dict[str, Any],
    poi_name: str,
    city_name: str,
    lat: Optional[float],
    lon: Optional[float],
) -> Dict[str, Any]:
    score = 0.0

    cand_name = ((candidate.get("displayName") or {}).get("text")) or ""
    cand_addr = candidate.get("formattedAddress", "") or ""

    poi_name_n = normalize_name(poi_name)
    cand_name_n = normalize_name(cand_name)
    city_name_l = city_name.lower().strip()
    cand_addr_l = cand_addr.lower().strip()

    name_sim = token_jaccard(poi_name_n, cand_name_n)

    if cand_name_n == poi_name_n:
        score += 5.0
    elif poi_name_n in cand_name_n or cand_name_n in poi_name_n:
        score += 3.0
    else:
        score += 2.0 * name_sim

    city_in_addr = city_name_l in cand_addr_l if city_name_l else False
    if city_in_addr:
        score += 2.0

    distance_km = None
    cand_loc = candidate.get("location") or {}
    c_lat = cand_loc.get("latitude")
    c_lon = cand_loc.get("longitude")

    if lat is not None and lon is not None and c_lat is not None and c_lon is not None:
        distance_km = haversine_km(lat, lon, c_lat, c_lon)
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
        "name_similarity": round(name_sim, 4),
        "city_in_addr": city_in_addr,
    }


def derive_match_quality(
    score: float,
    distance_km: Optional[float],
    name_similarity: float,
    city_in_addr: bool,
) -> str:
    if (
        score >= 5.0
        and (distance_km is None or distance_km <= 1.0)
        and name_similarity >= 0.5
        and city_in_addr
    ):
        return "high"

    if (
        score >= 4.0
        and (distance_km is None or distance_km <= 5.0)
        and name_similarity >= 0.3
    ):
        return "medium"

    return "low"


def choose_best_match(response: Dict[str, Any], row: pd.Series) -> Dict[str, Any]:
    places = response.get("places", []) or []
    if not places:
        return {
            "match_status": "no_match",
            "match_score": None,
            "match_quality": "low",
            "review_needed": True,
            "distance_km": None,
            "name_similarity": None,
            "city_in_addr": None,
            "matched_place_id": None,
            "matched_place_name": None,
            "matched_formatted_address": None,
            "matched_lat": None,
            "matched_lon": None,
            "rating": None,
            "user_ratings_total": None,
            "primary_type": None,
            "all_types": None,
            "business_status": None,
            "opening_hours_present": None,
        }

    poi_name = str(row["poiName"])
    city_name = str(row["city_name"])
    lat = safe_float(row.get("lat"))
    lon = safe_float(row.get("lon"))

    scored: List[tuple[float, Dict[str, Any], Dict[str, Any]]] = []
    for p in places:
        meta = score_candidate(p, poi_name, city_name, lat, lon)
        scored.append((meta["score"], p, meta))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best, meta = scored[0]

    status = "matched"
    if len(scored) > 1 and abs(best_score - scored[1][0]) < 0.5:
        status = "ambiguous"

    quality = derive_match_quality(
        score=best_score,
        distance_km=meta["distance_km"],
        name_similarity=meta["name_similarity"],
        city_in_addr=meta["city_in_addr"],
    )

    review_needed = status != "matched" or quality != "high"

    if status == "matched" and quality == "low":
        status = "flagged"

    disp = (best.get("displayName") or {}).get("text")
    loc = best.get("location") or {}

    return {
        "match_status": status,
        "match_score": round(best_score, 3),
        "match_quality": quality,
        "review_needed": review_needed,
        "distance_km": None if meta["distance_km"] is None else round(meta["distance_km"], 4),
        "name_similarity": meta["name_similarity"],
        "city_in_addr": meta["city_in_addr"],
        "matched_place_id": best.get("id"),
        "matched_place_name": disp,
        "matched_formatted_address": best.get("formattedAddress"),
        "matched_lat": loc.get("latitude"),
        "matched_lon": loc.get("longitude"),
        "rating": best.get("rating"),
        "user_ratings_total": best.get("userRatingCount"),
        "primary_type": best.get("primaryType"),
        "all_types": "|".join(best.get("types", []) or []),
        "business_status": best.get("businessStatus"),
        "opening_hours_present": bool(best.get("regularOpeningHours")),
    }


def write_csv_row(csv_path: Path, row_dict: Dict[str, Any], header: List[str]) -> None:
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input POI modeling CSV")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=SLEEP_BETWEEN_CALLS)
    parser.add_argument("--city_code", type=str, default=None, help="Optional: only enrich one city for audit")
    args = parser.parse_args()

    api_key = os.environ.get("GOOGLE_MAPS_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: set GOOGLE_MAPS_API_KEY or GOOGLE_API_KEY")
        return 1

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cache_jsonl = outdir / "google_places_cache.jsonl"
    errors_jsonl = outdir / "google_places_errors.jsonl"
    features_csv = outdir / "poi_google_features.csv"

    df = pd.read_csv(input_path)
    required_cols = {"city_code", "city_name", "poiID", "poiName", "lat", "lon"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"ERROR: missing required input columns: {missing}")
        return 1

    if args.city_code:
        df = df[df["city_code"] == args.city_code].copy()

    df["poi_key"] = df.apply(make_poi_key, axis=1)

    done_keys = load_done_keys(cache_jsonl)
    todo = df[~df["poi_key"].isin(done_keys)].copy()

    if args.limit is not None:
        todo = todo.head(args.limit)

    print(f"Input rows: {len(df)}")
    print(f"Already cached: {len(done_keys)}")
    print(f"To process now: {len(todo)}")

    header = [
        "poi_key",
        "city_code",
        "city_name",
        "poiID",
        "poiName",
        "lat",
        "lon",
        "match_status",
        "match_score",
        "match_quality",
        "review_needed",
        "distance_km",
        "name_similarity",
        "city_in_addr",
        "matched_place_id",
        "matched_place_name",
        "matched_formatted_address",
        "matched_lat",
        "matched_lon",
        "rating",
        "user_ratings_total",
        "primary_type",
        "all_types",
        "business_status",
        "opening_hours_present",
    ]

    consecutive_failures = 0
    for i, (_, row) in enumerate(todo.iterrows(), start=1):
        base_record = {
            "poi_key": row["poi_key"],
            "city_code": row["city_code"],
            "city_name": row["city_name"],
            "poiID": row["poiID"],
            "poiName": row["poiName"],
            "lat": row["lat"],
            "lon": row["lon"],
        }

        try:
            payload = build_payload(row)
            response = call_text_search(api_key, payload)
            match_info = choose_best_match(response, row)

            record = {**base_record, **match_info}
            append_jsonl(cache_jsonl, record)
            write_csv_row(features_csv, record, header)

            consecutive_failures = 0
            if i % 20 == 0:
                print(f"Processed {i}/{len(todo)}")
            time.sleep(args.sleep)

        except Exception as e:
            consecutive_failures += 1
            err = {**base_record, "error": str(e)}
            append_jsonl(errors_jsonl, err)
            print(f"[ERROR] {row['poi_key']}: {e}")

            if consecutive_failures >= 10:
                print("Stopping after 10 consecutive failures to avoid wasting quota.")
                break

    print("Done.")
    print(f"Cache:    {cache_jsonl}")
    print(f"Errors:   {errors_jsonl}")
    print(f"Features: {features_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())