import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
import requests


GOOGLE_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"

GOOGLE_FIELD_MASK = ",".join(
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
    ]
)

OVERPASS_ENDPOINTS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://z.overpass-api.de/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]

MODEL_FEATURES = [
    "category_group",
    "google_rating_filled",
    "log_google_user_rating_count_capped",
    "parking_availability",
    "public_transport_availability",
    "osm_safety_proxy",
    "osm_security_proxy",
    "dataset_city",
]


def normalize_text(x):
    if pd.isna(x):
        return ""
    x = str(x).lower().strip()
    x = x.replace("_", " ")
    x = re.sub(r"[^a-z0-9]+", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def normalize_city(x):
    return normalize_text(x).replace(" ", "_")


def safe_float(x):
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1 = math.radians(float(lat1))
    p2 = math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlambda = math.radians(float(lon2) - float(lon1))

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    )

    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def token_jaccard(a, b):
    a_set = set(normalize_text(a).split())
    b_set = set(normalize_text(b).split())

    if not a_set or not b_set:
        return 0.0

    return len(a_set & b_set) / len(a_set | b_set)


def attraction_key(row):
    return f"{row['City']}::{row['Name']}::{row['Latitude']}::{row['Longitude']}"


def load_json_cache(path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_json_cache(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_google_payload(row):
    name = str(row["Name"])
    city = str(row["City"])
    address = str(row.get("Address", ""))

    query = f"{name}, {city}"
    if address and address.lower() != "nan":
        query = f"{name}, {address}"

    payload = {
        "textQuery": query,
        "pageSize": 5,
        "languageCode": "en",
    }

    lat = safe_float(row["Latitude"])
    lon = safe_float(row["Longitude"])

    if lat is not None and lon is not None:
        payload["locationBias"] = {
            "circle": {
                "center": {"latitude": lat, "longitude": lon},
                "radius": 3000.0,
            }
        }

    return payload


def call_google_places(api_key, payload, max_retries=5):
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": GOOGLE_FIELD_MASK,
    }

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                GOOGLE_SEARCH_URL,
                headers=headers,
                json=payload,
                timeout=30,
            )

            if resp.status_code == 200:
                return resp.json()

            if resp.status_code in {429, 500, 502, 503, 504}:
                last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
                time.sleep(min(2 ** attempt, 30))
                continue

            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")

        except Exception as e:
            last_error = str(e)
            time.sleep(min(2 ** attempt, 30))

    raise RuntimeError(f"Google Places failed: {last_error}")


def score_google_candidate(candidate, row):
    name = str(row["Name"])
    city = str(row["City"])
    lat = safe_float(row["Latitude"])
    lon = safe_float(row["Longitude"])

    cand_name = ((candidate.get("displayName") or {}).get("text")) or ""
    cand_addr = candidate.get("formattedAddress", "") or ""
    cand_loc = candidate.get("location") or {}

    name_sim = token_jaccard(name, cand_name)

    score = 0.0

    name_n = normalize_text(name)
    cand_name_n = normalize_text(cand_name)

    if name_n == cand_name_n:
        score += 5.0
    elif name_n in cand_name_n or cand_name_n in name_n:
        score += 3.0
    else:
        score += 2.0 * name_sim

    city_in_addr = normalize_text(city) in normalize_text(cand_addr)
    if city_in_addr:
        score += 1.5

    distance_km = None
    if (
        lat is not None
        and lon is not None
        and cand_loc.get("latitude") is not None
        and cand_loc.get("longitude") is not None
    ):
        distance_km = haversine_km(
            lat,
            lon,
            cand_loc["latitude"],
            cand_loc["longitude"],
        )

        if distance_km <= 0.2:
            score += 3.0
        elif distance_km <= 1.0:
            score += 2.0
        elif distance_km <= 3.0:
            score += 1.0

    return score, name_sim, distance_km, city_in_addr


def choose_google_match(response, row):
    places = response.get("places", []) or []

    if not places:
        return {
            "google_match_status": "no_match",
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
            "google_match_score": None,
            "google_name_similarity": None,
            "google_distance_km": None,
        }

    scored = []

    for p in places:
        score, name_sim, distance_km, city_in_addr = score_google_candidate(p, row)
        scored.append((score, name_sim, distance_km, city_in_addr, p))

    scored.sort(key=lambda x: x[0], reverse=True)

    best_score, best_name_sim, best_distance_km, best_city_in_addr, best = scored[0]

    loc = best.get("location") or {}
    disp = (best.get("displayName") or {}).get("text")

    if best_distance_km is not None and best_distance_km > 3.0 and best_name_sim < 0.5:
        status = "low_confidence"
    elif best_score >= 5.0:
        status = "matched"
    else:
        status = "low_confidence"

    return {
        "google_match_status": status,
        "google_place_id": best.get("id"),
        "google_name": disp,
        "google_formatted_address": best.get("formattedAddress"),
        "google_latitude": loc.get("latitude"),
        "google_longitude": loc.get("longitude"),
        "google_rating": best.get("rating"),
        "google_user_rating_count": best.get("userRatingCount"),
        "google_primary_type": best.get("primaryType"),
        "google_types": "|".join(best.get("types", []) or []),
        "google_business_status": best.get("businessStatus"),
        "google_match_score": round(best_score, 3),
        "google_name_similarity": round(best_name_sim, 4),
        "google_distance_km": None if best_distance_km is None else round(best_distance_km, 4),
    }


def infer_category_group(row):
    text = " ".join(
        [
            str(row.get("google_primary_type", "")),
            str(row.get("google_types", "")),
            str(row.get("Name", "")),
        ]
    ).lower()

    shopping_terms = [
        "store",
        "shopping",
        "mall",
        "market",
        "supermarket",
        "book_store",
        "clothing",
        "department_store",
        "electronics_store",
        "jewelry",
        "shoe",
    ]

    food_terms = [
        "restaurant",
        "cafe",
        "bar",
        "bakery",
        "meal",
        "food",
        "coffee",
        "fast_food",
        "pub",
    ]

    tourism_terms = [
        "tourist",
        "museum",
        "park",
        "monument",
        "landmark",
        "church",
        "mosque",
        "temple",
        "zoo",
        "aquarium",
        "art_gallery",
        "stadium",
        "amusement",
        "point_of_interest",
        "national",
        "historic",
    ]

    if any(t in text for t in shopping_terms):
        return "shopping"

    if any(t in text for t in food_terms):
        return "food_leisure"

    if any(t in text for t in tourism_terms):
        return "tourism_attraction"

    return "tourism_attraction"


def build_overpass_query(lat, lon):
    return f"""
    [out:json][timeout:60];
    (
      node["amenity"="parking"](around:600,{lat},{lon});
      way["amenity"="parking"](around:600,{lat},{lon});
      relation["amenity"="parking"](around:600,{lat},{lon});

      node["amenity"="parking_space"](around:600,{lat},{lon});
      way["amenity"="parking_space"](around:600,{lat},{lon});

      node["amenity"="parking_entrance"](around:600,{lat},{lon});

      node["highway"="bus_stop"](around:600,{lat},{lon});
      node["amenity"="bus_station"](around:600,{lat},{lon});
      way["amenity"="bus_station"](around:600,{lat},{lon});

      node["railway"="station"](around:600,{lat},{lon});
      way["railway"="station"](around:600,{lat},{lon});
      node["railway"="halt"](around:600,{lat},{lon});
      node["railway"="tram_stop"](around:600,{lat},{lon});
      node["railway"="subway_entrance"](around:600,{lat},{lon});

      node["public_transport"](around:600,{lat},{lon});
      way["public_transport"](around:600,{lat},{lon});

      node["amenity"="police"](around:600,{lat},{lon});
      way["amenity"="police"](around:600,{lat},{lon});

      node["amenity"="hospital"](around:600,{lat},{lon});
      way["amenity"="hospital"](around:600,{lat},{lon});

      node["amenity"="clinic"](around:600,{lat},{lon});
      way["amenity"="clinic"](around:600,{lat},{lon});

      node["emergency"](around:600,{lat},{lon});
      way["emergency"](around:600,{lat},{lon});

      node["highway"="street_lamp"](around:600,{lat},{lon});

      node["man_made"="surveillance"](around:600,{lat},{lon});
      way["man_made"="surveillance"](around:600,{lat},{lon});

      node["surveillance"](around:600,{lat},{lon});
      way["surveillance"](around:600,{lat},{lon});

      node["office"="security"](around:600,{lat},{lon});
      way["office"="security"](around:600,{lat},{lon});
    );
    out center tags;
    """


def call_overpass(query, max_retries=3):
    headers = {
        "User-Agent": "travelplanner-dwell-inference/1.0",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    }

    last_error = None

    for endpoint in OVERPASS_ENDPOINTS:
        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.post(
                    endpoint,
                    data={"data": query},
                    headers=headers,
                    timeout=90,
                )

                if resp.status_code == 200:
                    return resp.json()

                last_error = f"{endpoint} HTTP {resp.status_code}: {resp.text[:300]}"
                time.sleep(min(2 ** attempt, 20))

            except Exception as e:
                last_error = str(e)
                time.sleep(min(2 ** attempt, 20))

    raise RuntimeError(f"Overpass failed: {last_error}")


def element_lat_lon(element):
    if "lat" in element and "lon" in element:
        return element["lat"], element["lon"]

    center = element.get("center", {})
    return center.get("lat"), center.get("lon")


def classify_osm(tags):
    classes = []

    amenity = tags.get("amenity")
    highway = tags.get("highway")
    railway = tags.get("railway")
    public_transport = tags.get("public_transport")
    emergency = tags.get("emergency")
    man_made = tags.get("man_made")
    surveillance = tags.get("surveillance")
    office = tags.get("office")

    if amenity in {"parking", "parking_space", "parking_entrance"}:
        classes.append("parking")

    if (
        highway == "bus_stop"
        or amenity == "bus_station"
        or railway in {"station", "halt", "tram_stop", "subway_entrance"}
        or public_transport is not None
    ):
        classes.append("public_transport")

    if (
        amenity in {"police", "hospital", "clinic"}
        or emergency is not None
        or highway == "street_lamp"
    ):
        classes.append("osm_safety_proxy")

    if (
        amenity == "police"
        or man_made == "surveillance"
        or surveillance is not None
        or office == "security"
    ):
        classes.append("osm_security_proxy")

    return classes


def summarize_osm(data, lat, lon):
    counts = {
        "parking_count_400m": 0,
        "parking_count_600m": 0,
        "public_transport_count_400m": 0,
        "public_transport_count_600m": 0,
        "osm_safety_proxy_count_400m": 0,
        "osm_safety_proxy_count_600m": 0,
        "osm_security_proxy_count_400m": 0,
        "osm_security_proxy_count_600m": 0,
    }

    seen = set()

    for element in data.get("elements", []) or []:
        tags = element.get("tags", {}) or {}
        e_lat, e_lon = element_lat_lon(element)

        if e_lat is None or e_lon is None:
            continue

        dist_m = haversine_km(lat, lon, e_lat, e_lon) * 1000.0

        for cls in classify_osm(tags):
            key = (cls, element.get("type"), element.get("id"))

            if key in seen:
                continue

            seen.add(key)

            if dist_m <= 600:
                counts[f"{cls}_count_600m"] += 1

            if dist_m <= 400:
                counts[f"{cls}_count_400m"] += 1

    def level(prefix):
        if counts[f"{prefix}_count_400m"] > 0:
            return "high"
        if counts[f"{prefix}_count_600m"] > 0:
            return "medium"
        return "low"

    counts["parking_availability"] = level("parking")
    counts["public_transport_availability"] = level("public_transport")
    counts["osm_safety_proxy"] = level("osm_safety_proxy")
    counts["osm_security_proxy"] = level("osm_security_proxy")

    return counts


def build_model_features(row):
    rating = safe_float(row.get("google_rating"))
    reviews = safe_float(row.get("google_user_rating_count"))

    return {
        "category_group": row.get("category_group", "tourism_attraction"),
        "google_rating_filled": rating,
        "log_google_user_rating_count_capped": None if reviews is None else math.log1p(reviews),
        "parking_availability": row.get("parking_availability", "unknown"),
        "public_transport_availability": row.get("public_transport_availability", "unknown"),
        "osm_safety_proxy": row.get("osm_safety_proxy", "unknown"),
        "osm_security_proxy": row.get("osm_security_proxy", "unknown"),
        "dataset_city": normalize_city(row.get("City", "")),
    }


def clip_duration(x, default=90.0, min_minutes=15.0, max_minutes=240.0):
    try:
        x = float(x)
        if math.isnan(x):
            x = default
    except Exception:
        x = default

    return round(max(min_minutes, min(max_minutes, x)), 1)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True, help="TravelPlanner attractions.csv")
    parser.add_argument("--output", required=True, help="Output enriched CSV")
    parser.add_argument("--model", required=True, help="robust_dwell_model.joblib")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep-google", type=float, default=0.2)
    parser.add_argument("--sleep-osm", type=float, default=1.0)
    parser.add_argument("--skip-osm", action="store_true")

    args = parser.parse_args()

    api_key = os.environ.get("GOOGLE_MAPS_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    if not api_key:
        raise RuntimeError("Set GOOGLE_MAPS_API_KEY or GOOGLE_API_KEY first.")

    input_path = Path(args.input)
    output_path = Path(args.output)
    cache_dir = Path(args.cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)

    google_cache_path = cache_dir / "google_cache.json"
    osm_cache_path = cache_dir / "osm_cache.json"

    google_cache = load_json_cache(google_cache_path)
    osm_cache = load_json_cache(osm_cache_path)

    model = joblib.load(args.model)

    df = pd.read_csv(input_path)

    required = ["Name", "Latitude", "Longitude", "Address", "Phone", "Website", "City"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required TravelPlanner columns: {missing}")

    if args.limit is not None:
        df = df.head(args.limit).copy()

    enriched_rows = []

    print("Rows to process:", len(df))

    for i, row in df.iterrows():
        key = attraction_key(row)
        print(f"[{i + 1}/{len(df)}] {row['Name']} - {row['City']}")

        base = row.to_dict()

        if key in google_cache:
            google_info = google_cache[key]
        else:
            try:
                payload = build_google_payload(row)
                response = call_google_places(api_key, payload)
                google_info = choose_google_match(response, row)
            except Exception as e:
                google_info = {
                    "google_match_status": "google_error",
                    "google_error": str(e),
                }

            google_cache[key] = google_info
            save_json_cache(google_cache_path, google_cache)
            time.sleep(args.sleep_google)

        base.update(google_info)

        lat = safe_float(row["Latitude"])
        lon = safe_float(row["Longitude"])

        if lat is None or lon is None:
            osm_info = {
                "parking_availability": "unknown",
                "public_transport_availability": "unknown",
                "osm_safety_proxy": "unknown",
                "osm_security_proxy": "unknown",
            }
        elif args.skip_osm:
            osm_info = {
                "parking_count_400m": 0,
                "parking_count_600m": 0,
                "public_transport_count_400m": 0,
                "public_transport_count_600m": 0,
                "osm_safety_proxy_count_400m": 0,
                "osm_safety_proxy_count_600m": 0,
                "osm_security_proxy_count_400m": 0,
                "osm_security_proxy_count_600m": 0,
                "parking_availability": "unknown",
                "public_transport_availability": "unknown",
                "osm_safety_proxy": "unknown",
                "osm_security_proxy": "unknown",
            }
        elif key in osm_cache:
            osm_info = osm_cache[key]
        else:
            try:
                query = build_overpass_query(lat, lon)
                osm_response = call_overpass(query)
                osm_info = summarize_osm(osm_response, lat, lon)
            except Exception as e:
                osm_info = {
                    "parking_availability": "unknown",
                    "public_transport_availability": "unknown",
                    "osm_safety_proxy": "unknown",
                    "osm_security_proxy": "unknown",
                    "osm_error": str(e),
                }

            osm_cache[key] = osm_info
            save_json_cache(osm_cache_path, osm_cache)
            time.sleep(args.sleep_osm)

        base.update(osm_info)

        base["category_group"] = infer_category_group(base)
        base["dataset_city"] = normalize_city(base["City"])

        model_features = build_model_features(base)

        X = pd.DataFrame([model_features])
        pred = model.predict(X)[0]

        base["predicted_dwell_minutes"] = clip_duration(pred)
        base["dwell_prediction_source"] = "robust_regression_model"

        enriched_rows.append(base)

    out = pd.DataFrame(enriched_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    print("Saved:", output_path)
    print("Rows:", len(out))


if __name__ == "__main__":
    sys.exit(main())

