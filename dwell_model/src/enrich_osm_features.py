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


NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

DEFAULT_TIMEOUT = 45
MAX_RETRIES = 5
SLEEP_BETWEEN_CALLS = 1.2  # be polite to public OSM services


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
    x = x.replace("#", " ")
    x = x.replace("(", " ").replace(")", " ")
    x = x.replace(",", " ")
    x = " ".join(x.strip().lower().split())
    return x


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


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


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
                if key:
                    done.add(key)
            except Exception:
                continue
    return done


def request_with_retries(
    method: str,
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    data: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> requests.Response:
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                data=data,
                timeout=timeout,
            )
            if resp.status_code == 200:
                return resp

            if resp.status_code in {429, 500, 502, 503, 504}:
                last_err = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
                time.sleep(min(2 ** attempt, 30))
                continue

            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:1000]}")
        except requests.RequestException as e:
            last_err = e
            time.sleep(min(2 ** attempt, 30))

    raise RuntimeError(f"Request failed after retries: {last_err}")


def token_jaccard(a: str, b: str) -> float:
    a_set = set(normalize_name(a).split())
    b_set = set(normalize_name(b).split())
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)


def nominatim_search(
    poi_name: str,
    city_name: str,
    *,
    user_agent: str,
    email: Optional[str] = None,
) -> List[Dict[str, Any]]:
    headers = {"User-Agent": user_agent}
    params = {
        "q": f"{normalize_name(poi_name)}, {city_name}",
        "format": "jsonv2",
        "limit": 5,
        "addressdetails": 1,
    }
    if email:
        params["email"] = email

    resp = request_with_retries(
        "GET",
        NOMINATIM_URL,
        headers=headers,
        params=params,
        timeout=DEFAULT_TIMEOUT,
    )
    return resp.json()


def choose_best_nominatim(
    candidates: List[Dict[str, Any]],
    row: pd.Series,
) -> Dict[str, Any]:
    if not candidates:
        return {
            "osm_match_status": "no_match",
            "osm_match_quality": "low",
            "osm_review_needed": True,
            "osm_match_score": None,
            "osm_distance_km": None,
            "osm_name_similarity": None,
            "osm_display_name": None,
            "osm_type": None,
            "osm_class": None,
            "osm_lat": None,
            "osm_lon": None,
            "osm_place_id": None,
            "osm_osm_type": None,
            "osm_osm_id": None,
        }

    poi_name = str(row["poiName"])
    city_name = str(row["city_name"])
    lat = safe_float(row.get("lat"))
    lon = safe_float(row.get("lon"))

    scored = []
    for c in candidates:
        disp = c.get("display_name", "")
        name_guess = c.get("name") or disp.split(",")[0]
        sim = token_jaccard(poi_name, name_guess)

        score = 0.0
        score += 3.0 * sim

        city_hit = city_name.lower() in disp.lower()
        if city_hit:
            score += 2.0

        c_lat = safe_float(c.get("lat"))
        c_lon = safe_float(c.get("lon"))
        dist_km = None
        if lat is not None and lon is not None and c_lat is not None and c_lon is not None:
            dist_km = haversine_km(lat, lon, c_lat, c_lon)
            if dist_km <= 0.2:
                score += 3.0
            elif dist_km <= 1.0:
                score += 2.0
            elif dist_km <= 5.0:
                score += 1.0

        scored.append((score, sim, city_hit, dist_km, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, sim, city_hit, dist_km, best = scored[0]

    quality = "low"
    if best_score >= 5.0 and sim >= 0.5 and (dist_km is None or dist_km <= 1.0):
        quality = "high"
    elif best_score >= 3.5 and sim >= 0.25 and (dist_km is None or dist_km <= 5.0):
        quality = "medium"

    status = "matched" if quality != "low" else "flagged"
    review_needed = quality != "high"

    return {
        "osm_match_status": status,
        "osm_match_quality": quality,
        "osm_review_needed": review_needed,
        "osm_match_score": round(best_score, 3),
        "osm_distance_km": None if dist_km is None else round(dist_km, 4),
        "osm_name_similarity": round(sim, 4),
        "osm_display_name": best.get("display_name"),
        "osm_type": best.get("type"),
        "osm_class": best.get("class"),
        "osm_lat": safe_float(best.get("lat")),
        "osm_lon": safe_float(best.get("lon")),
        "osm_place_id": best.get("place_id"),
        "osm_osm_type": best.get("osm_type"),
        "osm_osm_id": best.get("osm_id"),
    }


def build_overpass_query(lat: float, lon: float, radius_m: int) -> str:
    return f"""
    [out:json][timeout:25];
    (
      node(around:{radius_m},{lat},{lon})["public_transport"];
      way(around:{radius_m},{lat},{lon})["public_transport"];
      relation(around:{radius_m},{lat},{lon})["public_transport"];

      node(around:{radius_m},{lat},{lon})["railway"~"station|halt|tram_stop"];
      way(around:{radius_m},{lat},{lon})["railway"~"station|halt|tram_stop"];
      relation(around:{radius_m},{lat},{lon})["railway"~"station|halt|tram_stop"];

      node(around:{radius_m},{lat},{lon})["highway"="bus_stop"];
      way(around:{radius_m},{lat},{lon})["highway"="bus_stop"];
      relation(around:{radius_m},{lat},{lon})["highway"="bus_stop"];

      node(around:{radius_m},{lat},{lon})["amenity"="parking"];
      way(around:{radius_m},{lat},{lon})["amenity"="parking"];
      relation(around:{radius_m},{lat},{lon})["amenity"="parking"];

      node(around:{radius_m},{lat},{lon})["parking"];
      way(around:{radius_m},{lat},{lon})["parking"];
      relation(around:{radius_m},{lat},{lon})["parking"];
    );
    out center tags;
    """


def overpass_fetch(lat: float, lon: float, radius_m: int, *, user_agent: str) -> Dict[str, Any]:
    headers = {"User-Agent": user_agent, "Content-Type": "application/x-www-form-urlencoded"}
    query = build_overpass_query(lat, lon, radius_m)

    resp = request_with_retries(
        "POST",
        OVERPASS_URL,
        headers=headers,
        data=query,
        timeout=DEFAULT_TIMEOUT,
    )
    return resp.json()


def extract_center(element: Dict[str, Any]) -> Optional[tuple[float, float]]:
    if "lat" in element and "lon" in element:
        return float(element["lat"]), float(element["lon"])
    center = element.get("center")
    if center and "lat" in center and "lon" in center:
        return float(center["lat"]), float(center["lon"])
    return None


def classify_pt(tags: Dict[str, Any]) -> bool:
    if not tags:
        return False
    if "public_transport" in tags:
        return True
    if tags.get("railway") in {"station", "halt", "tram_stop"}:
        return True
    if tags.get("highway") == "bus_stop":
        return True
    return False


def classify_parking(tags: Dict[str, Any]) -> bool:
    if not tags:
        return False
    if tags.get("amenity") == "parking":
        return True
    if "parking" in tags:
        return True
    return False


def summarize_osm_context(lat: float, lon: float, elements: List[Dict[str, Any]]) -> Dict[str, Any]:
    pt_distances = []
    parking_distances = []

    for el in elements:
        tags = el.get("tags") or {}
        center = extract_center(el)
        if center is None:
            continue

        el_lat, el_lon = center
        d = haversine_km(lat, lon, el_lat, el_lon)

        if classify_pt(tags):
            pt_distances.append(d)

        if classify_parking(tags):
            parking_distances.append(d)

    pt_count_400m = sum(d <= 0.4 for d in pt_distances)
    pt_count_600m = sum(d <= 0.6 for d in pt_distances)
    parking_count_400m = sum(d <= 0.4 for d in parking_distances)
    parking_count_600m = sum(d <= 0.6 for d in parking_distances)

    nearest_pt_km = min(pt_distances) if pt_distances else None
    nearest_parking_km = min(parking_distances) if parking_distances else None

    def band(d: Optional[float]) -> Optional[str]:
        if d is None:
            return None
        if d <= 0.4:
            return "high"
        if d <= 0.6:
            return "medium"
        return "low"

    return {
        "osm_pt_count_400m": pt_count_400m,
        "osm_pt_count_600m": pt_count_600m,
        "osm_parking_count_400m": parking_count_400m,
        "osm_parking_count_600m": parking_count_600m,
        "osm_nearest_pt_km": None if nearest_pt_km is None else round(nearest_pt_km, 4),
        "osm_nearest_parking_km": None if nearest_parking_km is None else round(nearest_parking_km, 4),
        "osm_pt_access_band": band(nearest_pt_km),
        "osm_parking_access_band": band(nearest_parking_km),
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
    parser.add_argument("--outdir", required=True, help="Output directory for OSM enrichment")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--city_code", type=str, default=None)
    parser.add_argument("--sleep", type=float, default=SLEEP_BETWEEN_CALLS)
    parser.add_argument("--user_agent", type=str, default="travelplanner-dwell-thesis/1.0")
    parser.add_argument("--email", type=str, default=None)
    parser.add_argument("--radius_m", type=int, default=600)
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cache_jsonl = outdir / "osm_cache.jsonl"
    errors_jsonl = outdir / "osm_errors.jsonl"
    features_csv = outdir / "poi_osm_features.csv"

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
        "osm_match_status",
        "osm_match_quality",
        "osm_review_needed",
        "osm_match_score",
        "osm_distance_km",
        "osm_name_similarity",
        "osm_display_name",
        "osm_type",
        "osm_class",
        "osm_lat",
        "osm_lon",
        "osm_place_id",
        "osm_osm_type",
        "osm_osm_id",
        "osm_pt_count_400m",
        "osm_pt_count_600m",
        "osm_parking_count_400m",
        "osm_parking_count_600m",
        "osm_nearest_pt_km",
        "osm_nearest_parking_km",
        "osm_pt_access_band",
        "osm_parking_access_band",
    ]

    consecutive_failures = 0

    for i, (_, row) in enumerate(todo.iterrows(), start=1):
        base = {
            "poi_key": row["poi_key"],
            "city_code": row["city_code"],
            "city_name": row["city_name"],
            "poiID": row["poiID"],
            "poiName": row["poiName"],
            "lat": row["lat"],
            "lon": row["lon"],
        }

        try:
            # Step 1: Nominatim match
            candidates = nominatim_search(
                poi_name=str(row["poiName"]),
                city_name=str(row["city_name"]),
                user_agent=args.user_agent,
                email=args.email,
            )
            match_info = choose_best_nominatim(candidates, row)

            # Step 2: Nearby PT / parking using original coordinates
            lat = safe_float(row["lat"])
            lon = safe_float(row["lon"])
            osm_context = {
                "osm_pt_count_400m": None,
                "osm_pt_count_600m": None,
                "osm_parking_count_400m": None,
                "osm_parking_count_600m": None,
                "osm_nearest_pt_km": None,
                "osm_nearest_parking_km": None,
                "osm_pt_access_band": None,
                "osm_parking_access_band": None,
            }

            if lat is not None and lon is not None:
                overpass = overpass_fetch(lat, lon, args.radius_m, user_agent=args.user_agent)
                elements = overpass.get("elements", []) or []
                osm_context = summarize_osm_context(lat, lon, elements)

            record = {**base, **match_info, **osm_context}
            append_jsonl(cache_jsonl, record)
            write_csv_row(features_csv, record, header)

            consecutive_failures = 0

            if i % 20 == 0:
                print(f"Processed {i}/{len(todo)}")

            time.sleep(args.sleep)

        except Exception as e:
            consecutive_failures += 1
            append_jsonl(errors_jsonl, {**base, "error": str(e)})
            print(f"[ERROR] {row['poi_key']}: {e}")

            if consecutive_failures >= 10:
                print("Stopping after 10 consecutive failures to avoid bad loops.")
                break

    print("Done.")
    print(f"Cache:    {cache_jsonl}")
    print(f"Errors:   {errors_jsonl}")
    print(f"Features: {features_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())