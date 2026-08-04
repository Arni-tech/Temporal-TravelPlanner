import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.neighbors import BallTree


OVERPASS_ENDPOINTS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://z.overpass-api.de/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]


def build_tiles(pois, tile_size_deg=0.08, buffer_deg=0.01):
    min_lat = pois["Latitude"].min() - buffer_deg
    max_lat = pois["Latitude"].max() + buffer_deg
    min_lon = pois["Longitude"].min() - buffer_deg
    max_lon = pois["Longitude"].max() + buffer_deg

    lat_edges = np.arange(min_lat, max_lat + tile_size_deg, tile_size_deg)
    lon_edges = np.arange(min_lon, max_lon + tile_size_deg, tile_size_deg)

    tiles = []

    for i in range(len(lat_edges) - 1):
        for j in range(len(lon_edges) - 1):
            a = lat_edges[i]
            b = lon_edges[j]
            c = lat_edges[i + 1]
            d = lon_edges[j + 1]

            in_tile = pois[
                (pois["Latitude"] >= a)
                & (pois["Latitude"] <= c)
                & (pois["Longitude"] >= b)
                & (pois["Longitude"] <= d)
            ]

            if len(in_tile) > 0:
                tiles.append((a - buffer_deg, b - buffer_deg, c + buffer_deg, d + buffer_deg))

    return tiles


def build_overpass_query(min_lat, min_lon, max_lat, max_lon):
    bbox = f"{min_lat},{min_lon},{max_lat},{max_lon}"

    return f"""
    [out:json][timeout:90];
    (
      node["amenity"="parking"]({bbox});
      way["amenity"="parking"]({bbox});
      relation["amenity"="parking"]({bbox});

      node["amenity"="parking_space"]({bbox});
      way["amenity"="parking_space"]({bbox});
      relation["amenity"="parking_space"]({bbox});

      node["amenity"="parking_entrance"]({bbox});
      way["amenity"="parking_entrance"]({bbox});
      relation["amenity"="parking_entrance"]({bbox});

      node["highway"="bus_stop"]({bbox});
      node["amenity"="bus_station"]({bbox});
      way["amenity"="bus_station"]({bbox});
      relation["amenity"="bus_station"]({bbox});

      node["railway"="station"]({bbox});
      way["railway"="station"]({bbox});
      relation["railway"="station"]({bbox});

      node["railway"="halt"]({bbox});
      node["railway"="tram_stop"]({bbox});
      node["railway"="subway_entrance"]({bbox});

      node["public_transport"]({bbox});
      way["public_transport"]({bbox});
      relation["public_transport"]({bbox});

      node["amenity"="police"]({bbox});
      way["amenity"="police"]({bbox});
      relation["amenity"="police"]({bbox});

      node["amenity"="hospital"]({bbox});
      way["amenity"="hospital"]({bbox});
      relation["amenity"="hospital"]({bbox});

      node["amenity"="clinic"]({bbox});
      way["amenity"="clinic"]({bbox});
      relation["amenity"="clinic"]({bbox});

      node["emergency"]({bbox});
      way["emergency"]({bbox});
      relation["emergency"]({bbox});

      node["highway"="street_lamp"]({bbox});

      node["man_made"="surveillance"]({bbox});
      way["man_made"="surveillance"]({bbox});
      relation["man_made"="surveillance"]({bbox});

      node["surveillance"]({bbox});
      way["surveillance"]({bbox});
      relation["surveillance"]({bbox});

      node["office"="security"]({bbox});
      way["office"="security"]({bbox});
      relation["office"="security"]({bbox});
    );
    out center tags;
    """


def request_overpass(query, sleep=2.0):
    headers = {
        "User-Agent": "travelplanner-attractions-osm-context/1.0 academic research",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    }

    last_error = None

    for endpoint in OVERPASS_ENDPOINTS:
        try:
            response = requests.post(
                endpoint,
                data={"data": query},
                headers=headers,
                timeout=120,
            )

            if response.status_code == 200:
                time.sleep(sleep)
                return response.json()

            last_error = f"{endpoint} returned {response.status_code}: {response.text[:300]}"

        except Exception as e:
            last_error = f"{endpoint} exception: {str(e)[:300]}"

    raise RuntimeError(last_error)


def element_lat_lon(element):
    if "lat" in element and "lon" in element:
        return element["lat"], element["lon"]

    center = element.get("center", {})
    return center.get("lat"), center.get("lon")


def classify_osm_element(tags):
    if not isinstance(tags, dict):
        return []

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


def osm_elements_to_table(elements):
    rows = []
    seen = set()

    for element in elements:
        tags = element.get("tags", {}) or {}
        lat, lon = element_lat_lon(element)

        if lat is None or lon is None:
            continue

        osm_type = element.get("type")
        osm_id = element.get("id")

        for cls in classify_osm_element(tags):
            key = (cls, osm_type, osm_id)

            if key in seen:
                continue

            seen.add(key)

            rows.append({
                "osm_feature_class": cls,
                "osm_type": osm_type,
                "osm_id": osm_id,
                "Latitude": lat,
                "Longitude": lon,
            })

    return pd.DataFrame(rows)


def summarize_nearby_features(pois, feature_df, feature_class):
    result = pd.DataFrame(index=pois.index)

    result[f"{feature_class}_count_400m"] = 0
    result[f"{feature_class}_count_600m"] = 0
    result[f"{feature_class}_nearest_m"] = np.nan

    subset = feature_df[feature_df["osm_feature_class"] == feature_class].copy()

    if subset.empty:
        return result

    earth_radius_m = 6371000

    poi_coords = np.radians(pois[["Latitude", "Longitude"]].astype(float).values)
    feat_coords = np.radians(subset[["Latitude", "Longitude"]].astype(float).values)

    tree = BallTree(feat_coords, metric="haversine")

    for radius_m in [400, 600]:
        ind = tree.query_radius(poi_coords, r=radius_m / earth_radius_m)
        result[f"{feature_class}_count_{radius_m}m"] = [len(x) for x in ind]

    dist, _ = tree.query(poi_coords, k=1)
    result[f"{feature_class}_nearest_m"] = dist[:, 0] * earth_radius_m

    return result


def level_from_counts(row, prefix):
    c400 = row.get(f"{prefix}_count_400m", 0)
    c600 = row.get(f"{prefix}_count_600m", 0)

    if c400 > 0:
        return "high"
    if c600 > 0:
        return "medium"
    return "low"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--sleep", type=float, default=2.0)
    parser.add_argument("--city", default=None)
    parser.add_argument("--tile-size-deg", type=float, default=0.08)

    args = parser.parse_args()

    df = pd.read_csv(args.input)

    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df = df.dropna(subset=["Latitude", "Longitude"]).copy()

    if args.city:
        df = df[df["City"] == args.city].copy()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    all_outputs = []

    for city, city_df in df.groupby("City"):
        print(f"Processing city: {city}, rows: {len(city_df)}")

        tiles = build_tiles(city_df, tile_size_deg=args.tile_size_deg)

        city_elements = []

        for tile_idx, (min_lat, min_lon, max_lat, max_lon) in enumerate(tiles):
            safe_city = str(city).lower().replace(" ", "_").replace("/", "_")
            cache_path = cache_dir / f"osm_context_{safe_city}_tile_{tile_idx}.json"

            if cache_path.exists():
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                query = build_overpass_query(min_lat, min_lon, max_lat, max_lon)

                try:
                    data = request_overpass(query, sleep=args.sleep)
                except Exception as e:
                    print(f"Tile failed for {city} tile {tile_idx}: {e}")
                    data = {"elements": []}

                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(data, f)

            city_elements.extend(data.get("elements", []))

        feature_df = osm_elements_to_table(city_elements)

        city_out = city_df.copy().reset_index(drop=True)
        # Remove existing placeholder OSM columns before adding real OSM features.
        osm_cols_to_replace = [
            "parking_count_400m",
            "parking_count_600m",
            "parking_nearest_m",
            "public_transport_count_400m",
            "public_transport_count_600m",
            "public_transport_nearest_m",
            "osm_safety_proxy_count_400m",
            "osm_safety_proxy_count_600m",
            "osm_safety_proxy_nearest_m",
            "osm_security_proxy_count_400m",
            "osm_security_proxy_count_600m",
            "osm_security_proxy_nearest_m",
            "parking_availability",
            "public_transport_availability",
            "osm_safety_proxy",
            "osm_security_proxy",
        ]

        city_out = city_out.drop(
            columns=[c for c in osm_cols_to_replace if c in city_out.columns],
            errors="ignore",
        )

        for feature_class in [
            "parking",
            "public_transport",
            "osm_safety_proxy",
            "osm_security_proxy",
        ]:
            summary = summarize_nearby_features(
                city_out.reset_index(drop=True),
                feature_df,
                feature_class,
            )
            city_out = pd.concat([city_out, summary.reset_index(drop=True)], axis=1)

        city_out["parking_availability"] = city_out.apply(
            lambda r: level_from_counts(r, "parking"), axis=1
        )

        city_out["public_transport_availability"] = city_out.apply(
            lambda r: level_from_counts(r, "public_transport"), axis=1
        )

        city_out["osm_safety_proxy"] = city_out.apply(
            lambda r: level_from_counts(r, "osm_safety_proxy"), axis=1
        )

        city_out["osm_security_proxy"] = city_out.apply(
            lambda r: level_from_counts(r, "osm_security_proxy"), axis=1
        )

        all_outputs.append(city_out)

    out = pd.concat(all_outputs, ignore_index=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    out.to_csv(output_path, index=False)

    print("Saved:", output_path)
    print("Rows:", len(out))

    for col in [
        "parking_availability",
        "public_transport_availability",
        "osm_safety_proxy",
        "osm_security_proxy",
    ]:
        print(f"\n{col}:")
        print(out[col].value_counts(dropna=False))


if __name__ == "__main__":
    main()

