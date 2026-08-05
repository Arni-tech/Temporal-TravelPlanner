from pathlib import Path
import re
import numpy as np
import pandas as pd


labels_path = Path(
    r"C:\Users\negia\trip_plan\dwell_model\adjusted_dwell_model_pipeline\data\best_eps_100m_min_5\episode_dwell_pois_debug.csv"
)

features_path = Path(
    r"C:\Users\negia\trip_plan\dwell_model\adjusted_dwell_model_pipeline\data\adjusted_google_osm_pois.csv"
)

output_path = Path(
    r"C:\Users\negia\trip_plan\dwell_model\adjusted_dwell_model_pipeline\data\dbscan_adjusted_google_osm_pois.csv"
)


def split_source_ids(value):
    """
    Handles source_global_venue_ids stored as:
    - a single id: beijing::123
    - pipe/semicolon/comma separated: beijing::123|beijing::456
    - list-like string: ['beijing::123', 'beijing::456']
    """
    if pd.isna(value):
        return []

    text = str(value).strip()

    # Remove common list-like wrappers/quotes.
    text = text.replace("[", "").replace("]", "")
    text = text.replace("'", "").replace('"', "")

    # Split on common separators.
    parts = re.split(r"\s*[|;,]\s*", text)

    return [p.strip() for p in parts if p.strip()]


def weighted_mean(values, weights):
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce")

    mask = values.notna() & weights.notna() & (weights > 0)

    if not mask.any():
        return np.nan

    return np.average(values[mask], weights=weights[mask])


labels = pd.read_csv(labels_path)
features = pd.read_csv(features_path)

print("Labels rows:", len(labels))
print("Google/OSM feature rows:", len(features))

required_label_cols = [
    "global_venue_id",
    "obs_count",
    "p25_episode_dwell_minutes",
    "median_episode_dwell_minutes",
    "p75_episode_dwell_minutes",
    "mean_episode_dwell_minutes",
    "std_episode_dwell_minutes",
    "multi_poi_episode_rate",
    "median_distance_to_next_episode_km",
    "median_estimated_travel_time_to_next_episode_min",
    "log_obs_count",
]

missing = [c for c in required_label_cols if c not in labels.columns]
if missing:
    raise ValueError(f"Missing required columns in label file: {missing}")

if "source_global_venue_ids" not in features.columns:
    raise ValueError("Feature file does not contain source_global_venue_ids")

# Explode Google/OSM rows so one Google Place row can match multiple source global venue ids.
features = features.copy()
features["source_global_venue_id_list"] = features["source_global_venue_ids"].apply(split_source_ids)

exploded = features.explode("source_global_venue_id_list").rename(
    columns={"source_global_venue_id_list": "global_venue_id"}
)

exploded["global_venue_id"] = exploded["global_venue_id"].astype(str).str.strip()
labels["global_venue_id"] = labels["global_venue_id"].astype(str).str.strip()

merged_long = exploded.merge(
    labels[required_label_cols],
    on="global_venue_id",
    how="inner",
    suffixes=("", "_dbscan"),
)

print("Exploded feature rows:", len(exploded))
print("Matched exploded rows:", len(merged_long))
print("Matched unique Google places:", merged_long["google_place_id"].nunique())

if merged_long.empty:
    raise ValueError(
        "No matches found. Check whether source_global_venue_ids uses the same "
        "format as episode_dwell_pois_debug.global_venue_id."
    )

# Re-aggregate back to one row per Google place / feature row.
# Use poi_id as the safest row identity if present, otherwise google_place_id.
group_key = "poi_id" if "poi_id" in merged_long.columns else "google_place_id"

base_cols = list(features.columns)
base_cols = [c for c in base_cols if c != "source_global_venue_id_list"]

# Keep the first feature row per group because Google/OSM features are already row-level features.
first_features = (
    merged_long
    .sort_values(["obs_count"], ascending=False)
    .groupby(group_key, as_index=False)
    .first()[base_cols]
)

agg_labels = (
    merged_long
    .groupby(group_key)
    .apply(
        lambda g: pd.Series(
            {
                "dbscan_source_global_venue_ids": "|".join(sorted(set(g["global_venue_id"].astype(str)))),
                "dbscan_matched_source_count": int(g["global_venue_id"].nunique()),

                # Total clean observations contributing to this Google Place.
                "dbscan_episode_obs_count": int(g["obs_count"].sum()),

                # Weighted averages across source venue labels.
                # This is not a true combined median, but it is defensible as an
                # obs-count-weighted average of source POI medians.
                "p25_episode_dwell_minutes": weighted_mean(
                    g["p25_episode_dwell_minutes"],
                    g["obs_count"],
                ),
                "median_episode_dwell_minutes": weighted_mean(
                    g["median_episode_dwell_minutes"],
                    g["obs_count"],
                ),
                "p75_episode_dwell_minutes": weighted_mean(
                    g["p75_episode_dwell_minutes"],
                    g["obs_count"],
                ),
                "mean_episode_dwell_minutes": weighted_mean(
                    g["mean_episode_dwell_minutes"],
                    g["obs_count"],
                ),
                "std_episode_dwell_minutes": weighted_mean(
                    g["std_episode_dwell_minutes"],
                    g["obs_count"],
                ),
                "multi_poi_episode_rate": weighted_mean(
                    g["multi_poi_episode_rate"],
                    g["obs_count"],
                ),
                "median_distance_to_next_episode_km": weighted_mean(
                    g["median_distance_to_next_episode_km"],
                    g["obs_count"],
                ),
                "median_estimated_travel_time_to_next_episode_min": weighted_mean(
                    g["median_estimated_travel_time_to_next_episode_min"],
                    g["obs_count"],
                ),
                "dbscan_log_obs_count": np.log1p(g["obs_count"].sum()),
            }
        )
    )
    .reset_index()
)

final = first_features.merge(agg_labels, on=group_key, how="inner")

# Preserve old direct-TTNE target columns by renaming them, then create compatibility columns.
old_target_cols = [
    "median_adjusted_dwell_minutes",
    "mean_adjusted_dwell_minutes",
    "std_adjusted_dwell_minutes",
    "obs_count",
    "log_obs_count",
]

for col in old_target_cols:
    if col in final.columns:
        final = final.rename(columns={col: f"old_direct_ttne_{col}"})

# Compatibility columns for existing model-training code.
final["median_adjusted_dwell_minutes"] = final["median_episode_dwell_minutes"]
final["mean_adjusted_dwell_minutes"] = final["mean_episode_dwell_minutes"]
final["std_adjusted_dwell_minutes"] = final["std_episode_dwell_minutes"]
final["obs_count"] = final["dbscan_episode_obs_count"]
final["log_obs_count"] = final["dbscan_log_obs_count"]

# Keep only valid Google place rows.
if "google_place_id" in final.columns:
    before = len(final)
    final = final[final["google_place_id"].notna()].copy()
    print("Rows after google_place_id filter:", len(final), "from", before)

final.to_csv(output_path, index=False)

print("\nSaved DBSCAN episode-labelled Google/OSM feature table:")
print(output_path)

print("\nFinal rows:", len(final))
print("Unique google_place_id:", final["google_place_id"].nunique() if "google_place_id" in final.columns else "N/A")

print("\nTarget columns now available:")
print("median_episode_dwell_minutes")
print("median_adjusted_dwell_minutes  # compatibility alias for training scripts")
