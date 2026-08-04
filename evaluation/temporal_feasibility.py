import json
import re
import sys
from pathlib import Path
from difflib import get_close_matches

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


DEFAULT_ATTRACTIONS_CSV = (
    PROJECT_ROOT
    / "database"
    / "attractions"
    / "attractions_google_osm_features_dbscan_dwell.csv"
)

DEFAULT_RESTAURANTS_CSV = (
    PROJECT_ROOT
    / "database"
    / "restaurants"
    / "restaurants_dbscan_empirical_dwell.csv"
)


def normalize_text(x):
    if x is None:
        return ""

    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass

    x = str(x).lower().strip()
    x = re.sub(r"[^a-z0-9]+", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def is_filled(x):
    if x is None:
        return False
    x = str(x).strip()
    return x != "" and x != "-"


def clean_value(x):
    if x is None:
        return "-"
    x = str(x).strip()
    return x if x else "-"


def strip_city_from_entity(entity):
    if not is_filled(entity):
        return ""
    parts = [p.strip() for p in str(entity).split(",")]
    if len(parts) >= 2:
        return ",".join(parts[:-1]).strip()
    return str(entity).strip()


def extract_city_from_entity(entity):
    if not is_filled(entity):
        return ""
    parts = [p.strip() for p in str(entity).split(",")]
    if len(parts) >= 2:
        return parts[-1].strip()
    return ""


def split_attractions(attraction_field):
    if not is_filled(attraction_field):
        return []
    parts = [p.strip() for p in str(attraction_field).split(";")]
    return [p for p in parts if p and p != "-"]


def safe_float(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def parse_time_to_minutes(time_str):
    m = re.search(r"(\d{1,2}):(\d{2})", str(time_str))
    if not m:
        return None
    return int(m.group(1)) * 60 + int(m.group(2))


def extract_departure_arrival_minutes(transportation):
    if not is_filled(transportation):
        return None, None

    text = str(transportation)
    dep_match = re.search(r"Departure Time:\s*(\d{1,2}:\d{2})", text, flags=re.IGNORECASE)
    arr_match = re.search(r"Arrival Time:\s*(\d{1,2}:\d{2})", text, flags=re.IGNORECASE)

    dep = parse_time_to_minutes(dep_match.group(1)) if dep_match else None
    arr = parse_time_to_minutes(arr_match.group(1)) if arr_match else None
    return dep, arr


def parse_transportation_minutes(transportation):
    if not is_filled(transportation):
        return 0.0

    text = str(transportation)

    duration_match = re.search(
        r"duration:\s*(?:(\d+)\s*hours?)?\s*(?:(\d+)\s*mins?)?",
        text,
        flags=re.IGNORECASE,
    )

    if duration_match:
        total = 0
        if duration_match.group(1):
            total += int(duration_match.group(1)) * 60
        if duration_match.group(2):
            total += int(duration_match.group(2))
        if total > 0:
            return float(total)

    dep, arr = extract_departure_arrival_minutes(transportation)
    if dep is not None and arr is not None:
        if arr < dep:
            arr += 24 * 60
        return float(arr - dep)

    return 0.0


def parse_raw_plan_text(text):
    if not text or not isinstance(text, str):
        return []

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"^\s*Travel Plan:\s*", "", text.strip(), flags=re.IGNORECASE)

    day_matches = list(re.finditer(r"(?im)^\s*Day\s*(\d+)\s*:\s*", text))
    if not day_matches:
        return []

    labels = {
        "Current City": "current_city",
        "Transportation": "transportation",
        "Breakfast": "breakfast",
        "Attraction": "attraction",
        "Lunch": "lunch",
        "Dinner": "dinner",
        "Accommodation": "accommodation",
    }

    label_regex = "|".join(re.escape(x) for x in labels.keys())
    parsed_days = []

    for i, match in enumerate(day_matches):
        day_num = int(match.group(1))
        start = match.end()
        end = day_matches[i + 1].start() if i + 1 < len(day_matches) else len(text)
        block = text[start:end].strip()
        day = {"day": day_num}

        for label, key in labels.items():
            pattern = rf"(?is){re.escape(label)}\s*:\s*(.*?)(?=\n\s*(?:{label_regex})\s*:|\Z)"
            m = re.search(pattern, block)
            day[key] = clean_value(m.group(1)) if m else "-"

        parsed_days.append(day)

    return parsed_days


class AttractionDwellLookup:
    """
    Looks up predicted attraction dwell directly from the active DBSCAN attraction file.
    This keeps evaluation aligned with the same file used by AttractionSearch.
    """

    def __init__(self, attractions_csv=None, default_dwell_minutes=90.0):
        self.default_dwell_minutes = float(default_dwell_minutes)
        self.attractions_csv = Path(attractions_csv or DEFAULT_ATTRACTIONS_CSV)

        if not self.attractions_csv.exists():
            raise FileNotFoundError(f"Could not find attractions CSV: {self.attractions_csv}")

        self.attractions = pd.read_csv(self.attractions_csv).copy()
        required = ["Name", "City", "predicted_dwell_minutes"]
        missing = [c for c in required if c not in self.attractions.columns]
        if missing:
            raise ValueError(f"Attraction CSV missing required columns: {missing}")

        self.attractions = self.attractions.dropna(subset=["Name", "City"]).copy()
        self.attractions["predicted_dwell_minutes"] = pd.to_numeric(
            self.attractions["predicted_dwell_minutes"], errors="coerce"
        ).fillna(self.default_dwell_minutes)

        self.attractions["name_norm"] = self.attractions["Name"].apply(normalize_text)
        self.attractions["city_norm"] = self.attractions["City"].apply(normalize_text)

        self.exact_lookup = {}
        for _, row in self.attractions.iterrows():
            self.exact_lookup[(row["name_norm"], row["city_norm"])] = row

        self.city_to_rows = {}
        self.city_to_names = {}
        for city_norm, group in self.attractions.groupby("city_norm"):
            self.city_to_rows[city_norm] = list(group.iterrows())
            self.city_to_names[city_norm] = group["name_norm"].tolist()

        self.name_to_first_row = {}
        for _, row in self.attractions.iterrows():
            if row["name_norm"] not in self.name_to_first_row:
                self.name_to_first_row[row["name_norm"]] = row

        self.match_cache = {}

    def _find_row(self, attraction_text, city_hint=""):
        name = strip_city_from_entity(attraction_text)
        city = extract_city_from_entity(attraction_text) or city_hint

        if " from " in f" {str(city).lower()} " and " to " in f" {str(city).lower()} ":
            city = extract_city_from_entity(attraction_text)

        name_norm = normalize_text(name)
        city_norm = normalize_text(city)
        cache_key = (name_norm, city_norm)

        if cache_key in self.match_cache:
            return self.match_cache[cache_key]

        row = self.exact_lookup.get((name_norm, city_norm))
        if row is not None:
            result = (row, "exact_attraction_db")
            self.match_cache[cache_key] = result
            return result

        row = self.name_to_first_row.get(name_norm)
        if row is not None:
            result = (row, "exact_name_attraction_db")
            self.match_cache[cache_key] = result
            return result

        if city_norm in self.city_to_names:
            matches = get_close_matches(name_norm, self.city_to_names[city_norm], n=1, cutoff=0.72)
            if matches:
                matched_name = matches[0]
                for _, candidate_row in self.city_to_rows[city_norm]:
                    if candidate_row["name_norm"] == matched_name:
                        result = (candidate_row, "fuzzy_attraction_db")
                        self.match_cache[cache_key] = result
                        return result

        result = (None, "fallback_not_in_attraction_db")
        self.match_cache[cache_key] = result
        return result

    def predict(self, attraction_text, city_hint=""):
        if not is_filled(attraction_text):
            return 0.0, "empty"

        row, match_source = self._find_row(attraction_text, city_hint=city_hint)
        if row is None:
            return self.default_dwell_minutes, "fallback_not_in_attraction_db"

        dwell = safe_float(row.get("predicted_dwell_minutes"), self.default_dwell_minutes)
        source = str(row.get("dwell_prediction_source", "dbscan_attraction_file"))
        return float(dwell), f"{source}_{match_source}"


class RestaurantDwellLookup:
    """
    Looks up empirical meal-duration defaults from restaurants_dbscan_empirical_dwell.csv.
    These are DBSCAN food/leisure empirical defaults, not restaurant-specific ML predictions.
    """

    def __init__(
        self,
        restaurants_csv=None,
        default_breakfast_minutes=49.3,
        default_lunch_minutes=57.9,
        default_dinner_minutes=63.9,
        default_meal_minutes=56.9,
    ):
        self.restaurants_csv = Path(restaurants_csv or DEFAULT_RESTAURANTS_CSV)
        if not self.restaurants_csv.exists():
            raise FileNotFoundError(f"Could not find restaurants CSV: {self.restaurants_csv}")

        self.defaults = {
            "breakfast": float(default_breakfast_minutes),
            "lunch": float(default_lunch_minutes),
            "dinner": float(default_dinner_minutes),
            "default": float(default_meal_minutes),
        }

        self.restaurants = pd.read_csv(self.restaurants_csv).copy()
        required = ["Name", "City"]
        missing = [c for c in required if c not in self.restaurants.columns]
        if missing:
            raise ValueError(f"Restaurants CSV missing required columns: {missing}")

        self.restaurants = self.restaurants.dropna(subset=["Name", "City"]).copy()

        for col, default in [
            ("predicted_breakfast_dwell_minutes", self.defaults["breakfast"]),
            ("predicted_lunch_dwell_minutes", self.defaults["lunch"]),
            ("predicted_dinner_dwell_minutes", self.defaults["dinner"]),
            ("predicted_dwell_minutes", self.defaults["default"]),
        ]:
            if col not in self.restaurants.columns:
                self.restaurants[col] = default
            self.restaurants[col] = pd.to_numeric(self.restaurants[col], errors="coerce").fillna(default)

        self.restaurants["name_norm"] = self.restaurants["Name"].apply(normalize_text)
        self.restaurants["city_norm"] = self.restaurants["City"].apply(normalize_text)

        self.exact_lookup = {}
        for _, row in self.restaurants.iterrows():
            self.exact_lookup[(row["name_norm"], row["city_norm"])] = row

        self.city_to_rows = {}
        self.city_to_names = {}
        for city_norm, group in self.restaurants.groupby("city_norm"):
            self.city_to_rows[city_norm] = list(group.iterrows())
            self.city_to_names[city_norm] = group["name_norm"].tolist()

        self.name_to_first_row = {}
        for _, row in self.restaurants.iterrows():
            if row["name_norm"] not in self.name_to_first_row:
                self.name_to_first_row[row["name_norm"]] = row

        self.match_cache = {}

    def _find_row(self, restaurant_text, city_hint=""):
        name = strip_city_from_entity(restaurant_text)
        city = extract_city_from_entity(restaurant_text) or city_hint

        if " from " in f" {str(city).lower()} " and " to " in f" {str(city).lower()} ":
            city = extract_city_from_entity(restaurant_text)

        name_norm = normalize_text(name)
        city_norm = normalize_text(city)
        cache_key = (name_norm, city_norm)

        if cache_key in self.match_cache:
            return self.match_cache[cache_key]

        row = self.exact_lookup.get((name_norm, city_norm))
        if row is not None:
            result = (row, "exact_restaurant_db")
            self.match_cache[cache_key] = result
            return result

        row = self.name_to_first_row.get(name_norm)
        if row is not None:
            result = (row, "exact_name_restaurant_db")
            self.match_cache[cache_key] = result
            return result

        if city_norm in self.city_to_names:
            matches = get_close_matches(name_norm, self.city_to_names[city_norm], n=1, cutoff=0.72)
            if matches:
                matched_name = matches[0]
                for _, candidate_row in self.city_to_rows[city_norm]:
                    if candidate_row["name_norm"] == matched_name:
                        result = (candidate_row, "fuzzy_restaurant_db")
                        self.match_cache[cache_key] = result
                        return result

        result = (None, "fallback_not_in_restaurant_db")
        self.match_cache[cache_key] = result
        return result

    def predict(self, restaurant_text, meal_slot, city_hint=""):
        if not is_filled(restaurant_text):
            return 0.0, "empty"

        meal_slot = str(meal_slot).lower().strip()
        if meal_slot == "breakfast":
            col = "predicted_breakfast_dwell_minutes"
            fallback = self.defaults["breakfast"]
        elif meal_slot == "lunch":
            col = "predicted_lunch_dwell_minutes"
            fallback = self.defaults["lunch"]
        elif meal_slot == "dinner":
            col = "predicted_dinner_dwell_minutes"
            fallback = self.defaults["dinner"]
        else:
            col = "predicted_dwell_minutes"
            fallback = self.defaults["default"]

        row, match_source = self._find_row(restaurant_text, city_hint=city_hint)
        if row is None:
            return float(fallback), "fallback_not_in_restaurant_db"

        dwell = safe_float(row.get(col), fallback)
        source = str(row.get("dwell_prediction_source", "dbscan_food_leisure_empirical"))
        return float(dwell), f"{source}_{match_source}"


def infer_day_budget(day, is_last_day=False):
    """
    Coarse TTDP-inspired duration-budget heuristic.
    Kept for continuity with earlier results.
    """
    current_city = str(day.get("current_city", "")).lower()
    transportation = str(day.get("transportation", "")).lower().strip()

    has_transport = transportation not in ["", "-"]
    is_from_to = "from" in current_city and "to" in current_city

    if is_last_day and (has_transport or is_from_to):
        return 360.0, "return_travel_day"
    if has_transport or is_from_to:
        return 480.0, "travel_day"
    return 600.0, "full_day"


def infer_available_sightseeing_window(
    day,
    is_last_day=False,
    day_start_minutes=9 * 60,
    day_end_minutes=21 * 60,
    arrival_buffer_minutes=60.0,
    departure_buffer_minutes=120.0,
):
    """
    Arrival/departure-aware sightseeing window for attraction-only feasibility.
    """
    current_city = str(day.get("current_city", "")).lower()
    transportation = str(day.get("transportation", "")).strip()
    has_transport = is_filled(transportation)
    is_from_to = "from" in current_city and "to" in current_city

    dep, arr = extract_departure_arrival_minutes(transportation)
    start = float(day_start_minutes)
    end = float(day_end_minutes)
    notes = []

    if has_transport or is_from_to:
        if arr is not None and not is_last_day:
            start = max(start, float(arr) + float(arrival_buffer_minutes))
            notes.append("arrival_buffer_applied")

        if dep is not None and is_last_day:
            end = min(end, float(dep) - float(departure_buffer_minutes))
            notes.append("departure_buffer_applied")
        elif dep is not None and is_from_to:
            notes.append("travel_day_detected")

    window = max(0.0, end - start)
    return window, ";".join(notes) if notes else "standard_full_day_window"


def load_raw_generated_plans(output_dir, set_type, model_name, mode, num_samples):
    folder = Path(output_dir) / set_type
    result_key = f"{model_name}_{mode}_results"
    plans = []

    for sample_id in range(1, num_samples + 1):
        path = folder / f"generated_plan_{sample_id}.json"
        if not path.exists():
            plans.append({"sample_id": sample_id, "status": "missing_file", "days": []})
            continue

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            obj = data[0]
            raw_text = obj.get(result_key, "")
            days = parse_raw_plan_text(raw_text)
            plans.append({"sample_id": sample_id, "status": "ok" if days else "unparsed_raw", "days": days})
        except Exception as e:
            plans.append({"sample_id": sample_id, "status": f"error: {e}", "days": []})

    return plans


def compute_temporal_eval(
    output_dir,
    system_name,
    set_type="validation",
    model_name="gpt-3.5-turbo-0125",
    mode="two-stage",
    num_samples=100,
    dwell_model_dir=None,
    meal_minutes=None,
    local_movement_minutes_per_attraction=30.0,
    save_dir=None,
    attractions_csv=None,
    restaurants_csv=None,
):
    attraction_lookup = AttractionDwellLookup(attractions_csv=attractions_csv, default_dwell_minutes=90.0)
    restaurant_lookup = RestaurantDwellLookup(restaurants_csv=restaurants_csv)

    plans = load_raw_generated_plans(
        output_dir=output_dir,
        set_type=set_type,
        model_name=model_name,
        mode=mode,
        num_samples=num_samples,
    )

    day_rows = []
    plan_rows = []

    for plan in plans:
        sample_id = plan["sample_id"]
        days = plan["days"]

        if not days:
            plan_rows.append({
                "system": system_name,
                "sample_id": sample_id,
                "evaluated_days": 0,
                "plan_temporally_feasible": False,
                "plan_attraction_temporally_feasible": False,
                "plan_full_day_temporally_feasible": False,
                "total_full_day_overload_minutes": None,
                "total_attraction_overload_minutes": None,
                "total_attractions": 0,
                "unique_attractions": 0,
            })
            continue

        max_day = max(day.get("day", 0) for day in days)
        full_day_feasible_flags = []
        attraction_feasible_flags = []
        feasible_complete_flags = []
        balanced_day_flags = []
        underplanned_day_flags = []
        full_day_overloads = []
        attraction_overloads = []
        all_attractions = []

        for day in days:
            is_last_day = day.get("day") == max_day
            coarse_budget, day_type = infer_day_budget(day, is_last_day=is_last_day)
            attraction_window, window_note = infer_available_sightseeing_window(day, is_last_day=is_last_day)

            current_city = day.get("current_city", "-")
            transportation = day.get("transportation", "-")
            breakfast = day.get("breakfast", "-")
            lunch = day.get("lunch", "-")
            dinner = day.get("dinner", "-")
            accommodation = day.get("accommodation", "-")
            attraction_field = day.get("attraction", "-")

            transport_minutes = parse_transportation_minutes(transportation)

            breakfast_minutes, breakfast_source = restaurant_lookup.predict(breakfast, "breakfast", current_city)
            lunch_minutes, lunch_source = restaurant_lookup.predict(lunch, "lunch", current_city)
            dinner_minutes, dinner_source = restaurant_lookup.predict(dinner, "dinner", current_city)

            meal_count = sum(is_filled(x) for x in [breakfast, lunch, dinner])
            total_meal_minutes = breakfast_minutes + lunch_minutes + dinner_minutes

            attractions = split_attractions(attraction_field)
            attraction_count = len(attractions)
            attraction_dwell_total = 0.0
            attraction_match_sources = []

            for attraction in attractions:
                dwell, source = attraction_lookup.predict(attraction, city_hint=current_city)
                attraction_dwell_total += dwell
                attraction_match_sources.append(source)
                all_attractions.append(normalize_text(strip_city_from_entity(attraction)))

            local_movement = attraction_count * local_movement_minutes_per_attraction

            attraction_only_load = attraction_dwell_total + local_movement
            attraction_overload = max(0.0, attraction_only_load - attraction_window)
            attraction_temporally_feasible = attraction_only_load <= attraction_window

            full_day_load = transport_minutes + total_meal_minutes + attraction_dwell_total + local_movement
            full_day_overload = max(0.0, full_day_load - coarse_budget)
            full_day_temporally_feasible = full_day_load <= coarse_budget

            temporal_utilisation_ratio = (
                full_day_load / coarse_budget if coarse_budget and coarse_budget > 0 else 0.0
            )
            attraction_utilisation_ratio = (
                attraction_only_load / attraction_window
                if attraction_window and attraction_window > 0
                else 0.0
            )

            # Feasible-complete day:
            # - For full destination days, require at least one attraction and at least two meals.
            # - For travel/return days, use a softer rule because travel reduces available time.
            if day_type == "full_day":
                feasible_complete_day = (
                    full_day_temporally_feasible
                    and attraction_count >= 1
                    and meal_count >= 2
                )
            else:
                feasible_complete_day = (
                    full_day_temporally_feasible
                    and meal_count >= 1
                )

            # Under-planned is only applied to full destination days so realistic
            # light arrival/departure days are not unfairly penalised.
            underplanned_day = (
                full_day_temporally_feasible
                and day_type == "full_day"
                and temporal_utilisation_ratio < 0.35
            )

            # Balanced day is deliberately a broad band. It is a diagnostic, not
            # a hard TravelPlanner constraint.
            balanced_day = (
                full_day_temporally_feasible
                and 0.35 <= temporal_utilisation_ratio <= 0.85
            )

            full_day_feasible_flags.append(full_day_temporally_feasible)
            attraction_feasible_flags.append(attraction_temporally_feasible)
            feasible_complete_flags.append(feasible_complete_day)
            balanced_day_flags.append(balanced_day)
            underplanned_day_flags.append(underplanned_day)
            full_day_overloads.append(full_day_overload)
            attraction_overloads.append(attraction_overload)

            nonempty_fields = sum(
                is_filled(day.get(k, "-"))
                for k in ["current_city", "transportation", "breakfast", "attraction", "lunch", "dinner", "accommodation"]
            )

            day_rows.append({
                "system": system_name,
                "sample_id": sample_id,
                "day": day.get("day"),
                "day_type": day_type,
                "window_note": window_note,
                "nonempty_fields": nonempty_fields,
                "meal_count": meal_count,
                "has_accommodation": is_filled(accommodation),
                "transport_minutes": transport_minutes,
                "breakfast_minutes": breakfast_minutes,
                "lunch_minutes": lunch_minutes,
                "dinner_minutes": dinner_minutes,
                "meal_dwell_minutes": total_meal_minutes,
                "meal_dwell_sources": "|".join([breakfast_source, lunch_source, dinner_source]),
                "attraction_count": attraction_count,
                "attraction_dwell_minutes": attraction_dwell_total,
                "local_movement_minutes": local_movement,
                "attraction_available_window_minutes": attraction_window,
                "attraction_only_load_minutes": attraction_only_load,
                "attraction_overload_minutes": attraction_overload,
                "attraction_temporally_feasible": attraction_temporally_feasible,
                "day_budget_minutes": coarse_budget,
                "full_day_load_minutes": full_day_load,
                "full_day_overload_minutes": full_day_overload,
                "full_day_temporally_feasible": full_day_temporally_feasible,
                "temporal_utilisation_ratio": temporal_utilisation_ratio,
                "attraction_utilisation_ratio": attraction_utilisation_ratio,
                "feasible_complete_day": feasible_complete_day,
                "balanced_day": balanced_day,
                "underplanned_day": underplanned_day,
                "estimated_day_load_minutes": full_day_load,
                "overload_minutes": full_day_overload,
                "temporally_feasible": full_day_temporally_feasible,
                "attraction_dwell_match_sources": "|".join(attraction_match_sources),
            })

        plan_rows.append({
            "system": system_name,
            "sample_id": sample_id,
            "evaluated_days": len(days),
            "plan_temporally_feasible": all(full_day_feasible_flags) if full_day_feasible_flags else False,
            "plan_full_day_temporally_feasible": all(full_day_feasible_flags) if full_day_feasible_flags else False,
            "plan_attraction_temporally_feasible": all(attraction_feasible_flags) if attraction_feasible_flags else False,
            "plan_feasible_complete": all(feasible_complete_flags) if feasible_complete_flags else False,
            "feasible_complete_days": int(sum(feasible_complete_flags)),
            "balanced_days": int(sum(balanced_day_flags)),
            "underplanned_days": int(sum(underplanned_day_flags)),
            "total_full_day_overload_minutes": sum(full_day_overloads) if full_day_overloads else None,
            "total_attraction_overload_minutes": sum(attraction_overloads) if attraction_overloads else None,
            "total_overload_minutes": sum(full_day_overloads) if full_day_overloads else None,
            "total_attractions": len(all_attractions),
            "unique_attractions": len(set(x for x in all_attractions if x)),
        })

    day_df = pd.DataFrame(day_rows)
    plan_df = pd.DataFrame(plan_rows)

    if len(day_df) == 0:
        summary = {"System": system_name, "Evaluated Plans": 0, "Evaluated Days": 0}
        return summary, day_df, plan_df

    evaluated_plans = int((plan_df["evaluated_days"] > 0).sum())
    evaluated_days = int(len(day_df))

    all_attraction_sources = []
    for src in day_df["attraction_dwell_match_sources"].dropna().tolist():
        if src:
            all_attraction_sources.extend([x for x in src.split("|") if x])

    attraction_fallback_count = sum(1 for x in all_attraction_sources if x.startswith("fallback"))
    attraction_model_count = sum(
        1 for x in all_attraction_sources
        if "dbscan_episode_random_forest" in x or "dbscan_attraction_file" in x
    )
    attraction_source_count = len(all_attraction_sources)
    attraction_fallback_rate = (attraction_fallback_count / attraction_source_count * 100) if attraction_source_count else 0.0
    attraction_model_rate = (attraction_model_count / attraction_source_count * 100) if attraction_source_count else 0.0

    summary = {
        "System": system_name,
        "Evaluated Plans": evaluated_plans,
        "Evaluated Days": evaluated_days,
        "Attraction-Only Day Temporal Feasibility Rate (%)": float(day_df["attraction_temporally_feasible"].mean() * 100),
        "Attraction-Only Plan Temporal Feasibility Rate (%)": float(
            plan_df.loc[plan_df["evaluated_days"] > 0, "plan_attraction_temporally_feasible"].mean() * 100
        ) if evaluated_plans > 0 else 0.0,
        "Attraction-Only Overloaded Day Rate (%)": float((~day_df["attraction_temporally_feasible"]).mean() * 100),
        "Mean Attraction-Only Load Minutes": float(day_df["attraction_only_load_minutes"].mean()),
        "Mean Attraction-Only Overload Minutes": float(day_df["attraction_overload_minutes"].mean()),
        "Full-Day Temporal Feasibility Rate (%)": float(day_df["full_day_temporally_feasible"].mean() * 100),
        "Full-Plan Temporal Feasibility Rate (%)": float(
            plan_df.loc[plan_df["evaluated_days"] > 0, "plan_full_day_temporally_feasible"].mean() * 100
        ) if evaluated_plans > 0 else 0.0,
        "Full-Day Overloaded Day Rate (%)": float((~day_df["full_day_temporally_feasible"]).mean() * 100),
        "Mean Full-Day Load Minutes": float(day_df["full_day_load_minutes"].mean()),
        "Mean Full-Day Overload Minutes": float(day_df["full_day_overload_minutes"].mean()),
        "Day Temporal Feasibility Rate (%)": float(day_df["full_day_temporally_feasible"].mean() * 100),
        "Plan Temporal Feasibility Rate (%)": float(
            plan_df.loc[plan_df["evaluated_days"] > 0, "plan_full_day_temporally_feasible"].mean() * 100
        ) if evaluated_plans > 0 else 0.0,
        "Overloaded Day Rate (%)": float((~day_df["full_day_temporally_feasible"]).mean() * 100),
        "Mean Day Load Minutes": float(day_df["full_day_load_minutes"].mean()),
        "Mean Overload Minutes": float(day_df["full_day_overload_minutes"].mean()),
        "Mean Attraction Count Per Day": float(day_df["attraction_count"].mean()),
        "Mean Attraction Dwell Minutes Per Day": float(day_df["attraction_dwell_minutes"].mean()),
        "Mean Meal Dwell Minutes Per Day": float(day_df["meal_dwell_minutes"].mean()),
        "Mean Meals Per Day": float(day_df["meal_count"].mean()),
        "Accommodation Day Rate (%)": float(day_df["has_accommodation"].mean() * 100),

        "Feasible-Complete Day Rate (%)": float(
            day_df["feasible_complete_day"].mean() * 100
        ),
        "Feasible-Complete Plan Rate (%)": float(
            plan_df.loc[
                plan_df["evaluated_days"] > 0,
                "plan_feasible_complete",
            ].mean()
            * 100
        ) if evaluated_plans > 0 else 0.0,
        "Balanced Day Rate (%)": float(day_df["balanced_day"].mean() * 100),
        "Under-planned Day Rate (%)": float(day_df["underplanned_day"].mean() * 100),
        "Mean Temporal Utilisation Ratio": float(day_df["temporal_utilisation_ratio"].mean()),
        "Mean Attraction Utilisation Ratio": float(day_df["attraction_utilisation_ratio"].mean()),

        "DBSCAN Attraction Dwell Match Rate (%)": float(attraction_model_rate),
        "Attraction Fallback Dwell Rate (%)": float(attraction_fallback_rate),
    }

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        day_df.to_csv(save_dir / f"{system_name}_temporal_day_details.csv", index=False)
        plan_df.to_csv(save_dir / f"{system_name}_temporal_plan_details.csv", index=False)
        with open(save_dir / f"{system_name}_temporal_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

    return summary, day_df, plan_df


def compare_temporal_summaries(primary_summary, compare_summary):
    rows = []
    metric_keys = [
        "Attraction-Only Day Temporal Feasibility Rate (%)",
        "Attraction-Only Plan Temporal Feasibility Rate (%)",
        "Attraction-Only Overloaded Day Rate (%)",
        "Mean Attraction-Only Load Minutes",
        "Mean Attraction-Only Overload Minutes",
        "Full-Day Temporal Feasibility Rate (%)",
        "Full-Plan Temporal Feasibility Rate (%)",
        "Full-Day Overloaded Day Rate (%)",
        "Mean Full-Day Load Minutes",
        "Mean Full-Day Overload Minutes",
        "Mean Attraction Count Per Day",
        "Mean Attraction Dwell Minutes Per Day",
        "Mean Meal Dwell Minutes Per Day",
        "Mean Meals Per Day",
        "Accommodation Day Rate (%)",
        "Feasible-Complete Day Rate (%)",
        "Feasible-Complete Plan Rate (%)",
        "Balanced Day Rate (%)",
        "Under-planned Day Rate (%)",
        "Mean Temporal Utilisation Ratio",
        "Mean Attraction Utilisation Ratio",
        "DBSCAN Attraction Dwell Match Rate (%)",
        "Attraction Fallback Dwell Rate (%)",
    ]

    for key in metric_keys:
        if key in primary_summary and key in compare_summary:
            rows.append({
                "Metric": key,
                compare_summary["System"]: compare_summary[key],
                primary_summary["System"]: primary_summary[key],
                "Delta Primary - Compare": primary_summary[key] - compare_summary[key],
            })

    return pd.DataFrame(rows)


def print_temporal_summary(summary):
    print(f"\nSystem: {summary.get('System', '-')}")
    for key, value in summary.items():
        if key == "System":
            continue
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")


def run_temporal_eval_block(
    output_dir,
    system_name,
    set_type,
    model_name,
    mode,
    num_samples,
    dwell_model_dir=None,
    compare_output_dir=None,
    compare_name="baseline",
    save_dir=None,
):
    print("\n===== DBSCAN Dwell-Aware Temporal Feasibility Evaluation =====")
    print("Attraction dwell source: TravelPlanner DBSCAN attraction prediction file")
    print("Meal dwell source: DBSCAN food/leisure empirical restaurant defaults")

    primary_summary, _, _ = compute_temporal_eval(
        output_dir=output_dir,
        system_name=system_name,
        set_type=set_type,
        model_name=model_name,
        mode=mode,
        num_samples=num_samples,
        dwell_model_dir=dwell_model_dir,
        save_dir=save_dir,
    )

    print_temporal_summary(primary_summary)

    if compare_output_dir:
        compare_summary, _, _ = compute_temporal_eval(
            output_dir=compare_output_dir,
            system_name=compare_name,
            set_type=set_type,
            model_name=model_name,
            mode=mode,
            num_samples=num_samples,
            dwell_model_dir=dwell_model_dir,
            save_dir=save_dir,
        )

        print("\n----- Comparison system -----")
        print_temporal_summary(compare_summary)

        comparison_df = compare_temporal_summaries(primary_summary, compare_summary)

        print("\n----- Temporal Feasibility Comparison -----")
        print(comparison_df.to_string(index=False))

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            comparison_df.to_csv(save_dir / "temporal_comparison_summary.csv", index=False)

    return primary_summary
