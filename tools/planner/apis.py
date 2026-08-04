import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from langchain.prompts import PromptTemplate
from agents.prompts import (
    planner_agent_prompt,
    cot_planner_agent_prompt,
    react_planner_agent_prompt,
    reflect_prompt,
    react_reflect_planner_agent_prompt,
    REFLECTION_HEADER,
)
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from env import ReactEnv, ReactReflectEnv
import tiktoken
import re
import openai
import time
from enum import Enum
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI


OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]


def catch_openai_api_error():
    error = sys.exc_info()[0]

    if error == openai.error.APIConnectionError:
        print("APIConnectionError")
    elif error == openai.error.RateLimitError:
        print("RateLimitError")
        time.sleep(60)
    elif error == openai.error.APIError:
        print("APIError")
    elif error == openai.error.AuthenticationError:
        print("AuthenticationError")
    else:
        print("API error:", error)


class ReflexionStrategy(Enum):
    """
    REFLEXION: Apply reflexion to the next reasoning trace
    """

    REFLEXION = "reflexion"


class Planner:
    def __init__(
        self,
        agent_prompt: PromptTemplate = planner_agent_prompt,
        model_name: str = "gpt-3.5-turbo-1106",
    ) -> None:
        self.agent_prompt = agent_prompt
        self.scratchpad: str = ""
        self.model_name = model_name
        self.enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

        if model_name in ["mistral-7B-32K"]:
            self.llm = ChatOpenAI(
                temperature=0,
                max_tokens=4096,
                openai_api_key="EMPTY",
                openai_api_base="http://localhost:8301/v1",
                model_name="gpt-3.5-turbo",
            )

        elif model_name in ["ChatGLM3-6B-32K"]:
            self.llm = ChatOpenAI(
                temperature=0,
                max_tokens=4096,
                openai_api_key="EMPTY",
                openai_api_base="http://localhost:8501/v1",
                model_name="gpt-3.5-turbo",
            )

        elif model_name in ["mixtral"]:
            self.max_token_length = 30000
            self.llm = ChatOpenAI(
                temperature=0,
                max_tokens=4096,
                openai_api_key="EMPTY",
                openai_api_base="http://localhost:8501/v1",
                model_name="YOUR/MODEL/PATH",
            )

        elif model_name in ["gemini"]:
            self.llm = ChatGoogleGenerativeAI(
                temperature=0,
                model="gemini-pro",
                google_api_key=GOOGLE_API_KEY,
            )

        else:
            self.llm = ChatOpenAI(
                model_name=model_name,
                temperature=0,
                max_tokens=4096,
                openai_api_key=OPENAI_API_KEY,
            )

        print(f"PlannerAgent {model_name} loaded.")

    def run(self, text, query, log_file=None) -> str:
        prompt = self._build_agent_prompt(text, query)

        if log_file:
            log_file.write("\n---------------Planner\n" + prompt)

        if self.model_name not in ["gemini"] and len(self.enc.encode(prompt)) > 12000:
            return "Max Token Length Exceeded."

        # 1. Generate the initial plan using the normal TravelPlanner planner.
        if self.model_name in ["gemini"]:
            initial_plan = str(self.llm.invoke(prompt).content)
        else:
            initial_plan = self.llm([HumanMessage(content=prompt)]).content

        # 2. Optional LLM repair if lightweight temporal check detects overload.
        repaired_plan = self._repair_temporal_overload_if_needed(
            notebook_text=text,
            query=query,
            initial_plan=initial_plan,
            log_file=log_file,
        )

        # 3. Deterministic final guardrail.
        # This is the important part: GPT-3.5 may ignore long repair prompts,
        # so final attraction-field cleanup is enforced in code.
        final_plan = apply_deterministic_attraction_guardrail(
            plan_text=repaired_plan,
            notebook_text=text,
            log_file=log_file,
        )

        return final_plan

    def _repair_temporal_overload_if_needed(
        self,
        notebook_text: str,
        query: str,
        initial_plan: str,
        log_file=None,
    ) -> str:
        """
        Revise only Attraction fields if the generated plan appears temporally
        overloaded.

        This LLM repair is useful, but not fully trusted. A deterministic
        attraction guardrail is applied after this method.
        """

        report = build_temporal_repair_report(
            plan_text=initial_plan,
            notebook_text=notebook_text,
        )

        if not report["needs_repair"]:
            return initial_plan

        repair_prompt = f"""
You are revising a TravelPlanner itinerary for temporal feasibility.

Original user query:
{query}

Available information from Notebook:
{notebook_text}

Initial generated Travel Plan:
{initial_plan}

Temporal feasibility report:
{report["text_report"]}

Revise the Travel Plan to fix only the overloaded Attraction fields.

Rules:
- Return the complete Travel Plan in the same format.
- Keep every Current City field unchanged.
- Keep every Transportation field unchanged.
- Keep every Breakfast, Lunch, and Dinner field unchanged.
- Keep every Accommodation field unchanged.
- Revise only Attraction fields.
- Use only exact attraction names from the provided attraction information.
- Prefer removing attractions over adding attractions.
- On a crowded travel day, use one attraction or '-'.
- On a crowded full day, use one or two attractions.
- Do not include dwell minutes, calculations, notes, or explanations.
- Return only the revised Travel Plan.
"""

        if log_file:
            log_file.write("\n---------------Temporal Repair Report\n" + report["text_report"])
            log_file.write("\n---------------Temporal Repair Prompt\n" + repair_prompt)

        if self.model_name in ["gemini"]:
            repaired = str(self.llm.invoke(repair_prompt).content)
        else:
            if len(self.enc.encode(repair_prompt)) > 12000:
                return initial_plan

            repaired = self.llm([HumanMessage(content=repair_prompt)]).content

        if "Day 1" not in repaired:
            return initial_plan

        return repaired

    def _build_agent_prompt(self, text, query) -> str:
        return self.agent_prompt.format(text=text, query=query)


# -------------------------------------------------------------------------
# Lightweight temporal helpers for Planner repair/guardrail
# -------------------------------------------------------------------------

def _planner_is_filled(x):
    if x is None:
        return False

    x = str(x).strip()
    return x != "" and x != "-"


def _planner_clean_value(x):
    if x is None:
        return "-"

    x = str(x).strip()
    return x if x else "-"


def _planner_normalize_text(x):
    if x is None:
        return ""

    x = str(x).lower().strip()
    x = re.sub(r"[^a-z0-9]+", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def _planner_strip_city_from_entity(entity):
    if not _planner_is_filled(entity):
        return ""

    parts = [p.strip() for p in str(entity).split(",")]

    if len(parts) >= 2:
        return ",".join(parts[:-1]).strip()

    return str(entity).strip()


def _planner_split_attractions(attraction_field):
    if not _planner_is_filled(attraction_field):
        return []

    return [
        p.strip()
        for p in str(attraction_field).split(";")
        if p.strip() and p.strip() != "-"
    ]


def _planner_parse_time_to_minutes(time_str):
    m = re.search(r"(\d{1,2}):(\d{2})", str(time_str))

    if not m:
        return None

    return int(m.group(1)) * 60 + int(m.group(2))


def _planner_extract_departure_arrival_minutes(transportation):
    if not _planner_is_filled(transportation):
        return None, None

    dep_match = re.search(
        r"Departure Time:\s*(\d{1,2}:\d{2})",
        str(transportation),
        flags=re.IGNORECASE,
    )

    arr_match = re.search(
        r"Arrival Time:\s*(\d{1,2}:\d{2})",
        str(transportation),
        flags=re.IGNORECASE,
    )

    dep = _planner_parse_time_to_minutes(dep_match.group(1)) if dep_match else None
    arr = _planner_parse_time_to_minutes(arr_match.group(1)) if arr_match else None

    return dep, arr


def _planner_parse_transportation_minutes(transportation):
    if not _planner_is_filled(transportation):
        return 0.0

    dep, arr = _planner_extract_departure_arrival_minutes(transportation)

    if dep is not None and arr is not None:
        if arr < dep:
            arr += 24 * 60

        return float(arr - dep)

    duration_match = re.search(
        r"duration:\s*(?:(\d+)\s*hours?)?\s*(?:(\d+)\s*mins?)?",
        str(transportation),
        flags=re.IGNORECASE,
    )

    if duration_match:
        total = 0

        if duration_match.group(1):
            total += int(duration_match.group(1)) * 60

        if duration_match.group(2):
            total += int(duration_match.group(2))

        return float(total)

    return 0.0


def _planner_parse_generated_plan_days(plan_text):
    if not plan_text or not isinstance(plan_text, str):
        return []

    text = plan_text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"^\s*Revised Travel Plan:\s*", "", text.strip(), flags=re.IGNORECASE)
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
            day[key] = _planner_clean_value(m.group(1)) if m else "-"

        parsed_days.append(day)

    return parsed_days


def _planner_extract_attraction_dwell_lookup(notebook_text):
    """
    Recover attraction dwell values from the plain-text AttractionSearch table
    written into Notebook.

    If a row cannot be recovered, the guardrail uses 90 minutes as a fallback.
    """

    lookup = {}

    if not notebook_text:
        return lookup

    for raw_line in str(notebook_text).splitlines():
        line = raw_line.rstrip()

        if "predicted_dwell_minutes" in line:
            continue

        has_dwell_marker = (
            "dbscan_episode_random_forest" in line
            or "medium_visit" in line
            or "short_visit" in line
            or "long_visit" in line
        )

        if not has_dwell_marker:
            continue

        parts = re.split(r"\s{2,}", line.strip())

        if len(parts) < 3:
            continue

        name = parts[0].strip()

        dwell = None

        for part in parts[1:]:
            try:
                value = float(part)

                if 15 <= value <= 240:
                    dwell = value
                    break

            except Exception:
                continue

        if name and dwell is not None:
            lookup[_planner_normalize_text(name)] = float(dwell)

    return lookup


def _planner_available_sightseeing_window(
    day,
    is_last_day=False,
    day_start_minutes=9 * 60,
    day_end_minutes=21 * 60,
    arrival_buffer_minutes=60.0,
    departure_buffer_minutes=120.0,
):
    current_city = str(day.get("current_city", "")).lower()
    transportation = str(day.get("transportation", "")).strip()
    has_transport = _planner_is_filled(transportation)
    is_from_to = "from" in current_city and "to" in current_city

    dep, arr = _planner_extract_departure_arrival_minutes(transportation)

    start = float(day_start_minutes)
    end = float(day_end_minutes)

    if has_transport or is_from_to:
        if arr is not None and not is_last_day:
            start = max(start, float(arr) + float(arrival_buffer_minutes))

        if dep is not None and is_last_day:
            end = min(end, float(dep) - float(departure_buffer_minutes))

    return max(0.0, end - start)


def _planner_infer_day_budget(day, is_last_day=False):
    current_city = str(day.get("current_city", "")).lower()
    transportation = str(day.get("transportation", "")).lower().strip()

    has_transport = transportation not in ["", "-"]
    is_from_to = "from" in current_city and "to" in current_city

    if is_last_day and (has_transport or is_from_to):
        return 360.0, "return_travel_day"

    if has_transport or is_from_to:
        return 480.0, "travel_day"

    return 600.0, "full_day"


def _planner_get_attraction_dwell(attraction_text, dwell_lookup, default=90.0):
    name = _planner_strip_city_from_entity(attraction_text)
    return dwell_lookup.get(_planner_normalize_text(name), float(default))


def _planner_compute_day_load(day, dwell_lookup, is_last_day):
    attraction_window = _planner_available_sightseeing_window(
        day,
        is_last_day=is_last_day,
    )

    day_budget, day_type = _planner_infer_day_budget(
        day,
        is_last_day=is_last_day,
    )

    attractions = _planner_split_attractions(day.get("attraction", "-"))

    attraction_dwell_total = sum(
        _planner_get_attraction_dwell(a, dwell_lookup) for a in attractions
    )

    local_movement = len(attractions) * 30.0
    attraction_load = attraction_dwell_total + local_movement

    meal_load = 0.0

    if _planner_is_filled(day.get("breakfast", "-")):
        meal_load += 49.3

    if _planner_is_filled(day.get("lunch", "-")):
        meal_load += 57.9

    if _planner_is_filled(day.get("dinner", "-")):
        meal_load += 63.9

    transport_minutes = _planner_parse_transportation_minutes(
        day.get("transportation", "-")
    )

    full_day_load = transport_minutes + meal_load + attraction_load

    return {
        "day_type": day_type,
        "day_budget": day_budget,
        "attractions": attractions,
        "attraction_window": attraction_window,
        "attraction_dwell_total": attraction_dwell_total,
        "local_movement": local_movement,
        "attraction_load": attraction_load,
        "meal_load": meal_load,
        "transport_minutes": transport_minutes,
        "full_day_load": full_day_load,
        "attraction_overload": max(0.0, attraction_load - attraction_window),
        "full_day_overload": max(0.0, full_day_load - day_budget),
    }


def build_temporal_repair_report(plan_text, notebook_text):
    days = _planner_parse_generated_plan_days(plan_text)
    dwell_lookup = _planner_extract_attraction_dwell_lookup(notebook_text)

    if not days:
        return {
            "needs_repair": False,
            "text_report": "Could not parse generated plan. No repair attempted.",
        }

    max_day = max(day.get("day", 0) for day in days)

    report_lines = []
    needs_repair = False

    for day in days:
        day_num = day.get("day")
        is_last_day = day_num == max_day

        load = _planner_compute_day_load(
            day=day,
            dwell_lookup=dwell_lookup,
            is_last_day=is_last_day,
        )

        day_needs_repair = (
            len(load["attractions"]) > 0
            and (load["attraction_overload"] > 0 or load["full_day_overload"] > 0)
        )

        if day_needs_repair:
            needs_repair = True

        report_lines.append(
            f"Day {day_num} ({load['day_type']}): "
            f"{len(load['attractions'])} attractions; "
            f"attraction_dwell={load['attraction_dwell_total']:.1f}; "
            f"local_movement={load['local_movement']:.1f}; "
            f"attraction_load={load['attraction_load']:.1f}; "
            f"available_sightseeing_window={load['attraction_window']:.1f}; "
            f"meal_load={load['meal_load']:.1f}; "
            f"transport={load['transport_minutes']:.1f}; "
            f"full_day_load={load['full_day_load']:.1f}; "
            f"day_budget={load['day_budget']:.1f}; "
            f"attraction_overload={load['attraction_overload']:.1f}; "
            f"full_day_overload={load['full_day_overload']:.1f}; "
            f"repair={'YES' if day_needs_repair else 'NO'}."
        )

    return {
        "needs_repair": needs_repair,
        "text_report": "\n".join(report_lines),
    }


def _planner_rebuild_plan_from_days(days, title="Travel Plan:"):
    lines = [title]

    for day in days:
        lines.append(f"Day {day.get('day')}:")
        lines.append(f"Current City: {day.get('current_city', '-')}")
        lines.append(f"Transportation: {day.get('transportation', '-')}")
        lines.append(f"Breakfast: {day.get('breakfast', '-')}")
        lines.append(f"Attraction: {day.get('attraction', '-')}")
        lines.append(f"Lunch: {day.get('lunch', '-')}")
        lines.append(f"Dinner: {day.get('dinner', '-')}")
        lines.append(f"Accommodation: {day.get('accommodation', '-')}")
        lines.append("")

    return "\n".join(lines).strip()


def _planner_choose_feasible_attractions(day, dwell_lookup, is_last_day):
    """
    Deterministically reduce Attraction field until it fits.

    This only changes Attraction. Meals, transport, accommodation, and city
    fields are not touched.
    """

    original_attractions = _planner_split_attractions(day.get("attraction", "-"))

    if not original_attractions:
        return "-"

    # If there is no sightseeing window, no attraction should remain.
    window = _planner_available_sightseeing_window(day, is_last_day=is_last_day)

    if window <= 0:
        return "-"

    day_budget, day_type = _planner_infer_day_budget(day, is_last_day=is_last_day)

    meal_load = 0.0

    if _planner_is_filled(day.get("breakfast", "-")):
        meal_load += 49.3

    if _planner_is_filled(day.get("lunch", "-")):
        meal_load += 57.9

    if _planner_is_filled(day.get("dinner", "-")):
        meal_load += 63.9

    transport_minutes = _planner_parse_transportation_minutes(
        day.get("transportation", "-")
    )

    available_after_fixed_items = max(0.0, day_budget - transport_minutes - meal_load)

    # Travel days are deliberately conservative.
    if day_type in ["travel_day", "return_travel_day"]:
        candidates = original_attractions[:1]
    else:
        candidates = original_attractions[:]

    while candidates:
        attraction_dwell = sum(
            _planner_get_attraction_dwell(a, dwell_lookup) for a in candidates
        )
        local_movement = len(candidates) * 30.0
        attraction_load = attraction_dwell + local_movement

        if attraction_load <= window and attraction_load <= available_after_fixed_items:
            return "; ".join(candidates)

        candidates = candidates[:-1]

    return "-"


def apply_deterministic_attraction_guardrail(plan_text, notebook_text, log_file=None):
    """
    Final deterministic cleanup after LLM generation/repair.

    Rules:
    - If a day has no sightseeing window, set Attraction to '-'.
    - If a travel day has too many attractions, keep at most one.
    - If attractions overload the day/window, remove attractions from the end
      until the day fits.
    - Do not change meals, transportation, accommodation, or current city.
    """

    days = _planner_parse_generated_plan_days(plan_text)

    if not days:
        return plan_text

    dwell_lookup = _planner_extract_attraction_dwell_lookup(notebook_text)

    max_day = max(day.get("day", 0) for day in days)

    changed = False
    guardrail_lines = []

    for day in days:
        day_num = day.get("day")
        is_last_day = day_num == max_day
        original_attraction = day.get("attraction", "-")

        if not _planner_is_filled(original_attraction):
            continue

        load_before = _planner_compute_day_load(
            day=day,
            dwell_lookup=dwell_lookup,
            is_last_day=is_last_day,
        )

        needs_guardrail = (
            load_before["attraction_window"] <= 0
            or (
                load_before["day_type"] in ["travel_day", "return_travel_day"]
                and len(load_before["attractions"]) > 1
            )
            or load_before["attraction_overload"] > 0
            or load_before["full_day_overload"] > 0
        )

        if not needs_guardrail:
            continue

        new_attraction = _planner_choose_feasible_attractions(
            day=day,
            dwell_lookup=dwell_lookup,
            is_last_day=is_last_day,
        )

        if new_attraction != original_attraction:
            day["attraction"] = new_attraction
            changed = True

            guardrail_lines.append(
                f"Day {day_num}: Attraction changed from "
                f"'{original_attraction}' to '{new_attraction}'. "
                f"Reason: day_type={load_before['day_type']}, "
                f"window={load_before['attraction_window']:.1f}, "
                f"attraction_overload={load_before['attraction_overload']:.1f}, "
                f"full_day_overload={load_before['full_day_overload']:.1f}."
            )

    if not changed:
        return plan_text

    final_plan = _planner_rebuild_plan_from_days(days, title="Travel Plan:")

    if log_file:
        log_file.write(
            "\n---------------Deterministic Attraction Guardrail\n"
            + "\n".join(guardrail_lines)
        )

    return final_plan


class ReactPlanner:
    """
    A question answering ReAct Agent.
    """

    def __init__(
        self,
        agent_prompt: PromptTemplate = react_planner_agent_prompt,
        model_name: str = "gpt-3.5-turbo-1106",
    ) -> None:
        self.agent_prompt = agent_prompt
        self.react_llm = ChatOpenAI(
            model_name=model_name,
            temperature=0,
            max_tokens=1024,
            openai_api_key=OPENAI_API_KEY,
            model_kwargs={"stop": ["Action", "Thought", "Observation"]},
        )
        self.env = ReactEnv()
        self.query = None
        self.max_steps = 30
        self.reset()
        self.finished = False
        self.answer = ""
        self.enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def run(self, text, query, reset=True) -> None:
        self.query = query
        self.text = text

        if reset:
            self.reset()

        while not (self.is_halted() or self.is_finished()):
            self.step()

        return self.answer, self.scratchpad

    def step(self) -> None:
        self.scratchpad += f"\nThought {self.curr_step}:"
        self.scratchpad += " " + self.prompt_agent()
        print(self.scratchpad.split("\n")[-1])

        self.scratchpad += f"\nAction {self.curr_step}:"
        action = self.prompt_agent()
        self.scratchpad += " " + action
        print(self.scratchpad.split("\n")[-1])

        self.scratchpad += f"\nObservation {self.curr_step}: "

        action_type, action_arg = parse_action(action)

        if action_type == "CostEnquiry":
            try:
                input_arg = eval(action_arg)

                if type(input_arg) != dict:
                    raise ValueError(
                        "The sub plan can not be parsed into json format, please check. Only one day plan is supported."
                    )

                observation = f"Cost: {self.env.run(input_arg)}"

            except SyntaxError:
                observation = "The sub plan can not be parsed into json format, please check."

            except ValueError as e:
                observation = str(e)

        elif action_type == "Finish":
            self.finished = True
            observation = "The plan is finished."
            self.answer = action_arg

        else:
            observation = f"Action {action_type} is not supported."

        self.curr_step += 1

        self.scratchpad += observation
        print(self.scratchpad.split("\n")[-1])

    def prompt_agent(self) -> str:
        while True:
            try:
                return format_step(
                    self.react_llm(
                        [HumanMessage(content=self._build_agent_prompt())]
                    ).content
                )

            except Exception:
                catch_openai_api_error()
                print(self._build_agent_prompt())
                print(len(self.enc.encode(self._build_agent_prompt())))
                time.sleep(5)

    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
            query=self.query,
            text=self.text,
            scratchpad=self.scratchpad,
        )

    def is_finished(self) -> bool:
        return self.finished

    def is_halted(self) -> bool:
        return (
            self.curr_step > self.max_steps
            or len(self.enc.encode(self._build_agent_prompt())) > 14000
        ) and not self.finished

    def reset(self) -> None:
        self.scratchpad = ""
        self.answer = ""
        self.curr_step = 1
        self.finished = False


class ReactReflectPlanner:
    """
    A question answering Self-Reflecting React Agent.
    """

    def __init__(
        self,
        agent_prompt: PromptTemplate = react_reflect_planner_agent_prompt,
        reflect_prompt: PromptTemplate = reflect_prompt,
        model_name: str = "gpt-3.5-turbo-1106",
    ) -> None:
        self.agent_prompt = agent_prompt
        self.reflect_prompt = reflect_prompt

        if model_name in ["gemini"]:
            self.react_llm = ChatGoogleGenerativeAI(
                temperature=0,
                model="gemini-pro",
                google_api_key=GOOGLE_API_KEY,
            )
            self.reflect_llm = ChatGoogleGenerativeAI(
                temperature=0,
                model="gemini-pro",
                google_api_key=GOOGLE_API_KEY,
            )

        else:
            self.react_llm = ChatOpenAI(
                model_name=model_name,
                temperature=0,
                max_tokens=1024,
                openai_api_key=OPENAI_API_KEY,
                model_kwargs={"stop": ["Action", "Thought", "Observation,'\n"]},
            )
            self.reflect_llm = ChatOpenAI(
                model_name=model_name,
                temperature=0,
                max_tokens=1024,
                openai_api_key=OPENAI_API_KEY,
                model_kwargs={"stop": ["Action", "Thought", "Observation,'\n"]},
            )

        self.model_name = model_name
        self.env = ReactReflectEnv()
        self.query = None
        self.max_steps = 30
        self.reset()
        self.finished = False
        self.answer = ""
        self.reflections: List[str] = []
        self.reflections_str: str = ""
        self.enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def run(self, text, query, reset=True) -> None:
        self.query = query
        self.text = text

        if reset:
            self.reset()

        while not (self.is_halted() or self.is_finished()):
            self.step()

            if self.env.is_terminated and not self.finished:
                self.reflect(ReflexionStrategy.REFLEXION)

        return self.answer, self.scratchpad

    def step(self) -> None:
        self.scratchpad += f"\nThought {self.curr_step}:"
        self.scratchpad += " " + self.prompt_agent()
        print(self.scratchpad.split("\n")[-1])

        self.scratchpad += f"\nAction {self.curr_step}:"
        action = self.prompt_agent()
        self.scratchpad += " " + action
        print(self.scratchpad.split("\n")[-1])

        self.scratchpad += f"\nObservation {self.curr_step}: "

        action_type, action_arg = parse_action(action)

        if action_type == "CostEnquiry":
            try:
                input_arg = eval(action_arg)

                if type(input_arg) != dict:
                    raise ValueError(
                        "The sub plan can not be parsed into json format, please check. Only one day plan is supported."
                    )

                observation = f"Cost: {self.env.run(input_arg)}"

            except SyntaxError:
                observation = "The sub plan can not be parsed into json format, please check."

            except ValueError as e:
                observation = str(e)

        elif action_type == "Finish":
            self.finished = True
            observation = "The plan is finished."
            self.answer = action_arg

        else:
            observation = f"Action {action_type} is not supported."

        self.curr_step += 1

        self.scratchpad += observation
        print(self.scratchpad.split("\n")[-1])

    def reflect(self, strategy: ReflexionStrategy) -> None:
        print("Reflecting...")

        if strategy == ReflexionStrategy.REFLEXION:
            self.reflections += [self.prompt_reflection()]
            self.reflections_str = format_reflections(self.reflections)
        else:
            raise NotImplementedError(f"Unknown reflection strategy: {strategy}")

        print(self.reflections_str)

    def prompt_agent(self) -> str:
        while True:
            try:
                if self.model_name in ["gemini"]:
                    return format_step(
                        self.react_llm.invoke(self._build_agent_prompt()).content
                    )

                return format_step(
                    self.react_llm(
                        [HumanMessage(content=self._build_agent_prompt())]
                    ).content
                )

            except Exception:
                catch_openai_api_error()
                print(self._build_agent_prompt())
                print(len(self.enc.encode(self._build_agent_prompt())))
                time.sleep(5)

    def prompt_reflection(self) -> str:
        while True:
            try:
                if self.model_name in ["gemini"]:
                    return format_step(
                        self.reflect_llm.invoke(
                            self._build_reflection_prompt()
                        ).content
                    )

                return format_step(
                    self.reflect_llm(
                        [HumanMessage(content=self._build_reflection_prompt())]
                    ).content
                )

            except Exception:
                catch_openai_api_error()
                print(self._build_reflection_prompt())
                print(len(self.enc.encode(self._build_reflection_prompt())))
                time.sleep(5)

    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
            query=self.query,
            text=self.text,
            scratchpad=self.scratchpad,
            reflections=self.reflections_str,
        )

    def _build_reflection_prompt(self) -> str:
        return self.reflect_prompt.format(
            query=self.query,
            text=self.text,
            scratchpad=self.scratchpad,
        )

    def is_finished(self) -> bool:
        return self.finished

    def is_halted(self) -> bool:
        return (
            self.curr_step > self.max_steps
            or len(self.enc.encode(self._build_agent_prompt())) > 14000
        ) and not self.finished

    def reset(self) -> None:
        self.scratchpad = ""
        self.answer = ""
        self.curr_step = 1
        self.finished = False
        self.reflections = []
        self.reflections_str = ""
        self.env.reset()


def format_step(step: str) -> str:
    return step.strip("\n").strip().replace("\n", "")


def parse_action(string):
    pattern = r"^(\w+)\[(.+)\]$"
    match = re.match(pattern, string)

    try:
        if match:
            action_type = match.group(1)
            action_arg = match.group(2)
            return action_type, action_arg

        return None, None

    except Exception:
        return None, None


def format_reflections(reflections: List[str], header: str = REFLECTION_HEADER) -> str:
    if reflections == []:
        return ""

    return header + "Reflections:\n- " + "\n- ".join(
        [r.strip() for r in reflections]
    )


# if __name__ == '__main__':