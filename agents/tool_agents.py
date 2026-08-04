import re
import string
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "tools/planner")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../tools/planner")))

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import importlib
from typing import List, Dict, Any

import tiktoken
from pandas import DataFrame
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.schema import HumanMessage
from prompts import zeroshot_react_agent_prompt
import json
import openai
import time
import pandas as pd
from tqdm import tqdm
from langchain_google_genai import ChatGoogleGenerativeAI
import argparse
from datasets import load_dataset


OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
GOOGLE_API_KEY = dummy_google_api_key = "EMPTY"

pd.options.display.max_info_columns = 200
os.environ["TIKTOKEN_CACHE_DIR"] = "./tmp"

actionMapping = {
    "FlightSearch": "flights",
    "AttractionSearch": "attractions",
    "GoogleDistanceMatrix": "googleDistanceMatrix",
    "AccommodationSearch": "accommodation",
    "RestaurantSearch": "restaurants",
    "Planner": "planner",
    "NotebookWrite": "notebook",
    "CitySearch": "cities",
}


class CityError(Exception):
    pass


class DateError(Exception):
    pass


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


class ReactAgent:
    def __init__(
        self,
        args,
        mode: str = "zero_shot",
        tools: List[str] = None,
        max_steps: int = 30,
        max_retries: int = 3,
        illegal_early_stop_patience: int = 3,
        react_llm_name="gpt-3.5-turbo-1106",
        planner_llm_name="gpt-3.5-turbo-1106",
        city_file_path="../database/background/citySet.txt",
    ) -> None:

        self.answer = ""
        self.max_steps = max_steps
        self.mode = mode

        self.react_name = react_llm_name
        self.planner_name = planner_llm_name

        if self.mode == "zero_shot":
            self.agent_prompt = zeroshot_react_agent_prompt

        self.json_log = []
        self.current_observation = ""
        self.current_data = None

        if "gpt-3.5" in react_llm_name:
            stop_list = ["\n"]
            self.max_token_length = 15000
            self.llm = ChatOpenAI(
                temperature=1,
                max_tokens=256,
                model_name=react_llm_name,
                openai_api_key=OPENAI_API_KEY,
                model_kwargs={"stop": stop_list},
            )

        elif "gpt-4" in react_llm_name:
            stop_list = ["\n"]
            self.max_token_length = 30000
            self.llm = ChatOpenAI(
                temperature=0,
                max_tokens=256,
                model_name=react_llm_name,
                openai_api_key=OPENAI_API_KEY,
                model_kwargs={"stop": stop_list},
            )

        elif react_llm_name in ["mistral-7B-32K"]:
            stop_list = ["\n"]
            self.max_token_length = 30000
            self.llm = ChatOpenAI(
                temperature=0,
                max_tokens=256,
                openai_api_key="EMPTY",
                openai_api_base="http://localhost:8301/v1",
                model_name="gpt-3.5-turbo",
                model_kwargs={"stop": stop_list},
            )

        elif react_llm_name in ["mixtral"]:
            stop_list = ["\n"]
            self.max_token_length = 30000
            self.llm = ChatOpenAI(
                temperature=0,
                max_tokens=256,
                openai_api_key="EMPTY",
                openai_api_base="http://localhost:8501/v1",
                model_name="gpt-3.5-turbo",
                model_kwargs={"stop": stop_list},
            )

        elif react_llm_name in ["ChatGLM3-6B-32K"]:
            stop_list = ["\n"]
            self.max_token_length = 30000
            self.llm = ChatOpenAI(
                temperature=0,
                max_tokens=256,
                openai_api_key="EMPTY",
                openai_api_base="http://localhost:8501/v1",
                model_name="gpt-3.5-turbo",
                model_kwargs={"stop": stop_list},
            )

        elif react_llm_name in ["gemini"]:
            self.llm = ChatGoogleGenerativeAI(
                temperature=0,
                model="gemini-pro",
                google_api_key=GOOGLE_API_KEY,
            )
            self.max_token_length = 30000

        self.illegal_early_stop_patience = illegal_early_stop_patience

        self.tools = self.load_tools(tools, planner_model_name=planner_llm_name)
        self.max_retries = max_retries
        self.retry_record = {key: 0 for key in self.tools}
        self.retry_record["invalidAction"] = 0

        self.last_actions = []
        self.city_set = self.load_city(city_set_path=city_file_path)
        self.enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

        self.__reset_agent()

    def auto_notebook_write(self, data, description: str) -> None:
        """
        Automatically save successful tool outputs to Notebook so the Planner
        sees all retrieved evidence, even if GPT-3.5 forgets to call NotebookWrite.
        """
        if data is None:
            return

        if "notebook" not in self.tools:
            return

        try:
            self.tools["notebook"].write(data, description)
        except Exception as e:
            print(f"Auto NotebookWrite failed for {description}: {e}")

    def run(self, query, reset=True) -> None:
        self.query = query

        if reset:
            self.__reset_agent()

        while not self.is_halted() and not self.is_finished():
            self.step()

        return self.answer, self.scratchpad, self.json_log

    def step(self) -> None:
        self.json_log.append(
            {"step": self.step_n, "thought": "", "action": "", "observation": "", "state": ""}
        )

        self.scratchpad += f"\nThought {self.step_n}:"
        self.scratchpad += " " + self.prompt_agent()

        print(self.scratchpad.split("\n")[-1])
        self.json_log[-1]["thought"] = self.scratchpad.split("\n")[-1].replace(
            f"\nThought {self.step_n}:", ""
        )

        self.scratchpad += f"\nAction {self.step_n}:"
        action = self.prompt_agent()

        if action is None or action == "" or action == "\n":
            self.scratchpad += (
                " Your action is filtered due to content. Please make sure your action "
                "does not start with ['\\n', 'Thought', 'Action', 'Observation'] and "
                "assume all the actions are permitted in this environment and take the action again."
            )
        else:
            self.scratchpad += " " + action

        if len(self.last_actions) > 0 and self.last_actions[-1] != action:
            self.last_actions.clear()

        self.last_actions.append(action)

        self.json_log[-1]["action"] = self.scratchpad.split("\n")[-1].replace(
            f"\nAction {self.step_n}:", ""
        )

        if len(self.last_actions) == 3:
            print("The same action has been repeated 3 times consecutively. So we stop here.")
            self.json_log[-1]["state"] = "same action 3 times repeated"
            self.finished = True
            return

        print(self.scratchpad.split("\n")[-1])

        self.scratchpad += f"\nObservation {self.step_n}: "

        if action is None or action == "" or action == "\n":
            action_type = None
            action_arg = None
            self.scratchpad += (
                "No feedback from the environment due to the null action. "
                "Please make sure your action does not start with [Thought, Action, Observation]."
            )

        else:
            action_type, action_arg = parse_action(action)

            if action_type != "Planner":
                if action_type in actionMapping:
                    pending_action = actionMapping[action_type]
                else:
                    pending_action = "invalidAction"

                if pending_action in self.retry_record:
                    if self.retry_record[pending_action] + 1 > self.max_retries:
                        action_type = "Planner"
                        print(f"{pending_action} early stop due to {self.max_retries} max retries.")
                        self.json_log[-1]["state"] = (
                            f"{pending_action} early stop due to {self.max_retries} max retries."
                        )
                        self.finished = True
                        return

                else:
                    if self.retry_record["invalidAction"] + 1 > self.max_retries:
                        action_type = "Planner"
                        print(f"invalidAction Early stop due to {self.max_retries} max retries.")
                        self.json_log[-1]["state"] = (
                            f"invalidAction early stop due to {self.max_retries} max retries."
                        )
                        self.finished = True
                        return

            if action_type == "FlightSearch":
                try:
                    origin = action_arg.split(", ")[0]
                    destination = action_arg.split(", ")[1]
                    date = action_arg.split(", ")[2]

                    if (
                        validate_date_format(date)
                        and validate_city_format(origin, self.city_set)
                        and validate_city_format(destination, self.city_set)
                    ):
                        self.scratchpad = self.scratchpad.replace(
                            to_string(self.current_data).strip(),
                            "Masked due to limited length. Make sure the data has been written in Notebook.",
                        )

                        self.current_data = self.tools["flights"].run(origin, destination, date)

                        self.auto_notebook_write(
                            self.current_data,
                            f"Auto-saved FlightSearch result: {action_arg}",
                        )

                        self.current_observation = str(to_string(self.current_data))
                        self.scratchpad += self.current_observation
                        self.__reset_record()
                        self.json_log[-1]["state"] = "Successful"

                except DateError:
                    self.retry_record["flights"] += 1
                    self.current_observation = f"'{action_arg.split(', ')[2]}' is not in the format YYYY-MM-DD"
                    self.scratchpad += self.current_observation
                    self.json_log[-1]["state"] = "Illegal args. DateError"

                except ValueError as e:
                    self.retry_record["flights"] += 1
                    self.current_observation = str(e)
                    self.scratchpad += str(e)
                    self.json_log[-1]["state"] = "Illegal args. City Error"

                except Exception as e:
                    print(e)
                    self.retry_record["flights"] += 1
                    self.current_observation = "Illegal Flight Search. Please try again."
                    self.scratchpad += self.current_observation
                    self.json_log[-1]["state"] = "Illegal args. Other Error"

            elif action_type == "AttractionSearch":
                try:
                    if validate_city_format(action_arg, self.city_set):
                        self.scratchpad = self.scratchpad.replace(
                            to_string(self.current_data).strip().strip(),
                            "Masked due to limited length. Make sure the data has been written in Notebook.",
                        )

                        self.current_data = self.tools["attractions"].run(action_arg)

                        self.auto_notebook_write(
                            self.current_data,
                            f"Auto-saved AttractionSearch result: {action_arg}",
                        )

                        self.current_observation = to_string(self.current_data).strip("\n").strip()
                        self.scratchpad += self.current_observation
                        self.__reset_record()
                        self.json_log[-1]["state"] = "Successful"

                except ValueError as e:
                    self.retry_record["attractions"] += 1
                    self.current_observation = str(e)
                    self.scratchpad += str(e)
                    self.json_log[-1]["state"] = "Illegal args. City Error"

                except Exception as e:
                    print(e)
                    self.retry_record["attractions"] += 1
                    self.current_observation = "Illegal Attraction Search. Please try again."
                    self.scratchpad += self.current_observation
                    self.json_log[-1]["state"] = "Illegal args. Other Error"

            elif action_type == "AccommodationSearch":
                try:
                    if validate_city_format(action_arg, self.city_set):
                        self.scratchpad = self.scratchpad.replace(
                            to_string(self.current_data).strip().strip(),
                            "Masked due to limited length. Make sure the data has been written in Notebook.",
                        )

                        self.current_data = self.tools["accommodations"].run(action_arg)

                        self.auto_notebook_write(
                            self.current_data,
                            f"Auto-saved AccommodationSearch result: {action_arg}",
                        )

                        self.current_observation = to_string(self.current_data).strip("\n").strip()
                        self.scratchpad += self.current_observation
                        self.__reset_record()
                        self.json_log[-1]["state"] = "Successful"

                except ValueError as e:
                    self.retry_record["accommodations"] += 1
                    self.current_observation = str(e)
                    self.scratchpad += str(e)
                    self.json_log[-1]["state"] = "Illegal args. City Error"

                except Exception as e:
                    print(e)
                    self.retry_record["accommodations"] += 1
                    self.current_observation = "Illegal Accommodation Search. Please try again."
                    self.scratchpad += self.current_observation
                    self.json_log[-1]["state"] = "Illegal args. Other Error"

            elif action_type == "RestaurantSearch":
                try:
                    if validate_city_format(action_arg, self.city_set):
                        self.scratchpad = self.scratchpad.replace(
                            to_string(self.current_data).strip().strip(),
                            "Masked due to limited length. Make sure the data has been written in Notebook.",
                        )

                        self.current_data = self.tools["restaurants"].run(action_arg)

                        self.auto_notebook_write(
                            self.current_data,
                            f"Auto-saved RestaurantSearch result: {action_arg}",
                        )

                        self.current_observation = to_string(self.current_data).strip()
                        self.scratchpad += self.current_observation
                        self.__reset_record()
                        self.json_log[-1]["state"] = "Successful"

                except ValueError as e:
                    self.retry_record["restaurants"] += 1
                    self.current_observation = str(e)
                    self.scratchpad += str(e)
                    self.json_log[-1]["state"] = "Illegal args. City Error"

                except Exception as e:
                    print(e)
                    self.retry_record["restaurants"] += 1
                    self.current_observation = "Illegal Restaurant Search. Please try again."
                    self.scratchpad += self.current_observation
                    self.json_log[-1]["state"] = "Illegal args. Other Error"

            elif action_type == "CitySearch":
                try:
                    self.scratchpad = self.scratchpad.replace(
                        to_string(self.current_data).strip(),
                        "Masked due to limited length. Make sure the data has been written in Notebook.",
                    )

                    self.current_data = self.tools["cities"].run(action_arg)

                    self.auto_notebook_write(
                        self.current_data,
                        f"Auto-saved CitySearch result: {action_arg}",
                    )

                    self.current_observation = to_string(self.current_data).strip()
                    self.scratchpad += self.current_observation
                    self.__reset_record()
                    self.json_log[-1]["state"] = "Successful"

                except ValueError as e:
                    self.retry_record["cities"] += 1
                    self.current_observation = str(e)
                    self.scratchpad += str(e)
                    self.json_log[-1]["state"] = "Illegal args. State Error"

                except Exception as e:
                    print(e)
                    self.retry_record["cities"] += 1
                    self.current_observation = "Illegal City Search. Please try again."
                    self.scratchpad += self.current_observation
                    self.json_log[-1]["state"] = "Illegal args. Other Error"

            elif action_type == "GoogleDistanceMatrix":
                try:
                    origin = action_arg.split(", ")[0]
                    destination = action_arg.split(", ")[1]
                    mode = action_arg.split(", ")[2]

                    self.scratchpad = self.scratchpad.replace(
                        to_string(self.current_data).strip(),
                        "Masked due to limited length. Make sure the data has been written in Notebook.",
                    )

                    self.current_data = self.tools["googleDistanceMatrix"].run(
                        origin,
                        destination,
                        mode,
                    )

                    self.auto_notebook_write(
                        self.current_data,
                        f"Auto-saved GoogleDistanceMatrix result: {action_arg}",
                    )

                    self.current_observation = to_string(self.current_data)
                    self.scratchpad += self.current_observation
                    self.__reset_record()
                    self.json_log[-1]["state"] = "Successful"

                except Exception as e:
                    print(e)
                    self.retry_record["googleDistanceMatrix"] += 1
                    self.current_observation = "Illegal GoogleDistanceMatrix. Please try again."
                    self.scratchpad += self.current_observation
                    self.json_log[-1]["state"] = "Illegal args. Other Error"

            elif action_type == "NotebookWrite":
                try:
                    self.scratchpad = self.scratchpad.replace(
                        to_string(self.current_data).strip(),
                        "Masked due to limited length. Make sure the data has been written in Notebook.",
                    )

                    self.current_observation = str(
                        self.tools["notebook"].write(self.current_data, action_arg)
                    )
                    self.scratchpad += self.current_observation
                    self.__reset_record()
                    self.json_log[-1]["state"] = "Successful"

                except Exception as e:
                    print(e)
                    self.retry_record["notebook"] += 1
                    self.current_observation = f"{e}"
                    self.scratchpad += self.current_observation
                    self.json_log[-1]["state"] = "Illegal args. Other Error"

            elif action_type == "Planner":
                self.current_observation = str(
                    self.tools["planner"].run(
                        str(self.tools["notebook"].list_all()),
                        action_arg,
                    )
                )
                self.scratchpad += self.current_observation
                self.answer = self.current_observation
                self.__reset_record()
                self.json_log[-1]["state"] = "Successful"

            else:
                self.retry_record["invalidAction"] += 1
                self.current_observation = (
                    "Invalid Action. Valid Actions are FlightSearch[Departure City, Destination City, Date] / "
                    "AccommodationSearch[City] / RestaurantSearch[City] / NotebookWrite[Short Description] / "
                    "AttractionSearch[City] / CitySearch[State] / GoogleDistanceMatrix[Origin, Destination, Mode] "
                    "and Planner[Query]."
                )
                self.scratchpad += self.current_observation
                self.json_log[-1]["state"] = "invalidAction"

        if action is None or action == "" or action == "\n":
            print(f"Observation {self.step_n}: No feedback from the environment due to the null action.")
            self.json_log[-1]["observation"] = "No feedback from the environment due to the null action."
        else:
            print(f"Observation {self.step_n}: " + self.current_observation + "\n")
            self.json_log[-1]["observation"] = self.current_observation

        self.step_n += 1

        if action_type and action_type == "Planner" and self.retry_record["planner"] == 0:
            self.finished = True
            self.answer = self.current_observation
            self.step_n += 1
            return

    def prompt_agent(self) -> str:
        while True:
            try:
                if self.react_name == "gemini":
                    request = format_step(
                        self.llm.invoke(
                            self._build_agent_prompt(),
                            stop=["\n"],
                        ).content
                    )
                else:
                    request = format_step(
                        self.llm([HumanMessage(content=self._build_agent_prompt())]).content
                    )
                return request

            except Exception:
                catch_openai_api_error()
                print(self._build_agent_prompt())
                print(len(self.enc.encode(self._build_agent_prompt())))
                time.sleep(5)

    def _build_agent_prompt(self) -> str:
        if self.mode == "zero_shot":
            return self.agent_prompt.format(
                query=self.query,
                scratchpad=self.scratchpad,
            )

    def is_finished(self) -> bool:
        return self.finished

    def is_halted(self) -> bool:
        return (
            (self.step_n > self.max_steps)
            or (len(self.enc.encode(self._build_agent_prompt())) > self.max_token_length)
        ) and not self.finished

    def __reset_agent(self) -> None:
        self.step_n = 1
        self.finished = False
        self.answer = ""
        self.scratchpad: str = ""
        self.__reset_record()
        self.json_log = []
        self.current_observation = ""
        self.current_data = None
        self.last_actions = []

        if "notebook" in self.tools:
            self.tools["notebook"].reset()

    def __reset_record(self) -> None:
        self.retry_record = {key: 0 for key in self.retry_record}
        self.retry_record["invalidAction"] = 0

    def load_tools(self, tools: List[str], planner_model_name=None) -> Dict[str, Any]:
        tools_map = {}

        for tool_name in tools:
            module = importlib.import_module(f"tools.{tool_name}.apis")

            if tool_name == "planner" and planner_model_name is not None:
                tools_map[tool_name] = getattr(
                    module,
                    tool_name[0].upper() + tool_name[1:],
                )(model_name=planner_model_name)
            else:
                tools_map[tool_name] = getattr(
                    module,
                    tool_name[0].upper() + tool_name[1:],
                )()

        return tools_map

    def load_city(self, city_set_path: str) -> List[str]:
        city_set = []
        lines = open(city_set_path, "r").read().strip().split("\n")

        for unit in lines:
            city_set.append(unit)

        return city_set


gpt2_enc = tiktoken.encoding_for_model("text-davinci-003")


def parse_action(string):
    pattern = r"^(\w+)\[(.+)\]$"
    match = re.match(pattern, string)

    try:
        if match:
            action_type = match.group(1)
            action_arg = match.group(2)
            return action_type, action_arg
        else:
            return None, None

    except Exception:
        return None, None


def format_step(step: str) -> str:
    return step.strip("\n").strip().replace("\n", "")


def truncate_scratchpad(scratchpad: str, n_tokens: int = 1600, tokenizer=gpt2_enc) -> str:
    lines = scratchpad.split("\n")
    observations = filter(lambda x: x.startswith("Observation"), lines)
    observations_by_tokens = sorted(observations, key=lambda x: len(tokenizer.encode(x)))

    while len(gpt2_enc.encode("\n".join(lines))) > n_tokens:
        largest_observation = observations_by_tokens.pop(-1)
        ind = lines.index(largest_observation)
        lines[ind] = largest_observation.split(":")[0] + ": [truncated wikipedia excerpt]"

    return "\n".join(lines)


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the|usd)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def EM(answer, key) -> bool:
    return normalize_answer(str(answer)) == normalize_answer(str(key))


def remove_observation_lines(text, step_n):
    pattern = re.compile(rf"^Observation {step_n}.*", re.MULTILINE)
    return pattern.sub("", text)


def validate_date_format(date_str: str) -> bool:
    pattern = r"^\d{4}-\d{2}-\d{2}$"

    if not re.match(pattern, date_str):
        raise DateError

    return True


def validate_city_format(city_str: str, city_set: list) -> bool:
    if city_str not in city_set:
        raise ValueError(f"{city_str} is not valid city in {str(city_set)}.")

    return True


def parse_args_string(s: str) -> dict:
    segments = s.split(",")
    result = {}

    for segment in segments:
        if "contains" in segment:
            if "~contains" in segment:
                key, value = segment.split("~contains")
                operator = "~contains"
            else:
                key, value = segment.split("contains")
                operator = "contains"
        elif "<=" in segment:
            key, value = segment.split("<=")
            operator = "<="
        elif ">=" in segment:
            key, value = segment.split(">=")
            operator = ">="
        elif "=" in segment:
            key, value = segment.split("=")
            operator = "="
        else:
            continue

        key = key.strip()
        value = value.strip().strip("'")
        result[key] = (operator, value)

    return result


def to_string(data) -> str:
    if data is not None:
        if type(data) == DataFrame:
            return data.to_string(index=False)
        else:
            return str(data)
    else:
        return str(None)


if __name__ == "__main__":
    tools_list = [
        "notebook",
        "flights",
        "attractions",
        "accommodations",
        "restaurants",
        "googleDistanceMatrix",
        "planner",
        "cities",
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("--set_type", type=str, default="validation")
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo-1106")
    parser.add_argument("--output_dir", type=str, default="./")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of validation/test examples to run. Use 1 for smoke test, 100 for main validation run.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip samples whose generated_plan_{i}.json already exists and contains results for this model.",
    )

    args = parser.parse_args()

    if args.set_type == "validation":
        query_data_list = load_dataset("osunlp/TravelPlanner", "validation")["validation"]
    elif args.set_type == "test":
        query_data_list = load_dataset("osunlp/TravelPlanner", "test")["test"]
    else:
        raise ValueError(f"Unsupported set_type: {args.set_type}")

    query_data_list = query_data_list.select(range(args.num_samples))
    numbers = [i for i in range(1, len(query_data_list) + 1)]

    output_subdir = os.path.join(args.output_dir, args.set_type)
    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir)

    result_key = f"{args.model_name}_two-stage_results"

    agent = ReactAgent(
        None,
        tools=tools_list,
        max_steps=30,
        react_llm_name=args.model_name,
        planner_llm_name=args.model_name,
    )

    with get_openai_callback() as cb:
        for number in tqdm(numbers[:]):
            query = query_data_list[number - 1]["query"]
            output_file = os.path.join(output_subdir, f"generated_plan_{number}.json")

            if args.resume and os.path.exists(output_file):
                try:
                    existing_result = json.load(open(output_file, encoding="utf-8"))

                    if (
                        isinstance(existing_result, list)
                        and len(existing_result) > 0
                        and result_key in existing_result[-1]
                        and existing_result[-1][result_key]
                    ):
                        print(f"Skipping sample {number}: existing result found.")
                        continue

                except Exception as e:
                    print(f"Could not read existing file for sample {number}; regenerating. Error: {e}")

            if not os.path.exists(output_file):
                result = [{}]
            else:
                result = json.load(open(output_file, encoding="utf-8"))

                if not isinstance(result, list) or len(result) == 0:
                    result = [{}]

            while True:
                planner_results, scratchpad, action_log = agent.run(query)
                if planner_results is not None:
                    break

            if planner_results == "Max Token Length Exceeded.":
                result[-1][f"{args.model_name}_two-stage_results_logs"] = scratchpad
                result[-1][f"{args.model_name}_two-stage_results"] = "Max Token Length Exceeded."

                if action_log:
                    action_log[-1]["state"] = "Max Token Length of Planner Exceeded."

                result[-1][f"{args.model_name}_two-stage_action_logs"] = action_log

            else:
                result[-1][f"{args.model_name}_two-stage_results_logs"] = scratchpad
                result[-1][f"{args.model_name}_two-stage_results"] = planner_results
                result[-1][f"{args.model_name}_two-stage_action_logs"] = action_log

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4)

    print(cb)