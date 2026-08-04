"""
Temporal Commonsense Extension for TravelPlanner.

Purpose
-------
This file is intentionally separate from the original commonsense_constraint.py.

Use it when you want to report an enhanced commonsense layer without overwriting
or silently changing the original TravelPlanner benchmark metrics.

Recommended reporting:
    1. Original TravelPlanner metrics
    2. Temporal feasibility metrics from temporal_feasibility.py
    3. Temporal Commonsense Extension metrics from this file

This module operates on the day-level CSV produced by the enhanced
temporal_feasibility.py evaluator, not directly on raw generated plans.
That keeps the original TravelPlanner postprocessing/evaluation untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


TEMPORAL_COMMONSENSE_COLUMNS = [
    "attraction_temporally_feasible",
    "full_day_temporally_feasible",
    "feasible_complete_day",
]


OPTIONAL_DIAGNOSTIC_COLUMNS = [
    "balanced_day",
    "underplanned_day",
]


def load_temporal_day_details(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Temporal day details file not found: {path}")

    df = pd.read_csv(path)

    required = ["system", "sample_id", "day"] + TEMPORAL_COMMONSENSE_COLUMNS
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            "The day-level temporal file is missing required columns: "
            f"{missing}. Make sure you ran the enhanced temporal_feasibility.py."
        )

    return df


def compute_temporal_commonsense_from_day_df(day_df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """
    Compute TravelPlanner-style temporal commonsense micro/macro scores.

    Micro score:
        Each day-level temporal check counts as one constraint.

    Macro score:
        A plan passes if every required temporal commonsense check passes
        for every evaluated day in that plan.

    Required temporal commonsense checks:
        1. attraction_temporally_feasible
        2. full_day_temporally_feasible
        3. feasible_complete_day

    Diagnostic checks, reported but not included in the main macro pass:
        - balanced_day
        - underplanned_day
    """

    df = day_df.copy()

    for col in TEMPORAL_COMMONSENSE_COLUMNS + OPTIONAL_DIAGNOSTIC_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    constraint_rows = []

    for _, row in df.iterrows():
        for constraint in TEMPORAL_COMMONSENSE_COLUMNS:
            constraint_rows.append(
                {
                    "system": row["system"],
                    "sample_id": row["sample_id"],
                    "day": row["day"],
                    "constraint": constraint,
                    "passed": bool(row[constraint]),
                }
            )

    constraints_df = pd.DataFrame(constraint_rows)

    if len(constraints_df) == 0:
        summary = {
            "Temporal Commonsense Micro Pass Rate (%)": 0.0,
            "Temporal Commonsense Macro Pass Rate (%)": 0.0,
            "Temporal Commonsense Evaluated Plans": 0,
            "Temporal Commonsense Evaluated Days": 0,
        }
        return summary, constraints_df

    micro = constraints_df["passed"].mean() * 100

    plan_pass = (
        constraints_df
        .groupby("sample_id")["passed"]
        .all()
        .reset_index(name="temporal_commonsense_plan_pass")
    )

    macro = plan_pass["temporal_commonsense_plan_pass"].mean() * 100

    summary = {
        "Temporal Commonsense Micro Pass Rate (%)": float(micro),
        "Temporal Commonsense Macro Pass Rate (%)": float(macro),
        "Temporal Commonsense Evaluated Plans": int(plan_pass["sample_id"].nunique()),
        "Temporal Commonsense Evaluated Days": int(df[["sample_id", "day"]].drop_duplicates().shape[0]),
        "Attraction Temporal Feasibility Pass Rate (%)": float(
            df["attraction_temporally_feasible"].mean() * 100
        ),
        "Full-Day Temporal Feasibility Pass Rate (%)": float(
            df["full_day_temporally_feasible"].mean() * 100
        ),
        "Feasible-Complete Day Pass Rate (%)": float(
            df["feasible_complete_day"].mean() * 100
        ),
    }

    if "balanced_day" in df.columns:
        summary["Balanced Day Rate (%)"] = float(df["balanced_day"].mean() * 100)

    if "underplanned_day" in df.columns:
        summary["Under-planned Day Rate (%)"] = float(df["underplanned_day"].mean() * 100)

    if "temporal_utilisation_ratio" in df.columns:
        summary["Mean Temporal Utilisation Ratio"] = float(
            df["temporal_utilisation_ratio"].mean()
        )

    if "attraction_utilisation_ratio" in df.columns:
        summary["Mean Attraction Utilisation Ratio"] = float(
            df["attraction_utilisation_ratio"].mean()
        )

    return summary, constraints_df


def run_temporal_commonsense_extension(
    temporal_day_details_csv: str | Path,
    save_dir: str | Path | None = None,
    system_name: str | None = None,
) -> dict:
    """
    Convenience runner.

    Example:
        run_temporal_commonsense_extension(
            temporal_day_details_csv=r"C:\\...\\dwell_dbscan_temporal_day_details.csv",
            save_dir=r"C:\\...\\temporal_commonsense_outputs",
            system_name="dwell_dbscan",
        )
    """

    day_df = load_temporal_day_details(temporal_day_details_csv)

    if system_name is not None:
        day_df = day_df.copy()
        day_df["system"] = system_name

    summary, constraints_df = compute_temporal_commonsense_from_day_df(day_df)

    print("\n===== Temporal Commonsense Extension =====")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        constraints_df.to_csv(
            save_dir / "temporal_commonsense_constraint_details.csv",
            index=False,
        )

        with open(
            save_dir / "temporal_commonsense_summary.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(summary, f, indent=4)

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--temporal_day_details_csv", required=True)
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--system_name", default=None)

    args = parser.parse_args()

    run_temporal_commonsense_extension(
        temporal_day_details_csv=args.temporal_day_details_csv,
        save_dir=args.save_dir,
        system_name=args.system_name,
    )
