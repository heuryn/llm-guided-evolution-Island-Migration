#!/usr/bin/env python3
"""
Aggregate a fitness_diversity_summary.csv file by averaging the last N generations.

Example:
    python reduce_fitness_diversity.py deepseek-60-focus/fitness_diversity_summary.csv \
        --generations 3 --output deepseek-60-focus/fitness_diversity_summary_last3.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


FITNESS_COLUMNS: tuple[str, ...] = (
    "dominated_pareto_area",
    "crowding_total",
    "crowding_infinite",
    "crowding_min",
    "crowding_max",
    "crowding_mean",
    "crowding_median",
    "crowding_stdev",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Average the last N generations of a fitness diversity summary CSV."
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="Path to fitness_diversity_summary.csv produced by fitness_diversity_quantifier.py.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=3,
        help="Number of trailing generations to include in the average (default: 3).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional destination CSV for the aggregated row. Prints to stdout if omitted.",
    )
    return parser.parse_args()


def validate_columns(columns: Iterable[str]) -> None:
    missing = [col for col in FITNESS_COLUMNS if col not in columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    if df.empty:
        raise ValueError(f"Input CSV {args.input_csv} is empty.")

    validate_columns(df.columns)
    df = df.sort_values("generation")
    window = df.tail(args.generations).copy()

    result = {
        "input_csv": str(args.input_csv),
        "generations_averaged": len(window),
    }
    for column in FITNESS_COLUMNS:
        result[f"{column}_mean"] = window[column].mean()

    result_df = pd.DataFrame([result])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(args.output, index=False)
    else:
        csv_text = result_df.to_csv(index=False)
        print(csv_text.strip())


if __name__ == "__main__":
    main()
