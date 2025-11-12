#!/usr/bin/env python3
"""
Compute fitness diversity statistics for each generation of an LLM-guided
evolution run. The script scans the `global_data/global_gen_*.pkl` files that
are produced by runs launched via `pace_ice_island_controller.sbatch`, extracts
fitness tuples, and emits per-generation statistics along with a CSV summary.

Example:
    uv run fitness_diversity_quantifier.py deepseek_normal_3 --include-hist --front-only
"""

from __future__ import annotations

import argparse
import csv
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, stdev
from typing import Iterable, List, Sequence, Tuple


from deap import base, creator
from deap.tools import emo
from deap.tools import sortNondominated

# Default to the project's standard fitness weights. We avoid importing the
# project module directly to keep this script self-contained and to prevent
# environments without those dependencies from failing during import.
FITNESS_WEIGHTS = (1.0, -1.0)


@dataclass
class ObjectiveStats:
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    median: float | None = None
    stdev: float | None = None


@dataclass
class CrowdingStats:
    count: int = 0
    infinite_count: int = 0
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    median: float | None = None
    stdev: float | None = None

@dataclass
class LocalDensityStats:
    count: int = 0
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    median: float | None = None
    stdev: float | None = None


@dataclass
class GenerationSummary:
    generation: int
    file_path: Path
    total_entries: int
    valid_entries: int
    invalid_entries: int
    objectives: List[ObjectiveStats]
    crowding: CrowdingStats
    local_density: LocalDensityStats | None = None
    perf: LocalDensityStats | None = None
    combined: LocalDensityStats | None = None


# === performance tuning constants ===
#BASELINE_ACC = 0.928    # baseline accuracy you want to beat
BASELINE_ACC = 1.0
ACC_FLOOR = 0.80        # worst plausible accuracy (used to stabilize normalization)
P_LOG_MIN_DEFAULT = 4.0  # default log10(params) min (e.g., 10^4)
P_LOG_MAX_DEFAULT = 8.0  # default log10(params) max (e.g., 10^8)

# how much to weight performance vs diversity when forming a combined score
# alpha = weight on performance (0 = pure diversity, 1 = pure performance)
ALPHA_PERF = 0.15

# axis weights when computing weighted Euclidean distance for performance
W_ACC = 0.6
W_PARAM = 1 - W_ACC

EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantify fitness diversity per generation.")
    parser.add_argument(
        "run_directory",
        type=Path,
        help="Path to the run folder produced by pace_ice_island_controller (contains global_data/).",
    )
    parser.add_argument(
        "--include-hist",
        action="store_true",
        help="Include entries from GLOBAL_DATA_HIST alongside GLOBAL_DATA.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of generations to process (processed in ascending order).",
    )
    parser.add_argument(
        "--csv-name",
        type=str,
        default="fitness_diversity_summary.csv",
        help="Name of the CSV file to write inside the run directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process and print results without writing the CSV file.",
    )
    parser.add_argument(
        "--diversity-metric",
        choices=["crowding", "local_density"],
        default="local_density",
        help="Choose diversity metric: DEAP crowding distance or grid-based local density.",
    )
    parser.add_argument(
        "--grid-bins",
        type=int,
        default=10,
        help="Number of bins per objective for local-density grid.",
    )
    parser.add_argument(
        "--front-only",
        action="store_true",
        help="Compute stats using only the first (nondominated) Pareto front."
    )
    return parser.parse_args()


def extract_generation(filename: str) -> int:
    """Pull the generation number from a name like global_gen_12.pkl."""
    try:
        return int(filename.split("_")[2].split(".")[0])
    except (IndexError, ValueError):
        raise ValueError(f"Unable to extract generation from filename: {filename}")


def load_pickle(path: Path) -> dict:
    print(f"[DEBUG] Loading pickle: {path}")
    with path.open("rb") as handle:
        return pickle.load(handle)


def iter_fitness_values(
    dataset: dict | None,
) -> Iterable[Tuple[float, ...]]:
    if not dataset:
        return []

    valid_values: List[Tuple[float, ...]] = []
    for gene_id, attributes in dataset.items():
        fitness = attributes.get("fitness")
        if is_valid_fitness_tuple(fitness):
            valid_values.append(tuple(float(x) for x in fitness))
        else:
            print(f"[DEBUG] Skipping invalid fitness for gene {gene_id}: {fitness}")
    return valid_values


def is_valid_fitness_tuple(candidate: object) -> bool:
    if not isinstance(candidate, (tuple, list)):
        return False
    if not candidate:
        return False
    for value in candidate:
        if value is None:
            return False
        if not isinstance(value, (int, float)):
            return False
        if not math.isfinite(value):
            return False
    return True


def compute_objective_stats(values: Sequence[Tuple[float, ...]]) -> List[ObjectiveStats]:
    if not values:
        return []

    num_objectives = len(values[0])
    stats_per_objective: List[ObjectiveStats] = []
    for idx in range(num_objectives):
        column = [row[idx] for row in values]
        stats = ObjectiveStats(
            minimum=min(column) if column else None,
            maximum=max(column) if column else None,
            mean=safe_statistic(mean, column),
            median=safe_statistic(median, column),
            stdev=safe_statistic(stdev, column),
        )
        stats_per_objective.append(stats)
        print(
            "[DEBUG] Objective %d stats -> min: %s max: %s mean: %s median: %s stdev: %s"
            % (
                idx,
                stats.minimum,
                stats.maximum,
                stats.mean,
                stats.median,
                stats.stdev,
            )
        )
    return stats_per_objective


def safe_statistic(func, values: Sequence[float]) -> float | None:
    if not values:
        return None
    if func is stdev and len(values) < 2:
        return 0.0
    try:
        return func(values)
    except Exception as exc:  # Catch rare statistics errors.
        print(f"[DEBUG] Statistic {func.__name__} failed on {values}: {exc}")
        return None
    
def normalize_objectives_01(values: Sequence[Tuple[float, ...]]) -> List[Tuple[float, ...]]: #normalization for performance scores
    """Min–max normalize each objective column to [0,1]. Returns a new list."""
    if not values:
        return []
    m = len(values[0])
    cols = list(zip(*values))  # m columns
    mins = [min(c) for c in cols]
    maxs = [max(c) for c in cols]
    rngs = [mx - mn for mn, mx in zip(mins, maxs)]
    out: List[Tuple[float, ...]] = []
    for row in values:
        norm = []
        for j, x in enumerate(row):
            r = rngs[j]
            if r <= 0:
                norm.append(0.0)  # constant column -> collapse to 0
            else:
                norm.append((x - mins[j]) / r)
        out.append(tuple(norm))
    return out


def grid_cell_indices(norm_values: Sequence[Tuple[float, ...]], bins: int) -> List[Tuple[int, ...]]: #creating cells
    """Map each normalized point to a grid cell index (tuple of ints)."""
    if not norm_values:
        return []
    m = len(norm_values[0])
    cells: List[Tuple[int, ...]] = []
    for row in norm_values:
        idxs = []
        for j in range(m):
            # Clamp to [0, 1-eps) then floor into [0, bins-1]
            x = min(max(row[j], 0.0), 1.0 - 1e-12)
            idxs.append(int(x * bins))
        cells.append(tuple(idxs))
    return cells

def acc_component(acc: float, acc_floor: float = ACC_FLOOR, baseline: float = BASELINE_ACC) -> float:
    """
    Normalized accuracy deficit in [0,1], where 0 = meets/exceeds baseline and 1 = at/below floor.
    Uses a hinge: no penalty for exceeding the baseline.
    """
    deficit = max(0.0, baseline - acc)
    denom = max(EPS, baseline - acc_floor)
    return min(1.0, max(0.0, deficit / denom))


def params_component(params: float, p_log_min: float = P_LOG_MIN_DEFAULT, p_log_max: float = P_LOG_MAX_DEFAULT) -> float:
    """
    Normalized params in [0,1] operating in log10 space. Lower is better.
    """
    p_log = math.log10(max(params, 1.0))
    denom = max(EPS, p_log_max - p_log_min)
    return min(1.0, max(0.0, (p_log - p_log_min) / denom))


def perf_score_from_components(acc_norm: float, params_norm: float, w_acc: float = W_ACC, w_param: float = W_PARAM) -> float:
    """
    Weighted Euclidean distance on the normalized deficit axes, flipped to [0,1]
    so that 1.0 is best (closest to origin).
    """
    a = w_acc * acc_norm
    p = w_param * params_norm
    norm = math.hypot(a, p)
    denom = math.hypot(w_acc, w_param)  # max possible norm if acc_norm=params_norm=1
    frac = norm / max(EPS, denom)
    return 1.0 - min(1.0, frac)


def compute_perf_for_individual(acc_value: float, params_value: float,
                                acc_floor: float = ACC_FLOOR, baseline: float = BASELINE_ACC,
                                p_log_min: float = P_LOG_MIN_DEFAULT, p_log_max: float = P_LOG_MAX_DEFAULT,
                                w_acc: float = W_ACC, w_param: float = W_PARAM) -> float:
    """
    Convenience wrapper: compute normalized components then return perf score in [0,1].
    """
    acc_norm = acc_component(acc_value, acc_floor, baseline)
    p_norm = params_component(params_value, p_log_min, p_log_max)
    return perf_score_from_components(acc_norm, p_norm, w_acc, w_param)

def infer_perf_scaling(values: Sequence[Tuple[float, ...]],
                       default_p_log_min: float = P_LOG_MIN_DEFAULT,
                       default_p_log_max: float = P_LOG_MAX_DEFAULT,
                       default_acc_floor: float = ACC_FLOOR) -> tuple[float, float, float]:
    """
    Given raw fitness tuples (assumes order: (accuracy, params, ...optional...)),
    return (acc_floor, p_log_min, p_log_max) suitable for normalization.
    Falls back to defaults when data is missing or degenerate.
    """
    if not values:
        return (default_acc_floor, default_p_log_min, default_p_log_max)

    # Extract first two dimensions (accuracy, params) defensively
    accs: List[float] = []
    params_raw: List[float] = []
    for row in values:
        try:
            accs.append(float(row[0]))
            params_raw.append(float(row[1]))
        except Exception:
            # skip malformed rows
            continue

    if not accs or not params_raw:
        return (default_acc_floor, default_p_log_min, default_p_log_max)

    # acc_floor: keep at most default but allow if data has lower minima
    min_acc = min(accs)
    acc_floor = min(default_acc_floor, min_acc)

    # params: compute log10 bounds; ensure defaults are included so we always have a non-zero range
    p_logs = [math.log10(max(p, 1.0)) for p in params_raw]
    p_log_min = min(p_logs + [default_p_log_min])
    p_log_max = max(p_logs + [default_p_log_max])

    # If the observed range is tiny, expand slightly to avoid zero denom
    if abs(p_log_max - p_log_min) < 1e-6:
        p_log_min = min(p_log_min, default_p_log_min)
        p_log_max = max(p_log_max, default_p_log_max + 1.0)

    return (acc_floor, p_log_min, p_log_max)

def select_first_front(values: Sequence[Tuple[float, ...]]) -> List[Tuple[float, ...]]:
    if not values:
        return []
    ensure_creator_classes()
    inds = []
    for f in values:
        fit = creator.FitnessDiversity(f)
        ind = creator.IndividualDiversity([])
        ind.fitness = fit
        inds.append(ind)
    fronts = sortNondominated(inds, k=len(inds), first_front_only=True)
    first = fronts[0] if fronts else []
    return [tuple(ind.fitness.values) for ind in first]


def ensure_creator_classes() -> None:
    # Avoid recreating classes if this module is executed multiple times.
    if not hasattr(creator, "FitnessDiversity"):
        creator.create("FitnessDiversity", base.Fitness, weights=FITNESS_WEIGHTS)
    if not hasattr(creator, "IndividualDiversity"):
        creator.create("IndividualDiversity", list, fitness=creator.FitnessDiversity)


def compute_crowding_stats(values: Sequence[Tuple[float, ...]]) -> CrowdingStats:
    if len(values) < 2:
        return CrowdingStats(count=len(values))

    ensure_creator_classes()

    individuals: List[object] = []
    for fitness_values in values:
        fitness = creator.FitnessDiversity(fitness_values)
        individual = creator.IndividualDiversity([])
        individual.fitness = fitness
        individuals.append(individual)

    emo.assignCrowdingDist(individuals)
    distances = [ind.fitness.crowding_dist for ind in individuals]
    finite_distances = [dist for dist in distances if math.isfinite(dist)]
    inf_count = len(distances) - len(finite_distances)

    stats = CrowdingStats(
        count=len(distances),
        infinite_count=inf_count,
        minimum=min(finite_distances) if finite_distances else None,
        maximum=max(finite_distances) if finite_distances else None,
        mean=safe_statistic(mean, finite_distances),
        median=safe_statistic(median, finite_distances),
        stdev=safe_statistic(stdev, finite_distances),
    )

    print(
        "[DEBUG] Crowding stats -> count: %d finite: %d infinite: %d mean: %s median: %s"
        % (
            stats.count,
            len(finite_distances),
            stats.infinite_count,
            stats.mean,
            stats.median,
        )
    )
    return stats

def compute_local_density_stats(
    values: Sequence[Tuple[float, ...]],
    bins_per_dim: int,
) -> tuple[LocalDensityStats, List[float], List[float], List[float]]:
    """
    Compute local-density + performance + combined scores.

    Returns:
      - LocalDensityStats (summary of div scores),
      - div_scores: List[float] (1/(1+n_c)) per input order,
      - perf_scores: List[float] per input order (in [0,1]),
      - combined_scores: List[float] per input order (in [0,1]).
    Assumes each value tuple has accuracy at index 0 and params at index 1.
    """
    if not values:
        return (LocalDensityStats(count=0), [], [], [])

    # --- grid-based diversity (same as before) ---
    norm = normalize_objectives_01(values)
    cells = grid_cell_indices(norm, bins_per_dim)

    from collections import Counter
    counts = Counter(cells)

    div_scores = [1.0 / (1 + counts[cell]) for cell in cells]

    # --- infer safe scaling for performance ---
    acc_floor, p_log_min, p_log_max = infer_perf_scaling(values)

    # compute perf_scores for each individual (assumes acc at idx 0, params at idx 1)
    perf_scores: List[float] = []
    for row in values:
        try:
            acc_val = float(row[0])
            params_val = float(row[1])
        except Exception:
            # malformed; give worst perf (0.0)
            perf_scores.append(0.0)
            continue
        perf = compute_perf_for_individual(
            acc_value=acc_val,
            params_value=params_val,
            acc_floor=acc_floor,
            baseline=BASELINE_ACC,
            p_log_min=p_log_min,
            p_log_max=p_log_max,
            w_acc=W_ACC,
            w_param=W_PARAM,
        )
        perf_scores.append(perf)

    # --- combined score ---
    combined_scores = [
        (1.0 - ALPHA_PERF) * d + ALPHA_PERF * p for d, p in zip(div_scores, perf_scores)
    ]

    # summary stats for div (keep same shape as LocalDensityStats)
    stats = LocalDensityStats(
        count=len(div_scores),
        minimum=min(div_scores) if div_scores else None,
        maximum=max(div_scores) if div_scores else None,
        mean=safe_statistic(mean, div_scores),
        median=safe_statistic(median, div_scores),
        stdev=safe_statistic(stdev, div_scores),
    )

    print(
        "[DEBUG] Local density -> count: %d cells: %d div_min: %s div_max: %s div_mean: %s"
        % (
            stats.count,
            len(counts),
            stats.minimum,
            stats.maximum,
            stats.mean,
        )
    )

    return (stats, div_scores, perf_scores, combined_scores)

def summarize_generation(
    generation: int,
    file_path: Path,
    data: dict,
    include_hist: bool,
    diversity_metric: str = "crowding",
    grid_bins: int = 10,
    front_only: bool = False,
) -> GenerationSummary:
    global_data = data.get("GLOBAL_DATA", {})
    hist_data = data.get("GLOBAL_DATA_HIST", {}) if include_hist else {}

    global_values = list(iter_fitness_values(global_data))
    hist_values = list(iter_fitness_values(hist_data))
    combined_values = global_values + hist_values
    if front_only:  # thread the flag down from main/parse_args
        combined_values = select_first_front(combined_values)

    total_entries = len(global_data) + (len(hist_data) if include_hist else 0)
    valid_entries = len(combined_values)
    invalid_entries = total_entries - valid_entries

    print(
        "[DEBUG] Generation %d -> total entries: %d valid: %d invalid: %d"
        % (generation, total_entries, valid_entries, invalid_entries)
    )

    objective_stats = compute_objective_stats(combined_values)

    # Default placeholders
    crowding_stats = CrowdingStats() #turn off (not used anymore)
    local_density_stats = None
    perf_stats = None
    combined_stats = None

    ''' UNUSED AT THE MOMENT
    if diversity_metric == "crowding":
        # existing behavior
        crowding_stats = compute_crowding_stats(combined_values)
    '''
    if diversity_metric == "local_density":
        # new behavior: get div/perf/combined per-individual lists and summary
        ld_stats, div_scores, perf_scores, combined_scores = compute_local_density_stats(
            combined_values, bins_per_dim=grid_bins
        )
        local_density_stats = ld_stats
        # create simple LocalDensityStats-like summaries for perf & combined
        perf_stats = LocalDensityStats(
            count=len(perf_scores),
            minimum=min(perf_scores) if perf_scores else None,
            maximum=max(perf_scores) if perf_scores else None,
            mean=safe_statistic(mean, perf_scores),
            median=safe_statistic(median, perf_scores),
            stdev=safe_statistic(stdev, perf_scores),
        )
        combined_stats = LocalDensityStats(
            count=len(combined_scores),
            minimum=min(combined_scores) if combined_scores else None,
            maximum=max(combined_scores) if combined_scores else None,
            mean=safe_statistic(mean, combined_scores),
            median=safe_statistic(median, combined_scores),
            stdev=safe_statistic(stdev, combined_scores),
        )
        # For backward compatibility, compute crowding as well (optional)
        #crowding_stats = compute_crowding_stats(combined_values)
    else:
        #print(f"[DEBUG] Unknown diversity metric '{diversity_metric}'. Falling back to crowding.")
        print(f"[DEBUG] Unknown diversity metric '{diversity_metric}'. Using local density.")
        #crowding_stats = compute_crowding_stats(combined_values)
        ld_stats, div_scores, perf_scores, combined_scores = compute_local_density_stats(
            combined_values, bins_per_dim=grid_bins
        )
        local_density_stats = ld_stats
        perf_stats = LocalDensityStats(
            count=len(perf_scores),
            minimum=min(perf_scores) if perf_scores else None,
            maximum=max(perf_scores) if perf_scores else None,
            mean=safe_statistic(mean, perf_scores),
            median=safe_statistic(median, perf_scores),
            stdev=safe_statistic(stdev, perf_scores),
        )
        combined_stats = LocalDensityStats(
            count=len(combined_scores),
            minimum=min(combined_scores) if combined_scores else None,
            maximum=max(combined_scores) if combined_scores else None,
            mean=safe_statistic(mean, combined_scores),
            median=safe_statistic(median, combined_scores),
            stdev=safe_statistic(stdev, combined_scores),
        )

    return GenerationSummary(
        generation=generation,
        file_path=file_path,
        total_entries=total_entries,
        valid_entries=valid_entries,
        invalid_entries=invalid_entries,
        objectives=objective_stats,
        crowding=crowding_stats,
        local_density=local_density_stats,
        perf=perf_stats,
        combined=combined_stats,
    )

def summarize_run(
    run_directory: Path,
    include_hist: bool,
    limit: int | None,
    diversity_metric: str = "crowding",
    grid_bins: int = 10,
    front_only: bool = False,
) -> List[GenerationSummary]:
    global_data_dir = run_directory / "global_data"
    if not global_data_dir.is_dir():
        raise FileNotFoundError(f"No global_data/ directory found at {global_data_dir}")

    generation_files = sorted(global_data_dir.glob("global_gen_*.pkl"), key=lambda path: extract_generation(path.name))
    if limit is not None:
        generation_files = generation_files[:limit]

    print(f"[DEBUG] Found {len(generation_files)} generation files (limit={limit}).")

    summaries: List[GenerationSummary] = []
    for file_path in generation_files:
        generation_id = extract_generation(file_path.name)
        data = load_pickle(file_path)
        summary = summarize_generation(
            generation_id,
            file_path,
            data,
            include_hist,
            diversity_metric=diversity_metric,
            grid_bins=grid_bins,
            front_only=front_only,
        )
        summaries.append(summary)
    return summaries



def print_summary_table(summaries: Sequence[GenerationSummary]) -> None:
    if not summaries:
        print("[DEBUG] No generation summaries to display.")
        return

    # Determine the maximum number of objectives across generations.
    max_objectives = max((len(summary.objectives) for summary in summaries), default=0)

    header_cells = [
        "Gen",
        "Total",
        "Valid",
        "Invalid",
        "Pareto_Area",
        "Dom_Pareto_Area",
    ]
    for idx in range(max_objectives):
        header_cells.extend(
            [
                f"Obj{idx}_Min",
                f"Obj{idx}_Max",
                f"Obj{idx}_Mean",
                f"Obj{idx}_Median",
                f"Obj{idx}_Std",
            ]
        )

    # crowding
    '''
    header_cells.extend(
        [
            "Crowd_Finite",
            "Crowd_Inf",
            "Crowd_Min",
            "Crowd_Max",
            "Crowd_Mean",
            "Crowd_Median",
            "Crowd_Std",
        ]
    )
    '''

    # local_density summaries
    header_cells.extend(
        [
            "LocalDiv_Count",
            "LocalDiv_Min",
            "LocalDiv_Max",
            "LocalDiv_Mean",
            "LocalDiv_Median",
            "LocalDiv_Std",
        ]
    )

    # perf summaries
    header_cells.extend(
        [
            "Perf_Count",
            "Perf_Min",
            "Perf_Max",
            "Perf_Mean",
            "Perf_Median",
            "Perf_Std",
        ]
    )

    # combined summaries
    header_cells.extend(
        [
            "Comb_Count",
            "Comb_Min",
            "Comb_Max",
            "Comb_Mean",
            "Comb_Median",
            "Comb_Std",
        ]
    )

    print("[DEBUG] Summary Table:")
    print("\t".join(header_cells))

    for summary in summaries:
        row: List[str] = [
            str(summary.generation),
            str(summary.total_entries),
            str(summary.valid_entries),
            str(summary.invalid_entries),
            format_float(summary.pareto_front_area),
            format_float(summary.dominated_pareto_area),
        ]
        for idx in range(max_objectives):
            stats = summary.objectives[idx] if idx < len(summary.objectives) else ObjectiveStats()
            row.extend(
                [
                    format_float(stats.minimum),
                    format_float(stats.maximum),
                    format_float(stats.mean),
                    format_float(stats.median),
                    format_float(stats.stdev),
                ]
            )

        # crowding
        '''
        row.extend(
            [
                str(summary.crowding.count - summary.crowding.infinite_count),
                str(summary.crowding.infinite_count),
                format_float(summary.crowding.minimum),
                format_float(summary.crowding.maximum),
                format_float(summary.crowding.mean),
                format_float(summary.crowding.median),
                format_float(summary.crowding.stdev),
            ]
        )
        '''

        # local_density (may be None)
        ld = summary.local_density
        if ld is None:
            row.extend(["", "", "", "", "", ""])
        else:
            row.extend(
                [
                    str(ld.count),
                    format_float(ld.minimum),
                    format_float(ld.maximum),
                    format_float(ld.mean),
                    format_float(ld.median),
                    format_float(ld.stdev),
                ]
            )

        # perf summary
        pf = summary.perf
        if pf is None:
            row.extend(["", "", "", "", "", ""])
        else:
            row.extend(
                [
                    str(pf.count),
                    format_float(pf.minimum),
                    format_float(pf.maximum),
                    format_float(pf.mean),
                    format_float(pf.median),
                    format_float(pf.stdev),
                ]
            )

        # combined summary
        cb = summary.combined
        if cb is None:
            row.extend(["", "", "", "", "", ""])
        else:
            row.extend(
                [
                    str(cb.count),
                    format_float(cb.minimum),
                    format_float(cb.maximum),
                    format_float(cb.mean),
                    format_float(cb.median),
                    format_float(cb.stdev),
                ]
            )

        print("\t".join(row))


def write_csv(summaries: Sequence[GenerationSummary], destination: Path) -> None:
    if not summaries:
        print(f"[DEBUG] No summaries available; skipping CSV write to {destination}.")
        return

    max_objectives = max((len(summary.objectives) for summary in summaries), default=0)
    fieldnames = [
        "generation",
        "file_path",
        "total_entries",
        "valid_entries",
        "invalid_entries",
        "pareto_front_area",
        "dominated_pareto_area",
    ]
    for idx in range(max_objectives):
        fieldnames.extend(
            [
                f"objective_{idx}_min",
                f"objective_{idx}_max",
                f"objective_{idx}_mean",
                f"objective_{idx}_median",
                f"objective_{idx}_stdev",
            ]
        )

    # crowding fields
    '''
    fieldnames.extend(
        [
            "crowding_total",
            "crowding_infinite",
            "crowding_min",
            "crowding_max",
            "crowding_mean",
            "crowding_median",
            "crowding_stdev",
        ]
    )
    '''

    # local density fields
    fieldnames.extend(
        [
            "localdiv_count",
            "localdiv_min",
            "localdiv_max",
            "localdiv_mean",
            "localdiv_median",
            "localdiv_stdev",
        ]
    )

    # perf summary fields
    fieldnames.extend(
        [
            "perf_count",
            "perf_min",
            "perf_max",
            "perf_mean",
            "perf_median",
            "perf_stdev",
        ]
    )

    # combined summary fields
    fieldnames.extend(
        [
            "combined_count",
            "combined_min",
            "combined_max",
            "combined_mean",
            "combined_median",
            "combined_stdev",
        ]
    )

    print(f"[DEBUG] Writing CSV to: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            row = {
                "generation": summary.generation,
                "file_path": summary.file_path.name,
                "total_entries": summary.total_entries,
                "valid_entries": summary.valid_entries,
                "invalid_entries": summary.invalid_entries,
                #"crowding_total": summary.crowding.count,
                #"crowding_infinite": summary.crowding.infinite_count,
                #"crowding_min": summary.crowding.minimum,
                #"crowding_max": summary.crowding.maximum,
                #"crowding_mean": summary.crowding.mean,
                #"crowding_median": summary.crowding.median,
                #"crowding_stdev": summary.crowding.stdev,
            }

            # objectives
            for idx in range(max_objectives):
                stats = summary.objectives[idx] if idx < len(summary.objectives) else ObjectiveStats()
                row[f"objective_{idx}_min"] = stats.minimum
                row[f"objective_{idx}_max"] = stats.maximum
                row[f"objective_{idx}_mean"] = stats.mean
                row[f"objective_{idx}_median"] = stats.median
                row[f"objective_{idx}_stdev"] = stats.stdev

            # localdiv summary
            ld = summary.local_density
            if ld is None:
                row["localdiv_count"] = None
                row["localdiv_min"] = None
                row["localdiv_max"] = None
                row["localdiv_mean"] = None
                row["localdiv_median"] = None
                row["localdiv_stdev"] = None
            else:
                row["localdiv_count"] = ld.count
                row["localdiv_min"] = ld.minimum
                row["localdiv_max"] = ld.maximum
                row["localdiv_mean"] = ld.mean
                row["localdiv_median"] = ld.median
                row["localdiv_stdev"] = ld.stdev

            # perf summary
            pf = summary.perf
            if pf is None:
                row["perf_count"] = None
                row["perf_min"] = None
                row["perf_max"] = None
                row["perf_mean"] = None
                row["perf_median"] = None
                row["perf_stdev"] = None
            else:
                row["perf_count"] = pf.count
                row["perf_min"] = pf.minimum
                row["perf_max"] = pf.maximum
                row["perf_mean"] = pf.mean
                row["perf_median"] = pf.median
                row["perf_stdev"] = pf.stdev

            # combined summary
            cb = summary.combined
            if cb is None:
                row["combined_count"] = None
                row["combined_min"] = None
                row["combined_max"] = None
                row["combined_mean"] = None
                row["combined_median"] = None
                row["combined_stdev"] = None
            else:
                row["combined_count"] = cb.count
                row["combined_min"] = cb.minimum
                row["combined_max"] = cb.maximum
                row["combined_mean"] = cb.mean
                row["combined_median"] = cb.median
                row["combined_stdev"] = cb.stdev

            writer.writerow(row)


def format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def main() -> None:
    args = parse_args()
    run_directory = args.run_directory.resolve()
    print(f"[DEBUG] Run directory resolved to: {run_directory}")

    summaries = summarize_run(
        run_directory,
        include_hist=args.include_hist,
        limit=args.limit,
        diversity_metric=args.diversity_metric,
        grid_bins=args.grid_bins,
        front_only=args.front_only,
    )

    print_summary_table(summaries)

    if not args.dry_run:
        csv_path = run_directory / args.csv_name
        write_csv(summaries, csv_path)
        print(f"[DEBUG] CSV written to {csv_path}")
    else:
        print("[DEBUG] Dry-run mode enabled; CSV not written.")


if __name__ == "__main__":
    main()
