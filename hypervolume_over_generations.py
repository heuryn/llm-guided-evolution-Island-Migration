import argparse
import csv
import os
import pickle
import re
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np


GEN_RE = re.compile(r"checkpoint_gen_(\d+)\.pkl$")


def parse_ref_point(ref_point: str) -> Tuple[float, float]:
    try:
        x_str, y_str = ref_point.split(",")
        return float(x_str), float(y_str)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--ref-point must be in the form \"R1,R2\" (comma-separated floats)"
        ) from exc


def extract_generation(path: Path) -> int:
    match = GEN_RE.search(path.name)
    if not match:
        return -1
    return int(match.group(1))


def find_checkpoints(run_dir: Path) -> List[Path]:
    """Return a sorted list of checkpoint files for the run."""
    direct = sorted(
        p for p in run_dir.glob("checkpoint_gen_*.pkl") if GEN_RE.match(p.name)
    )
    if direct:
        return direct

    island_dirs = sorted(
        p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("island_")
    )
    for island_dir in island_dirs:
        checkpoints = sorted(
            p for p in island_dir.glob("checkpoint_gen_*.pkl") if GEN_RE.match(p.name)
        )
        if checkpoints:
            print(f"Using checkpoints from {island_dir}")
            return checkpoints

    return []


def pareto_front(points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Return non-dominated points assuming maximization for both objectives."""
    front: List[Tuple[float, float]] = []
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if i == j:
                continue
            if q[0] >= p[0] and q[1] >= p[1] and (q[0] > p[0] or q[1] > p[1]):
                dominated = True
                break
        if not dominated:
            front.append(p)
    return front


def hypervolume_2d_max(points: Iterable[Tuple[float, float]],
                       ref: Tuple[float, float]) -> float:
    """
    Compute 2D hypervolume for a maximization problem with respect to ref.

    Assumes:
      - Both objectives are to be maximized.
      - ref is a "worse" point that is dominated by all relevant points.
    """
    points = list(points)
    if not points:
        return 0.0

    rx, ry = ref

    # Keep only points that dominate the reference; others don't contribute.
    dom_points = [(x, y) for (x, y) in points if x >= rx and y >= ry]
    if not dom_points:
        return 0.0

    # Transform to minimization by negating both coordinates.
    pts_min = [(-x, -y) for (x, y) in dom_points]
    ref_min = (-rx, -ry)

    # 2D hypervolume for minimization with ref at top-right.
    pts_min.sort(key=lambda t: t[0])  # sort by first objective (ascending)
    rxm, rym = ref_min

    hv = 0.0
    best_y = float("inf")
    prev_x = pts_min[0][0]

    for i, (x, y) in enumerate(pts_min):
        # Update best (smallest) y among points seen so far.
        best_y = min(best_y, y)

        # Next x boundary: either the next point or the reference.
        next_x = pts_min[i + 1][0] if i + 1 < len(pts_min) else rxm

        width = max(0.0, next_x - prev_x)
        height = max(0.0, rym - best_y)

        hv += width * height
        prev_x = next_x

    return hv



def load_population_points(path: Path) -> List[Tuple[float, float]]:
    with path.open("rb") as f:
        data = pickle.load(f)
    population = data.get("population", [])
    points: List[Tuple[float, float]] = []
    for ind in population:
        fitness = getattr(ind, "fitness", None)
        values = getattr(fitness, "values", None)
        if values and len(values) >= 2:
            f_max, f_min = values[0], values[1]
            g1 = float(f_max)
            g2 = -float(f_min)
            points.append((g1, g2))
    return points


def write_csv(run_dir: Path, rows: List[Tuple[int, float]]) -> None:
    csv_path = run_dir / "hypervolume_per_generation.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "hypervolume"])
        writer.writerows(rows)
    print(f"Wrote {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute hypervolume per generation for a run."
    )
    parser.add_argument("run_path", type=str, help="Path to run folder")
    parser.add_argument(
        "--ref-point",
        required=True,
        type=parse_ref_point,
        help='Reference point in "R1,R2" format (after transformation g1=f_max, g2=-f_min)',
    )
    args = parser.parse_args()

    run_dir = Path(os.path.expanduser(args.run_path)).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run folder not found: {run_dir}")

    ref_point = args.ref_point
    checkpoints = find_checkpoints(run_dir)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint_gen_*.pkl files found under {run_dir}")

    # order by generation number
    checkpoints.sort(key=extract_generation)

    hv_rows: List[Tuple[int, float]] = []
    for ckpt in checkpoints:
        gen = extract_generation(ckpt)
        points = load_population_points(ckpt)
        front = pareto_front(points)
        hv = hypervolume_2d_max(front, ref_point)
        hv_rows.append((gen, hv))
        print(f"Gen {gen}: {len(points)} pts, {len(front)} on Pareto front, HV={hv:.4f}")

    hv_values = [hv for _, hv in hv_rows]
    hv_final = hv_values[-1]
    hv_auc = float(np.sum(hv_values))
    hv_mean = float(np.mean(hv_values))

    print(f"\nHV_final: {hv_final}")
    print(f"HV_AUC:   {hv_auc}")
    print(f"HV_mean:  {hv_mean}")

    write_csv(run_dir, hv_rows)


if __name__ == "__main__":
    main()
