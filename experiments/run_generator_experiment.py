#!/usr/bin/env python3
"""Generator experiment runner — implements all 4 experiments from generator_experiment_plan.md.

Usage:
    python experiments/run_generator_experiment.py --all
    python experiments/run_generator_experiment.py --exp 1
    python experiments/run_generator_experiment.py --exp 2
    python experiments/run_generator_experiment.py --exp 3
    python experiments/run_generator_experiment.py --exp 4
    python experiments/run_generator_experiment.py --exp 1 --no-llm     # classical baselines only
    python experiments/run_generator_experiment.py --exp 1 --runs 3     # quick test
    python experiments/run_generator_experiment.py --exp 1 --model qwen2.5-coder:3b
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ── make all packages importable when run from repo root ──────────────────────
_repo = Path(__file__).parents[1]
for _pkg in ("core", "benchmarks", "classical", "llm", "evaluation",
             "refiner", "generator", "selector"):
    sys.path.insert(0, str(_repo / "packages" / _pkg / "src"))

from agenticpathplanner.benchmarks.cec2024.loader import CEC2024Loader
from agenticpathplanner.benchmarks.cec2024.metrics import evaluate_path, aggregate_runs
from agenticpathplanner.classical.astar import AStarPlanner
from agenticpathplanner.classical.theta_star import ThetaStarPlanner
from agenticpathplanner.classical.apf import APFRefiner
from agenticpathplanner.classical.rrt_star import RRTStarPlanner
from agenticpathplanner.classical.mpc import MPCPlanner
from agenticpathplanner.core.agent import AgentContext
from agenticpathplanner.core.types import MapInstance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Mission strings ────────────────────────────────────────────────────────────

MISSIONS: dict[str, dict[str, str]] = {
    "efficiency": {
        "simple":  (
            "Navigate from start to goal in a small open environment. "
            "Minimise total path distance. There are no special constraints."
        ),
        "medium":  (
            "Navigate a corridor maze quickly. Time is critical. "
            "Find the shortest valid path. Do not sacrifice length for clearance."
        ),
        "complex": (
            "Navigate a dense cluttered environment under time pressure. "
            "Find any valid path as quickly as possible; favour direct routes."
        ),
    },
    "safety": {
        "simple":  (
            "Navigate safely with maximum clearance from walls. "
            "Never pass closer than 2 cells to any obstacle. "
            "Accept longer paths to stay safe."
        ),
        "medium":  (
            "Navigate a corridor maze with maximum obstacle clearance. "
            "The platform is fragile; prioritise clearance over path length."
        ),
        "complex": (
            "Navigate a dense cluttered environment with extreme caution. "
            "Maximise the minimum distance to any obstacle at every waypoint."
        ),
    },
    "smoothness": {
        "simple":  (
            "Generate a smooth path with minimal direction changes. "
            "The vehicle cannot turn sharply. Prefer wide arcs over tight corners."
        ),
        "medium":  (
            "Navigate the corridor maze with the fewest and gentlest turns. "
            "Avoid zigzag paths. Smooth continuous movement is required."
        ),
        "complex": (
            "Navigate the cluttered environment as smoothly as possible. "
            "Minimise total turning angle. Straight or gently curved segments are preferred."
        ),
    },
    "multi": {
        "simple":  (
            "Find a path that is both efficient and safe. Stay reasonably close to "
            "the straight-line distance while keeping at least 1 cell from obstacles."
        ),
        "medium":  (
            "Navigate the corridor maze balancing speed and safety. Do not hug walls. "
            "Aim for a path that is both short and clear of obstacles."
        ),
        "complex": (
            "Navigate dense clutter efficiently without sacrificing safety entirely. "
            "Prioritise finding a valid path, but avoid unnecessary obstacle proximity."
        ),
    },
}

GENERIC_MISSION = "Find the shortest collision-free path from start to goal."

ALL_MAP_IDS = ["cec2024_s01", "cec2024_m01", "cec2024_c01"]

# ── Data containers ────────────────────────────────────────────────────────────

@dataclass
class RunRecord:
    """One row of raw result data — one LLM call or one classical run."""
    experiment: str
    method: str
    map_id: str
    difficulty: str
    mission_type: str
    run_index: int
    model: str
    temperature: float
    # path metrics
    valid: bool = False
    plr: float = float("inf")
    sd: float = 0.0
    sm: float = float("inf")
    path_length: float = 0.0
    elapsed_s: float = 0.0
    # generator-specific
    code_generated: bool = False
    code_valid: bool = False
    plan_fn_found: bool = False
    path_reached_goal: bool = False
    error: str = ""


# ── Core helpers ───────────────────────────────────────────────────────────────

def _load_maps(map_ids: list[str]) -> list[MapInstance]:
    loader = CEC2024Loader()
    return [loader.load(m) for m in map_ids]


def _run_classical(
    planner,
    instance: MapInstance,
    experiment: str,
    mission_type: str,
    run_index: int,
) -> RunRecord:
    t0 = time.perf_counter()
    result = planner.solve(instance)
    elapsed = time.perf_counter() - t0
    m = evaluate_path(result.path, instance.grid, instance.start, instance.goal)
    return RunRecord(
        experiment=experiment,
        method=planner.name,
        map_id=instance.map_id,
        difficulty=instance.difficulty,
        mission_type=mission_type,
        run_index=run_index,
        model="deterministic",
        temperature=0.0,
        valid=m.valid,
        plr=m.plr if m.valid else float("inf"),
        sd=m.sd if m.valid else 0.0,
        sm=m.sm if m.valid else float("inf"),
        path_length=m.path_length,
        elapsed_s=elapsed,
        code_generated=False,
        code_valid=False,
        plan_fn_found=False,
        path_reached_goal=result.success,
    )


def _run_generator(
    agent,
    instance: MapInstance,
    mission: str,
    experiment: str,
    mission_type: str,
    run_index: int,
    model: str,
    temperature: float,
) -> RunRecord:
    ctx = AgentContext(instance=instance, mission=mission)
    t0 = time.perf_counter()
    try:
        result = agent.run(ctx)
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        log.warning("GeneratorAgent raised: %s", exc)
        return RunRecord(
            experiment=experiment, method="generator", map_id=instance.map_id,
            difficulty=instance.difficulty, mission_type=mission_type,
            run_index=run_index, model=model, temperature=temperature,
            elapsed_s=elapsed, error=str(exc),
        )
    elapsed = time.perf_counter() - t0

    m = evaluate_path(result.path, instance.grid, instance.start, instance.goal)
    code = result.generated_code or ""

    return RunRecord(
        experiment=experiment,
        method="generator",
        map_id=instance.map_id,
        difficulty=instance.difficulty,
        mission_type=mission_type,
        run_index=run_index,
        model=model,
        temperature=temperature,
        valid=m.valid,
        plr=m.plr if m.valid else float("inf"),
        sd=m.sd if m.valid else 0.0,
        sm=m.sm if m.valid else float("inf"),
        path_length=m.path_length,
        elapsed_s=elapsed,
        code_generated=bool(code),
        code_valid=bool(code),       # if code was stored, exec succeeded
        plan_fn_found=bool(result.path is not None),
        path_reached_goal=result.success,
    )


def _save_results(records: list[RunRecord], out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}.json"
    csv_path  = out_dir / f"{stem}.csv"

    rows = [asdict(r) for r in records]
    json_path.write_text(json.dumps(rows, indent=2))

    if rows:
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    log.info("Saved %d records → %s", len(records), out_dir)


def _aggregate_records(records: list[RunRecord]) -> list[dict]:
    """Group records and compute SR, PLR_mean/std, SD_mean, SM_mean."""
    from collections import defaultdict
    import statistics

    groups: dict[tuple, list[RunRecord]] = defaultdict(list)
    for r in records:
        key = (r.experiment, r.method, r.map_id, r.mission_type, r.model, r.temperature)
        groups[key].append(r)

    rows = []
    for (experiment, method, map_id, mission_type, model, temperature), recs in groups.items():
        valid = [r for r in recs if r.valid]
        sr = len(valid) / len(recs) if recs else 0.0
        plr_vals = [r.plr for r in valid]
        sd_vals  = [r.sd  for r in valid]
        sm_vals  = [r.sm  for r in valid]

        rows.append({
            "experiment": experiment,
            "method": method,
            "map_id": map_id,
            "mission_type": mission_type,
            "model": model,
            "temperature": temperature,
            "n_runs": len(recs),
            "sr": round(sr, 3),
            "plr_mean": round(statistics.mean(plr_vals), 4) if plr_vals else None,
            "plr_std":  round(statistics.stdev(plr_vals), 4) if len(plr_vals) > 1 else 0.0,
            "sd_mean":  round(statistics.mean(sd_vals), 4) if sd_vals else None,
            "sm_mean":  round(statistics.mean(sm_vals), 4) if sm_vals else None,
            "cvr": round(sum(r.code_generated for r in recs) / len(recs), 3),
            "pfr": round(sum(r.plan_fn_found  for r in recs) / len(recs), 3),
            "phr": round(sum(r.path_reached_goal for r in recs) / len(recs), 3),
            "elapsed_mean": round(statistics.mean(r.elapsed_s for r in recs), 2),
        })
    return rows


def _save_summary(records: list[RunRecord], out_dir: Path, stem: str) -> None:
    rows = _aggregate_records(records)
    if not rows:
        return
    summary_path = out_dir / f"{stem}_summary.csv"
    with summary_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    log.info("Summary saved → %s", summary_path)


def _make_generator(model: str, temperature: float):
    from agenticpathplanner.generator.agent import GeneratorAgent
    return GeneratorAgent(model=model, temperature=temperature)


def _classical_planners():
    return [
        AStarPlanner(),
        ThetaStarPlanner(),
        APFRefiner(),
        RRTStarPlanner(seed=0),
        MPCPlanner(),
    ]


def _check_ollama(model: str) -> bool:
    """Return True if Ollama is reachable and the model is available."""
    try:
        import ollama
        models = [m.model for m in ollama.list().models]
        if not any(model in m for m in models):
            log.warning("Model '%s' not found. Run: ollama pull %s", model, model)
            return False
        return True
    except Exception as exc:
        log.warning("Ollama unavailable: %s", exc)
        return False


def _print_separator(title: str) -> None:
    print(f"\n{'='*70}\n  {title}\n{'='*70}")


# ── Experiment 1 — Main Comparison ────────────────────────────────────────────

def run_exp1(
    model: str,
    temperature: float,
    runs: int,
    no_llm: bool,
    out_root: Path,
) -> None:
    _print_separator("EXP 1 — Main Comparison: Generator vs Classical Baselines")
    out_dir = out_root / "exp1_main_comparison"
    instances = _load_maps(ALL_MAP_IDS)
    classical = _classical_planners()
    records: list[RunRecord] = []

    if not no_llm and not _check_ollama(model):
        log.warning("LLM unavailable — running classical only.")
        no_llm = True

    generator = None if no_llm else _make_generator(model, temperature)

    for mission_type, mission_by_diff in MISSIONS.items():
        log.info("── Mission: %s ──", mission_type)
        for instance in instances:
            mission = mission_by_diff[instance.difficulty]
            log.info("  Map: %s", instance.map_id)

            # Classical — deterministic, run once
            for planner in classical:
                rec = _run_classical(planner, instance, "exp1", mission_type, 0)
                records.append(rec)
                _log_record(rec)

            # Generator — N stochastic runs
            if generator is not None:
                for i in range(runs):
                    rec = _run_generator(
                        generator, instance, mission,
                        "exp1", mission_type, i, model, temperature
                    )
                    records.append(rec)
                    _log_record(rec)

    _save_results(records, out_dir, "exp1_results")
    _save_summary(records, out_dir, "exp1")
    _write_exp1_report(records, out_dir)
    log.info("Experiment 1 complete.")


def _write_exp1_report(records: list[RunRecord], out_dir: Path) -> None:
    from collections import defaultdict
    import statistics

    lines = [
        "# Experiment 1 — Main Comparison Report",
        "",
        "Generator vs classical baselines across 4 mission types × 3 maps.",
        "Metrics: PLR (lower ↓), SD (higher ↑), SM (lower ↓), SR (higher ↑).",
        "",
    ]

    for mission_type in MISSIONS:
        lines.append(f"## Mission: `{mission_type}`")
        lines.append("")

        for map_id in ALL_MAP_IDS:
            recs = [r for r in records if r.mission_type == mission_type and r.map_id == map_id]
            if not recs:
                continue

            lines.append(f"### {map_id}")
            lines.append("")
            lines.append("| Method | SR | PLR | SD | SM |")
            lines.append("|---|---:|---:|---:|---:|")

            methods = sorted(set(r.method for r in recs))
            for method in methods:
                mrecs = [r for r in recs if r.method == method]
                valid = [r for r in mrecs if r.valid]
                sr = len(valid) / len(mrecs) if mrecs else 0.0
                plr = f"{statistics.mean(r.plr for r in valid):.3f}" if valid else "—"
                sd  = f"{statistics.mean(r.sd  for r in valid):.2f}" if valid else "—"
                sm  = f"{statistics.mean(r.sm  for r in valid):.2f}" if valid else "—"
                lines.append(f"| {method} | {sr:.0%} | {plr} | {sd} | {sm} |")
            lines.append("")

    (out_dir / "exp1_report.md").write_text("\n".join(lines))


# ── Experiment 2 — Mission Ablation ───────────────────────────────────────────

def run_exp2(
    model: str,
    temperature: float,
    runs: int,
    no_llm: bool,
    out_root: Path,
) -> None:
    _print_separator("EXP 2 — Mission Ablation: Rich vs Generic Mission")
    out_dir = out_root / "exp2_mission_ablation"
    instances = _load_maps(["cec2024_m01"])  # medium map only
    records: list[RunRecord] = []

    if not no_llm and not _check_ollama(model):
        log.warning("LLM unavailable — skipping Exp 2.")
        return

    generator = _make_generator(model, temperature)

    for mission_type in ("safety", "smoothness"):
        log.info("── Mission: %s ──", mission_type)
        instance = instances[0]
        rich_mission = MISSIONS[mission_type][instance.difficulty]

        log.info("  Condition: rich")
        for i in range(runs):
            rec = _run_generator(
                generator, instance, rich_mission,
                "exp2", mission_type + "_rich", i, model, temperature
            )
            records.append(rec)
            _log_record(rec)

        log.info("  Condition: generic")
        for i in range(runs):
            rec = _run_generator(
                generator, instance, GENERIC_MISSION,
                "exp2", mission_type + "_generic", i, model, temperature
            )
            records.append(rec)
            _log_record(rec)

    _save_results(records, out_dir, "exp2_results")
    _save_summary(records, out_dir, "exp2")
    _write_exp2_report(records, out_dir)
    log.info("Experiment 2 complete.")


def _write_exp2_report(records: list[RunRecord], out_dir: Path) -> None:
    import statistics

    lines = [
        "# Experiment 2 — Mission Ablation Report",
        "",
        "Rich mission prompt vs generic prompt on cec2024_m01.",
        "Each condition: 10 runs. Metric: SD (safety), SM (smoothness).",
        "",
        "| Mission type | Condition | SR | PLR | SD | SM |",
        "|---|---|---:|---:|---:|---:|",
    ]

    for mission_type in ("safety", "smoothness"):
        for condition in ("rich", "generic"):
            tag = f"{mission_type}_{condition}"
            recs = [r for r in records if r.mission_type == tag]
            valid = [r for r in recs if r.valid]
            sr  = len(valid)/len(recs) if recs else 0.0
            plr = f"{statistics.mean(r.plr for r in valid):.3f}" if valid else "—"
            sd  = f"{statistics.mean(r.sd  for r in valid):.2f}" if valid else "—"
            sm  = f"{statistics.mean(r.sm  for r in valid):.2f}" if valid else "—"
            lines.append(f"| {mission_type} | {condition} | {sr:.0%} | {plr} | {sd} | {sm} |")

    lines += [
        "",
        "## Interpretation",
        "",
        "- Safety mission: SD_rich should exceed SD_generic if mission context matters.",
        "- Smoothness mission: SM_rich should be lower than SM_generic.",
        "- Statistical test: Wilcoxon signed-rank on per-run metric pairs.",
        "",
    ]

    # Append raw per-run data for manual statistical testing
    lines.append("## Raw Per-Run Data")
    lines.append("")
    lines.append("| Run | mission_type | condition | SD | SM |")
    lines.append("|---|---|---|---:|---:|")
    for r in records:
        parts = r.mission_type.rsplit("_", 1)
        mtype, cond = (parts[0], parts[1]) if len(parts) == 2 else (r.mission_type, "?")
        sd_str = f"{r.sd:.2f}" if r.valid else "—"
        sm_str = f"{r.sm:.2f}" if r.valid else "—"
        lines.append(f"| {r.run_index} | {mtype} | {cond} | {sd_str} | {sm_str} |")

    (out_dir / "exp2_report.md").write_text("\n".join(lines))


# ── Experiment 3 — Code Reliability ───────────────────────────────────────────

def run_exp3(
    model: str,
    temperature: float,
    runs: int,
    no_llm: bool,
    out_root: Path,
) -> None:
    _print_separator("EXP 3 — Code Reliability: CVR / PFR / PHR Across Maps")
    out_dir = out_root / "exp3_reliability"
    instances = _load_maps(ALL_MAP_IDS)
    records: list[RunRecord] = []

    if not no_llm and not _check_ollama(model):
        log.warning("LLM unavailable — skipping Exp 3.")
        return

    generator = _make_generator(model, temperature)
    mission_type = "efficiency"

    for instance in instances:
        mission = MISSIONS[mission_type][instance.difficulty]
        log.info("Map: %s — running %d reliability trials", instance.map_id, runs)
        for i in range(runs):
            rec = _run_generator(
                generator, instance, mission,
                "exp3", mission_type, i, model, temperature
            )
            records.append(rec)
            log.info(
                "  run %02d  code=%s  plan=%s  valid=%s  elapsed=%.1fs",
                i, rec.code_generated, rec.plan_fn_found, rec.valid, rec.elapsed_s
            )

    _save_results(records, out_dir, "exp3_results")
    _write_exp3_report(records, out_dir, runs)
    log.info("Experiment 3 complete.")


def _write_exp3_report(records: list[RunRecord], out_dir: Path, runs: int) -> None:
    import statistics

    lines = [
        "# Experiment 3 — Code Reliability Report",
        "",
        f"Model: {records[0].model if records else '?'}  |  Runs per map: {runs}",
        "",
        "## Failure Funnel",
        "",
        "| Map | Difficulty | LLM calls | CVR | PFR | PHR (SR) | Avg time (s) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]

    for map_id in ALL_MAP_IDS:
        recs = [r for r in records if r.map_id == map_id]
        if not recs:
            continue
        n = len(recs)
        difficulty = recs[0].difficulty
        cvr = sum(r.code_generated   for r in recs) / n
        pfr = sum(r.plan_fn_found    for r in recs) / n
        phr = sum(r.path_reached_goal for r in recs) / n
        avg_t = statistics.mean(r.elapsed_s for r in recs)
        lines.append(
            f"| {map_id} | {difficulty} | {n} | {cvr:.0%} | {pfr:.0%} | {phr:.0%} | {avg_t:.1f} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "- **CVR** (Code Validity Rate): fraction where `exec()` completed without exception.",
        "- **PFR** (Plan Function Rate): fraction where a callable `plan` function was found.",
        "- **PHR** (Path Hit Rate): fraction where `plan()` returned a valid path to goal — equals SR.",
        "- Gap CVR→PFR: LLM generated syntactically valid code but did not define `plan()`.",
        "- Gap PFR→PHR: `plan()` raised a runtime error or returned an invalid path.",
        "",
    ]

    (out_dir / "exp3_report.md").write_text("\n".join(lines))


# ── Experiment 4 — Model and Temperature Sweep ────────────────────────────────

def run_exp4(
    runs: int,
    no_llm: bool,
    out_root: Path,
) -> None:
    _print_separator("EXP 4 — Model and Temperature Sweep")
    out_dir = out_root / "exp4_model_sweep"
    instance = _load_maps(["cec2024_m01"])[0]
    mission = MISSIONS["efficiency"]["medium"]
    records: list[RunRecord] = []

    if no_llm:
        log.warning("--no-llm set; skipping Exp 4.")
        return

    # 4a — Model comparison (fixed temp 0.1)
    models = ["qwen2.5-coder:7b", "qwen2.5-coder:3b", "codellama:13b"]
    for model in models:
        if not _check_ollama(model):
            log.warning("Skipping model %s (not available).", model)
            continue
        log.info("── Model: %s ──", model)
        agent = _make_generator(model, temperature=0.1)
        for i in range(runs):
            rec = _run_generator(
                agent, instance, mission, "exp4a", "efficiency", i, model, 0.1
            )
            records.append(rec)
            _log_record(rec)

    # 4b — Temperature sweep (fixed model qwen2.5-coder:7b)
    primary_model = "qwen2.5-coder:7b"
    if _check_ollama(primary_model):
        for temp in (0.0, 0.1, 0.3, 0.5):
            log.info("── Temperature: %.1f ──", temp)
            agent = _make_generator(primary_model, temperature=temp)
            for i in range(runs):
                rec = _run_generator(
                    agent, instance, mission, "exp4b", "efficiency", i, primary_model, temp
                )
                records.append(rec)
                _log_record(rec)

    _save_results(records, out_dir, "exp4_results")
    _save_summary(records, out_dir, "exp4")
    _write_exp4_report(records, out_dir)
    log.info("Experiment 4 complete.")


def _write_exp4_report(records: list[RunRecord], out_dir: Path) -> None:
    import statistics

    lines = [
        "# Experiment 4 — Model and Temperature Sweep Report",
        "",
        "## 4a — Model Comparison (temperature=0.1, cec2024_m01, efficiency mission)",
        "",
        "| Model | SR | PLR | SD | SM | Avg time (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    models = sorted(set(r.model for r in records if r.experiment == "exp4a"))
    for model in models:
        recs = [r for r in records if r.experiment == "exp4a" and r.model == model]
        valid = [r for r in recs if r.valid]
        sr  = len(valid)/len(recs) if recs else 0.0
        plr = f"{statistics.mean(r.plr for r in valid):.3f}" if valid else "—"
        sd  = f"{statistics.mean(r.sd  for r in valid):.2f}" if valid else "—"
        sm  = f"{statistics.mean(r.sm  for r in valid):.2f}" if valid else "—"
        avg_t = f"{statistics.mean(r.elapsed_s for r in recs):.1f}" if recs else "—"
        lines.append(f"| {model} | {sr:.0%} | {plr} | {sd} | {sm} | {avg_t} |")

    lines += [
        "",
        "## 4b — Temperature Sweep (qwen2.5-coder:7b, cec2024_m01, efficiency mission)",
        "",
        "| Temperature | SR | PLR | SD | SM |",
        "|---|---:|---:|---:|---:|",
    ]

    temps = sorted(set(r.temperature for r in records if r.experiment == "exp4b"))
    for temp in temps:
        recs = [r for r in records if r.experiment == "exp4b" and r.temperature == temp]
        valid = [r for r in recs if r.valid]
        sr  = len(valid)/len(recs) if recs else 0.0
        plr = f"{statistics.mean(r.plr for r in valid):.3f}" if valid else "—"
        sd  = f"{statistics.mean(r.sd  for r in valid):.2f}" if valid else "—"
        sm  = f"{statistics.mean(r.sm  for r in valid):.2f}" if valid else "—"
        lines.append(f"| {temp:.1f} | {sr:.0%} | {plr} | {sd} | {sm} |")

    (out_dir / "exp4_report.md").write_text("\n".join(lines))


# ── Logging helper ─────────────────────────────────────────────────────────────

def _log_record(r: RunRecord) -> None:
    if r.method == "generator":
        log.info(
            "    %-26s  valid=%-5s  PLR=%-6s  SD=%-5s  cvr=%s  elapsed=%.1fs",
            f"{r.method}[{r.mission_type}]",
            str(r.valid),
            f"{r.plr:.3f}" if r.valid else "—",
            f"{r.sd:.2f}" if r.valid else "—",
            str(r.code_generated),
            r.elapsed_s,
        )
    else:
        log.info(
            "    %-26s  valid=%-5s  PLR=%-6s  SD=%-5s  SM=%.2f",
            r.method,
            str(r.valid),
            f"{r.plr:.3f}" if r.valid else "—",
            f"{r.sd:.2f}" if r.valid else "—",
            r.sm if r.valid else 0.0,
        )


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generator experiment runner (see generator_experiment_plan.md)"
    )
    p.add_argument("--all", action="store_true", help="Run all 4 experiments")
    p.add_argument("--exp", type=int, choices=[1, 2, 3, 4], help="Run a single experiment")
    p.add_argument("--model",       default="qwen2.5-coder:7b")
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--runs",        type=int, default=10,
                   help="Runs per (method × map × mission) for stochastic methods")
    p.add_argument("--no-llm", action="store_true",
                   help="Skip LLM (Exp 1 classical baselines only; Exps 2-4 are skipped)")
    p.add_argument("--output", default="results/generator",
                   help="Root output directory")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.all and args.exp is None:
        print("Specify --all or --exp {1,2,3,4}.", file=sys.stderr)
        sys.exit(1)

    out_root = Path(args.output)
    kwargs = dict(model=args.model, temperature=args.temperature,
                  runs=args.runs, no_llm=args.no_llm, out_root=out_root)

    exps = [1, 2, 3, 4] if args.all else [args.exp]

    for exp in exps:
        if exp == 1:
            run_exp1(**kwargs)
        elif exp == 2:
            run_exp2(**kwargs)
        elif exp == 3:
            run_exp3(**kwargs)
        elif exp == 4:
            run_exp4(runs=args.runs, no_llm=args.no_llm, out_root=out_root)

    print(f"\nAll done. Results saved to: {out_root.resolve()}")


if __name__ == "__main__":
    main()
