#!/usr/bin/env python3
"""Standalone Exp2 runner: mission-specific LLM path-planning generation.

Run from repository root:
  python experiments/run_exp2.py --model qwen2.5-coder:7b --temperature 0.1 --runs 1
"""
from __future__ import annotations

import argparse
import ast
import csv
import difflib
import hashlib
import json
import logging
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for package in ("core", "benchmarks", "llm", "generator"):
    package_src = ROOT / "packages" / package / "src"
    if str(package_src) not in sys.path:
        sys.path.insert(0, str(package_src))

EXP2_DIR = Path(__file__).resolve().parent / "exp2"
if str(EXP2_DIR) not in sys.path:
    sys.path.insert(0, str(EXP2_DIR))

from agenticpathplanner.benchmarks.cec2024.loader import CEC2024Loader
from agenticpathplanner.generator.executor import CodeExecutor
from agenticpathplanner.llm.client import GenerationConfig, OllamaClient
from mission_prompts import MISSION_PROMPTS, SYSTEM_PROMPT, PromptInput, build_user_prompt
from mission_validation import validate_path

LOG = logging.getLogger("exp2")
DEFAULT_MAPS = ("cec2024_s01", "cec2024_m01", "cec2024_c01")
DEFAULT_MISSIONS = ("efficiency", "safety", "smoothness", "multi")
ALLOWED_BUILTINS = (
    "range", "len", "list", "tuple", "int", "float", "bool", "min", "max",
    "abs", "sum", "sorted", "enumerate", "zip", "print", "isinstance", "type",
    "hasattr", "__import__", "set", "frozenset", "dict", "all", "any",
    "reversed", "round", "next", "iter", "ValueError", "RuntimeError",
)


def _json_safe(value: Any) -> Any:
    """Convert numpy and non-finite values to strict JSON-compatible values."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity" if value < 0 else "NaN"
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def execute_generated_source(raw_response: str, grid: np.ndarray,
                             start: tuple[int, int], goal: tuple[int, int]) -> dict:
    """Extract, compile, and execute generated code; return explicit stage statuses.

    This uses the same restricted namespace pattern as the project's CodeExecutor.
    It is not a security sandbox: generated Python must only be run in an isolated,
    trusted research environment.
    """
    source = CodeExecutor.extract_code(raw_response or "")
    result: dict[str, Any] = {
        "extracted_code": source,
        "code_extracted": bool(source.strip()),
        "compile_success": False,
        "plan_fn_found": False,
        "execution_success": False,
        "path": [],
        "compile_error": None,
        "runtime_error": None,
        "planning_runtime_s": 0.0,
    }
    if not source.strip():
        result["compile_error"] = "empty_generated_source"
        return result

    try:
        compiled = compile(source, "<llm_generated_plan>", "exec")
    except Exception as exc:
        result["compile_error"] = f"{type(exc).__name__}: {exc}"
        return result
    result["compile_success"] = True

    namespace: dict[str, Any] = {
        "numpy": np,
        "np": np,
        "math": math,
        "heapq": __import__("heapq"),
        "collections": __import__("collections"),
        "__builtins__": {name: getattr(__builtins__, name)
                         for name in ALLOWED_BUILTINS
                         if hasattr(__builtins__, name)}
        if not isinstance(__builtins__, dict)
        else {name: __builtins__[name] for name in ALLOWED_BUILTINS
              if name in __builtins__},
    }
    try:
        exec(compiled, namespace)
    except Exception as exc:
        result["runtime_error"] = f"module execution: {type(exc).__name__}: {exc}"
        return result

    plan_fn = namespace.get("plan")
    result["plan_fn_found"] = callable(plan_fn)
    if not callable(plan_fn):
        result["runtime_error"] = "No callable plan(grid, start, goal) function found."
        return result

    started = time.perf_counter()
    try:
        raw_path = plan_fn(grid, start, goal)
        if raw_path is None:
            path = []
        else:
            path = list(raw_path)
        result["execution_success"] = True
        result["path"] = path
    except Exception as exc:
        result["runtime_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["planning_runtime_s"] = time.perf_counter() - started
    return result


def _ascii_grid(instance) -> str:
    return instance.as_text_grid()


def _flatten_record(record: dict) -> dict:
    flat = dict(record)
    validation = flat.pop("validation", {}) or {}
    for key, value in validation.items():
        flat[f"validation_{key}"] = json.dumps(_json_safe(value), ensure_ascii=False) \
            if isinstance(value, (list, dict)) else _json_safe(value)
    return flat


def _write_csv(path: Path, rows: list[dict], empty_columns: tuple[str, ...] = ()) -> None:
    flat_rows = [_flatten_record(row) for row in rows]
    columns = (list(dict.fromkeys(key for row in flat_rows for key in row))
               if flat_rows else list(empty_columns))
    if not columns:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flat_rows)


def _summaries(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    summary: list[dict] = []
    metrics = (
        "mission_objective_value", "mission_optimum_value", "mission_objective_gap",
        "mission_objective_ratio", "efficiency_ratio", "min_clearance",
        "total_turn_angle_rad", "multi_weighted_cost",
        "generation_time_s", "planning_runtime_s",
    )
    for mission in DEFAULT_MISSIONS:
        group = [r for r in rows if r["mission"] == mission]
        valid = [r for r in group if r.get("validation", {}).get("valid")]
        item: dict[str, Any] = {
            "mission": mission,
            "runs": len(group),
            "valid_paths": len(valid),
            "valid_path_rate": len(valid) / len(group) if group else 0.0,
            "objective_optimal_paths": sum(bool(r.get("validation", {}).get("mission_objective_optimal")) for r in group),
            "objective_optimal_rate": (sum(bool(r.get("validation", {}).get("mission_objective_optimal")) for r in group) / len(group)) if group else 0.0,
        }
        for metric in metrics:
            vals = []
            for row in valid if metric not in ("generation_time_s", "planning_runtime_s") else group:
                value = row.get("validation", {}).get(metric) if metric in row.get("validation", {}) \
                    else row.get(metric)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    vals.append(float(value))
            item[f"{metric}_mean"] = statistics.mean(vals) if vals else None
            item[f"{metric}_median"] = statistics.median(vals) if vals else None
            item[f"{metric}_stdev"] = statistics.stdev(vals) if len(vals) > 1 else 0.0 if vals else None
        summary.append(item)

    similarity: list[dict] = []
    for i, left in enumerate(rows):
        a = left.get("extracted_code", "")
        if not a:
            continue
        for right in rows[i + 1:]:
            b = right.get("extracted_code", "")
            if not b:
                continue
            similarity.append({
                "mission_a": left["mission"], "map_a": left["map_id"],
                "run_a": left["run"], "mission_b": right["mission"],
                "map_b": right["map_id"], "run_b": right["run"],
                "sha256_identical": left.get("code_sha256") == right.get("code_sha256"),
                "sequence_similarity": difflib.SequenceMatcher(None, a, b).ratio(),
            })
    return summary, similarity


def run_exp2(model: str, temperature: float, runs: int, output: Path,
             map_ids: tuple[str, ...], missions: tuple[str, ...]) -> list[dict]:
    loader = CEC2024Loader()
    instances = {map_id: loader.load(map_id) for map_id in map_ids}
    client = OllamaClient(GenerationConfig(model=model, temperature=temperature))
    available = client.list_models()
    if available and not any(name == model or name.split(":")[0] == model.split(":")[0]
                             for name in available):
        LOG.warning("Model '%s' not found in Ollama model list: %s", model, available)

    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    total = len(missions) * len(map_ids) * runs
    call_number = 0

    for mission in missions:
        for map_id in map_ids:
            instance = instances[map_id]
            obstacle_ratio = float(np.mean(instance.grid == 1))
            prompt_data = PromptInput(
                map_id=instance.map_id,
                height=instance.height,
                width=instance.width,
                difficulty=instance.difficulty,
                start=tuple(instance.start),
                goal=tuple(instance.goal),
                obstacle_ratio=obstacle_ratio,
                ascii_grid=_ascii_grid(instance),
            )
            user_prompt = build_user_prompt(prompt_data, mission)
            for run_index in range(1, runs + 1):
                call_number += 1
                case_dir = output / mission / map_id / f"run_{run_index:03d}"
                case_dir.mkdir(parents=True, exist_ok=True)
                LOG.info("[%d/%d] mission=%s map=%s run=%d",
                         call_number, total, mission, map_id, run_index)
                (case_dir / "prompt.txt").write_text(
                    "===== SYSTEM =====\n" + SYSTEM_PROMPT
                    + "\n\n===== USER =====\n" + user_prompt,
                    encoding="utf-8",
                )

                raw_response = ""
                response_error = None
                generation_time = 0.0
                prompt_tokens = completion_tokens = 0
                response_model = model
                started = time.perf_counter()
                try:
                    response = client.generate(prompt=user_prompt, system=SYSTEM_PROMPT)
                    raw_response = response.content or ""
                    response_model = response.model
                    prompt_tokens = response.prompt_tokens
                    completion_tokens = response.completion_tokens
                except Exception as exc:
                    response_error = f"{type(exc).__name__}: {exc}"
                finally:
                    generation_time = time.perf_counter() - started

                (case_dir / "raw_response.txt").write_text(raw_response, encoding="utf-8")
                execution = execute_generated_source(
                    raw_response, instance.grid, tuple(instance.start), tuple(instance.goal)
                )
                extracted = execution.pop("extracted_code")
                (case_dir / "extracted_code.py").write_text(extracted, encoding="utf-8")
                path = execution.pop("path")

                validation = validate_path(
                    path, instance.grid, tuple(instance.start), tuple(instance.goal), mission
                )
                path_payload = {
                    "map_id": map_id, "mission": mission, "run": run_index,
                    "start": instance.start, "goal": instance.goal, "path": path,
                }
                _write_json(case_dir / "path.json", path_payload)
                _write_json(case_dir / "validation.json", validation.as_dict())

                code_hash = hashlib.sha256(extracted.encode("utf-8")).hexdigest() \
                    if extracted.strip() else ""
                record = {
                    "experiment": "exp2.1",
                    "mission": mission,
                    "map_id": map_id,
                    "difficulty": instance.difficulty,
                    "run": run_index,
                    "model": response_model,
                    "temperature": temperature,
                    "generation_time_s": generation_time,
                    "planning_runtime_s": execution["planning_runtime_s"],
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "response_error": response_error,
                    "code_extracted": execution["code_extracted"],
                    "compile_success": execution["compile_success"],
                    "plan_fn_found": execution["plan_fn_found"],
                    "execution_success": execution["execution_success"],
                    "compile_error": execution["compile_error"],
                    "runtime_error": execution["runtime_error"],
                    "path_returned": bool(path),
                    "code_sha256": code_hash,
                    "extracted_code": extracted,
                    "validation": validation.as_dict(),
                    "case_dir": str(case_dir),
                }
                _write_json(case_dir / "metadata.json", {
                    key: value for key, value in record.items()
                    if key not in ("extracted_code", "validation")
                } | {"validation_file": "validation.json", "path_file": "path.json"})
                rows.append(record)

                LOG.info(
                    "  extracted=%s compile=%s plan=%s exec=%s valid=%s "
                    "objective=%s generation=%.1fs planning=%.4fs",
                    execution["code_extracted"], execution["compile_success"],
                    execution["plan_fn_found"], execution["execution_success"],
                    validation.valid, validation.mission_objective_value,
                    generation_time, execution["planning_runtime_s"],
                )

                # Incremental CSV means a stopped run still preserves completed cases.
                _write_csv(output / "exp2_runs.csv", rows)

    summary, similarity = _summaries(rows)
    _write_csv(output / "exp2_runs.csv", rows)
    _write_csv(output / "exp2_mission_summary.csv", summary)
    _write_csv(
        output / "exp2_code_similarity.csv",
        similarity,
        empty_columns=(
            "mission_a", "map_a", "run_a", "mission_b", "map_b", "run_b",
            "sha256_identical", "sequence_similarity",
        ),
    )
    _write_json(output / "exp2_results.json", rows)

    report = [
        "# Exp2.1 Pilot Report",
        "",
        f"- Model: `{model}`",
        f"- Temperature: `{temperature}`",
        f"- Runs per mission/map: {runs}",
        f"- Total generated cases: {len(rows)}",
        f"- Valid paths: {sum(bool(r['validation']['valid']) for r in rows)}",
        "",
        "| Mission | Runs | Valid paths | Valid rate | Mean objective |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in summary:
        val = item["mission_objective_value_mean"]
        report.append(
            f"| {item['mission']} | {item['runs']} | {item['valid_paths']} | "
            f"{item['valid_path_rate']:.3f} | "
            f"{val if val is not None else 'N/A'} |"
        )
    report += [
        "",
        "Generation time and planner runtime are recorded separately.",
        "A valid path only establishes path feasibility; mission-objective compliance "
        "must be interpreted from the independent mission metrics and code review.",
        "",
    ]
    (output / "exp2_report.md").write_text("\n".join(report), encoding="utf-8")
    LOG.info("Exp2 completed: %d cases; outputs: %s", len(rows), output)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the standalone Exp2 LLM experiment.")
    parser.add_argument("--model", default="qwen2.5-coder:7b")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("results/generator/exp2_pilot"))
    parser.add_argument("--maps", nargs="+", default=list(DEFAULT_MAPS))
    parser.add_argument("--missions", nargs="+", default=list(DEFAULT_MISSIONS),
                        choices=list(DEFAULT_MISSIONS))
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be >= 1")
    if not 0.0 <= args.temperature <= 2.0:
        parser.error("--temperature must be between 0 and 2")
    unknown = sorted(set(args.maps) - set(DEFAULT_MAPS))
    if unknown:
        parser.error(f"Unknown map IDs: {', '.join(unknown)}")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    run_exp2(
        model=args.model,
        temperature=args.temperature,
        runs=args.runs,
        output=args.output,
        map_ids=tuple(args.maps),
        missions=tuple(args.missions),
    )


if __name__ == "__main__":
    main()
