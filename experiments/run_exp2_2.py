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
import re
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

EXP2_DIR = Path(__file__).resolve().parent / "exp2_2"
if str(EXP2_DIR) not in sys.path:
    sys.path.insert(0, str(EXP2_DIR))

from agenticpathplanner.benchmarks.cec2024.loader import CEC2024Loader
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


def extract_python_source(raw_response: str) -> str:
    """Extract a Python fence, including an unclosed fence from truncated output."""
    text = (raw_response or "").lstrip("\ufeff").strip()
    match = re.search(r"```(?:python|py)?[ \t]*\r?\n", text, flags=re.IGNORECASE)
    if not match:
        return text
    start = match.end()
    end = text.find("```", start)
    return (text[start:end] if end >= 0 else text[start:]).strip()


def execute_generated_source(raw_response: str, grid: np.ndarray,
                             start: tuple[int, int], goal: tuple[int, int]) -> dict:
    """Extract, compile, and execute generated code; return explicit stage statuses.

    This uses the same restricted namespace pattern as the project's CodeExecutor.
    It is not a security sandbox: generated Python must only be run in an isolated,
    trusted research environment.
    """
    source = extract_python_source(raw_response or "")
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


def _attempt_rank(mission: str, validation: dict) -> tuple:
    if not validation.get("valid"):
        return (0, math.inf, math.inf)
    value = float(validation.get("mission_objective_value", math.inf))
    length = float(validation.get("path_length", math.inf))
    if mission == "safety":
        return (1, -value, length)  # maximize bottleneck, then minimize length
    if mission == "smoothness":
        return (1, value, length)   # minimize total turn, then minimize length
    return (1, value, 0.0)


def _retry_feedback(execution: dict, validation: dict) -> str:
    problems = []
    if execution.get("compile_error"):
        problems.append("Compile error: " + str(execution["compile_error"]))
    if execution.get("runtime_error"):
        problems.append("Runtime error: " + str(execution["runtime_error"]))
    if not validation.get("valid"):
        problems.append("Path validation errors: " + ", ".join(validation.get("errors", [])))
    elif not validation.get("mission_objective_optimal"):
        problems.append(
            "Feasible path is not at the independent mission optimum: "
            f"objective={validation.get('mission_objective_value')}, "
            f"reference={validation.get('mission_optimum_value')}, "
            f"path_length={validation.get('path_length')}, "
            f"reference_secondary_length={validation.get('mission_optimum_secondary_value')}"
        )
    return ("\n\nREPAIR FEEDBACK FROM INDEPENDENT VALIDATOR:\n- "
            + ("\n- ".join(problems) if problems else "Previous attempt failed.")
            + "\nRegenerate the complete implementation. Fix all reported errors. "
              "Explicitly import every module used; keep state tuple shapes consistent; "
              "check bounds before indexing; implement the exact mission objective. "
              "Return only Python source.")


def run_exp2(model: str, temperature: float, runs: int, output: Path,
             map_ids: tuple[str, ...], missions: tuple[str, ...],
             max_attempts: int = 2) -> list[dict]:
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

                attempts: list[dict[str, Any]] = []
                total_generation_time = 0.0
                total_prompt_tokens = total_completion_tokens = 0
                current_prompt = user_prompt
                for attempt_index in range(1, max_attempts + 1):
                    attempt_dir = case_dir / "attempts" / f"attempt_{attempt_index:02d}"
                    attempt_dir.mkdir(parents=True, exist_ok=True)
                    response_error = None
                    started = time.perf_counter()
                    try:
                        response = client.generate(prompt=current_prompt, system=SYSTEM_PROMPT)
                        raw = response.content or ""
                        model_name = response.model
                        pt, ct = response.prompt_tokens, response.completion_tokens
                    except Exception as exc:
                        raw, model_name, pt, ct = "", model, 0, 0
                        response_error = f"{type(exc).__name__}: {exc}"
                    gen_time = time.perf_counter() - started
                    total_generation_time += gen_time
                    total_prompt_tokens += pt
                    total_completion_tokens += ct
                    (attempt_dir / "prompt.txt").write_text(current_prompt, encoding="utf-8")
                    (attempt_dir / "raw_response.txt").write_text(raw, encoding="utf-8")
                    execution = execute_generated_source(raw, instance.grid,
                        tuple(instance.start), tuple(instance.goal))
                    code = execution.pop("extracted_code")
                    path = execution.pop("path")
                    (attempt_dir / "extracted_code.py").write_text(code, encoding="utf-8")
                    validation = validate_path(path, instance.grid,
                        tuple(instance.start), tuple(instance.goal), mission).as_dict()
                    attempt = {"attempt": attempt_index, "response_error": response_error,
                        "model": model_name, "generation_time_s": gen_time,
                        "prompt_tokens": pt, "completion_tokens": ct,
                        "execution": execution, "path": path, "validation": validation,
                        "code": code}
                    _write_json(attempt_dir / "attempt.json", {
                        "attempt": attempt_index, "response_error": response_error,
                        "model": model_name, "generation_time_s": gen_time,
                        "execution": execution, "validation": validation, "path": path})
                    attempts.append(attempt)
                    if validation["valid"] and validation["mission_objective_optimal"]:
                        break
                    if attempt_index < max_attempts:
                        current_prompt = user_prompt + _retry_feedback(execution, validation)

                feasible = [a for a in attempts if a["validation"]["valid"]]
                selected = (min(feasible, key=lambda a: _attempt_rank(mission, a["validation"]))
                            if feasible else attempts[-1])
                selected_index = selected["attempt"]
                selected_dir = case_dir / "attempts" / f"attempt_{selected_index:02d}"
                raw_response = (selected_dir / "raw_response.txt").read_text(encoding="utf-8")
                extracted = selected["code"]
                path, execution, validation = selected["path"], selected["execution"], selected["validation"]
                (case_dir / "prompt.txt").write_text(user_prompt, encoding="utf-8")
                (case_dir / "raw_response.txt").write_text(raw_response, encoding="utf-8")
                (case_dir / "extracted_code.py").write_text(extracted, encoding="utf-8")
                _write_json(case_dir / "path.json", {"map_id": map_id, "mission": mission,
                    "run": run_index, "selected_attempt": selected_index, "path": path})
                _write_json(case_dir / "validation.json", validation)
                code_hash = hashlib.sha256(extracted.encode("utf-8")).hexdigest() if extracted.strip() else ""
                record = {
                    "experiment": "exp2.2", "mission": mission, "map_id": map_id,
                    "difficulty": instance.difficulty, "run": run_index,
                    "model": selected["model"], "temperature": temperature,
                    "attempts_used": len(attempts), "selected_attempt": selected_index,
                    "attempt_history": [{"attempt": a["attempt"],
                        "compile_error": a["execution"]["compile_error"],
                        "runtime_error": a["execution"]["runtime_error"],
                        "execution_success": a["execution"]["execution_success"],
                        "valid": a["validation"]["valid"],
                        "mission_objective_optimal": a["validation"]["mission_objective_optimal"],
                        "errors": a["validation"]["errors"],
                        "objective": a["validation"]["mission_objective_value"]} for a in attempts],
                    "generation_time_s": total_generation_time,
                    "selected_generation_time_s": selected["generation_time_s"],
                    "planning_runtime_s": execution["planning_runtime_s"],
                    "prompt_tokens": total_prompt_tokens, "completion_tokens": total_completion_tokens,
                    "response_error": selected["response_error"],
                    "code_extracted": execution["code_extracted"],
                    "compile_success": execution["compile_success"],
                    "plan_fn_found": execution["plan_fn_found"],
                    "execution_success": execution["execution_success"],
                    "compile_error": execution["compile_error"],
                    "runtime_error": execution["runtime_error"],
                    "path_returned": bool(path), "code_sha256": code_hash,
                    "extracted_code": extracted, "validation": validation,
                    "case_dir": str(case_dir)}
                _write_json(case_dir / "metadata.json", {
                    key: value for key, value in record.items()
                    if key not in ("extracted_code", "validation")}
                    | {"validation_file": "validation.json", "path_file": "path.json",
                       "attempts_dir": "attempts/"})
                rows.append(record)

                LOG.info(
                    "  extracted=%s compile=%s plan=%s exec=%s valid=%s "
                    "objective=%s generation=%.1fs planning=%.4fs",
                    execution["code_extracted"], execution["compile_success"],
                    execution["plan_fn_found"], execution["execution_success"],
                    validation["valid"], validation["mission_objective_value"],
                    total_generation_time, execution["planning_runtime_s"],
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
        "# Exp2.2 Pilot Report",
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
        "All generation/repair attempts are retained under each case.",
        "Total generation time includes all attempts; selected planner runtime is separate.",
        "A valid path only establishes path feasibility; mission-objective compliance "
        "must be interpreted from the independent mission metrics and code review.",
        "",
    ]
    (output / "exp2_report.md").write_text("\n".join(report), encoding="utf-8")
    LOG.info("Exp2.2 completed: %d cases; outputs: %s", len(rows), output)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the standalone Exp2 LLM experiment.")
    parser.add_argument("--model", default="qwen2.5-coder:7b")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=2, help="Maximum generation/repair attempts per case")
    parser.add_argument("--output", type=Path, default=Path("results/generator/exp2_pilot"))
    parser.add_argument("--maps", nargs="+", default=list(DEFAULT_MAPS))
    parser.add_argument("--missions", nargs="+", default=list(DEFAULT_MISSIONS),
                        choices=list(DEFAULT_MISSIONS))
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be >= 1")
    if args.max_attempts < 1 or args.max_attempts > 3:
        parser.error("--max-attempts must be between 1 and 3")
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
        max_attempts=args.max_attempts,
    )


if __name__ == "__main__":
    main()
