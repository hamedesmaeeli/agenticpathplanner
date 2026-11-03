from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "packages" / "core" / "src"))
sys.path.insert(0, str(ROOT / "packages" / "generator" / "src"))
sys.path.insert(0, str(Path(__file__).parents[2]))

from run_exp2 import execute_generated_source


def test_extracts_markdown_and_executes_plan():
    source = """```python
def plan(grid, start, goal):
    return [start, (start[0] + 1, start[1]), goal]
```"""
    result = execute_generated_source(source, np.zeros((3, 2), dtype=np.uint8),
                                      (0, 0), (2, 0))
    assert result["code_extracted"]
    assert result["compile_success"]
    assert result["plan_fn_found"]
    assert result["execution_success"]
    assert result["path"] == [(0, 0), (1, 0), (2, 0)]


def test_reports_syntax_error_separately():
    result = execute_generated_source("def plan(:\n pass", np.zeros((2, 2)), (0, 0), (1, 1))
    assert not result["compile_success"]
    assert result["compile_error"]


def test_reports_missing_plan_function():
    result = execute_generated_source("x = 1", np.zeros((2, 2)), (0, 0), (1, 1))
    assert result["compile_success"]
    assert not result["plan_fn_found"]
    assert not result["execution_success"]


def test_reports_runtime_error():
    source = "def plan(grid, start, goal):\n    raise ValueError('intentional test')"
    result = execute_generated_source(source, np.zeros((2, 2)), (0, 0), (1, 1))
    assert result["compile_success"]
    assert result["plan_fn_found"]
    assert not result["execution_success"]
    assert "intentional test" in result["runtime_error"]



def test_runner_writes_reproducible_case_artifacts(tmp_path, monkeypatch):
    import run_exp2
    from types import SimpleNamespace

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def list_models(self):
            return ["qwen2.5-coder:7b"]

        def generate(self, prompt, system=None):
            code = (
                "def plan(grid, start, goal):\n"
                "    return [(1,1),(2,2),(3,3),(4,4),(5,5),(6,6),(7,7),(8,8)]"
            )
            return SimpleNamespace(
                content="```python\n" + code + "\n```",
                model="qwen2.5-coder:7b",
                prompt_tokens=100,
                completion_tokens=50,
            )

    monkeypatch.setattr(run_exp2, "OllamaClient", FakeClient)
    out = tmp_path / "exp2"
    rows = run_exp2.run_exp2(
        model="qwen2.5-coder:7b",
        temperature=0.1,
        runs=1,
        output=out,
        map_ids=("cec2024_s01",),
        missions=("efficiency",),
    )

    case = out / "efficiency" / "cec2024_s01" / "run_001"
    assert len(rows) == 1
    assert rows[0]["execution_success"]
    assert rows[0]["validation"]["valid"]
    for filename in (
        "prompt.txt", "raw_response.txt", "extracted_code.py",
        "metadata.json", "path.json", "validation.json",
    ):
        assert (case / filename).is_file()
    for filename in (
        "exp2_runs.csv", "exp2_mission_summary.csv",
        "exp2_code_similarity.csv", "exp2_results.json", "exp2_report.md",
    ):
        assert (out / filename).is_file()
