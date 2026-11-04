from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "packages" / "core" / "src"))
sys.path.insert(0, str(ROOT / "packages" / "llm" / "src"))
sys.path.insert(0, str(Path(__file__).parents[2]))
from run_exp2_2 import extract_python_source, execute_generated_source
from mission_validation import _smoothness_optimum

def test_extract_complete_fence():
    assert extract_python_source("```python\ndef plan():\n    return []\n```") == "def plan():\n    return []"

def test_extract_unclosed_fence():
    assert extract_python_source("```python\ndef plan(:") == "def plan(:"

def test_extract_plain_source():
    assert extract_python_source("def plan():\n    return []") == "def plan():\n    return []"

def test_smoothness_reference_returns_pair_when_unreachable():
    result = _smoothness_optimum(np.ones((3,3),dtype=np.uint8),(0,0),(2,2))
    assert isinstance(result, tuple) and len(result) == 2

def test_smoothness_reference_straight_diagonal():
    result = _smoothness_optimum(np.zeros((4,4),dtype=np.uint8),(0,0),(3,3))
    assert result[0] == 0.0
    assert abs(result[1] - 3 * 2**0.5) < 1e-9

def test_execution_records_missing_import():
    result = execute_generated_source("def plan(grid,start,goal):\n    return list(itertools.chain([start],[goal]))", np.zeros((2,2)), (0,0),(1,1))
    assert result["compile_success"] and result["plan_fn_found"]
    assert not result["execution_success"]
    assert "itertools" in result["runtime_error"]

def test_runner_retries_after_runtime_error_and_preserves_attempts(tmp_path, monkeypatch):
    import run_exp2_2
    from types import SimpleNamespace

    class FakeClient:
        calls = 0
        def __init__(self, *args, **kwargs):
            pass
        def list_models(self):
            return ["qwen2.5-coder:7b"]
        def generate(self, prompt, system=None):
            self.calls += 1
            if self.calls == 1:
                code = "def plan(grid, start, goal):\n    raise RuntimeError('first attempt fails')"
            else:
                code = '''import heapq, itertools, math

def plan(grid, start, goal):
    moves=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    counter=itertools.count(); dist={start:0.0}; prev={}
    q=[(0.0,next(counter),start)]
    while q:
        cost,_,p=heapq.heappop(q)
        if cost != dist.get(p): continue
        if p == goal:
            path=[]
            while p in prev: path.append(p); p=prev[p]
            path.append(start); return path[::-1]
        for dr,dc in moves:
            n=(p[0]+dr,p[1]+dc)
            if not (0<=n[0]<grid.shape[0] and 0<=n[1]<grid.shape[1]) or grid[n] != 0: continue
            nc=cost+math.hypot(dr,dc)
            if nc < dist.get(n,math.inf):
                dist[n]=nc; prev[n]=p; heapq.heappush(q,(nc,next(counter),n))
    return []'''
            return SimpleNamespace(content=code, model="qwen2.5-coder:7b",
                                   prompt_tokens=100, completion_tokens=80)

    fake = FakeClient()
    monkeypatch.setattr(run_exp2_2, "OllamaClient", lambda *a, **k: fake)
    out = tmp_path / "exp2_2_retry"
    rows = run_exp2_2.run_exp2(
        model="qwen2.5-coder:7b", temperature=0.1, runs=1, output=out,
        map_ids=("cec2024_s01",), missions=("efficiency",), max_attempts=2)
    assert fake.calls == 2
    assert rows[0]["attempts_used"] == 2
    assert rows[0]["selected_attempt"] == 2
    assert rows[0]["validation"]["valid"]
    assert rows[0]["validation"]["mission_objective_optimal"]
    case = out / "efficiency" / "cec2024_s01" / "run_001"
    assert (case / "attempts" / "attempt_01" / "attempt.json").is_file()
    assert (case / "attempts" / "attempt_02" / "attempt.json").is_file()
