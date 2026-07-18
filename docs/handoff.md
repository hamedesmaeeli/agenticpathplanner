# Agent Handoff Document

**Project:** Agentic Path Planner (`agenticpathplanner`)  
**Repository:** https://github.com/faramarzsafi/agenticpathplanner  
**Local path:** `/data/fsafiesf/fsafiesf/agenticpathplanner`  
**Branch:** `master` (single branch, linear history)  
**Python:** 3.12 (`/usr/bin/python3.12`)  
**Last commit:** `d168b97` — Add comprehensive project documentation  
**Test status:** 49/49 passing (no Ollama required)  
**Date of handoff:** 2026-05-29

---

## 1. What This Project Is

A Python monorepo that implements three agentic roles for LLMs in UAV/robot path planning, evaluated on the **CEC2024 Path Planning Benchmark Suite**. All models run locally via [Ollama](https://ollama.com) — no cloud API.

### The Three Ideas (from `docs/ideas.pptx`)

| # | Name | Package | LLM Does | Key method |
|---|------|---------|----------|------------|
| 1 | **Refiner** | `agenticpathplanner.refiner` | Receives a classical A\* path + environmental context (wind, interference, safety margins) → outputs a semantically refined path | RefinerAgent |
| 2 | **Generator** | `agenticpathplanner.generator` | Receives a mission description → writes a complete `plan(grid, start, goal)` Python function → function is sandboxed-executed | GeneratorAgent + CodeExecutor |
| 3 | **Selector** | `agenticpathplanner.selector` | Receives real-time conditions (obstacle density, urgency) → selects the best algorithm from an AlgorithmRegistry → executes it | SelectorAgent + AlgorithmRegistry |

All three implement the **BaseAgent** agentic cycle from `agenticpathplanner.core`:
```python
agent.run(AgentContext) -> AgentResult
# internally: perceive → reason (LLM) → act → reflect
```

---

## 2. Monorepo Structure

```
agenticpathplanner/
├── packages/                       8 independently-installable Python packages
│   ├── core/                       agenticpathplanner.core
│   │   └── src/agenticpathplanner/core/
│   │       ├── types.py            MapInstance, Coord, Path
│   │       ├── environment.py      GridWorld (neighbours, render)
│   │       ├── planner.py          BasePlanner, PlanResult (abstract)
│   │       └── agent.py            BaseAgent, AgentContext, AgentResult (abstract)
│   │
│   ├── benchmarks/                 agenticpathplanner.benchmarks
│   │   └── src/agenticpathplanner/benchmarks/cec2024/
│   │       ├── maps.py             3 built-in MapInstances (s01/m01/c01)
│   │       ├── loader.py           CEC2024Loader (built-in + .npy files)
│   │       └── metrics.py          evaluate_path(), aggregate_runs(), PathMetrics
│   │
│   ├── classical/                  agenticpathplanner.classical
│   │   └── src/agenticpathplanner/classical/
│   │       ├── astar.py            AStarPlanner — 8-connected A* [Hart 1968]
│   │       ├── theta_star.py       ThetaStarPlanner — any-angle A* [Nash 2007]
│   │       └── apf.py              APFRefiner — potential-field post-processor [Khatib 1986]
│   │
│   ├── llm/                        agenticpathplanner.llm
│   │   └── src/agenticpathplanner/llm/
│   │       ├── client.py           OllamaClient, GenerationConfig, LLMResponse
│   │       ├── prompts.py          PromptBuilder (generate_path / generate_algorithm modes)
│   │       └── planner.py          LLMPlanner (BasePlanner wrapping Ollama)
│   │
│   ├── evaluation/                 agenticpathplanner.evaluation
│   │   └── src/agenticpathplanner/evaluation/
│   │       ├── evaluator.py        Evaluator — multi-run CSV/JSON output
│   │       ├── visualiser.py       render_comparison(), save_report()
│   │       └── cli.py              agentpp run / report / list
│   │
│   ├── refiner/                    agenticpathplanner.refiner
│   │   └── src/agenticpathplanner/refiner/
│   │       ├── agent.py            RefinerAgent (COMPLETE)
│   │       └── prompts.py          RefinerPromptBuilder
│   │
│   ├── generator/                  agenticpathplanner.generator
│   │   └── src/agenticpathplanner/generator/
│   │       ├── agent.py            GeneratorAgent (COMPLETE)
│   │       ├── executor.py         CodeExecutor — sandboxed Python execution
│   │       └── prompts.py          GeneratorPromptBuilder
│   │
│   └── selector/                   agenticpathplanner.selector
│       └── src/agenticpathplanner/selector/
│           ├── agent.py            SelectorAgent (COMPLETE)
│           ├── registry.py         AlgorithmRegistry — {name: BasePlanner}
│           ├── baselines.py        RandomSelector, RuleBasedSelector, OracleSelector
│           └── prompts.py          SelectorPromptBuilder
│
├── experiments/
│   ├── run_cec2024.py              Single-idea benchmark runner
│   ├── run_case_study.py           ★ End-to-end case study (all methods, all maps)
│   └── configs/
│       ├── cec2024.yaml            General config
│       ├── generator.yaml          Generator: qwen2.5-coder:7b, generate_algorithm
│       ├── refiner.yaml            Refiner: llama3.1:8b, temp 0.4, env scenarios
│       ├── selector.yaml           Selector: mistral:7b, temp 0.1, conditions
│       └── models.yaml             Full model registry + hardware tiers
│
├── case_study/                     V1 classical-only output (committed)
│   ├── figures/                    cec2024_{s01,m01,c01}.png
│   ├── results.json
│   └── case_study_report.md
│
├── docs/
│   ├── ideas.pptx                  Original 3 ideas (Farsi + English)
│   ├── research_positioning.md     ★ Literature review (64 refs), novelty, comparison tables
│   ├── development_log.md          Full project narrative
│   ├── handoff.md                  This file
│   └── handoff.md                  ← you are here
│
├── data/cec2024/maps/              Official .npy map files (NOT tracked; empty)
├── results/                        Experiment outputs (NOT tracked)
├── README.md                       Project overview
├── CHANGELOG.md                    Version history
└── Makefile                        install-dev, test-all, test PKG=<x>, clean
```

### Namespace Package Rule
`src/agenticpathplanner/` in every package has **no `__init__.py`** — this is the shared Python namespace. Sub-package directories (e.g., `core/`, `benchmarks/`) do have `__init__.py`.

### Dependency Graph
```
core  ←── benchmarks
core  ←── classical
core  ←── llm
core  ←── evaluation  ←── benchmarks
core  ←── refiner     ←── llm
core  ←── generator   ←── llm
core  ←── selector    ←── llm, classical
```

---

## 3. Environment Setup

```bash
# Install all 8 packages in editable mode
cd /data/fsafiesf/fsafiesf/agenticpathplanner
make install-dev
# equivalent to:
# for pkg in core benchmarks classical llm evaluation refiner generator selector; do
#   python3.12 -m pip install -e "packages/$pkg[dev]" -q
# done

# Verify
make test-all     # should print: 49 passed
```

No external services needed for tests. Ollama is only needed to run LLM agents.

---

## 4. Key Data Flows

### MapInstance (the universal currency)
```python
from agenticpathplanner.core.types import MapInstance
# grid: numpy uint8 (H×W), 0=free, 1=obstacle
# start, goal: (row, col) tuples
# Built-in maps:
from agenticpathplanner.benchmarks.cec2024.maps import BUILTIN_MAPS
inst = BUILTIN_MAPS["cec2024_s01"]  # 10×10 simple
inst = BUILTIN_MAPS["cec2024_m01"]  # 20×20 medium
inst = BUILTIN_MAPS["cec2024_c01"]  # 30×30 complex (28% obstacles)
```

### Running a classical planner
```python
from agenticpathplanner.classical.astar import AStarPlanner
result = AStarPlanner().solve(inst)   # PlanResult(path, elapsed_s, success)
```

### Running an LLM agent
```python
from agenticpathplanner.core.agent import AgentContext
from agenticpathplanner.refiner.agent import RefinerAgent

# Refiner needs an initial path + environmental context
initial = AStarPlanner().solve(inst).path
ctx = AgentContext(
    instance=inst,
    initial_path=initial,
    environment={"wind_speed": "15 knots N", "interference_zones": []},
)
result = RefinerAgent(model="llama3.1:8b").run(ctx)
# result.path, result.success, result.reasoning
```

```python
from agenticpathplanner.generator.agent import GeneratorAgent
ctx = AgentContext(
    instance=inst,
    mission="Search and rescue. Prioritise clearance from unstable walls.",
)
result = GeneratorAgent(model="qwen2.5-coder:7b").run(ctx)
# result.path, result.generated_code
```

```python
from agenticpathplanner.selector.agent import SelectorAgent
from agenticpathplanner.selector.registry import AlgorithmRegistry
ctx = AgentContext(
    instance=inst,
    environment={"obstacle_density": "high", "urgency": "high"},
)
result = SelectorAgent(model="mistral:7b", registry=AlgorithmRegistry.default()).run(ctx)
# result.path, result.selected_algorithm, result.reasoning
```

### Evaluating a path
```python
from agenticpathplanner.benchmarks.cec2024.metrics import evaluate_path
m = evaluate_path(result.path, inst.grid, inst.start, inst.goal)
# m.plr (path length ratio), m.sd (safety distance), m.sm (smoothness), m.valid
```

---

## 5. Running Experiments

### Classical-only case study (no Ollama needed)
```bash
python experiments/run_case_study.py --no-llm --output case_study/
# Produces: case_study/figures/*.png, case_study/results.json, case_study_report.md
```

### Full case study with LLM agents
```bash
# Pull models first
ollama pull qwen2.5-coder:7b
ollama pull llama3.1:8b
ollama pull mistral:7b

python experiments/run_case_study.py \
    --output case_study_llm/ \
    --model-generator qwen2.5-coder:7b \
    --model-refiner   llama3.1:8b \
    --model-selector  mistral:7b
```

### Single-idea benchmark
```bash
python experiments/run_cec2024.py --config experiments/configs/generator.yaml
python experiments/run_cec2024.py --config experiments/configs/refiner.yaml
python experiments/run_cec2024.py --config experiments/configs/selector.yaml
```

### Tests
```bash
make test-all                        # all packages
make test PKG=classical              # one package
python3.12 -m pytest packages/selector/tests/ -v
```

---

## 6. Current V1 Results (Classical-Only)

From `case_study/results.json` and `case_study/case_study_report.md`:

| Method | s01 PLR | s01 SD | m01 PLR | m01 SD | c01 PLR | c01 valid |
|--------|--------:|-------:|--------:|-------:|--------:|:---------:|
| A\* | 1.000 | 1.00 | 1.122 | 1.00 | 1.061 | ✅ |
| Theta\* | **1.000** | **3.00** | **1.080** | 1.00 | **1.015** | ✅ |
| APF refined | 1.082 | 1.00 | 1.345 | **2.00** | — | ❌ |
| Random Sel. | 1.082 | 1.00 | 1.122 | 1.00 | 1.061 | ✅ |
| Rule-based Sel. | 1.000 | 1.00 | 1.122 | 1.00 | 1.015 | ✅ |
| Oracle Sel. | 1.000 | 1.00 | 1.080 | 1.00 | 1.015 | ✅ |

**Key findings:**
- Theta\* consistently dominates A\* on PLR and SM across all maps
- APF fails on the complex map (28% obstacles) — waypoint snapping crosses obstacles under high density. This is a deliberate finding: it motivates the LLM Refiner
- Rule-based selector correctly picks Theta\* for the complex map (obstacle ratio > 15%)

---

## 7. What Is Done vs What Is Next

### Done ✅

| Area | Status |
|------|--------|
| Core types + agentic interface | Complete — `MapInstance`, `GridWorld`, `BasePlanner`, `BaseAgent` |
| CEC2024 benchmark | Complete — 3 built-in maps, loader, PLR/SD/SM/SR metrics |
| Classical planners | Complete — A\*, Theta\*, APF |
| Ollama LLM client | Complete — `OllamaClient`, `PromptBuilder`, `LLMPlanner` |
| Evaluator + CLI | Complete — `agentpp run/report/list`, CSV/JSON output |
| RefinerAgent | Complete — A\* → LLM refinement → collision validation |
| GeneratorAgent | Complete — mission → LLM code → sandboxed execution |
| SelectorAgent | Complete — conditions → LLM selection → registry dispatch |
| Selector baselines | Complete — Random, Rule-based, Oracle |
| Visualiser | Complete — side-by-side figures, markdown report |
| End-to-end case study runner | Complete — `experiments/run_case_study.py` |
| Research positioning doc | Complete — 64 references, §1–10 |
| V1 classical case study | Complete — figures in `case_study/figures/` |
| All tests | 49/49 passing |

### Pending / Next Steps 🔲

**Priority 1 — Run full LLM case study** (immediate, just needs Ollama)
```bash
ollama pull qwen2.5-coder:7b && ollama pull llama3.1:8b && ollama pull mistral:7b
python experiments/run_case_study.py --output case_study_llm/
```
This will produce the first LLM vs classical comparison. The key question: does the LLM Refiner improve over raw A\* on SD? Does Generator produce valid paths?

**Priority 2 — Add RRT\*** to `packages/classical/`
```python
# File: packages/classical/src/agenticpathplanner/classical/rrt_star.py
class RRTStarPlanner(BasePlanner):
    name = "rrt_star"
    # Standard RRT* with rewiring, 8-connected movement, Euclidean cost
```
Register in `AlgorithmRegistry.default()` and add to selector configs.
Needed for Generator comparison on complex maps and Selector registry expansion.

**Priority 3 — Add PSO path planner** to `packages/classical/`
```python
# File: packages/classical/src/agenticpathplanner/classical/pso.py
class PSOPlanner(BasePlanner):
    name = "pso"
    # Optimises N waypoints as a particle swarm over path cost function
    # Cost = path_length + obstacle_penalty + smoothness_penalty
```
Closest classical analogue to mission-aware planning — key Generator comparison.

**Priority 4 — Multi-run averaging in case study**
- Add `--runs N` flag to `run_case_study.py`
- Report mean ± std for each metric
- Currently single-pass (deterministic for classical, single LLM call for agents)

**Priority 5 — Write the paper**
Foundation is ready in `docs/research_positioning.md`. The paper structure:
- Abstract, Introduction → from §1–2 of research_positioning.md
- Background → §2 (path planning concepts, Double Q-learning)
- Literature Review → §3–5
- Methodology → describe the three agents + BaseAgent interface
- Experiments → run §9 baseline comparisons, report tables
- Conclusion → §6 novelty claims + §7 comparison table

---

## 8. Critical Design Decisions (Do Not Change)

| Decision | Rationale |
|----------|-----------|
| **Namespace package** — no `__init__.py` at `src/agenticpathplanner/` | Allows all 8 packages to share one import namespace without conflicts |
| **BaseAgent is in core** | All three idea-agents must implement the same interface for the case study runner to treat them uniformly |
| **AgentContext carries `initial_path`** | Refiner needs A\* output before calling the LLM; the runner pre-computes it in `_build_refiner_context()` |
| **CodeExecutor restricts builtins** | LLM-generated code runs in a sandboxed namespace — only `math`, `heapq`, `collections`, `numpy` allowed; prevents arbitrary code execution |
| **AlgorithmRegistry uses string keys** | LLM outputs algorithm names as strings; registry maps them to planners; unknown names fall back to default safely |
| **Theta\* uses Bresenham LoS** | Same `_los()` function logic as `metrics._segment_free()` — consistent collision semantics across planning and evaluation |
| **APF keeps start/goal fixed** | Refining interior waypoints only; ensures the path still begins and ends at the correct positions |
| **OracleSelector built post-hoc** | Oracle is computed from the first-pass results of all other methods, not pre-set |
| **`--no-llm` flag in case study** | Allows full classical pipeline to run and be verified without any LLM dependency |

---

## 9. Ollama Model Quick Reference

| Subproject | Primary | Balanced | CPU fallback | Temperature |
|-----------|---------|----------|--------------|-------------|
| Generator | `qwen2.5-coder:14b` | `qwen2.5-coder:7b` | `qwen2.5-coder:3b` | 0.1 |
| Refiner | `llama3.1:8b` | `mistral:7b` | `phi3.5:3.8b` | 0.4 |
| Selector | `llama3.1:8b` | `mistral:7b` | `phi3.5:3.8b` | 0.1 |

Full model registry with hardware tiers: `experiments/configs/models.yaml`

**Generator is most model-sensitive** — models below 7B frequently produce syntactically invalid Python.

---

## 10. Key Files Quick Reference

| Purpose | File |
|---------|------|
| Central types | `packages/core/src/agenticpathplanner/core/types.py` |
| Agentic interface | `packages/core/src/agenticpathplanner/core/agent.py` |
| CEC2024 maps | `packages/benchmarks/src/agenticpathplanner/benchmarks/cec2024/maps.py` |
| CEC2024 metrics | `packages/benchmarks/src/agenticpathplanner/benchmarks/cec2024/metrics.py` |
| A\* planner | `packages/classical/src/agenticpathplanner/classical/astar.py` |
| Theta\* planner | `packages/classical/src/agenticpathplanner/classical/theta_star.py` |
| APF refiner | `packages/classical/src/agenticpathplanner/classical/apf.py` |
| Ollama client | `packages/llm/src/agenticpathplanner/llm/client.py` |
| Refiner agent | `packages/refiner/src/agenticpathplanner/refiner/agent.py` |
| Generator agent | `packages/generator/src/agenticpathplanner/generator/agent.py` |
| Code executor | `packages/generator/src/agenticpathplanner/generator/executor.py` |
| Selector agent | `packages/selector/src/agenticpathplanner/selector/agent.py` |
| Algorithm registry | `packages/selector/src/agenticpathplanner/selector/registry.py` |
| Selector baselines | `packages/selector/src/agenticpathplanner/selector/baselines.py` |
| Visualiser | `packages/evaluation/src/agenticpathplanner/evaluation/visualiser.py` |
| Case study runner | `experiments/run_case_study.py` |
| Research doc | `docs/research_positioning.md` |
| Development history | `docs/development_log.md` |
| Changelog | `CHANGELOG.md` |

---

## 11. Git History Summary

```
d168b97  Add comprehensive project documentation (README, CHANGELOG, development_log)
31c0b9c  Implement V1 end-to-end case study pipeline (Theta*, APF, baselines, visualiser)
80974ef  Update experimental docs with LLM model recommendations
b1e35cc  Add Section 9: recommended comparison methods with numbered references
d463d38  Revise research_positioning.md: numbered refs, comparison table, novelty
ada2174  Add research positioning and literature review document
8058b76  Add three agentic subprojects: Refiner, Generator, Selector
31455c0  Restructure as namespace-package monorepo with shared core
66f1864  Rename package from llmpathplanner to agenticpathplanner
2915cf0  Scaffold LLM path planner for CEC2024 benchmarks
```

---

## 12. Owner Information

- **Supervisor:** Faramarz Safiesfahani
- **Email:** faramarz.safiesfahani@surroundaustralia.com
- **GitHub account:** faramarzsafi
- **Context:** This is a PhD thesis subproject. The owner has deep Java/cloud computing background and is new to Python research tooling. Explanations should be concrete and code-first.

---

## 13. Immediate First Action for a Continuing Agent

Run the verification sequence in order:

```bash
# 1. Confirm environment
cd /data/fsafiesf/fsafiesf/agenticpathplanner
python3.12 -m pytest packages/*/tests/ -q        # expect: 49 passed

# 2. Confirm classical case study runs end-to-end
python experiments/run_case_study.py --no-llm --output /tmp/cs_verify/
# expect: 3 PNG files, results.json, case_study_report.md

# 3. Check Ollama availability
ollama list                                        # see which models are pulled

# 4. If models available, run full case study
python experiments/run_case_study.py \
    --output case_study_llm/ \
    --model-generator qwen2.5-coder:7b \
    --model-refiner   llama3.1:8b \
    --model-selector  mistral:7b
```

If tests fail or the case study errors, check `packages/*/pyproject.toml` for missing deps and re-run `make install-dev`.
