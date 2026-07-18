# Agentic Path Planner

A modular monorepo for **LLM-based path planning**, evaluated on the **CEC2024 Path Planning Benchmark Suite**. The project implements three distinct agentic roles for LLMs in path planning — Refiner, Generator, and Selector — alongside classical and hybrid baselines, and provides a full end-to-end case study pipeline.

---

## Research Ideas

The three core ideas originate from `docs/ideas.pptx` and are fully implemented as agentic subprojects:

| Idea | Package | LLM Role | Description |
|------|---------|----------|-------------|
| **1 — Refiner** | `agenticpathplanner.refiner` | Semantic refiner | A classical planner (A\*) generates a valid baseline path; the LLM refines it using environmental context (wind, interference zones, safety margins) |
| **2 — Generator** | `agenticpathplanner.generator` | Algorithm generator | Given a mission description, the LLM writes a complete Python `plan(grid, start, goal)` function tailored to the mission; the code is sandboxed and executed |
| **3 — Selector** | `agenticpathplanner.selector` | Algorithm selector | The LLM reads real-time conditions (obstacle density, urgency, target motion) and selects the most appropriate algorithm from an `AlgorithmRegistry` |

All three implement the **BaseAgent** agentic cycle:
```
perceive → reason (LLM) → act → reflect
```

---

## Architecture

All subprojects share the `agenticpathplanner` Python **namespace package** and depend on `agenticpathplanner-core`.

```
agenticpathplanner/
├── packages/
│   ├── core/          agenticpathplanner.core        — MapInstance, GridWorld, BasePlanner, BaseAgent
│   ├── benchmarks/    agenticpathplanner.benchmarks  — CEC2024 maps, loader, PLR/SD/SM/SR metrics
│   ├── classical/     agenticpathplanner.classical   — A*, Theta*, APF baselines
│   ├── llm/           agenticpathplanner.llm         — Ollama client, prompt builder, LLMPlanner
│   ├── evaluation/    agenticpathplanner.evaluation  — Evaluator, visualiser, agentpp CLI
│   ├── refiner/       agenticpathplanner.refiner     — Idea 1: RefinerAgent
│   ├── generator/     agenticpathplanner.generator   — Idea 2: GeneratorAgent + CodeExecutor
│   └── selector/      agenticpathplanner.selector    — Idea 3: SelectorAgent + AlgorithmRegistry
├── experiments/
│   ├── run_cec2024.py          General benchmark runner
│   ├── run_case_study.py       End-to-end case study (all methods, all maps)
│   └── configs/
│       ├── cec2024.yaml        General config
│       ├── generator.yaml      Idea 2 config (qwen2.5-coder:7b)
│       ├── refiner.yaml        Idea 1 config (llama3.1:8b)
│       ├── selector.yaml       Idea 3 config (mistral:7b)
│       └── models.yaml         Full Ollama model registry + hardware tiers
├── docs/
│   ├── research_positioning.md — Literature review, novelty claims, comparison tables
│   └── development_log.md      — Full project development narrative
├── case_study/                 — V1 case study outputs (figures + report)
└── data/cec2024/               — Official map files (not tracked)
```

### Dependency Graph

```
core  ←── benchmarks
core  ←── classical
core  ←── llm
core  ←── evaluation  ←── benchmarks
core  ←── refiner     ←── llm
core  ←── generator   ←── llm
core  ←── selector    ←── llm · classical
```

---

## Quick Start

### 1. Install all packages

```bash
make install-dev
```

### 2. Pull Ollama models

```bash
ollama pull qwen2.5-coder:7b    # Generator
ollama pull llama3.1:8b         # Refiner + Selector
ollama pull mistral:7b          # Selector (alternative)
```

### 3. Run the end-to-end case study

```bash
# Classical baselines only (no Ollama required)
python experiments/run_case_study.py --no-llm --output case_study/

# Full run with LLM agents
python experiments/run_case_study.py \
    --output case_study/ \
    --model-generator qwen2.5-coder:7b \
    --model-refiner   llama3.1:8b \
    --model-selector  mistral:7b
```

Output:
```
case_study/
├── figures/
│   ├── cec2024_s01.png    # side-by-side comparison for all methods
│   ├── cec2024_m01.png
│   └── cec2024_c01.png
├── results.json
└── case_study_report.md   # metric tables + figure references
```

### 4. Run a single-idea benchmark

```bash
python experiments/run_cec2024.py --config experiments/configs/generator.yaml
python experiments/run_cec2024.py --config experiments/configs/refiner.yaml
python experiments/run_cec2024.py --config experiments/configs/selector.yaml
```

### 5. View results

```bash
agentpp report --results results/
agentpp list --maps
agentpp list --models
```

---

## Testing

```bash
make test-all                  # all 8 packages (49 tests)
make test PKG=core             # single package
pytest packages/classical/tests/ -v
```

---

## Implemented Algorithms

### Classical planners (`agenticpathplanner.classical`)

| Algorithm | Class | Reference | Notes |
|-----------|-------|-----------|-------|
| A\* | `AStarPlanner` | Hart et al. (1968) | 8-connected, Euclidean heuristic; optimal baseline |
| Theta\* | `ThetaStarPlanner` | Nash & Koenig (2007) | Any-angle A\* with LoS check; shorter smoother paths |
| APF post-processor | `APFRefiner` | Khatib (1986) | Repels A\* waypoints from obstacles to improve safety distance |

### Selector baselines (`agenticpathplanner.selector`)

| Baseline | Class | Role |
|----------|-------|------|
| Random Selector | `RandomSelector` | Floor — uniform random algorithm choice |
| Rule-based Selector | `RuleBasedSelector` | Expert — obstacle-ratio if/else rules |
| Oracle Selector | `OracleSelector` | Ceiling — post-hoc optimal per map |

---

## CEC2024 Benchmark Maps

| ID | Size | Difficulty | Description |
|----|------|-----------|-------------|
| `cec2024_s01` | 10×10 | Simple | Open room with central wall gap |
| `cec2024_m01` | 20×20 | Medium | Corridor maze |
| `cec2024_c01` | 30×30 | Complex | Randomly cluttered (28% obstacles, seed=42) |

### Evaluation Metrics

| Metric | Symbol | Direction | Definition |
|--------|--------|-----------|------------|
| Path Length Ratio | PLR | lower ↓ | path_length / straight_line_distance |
| Safety Distance | SD | higher ↑ | minimum clearance from any obstacle |
| Smoothness | SM | lower ↓ | sum of absolute heading changes (rad) |
| Success Rate | SR | higher ↑ | fraction of runs returning a valid path |

---

## V1 Classical-Only Case Study Results

| Method | s01 PLR | s01 SD | m01 PLR | m01 SD | c01 PLR | c01 SD |
|--------|--------:|-------:|--------:|-------:|--------:|-------:|
| A\* | 1.000 | 1.00 | 1.122 | 1.00 | 1.061 | 1.00 |
| Theta\* | **1.000** | **3.00** | **1.080** | 1.00 | **1.015** | 1.00 |
| APF refined | 1.082 | 1.00 | 1.345 | **2.00** | ❌ invalid | — |
| Rule-based | 1.000 | 1.00 | 1.122 | 1.00 | 1.015 | 1.00 |
| Oracle | 1.000 | 1.00 | 1.080 | 1.00 | 1.015 | 1.00 |

Key finding: Theta\* consistently achieves lower PLR and higher SD than A\*, confirming it as the stronger classical baseline for the Refiner comparison. APF fails on the dense complex map (28% obstacles), which is precisely the scenario the LLM Refiner is designed to handle.

---

## Recommended LLM Models

| Subproject | Primary | Balanced | CPU fallback |
|-----------|---------|----------|--------------|
| **Generator** | `qwen2.5-coder:14b` | `qwen2.5-coder:7b` | `qwen2.5-coder:3b` |
| **Refiner** | `llama3.1:8b` | `mistral:7b` | `phi3.5:3.8b` |
| **Selector** | `llama3.1:8b` | `mistral:7b` | `phi3.5:3.8b` |

See `experiments/configs/models.yaml` for the full registry.

---

## Research Documentation

| Document | Description |
|----------|-------------|
| `docs/research_positioning.md` | Literature review (64 refs), novelty analysis, 13-dimension comparison table, baseline recommendations (§9), LLM model recommendations (§10) |
| `docs/development_log.md` | Full project development narrative |
| `docs/ideas.pptx` | Original three idea slides |
| `case_study/case_study_report.md` | V1 case study results with figures |

---

## Adding a New Subproject

```bash
mkdir -p packages/<name>/src/agenticpathplanner/<name>
cp packages/classical/pyproject.toml packages/<name>/pyproject.toml
# edit: name → agenticpathplanner-<name>, add deps
# do NOT add __init__.py at src/agenticpathplanner/ level (namespace package)
pip install -e packages/<name>[dev]
```
