# Changelog

All notable changes to this project are documented here, in reverse chronological order.

---

## [V1.0] — 2026-05-29 — End-to-End Case Study Pipeline

### Added
- **Theta\* planner** (`packages/classical/theta_star.py`) — any-angle A\* with Bresenham line-of-sight check; produces shorter, smoother paths than A\* [Nash & Koenig 2007]
- **APF post-processor** (`packages/classical/apf.py`) — repels A\* waypoints from nearby obstacles using artificial potential fields to improve safety distance [Khatib 1986]
- **Selector baselines** (`packages/selector/baselines.py`):
  - `RandomSelector` — uniform random algorithm choice (performance floor)
  - `RuleBasedSelector` — expert hand-coded obstacle-ratio rules (human expert comparison)
  - `OracleSelector` — post-hoc optimal algorithm per map (theoretical ceiling)
- **Visualiser module** (`packages/evaluation/visualiser.py`):
  - `render_comparison()` — side-by-side subplot grid for all methods on one map
  - `render_single()` — single-path figure
  - `generate_markdown_table()` — per-map metric table as markdown string
  - `save_report()` — full `case_study_report.md` with figure references
- **Case study runner** (`experiments/run_case_study.py`) — end-to-end orchestrator:
  - Runs all classical planners, selector baselines, and all three LLM agents
  - Injects correct `AgentContext` per idea (Refiner: initial_path + env, Generator: mission, Selector: conditions)
  - Builds Oracle post-hoc from first-pass results
  - Saves `case_study/figures/*.png`, `results.json`, `case_study_report.md`
  - `--no-llm` flag for classical-only runs without Ollama
- **V1 case study outputs** (`case_study/`) — three PNG figures and report from the first classical run

### Changed
- `AlgorithmRegistry.default()` now registers `astar`, `theta_star`, `apf_refined`
- `classical/__init__.py` exports `ThetaStarPlanner`, `APFRefiner`
- `evaluation/__init__.py` exports `render_comparison`, `render_single`, `generate_markdown_table`, `save_report`
- `selector/__init__.py` exports `RandomSelector`, `RuleBasedSelector`, `OracleSelector`

### Results (classical-only, no LLM)
| Method | s01 PLR | s01 SD | m01 PLR | m01 SD | c01 PLR | c01 valid |
|--------|--------:|-------:|--------:|-------:|--------:|:---------:|
| A\* | 1.000 | 1.00 | 1.122 | 1.00 | 1.061 | ✅ |
| Theta\* | 1.000 | **3.00** | **1.080** | 1.00 | **1.015** | ✅ |
| APF | 1.082 | 1.00 | 1.345 | **2.00** | — | ❌ |

---

## [0.9] — 2026-05-29 — Experimental Documentation

### Added
- `docs/research_positioning.md` §10: recommended Ollama LLMs per subproject (capability requirements, ranked model tables, hardware tiers, temperature settings)
- `experiments/configs/generator.yaml` — Generator-specific config: `qwen2.5-coder:7b`, `generate_algorithm` mode, mission descriptions per difficulty tier
- `experiments/configs/refiner.yaml` — Refiner-specific config: `llama3.1:8b`, temperature 0.4, three environmental risk scenarios, baseline list
- `experiments/configs/selector.yaml` — Selector-specific config: `llama3.1:8b`/`mistral:7b`, temperature 0.1, algorithm registry, condition scenarios, four baselines
- `experiments/configs/models.yaml` — full model registry with hardware tiers and pull commands
- `experiments/README.md` — quick-start guide, per-idea run commands, model summary table, baseline tables, reproducibility notes

### Changed
- `experiments/configs/cec2024.yaml` updated with model guidance, temperature rationale, per-subproject notes

---

## [0.8] — 2026-05-29 — Research Positioning Document (Revised)

### Added / Changed
- `docs/research_positioning.md` revised with:
  - **§9** — Recommended comparison methods with numbered references [x]: Theta\* [40], APF [27], RRT\* [23], PSO [25], Rule-based/Random/Oracle selectors
  - All in-text citations converted to numbered references [1]–[64]
  - **§7** — Comprehensive 13-dimension comparison table: Refiner, Generator, Selector vs UAV-VLA, NeLV, CoT Planner, LLM-AUV, FM-Planner, FunSearch, Eureka, ReEvo
  - **§6** — Per-idea novelty analysis with prior-work comparison tables and gap statements
  - Full 64-entry numbered reference list with complete bibliographic details

---

## [0.7] — 2026-05-29 — Research Positioning Document (Initial)

### Added
- `docs/research_positioning.md` — comprehensive research document covering:
  - §1–2: Three core ideas with agentic cycles and example prompts
  - §3: Path planning algorithm literature (8 categories, 50+ algorithms)
  - §4: Gap analysis — no prior "path planner algorithm generator" in literature
  - §5: Five-paper analysis (UAV-VLA, NeLV, CoT Planner, LLM-AUV, FM-Planner)
  - §6: Novelty claim and correct positioning
  - §7: Comprehensive comparison table
  - §8: Repository structure reference

---

## [0.6] — 2026-05-29 — Three Agentic Subprojects

### Added
- `agenticpathplanner.core.agent` — shared agentic interface:
  - `AgentContext` — input state (instance, mission, initial_path, environment)
  - `AgentResult` — output (path, success, reasoning, selected_algorithm, generated_code)
  - `BaseAgent` — four-step agentic cycle (perceive → reason → act → reflect)
- **`packages/refiner/`** — Idea 1: `RefinerAgent`
  - Runs A\* first, passes path + environmental context to LLM for refinement
  - Validates refined path for collision safety in `reflect` step
  - Falls back to initial A\* path on LLM failure or invalid output
- **`packages/generator/`** — Idea 2: `GeneratorAgent` + `CodeExecutor`
  - LLM writes a `plan(grid, start, goal)` Python function from a mission description
  - `CodeExecutor` sandboxes execution with restricted builtins
  - Stores generated code in `AgentResult.generated_code`
- **`packages/selector/`** — Idea 3: `SelectorAgent` + `AlgorithmRegistry`
  - `AlgorithmRegistry` maps string names to `BasePlanner` instances
  - LLM outputs `Selected: <name>\nReasoning: <sentence>`
  - Falls back to default algorithm on invalid LLM selection
- Makefile updated to include `refiner`, `generator`, `selector`

---

## [0.5] — 2026-05-29 — Namespace-Package Monorepo Restructure

### Changed (breaking restructure)
- Flat `src/agenticpathplanner/` layout replaced with namespace-package monorepo
- Five independently installable packages under `packages/`:
  - `agenticpathplanner-core` — `MapInstance`, `GridWorld`, `BasePlanner`
  - `agenticpathplanner-benchmarks` — CEC2024 loader, maps, metrics
  - `agenticpathplanner-classical` — A\* baseline
  - `agenticpathplanner-llm` — Ollama client, prompt builder, `LLMPlanner`
  - `agenticpathplanner-evaluation` — `Evaluator`, `agentpp` CLI
- Python implicit namespace packages (`src/agenticpathplanner/` has no `__init__.py`)
- `MapInstance` moved to `agenticpathplanner.core.types` (was in benchmarks)
- `GridWorld` moved to `agenticpathplanner.core.environment`
- `BasePlanner`/`PlanResult` moved to `agenticpathplanner.core.planner`
- `AStarPlanner` moved to `agenticpathplanner.classical.astar`
- `LLMPlanner` moved to `agenticpathplanner.llm.planner`
- `Evaluator` moved to `agenticpathplanner.evaluation.evaluator`
- `Makefile` added with `install-dev`, `test-all`, `test PKG=<name>`, `lint`, `clean`
- Old `src/` directory and `tests/` removed
- All 26 tests ported to per-package `tests/` directories

---

## [0.4] — 2026-05-29 — Repository Rename

### Changed
- GitHub repository renamed: `llmpathplanner` → `agenticpathplanner`
- Python package renamed: `llmpathplanner` → `agenticpathplanner` throughout all source files
- Local directory renamed: `llmpathplanner/` → `agenticpathplanner/`
- Git remote URL updated to `git@github.com:faramarzsafi/agenticpathplanner.git`

---

## [0.3] — 2026-05-29 — Local/GitHub Connection

### Added
- Git repository initialised in `/data/fsafiesf/fsafiesf/llmpathplanner`
- Remote added: `git@github.com:faramarzsafi/llmpathplanner.git`

---

## [0.2] — 2026-05-29 — Initial Scaffold (llmpathplanner)

### Added
First complete scaffold of the project under the original name `llmpathplanner`:

**Core infrastructure**
- `pyproject.toml` — Python 3.10+, setuptools, `agenticpathplanner-dev` meta-package
- `Makefile` — install-dev, test-all, test, lint, clean
- `.gitignore` — Python, results, data

**CEC2024 benchmark** (`benchmarks/cec2024/`)
- `maps.py` — three built-in instances: s01 (10×10), m01 (20×20), c01 (30×30)
- `loader.py` — loads built-in + on-disk `.npy` files
- `metrics.py` — PLR, SD, SM, SR with Bresenham collision checking

**Environment** (`environment/`)
- `grid.py` — `GridWorld` with 8-connected neighbours, step cost, matplotlib render

**LLM** (`llm/`)
- `client.py` — `OllamaClient` wrapper (`generate`, `chat`, `list_models`)
- `prompts.py` — `PromptBuilder` with `generate_path` and `generate_algorithm` modes

**Planners** (`planner/`)
- `base.py` — `BasePlanner` abstract class, `PlanResult`
- `llm_planner.py` — `LLMPlanner`: direct path generation + sandboxed code generation
- `baselines/astar.py` — A\* with 8-connected Euclidean heuristic

**Evaluation** (`evaluation/`)
- `evaluator.py` — multi-run `Evaluator`, CSV/JSON output, Rich console table
- `cli.py` — `llmpp run / report / list`

**Experiments**
- `experiments/run_cec2024.py` — benchmark runner
- `experiments/configs/cec2024.yaml` — default config

**Tests** — 26 tests across benchmark, planner, and LLM modules (all passing)

---

## Test Coverage Summary

| Package | Tests | Status |
|---------|-------|--------|
| core | 9 | ✅ |
| benchmarks | 6 | ✅ |
| classical | 4 | ✅ |
| llm | 9 | ✅ |
| evaluation | 2 | ✅ |
| refiner | 4 | ✅ |
| generator | 7 | ✅ |
| selector | 8 | ✅ |
| **Total** | **49** | **✅ all passing** |
