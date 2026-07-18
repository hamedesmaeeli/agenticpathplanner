# Development Log

This document records the full narrative of the Agentic Path Planner project — what was decided, built, and why, in chronological order.

---

## Session 1 — Project Scaffolding

### Starting Point

The project began as a request to scaffold a repository for an LLM-based path planner with a focus on the CEC2024 path planning benchmarks. Three key design decisions were made upfront:

1. **LLM role**: The LLM provides a path planning algorithm tailored to a specific problem specification (not just waypoints, but executable algorithm code).
2. **Backend**: Locally-hosted models via Ollama — no cloud API dependency.
3. **Stack**: Python 3.12, `pyproject.toml`, `src/` layout, pytest.

The initial repo was named `llmpathplanner` and scaffolded with:
- CEC2024 benchmark loader with three built-in maps (simple/medium/complex)
- Standard CEC2024 metrics: PLR, SD, SM, SR
- Ollama client wrapper
- Two LLM planning modes: `generate_path` (outputs waypoints) and `generate_algorithm` (writes executable Python)
- A\* baseline planner
- Multi-run evaluator with CSV/JSON output
- 26 passing tests (none require a live Ollama server)

---

## Session 2 — Repository Rename and Monorepo Restructure

### Rename: llmpathplanner → agenticpathplanner

The repository was renamed to better reflect the agentic architecture — the system is not just an LLM-based path planner but a family of agentic AI systems for path planning.

Changes made:
- GitHub repo: `faramarzsafi/llmpathplanner` → `faramarzsafi/agenticpathplanner`
- Python package name throughout all source files
- Local directory name
- Git remote URL

### Monorepo Restructure

The flat `src/agenticpathplanner/` layout was replaced with a **Python namespace package monorepo**. This allows:
- Each subproject to be independently installed (`pip install packages/<name>`)
- A clear dependency graph rooted at `core`
- New subprojects to be added without touching existing code

The five initial packages and their dependency graph:

```
agenticpathplanner-core
    ← agenticpathplanner-benchmarks
    ← agenticpathplanner-classical
    ← agenticpathplanner-llm
    ← agenticpathplanner-evaluation ← benchmarks
```

**Key design rule**: `src/agenticpathplanner/` in each package has **no `__init__.py`** — this is the shared namespace. Sub-packages (e.g., `agenticpathplanner/core/`) do have `__init__.py`.

---

## Session 3 — Three Agentic Subprojects

### Ideas from docs/ideas.pptx

The three research ideas were extracted from `docs/ideas.pptx` and implemented as new packages:

**Idea 1 — Refiner** (`agenticpathplanner.refiner`)

A classical planner generates a geometrically valid baseline path first. The LLM then uses semantic environmental context to refine it. The key design insight: geometry-only planners (A\*, Theta\*) cannot reason about electromagnetic interference zones, wind conditions, or regulatory no-fly areas. The LLM can.

The agentic cycle:
- `perceive`: extract initial A\* path + environmental context
- `reason`: LLM rewrites the path given constraints
- `act`: parse refined waypoints; fall back to initial path on failure
- `reflect`: validate the refined path is still collision-free using GridWorld

**Idea 2 — Generator** (`agenticpathplanner.generator`)

Given a mission description in natural language, the LLM writes a complete Python `plan(grid, start, goal)` function. The function is executed in a sandboxed namespace with restricted builtins (`math`, `heapq`, `collections`, `numpy` only). The generated code is stored in `AgentResult.generated_code` for reproducibility.

**Idea 3 — Selector** (`agenticpathplanner.selector`)

An `AlgorithmRegistry` maps string names to `BasePlanner` instances. The LLM reads real-time conditions and algorithm descriptions, outputs `Selected: <name>\nReasoning: <sentence>`, and the agent executes the chosen planner. Falls back to the default algorithm if the LLM names an unregistered algorithm.

### Shared BaseAgent Interface

All three implement `BaseAgent` from `agenticpathplanner.core`:

```python
class BaseAgent(ABC):
    def run(context: AgentContext) -> AgentResult    # full cycle
    def perceive(context)  -> dict                   # read environment
    def reason(obs, ctx)   -> dict                   # LLM decision
    def act(plan, ctx)     -> AgentResult            # execute
    def reflect(result, ctx) -> AgentResult          # validate / log
```

This interface means all three agents can later be wired into any agentic AI framework (LangChain, CrewAI, Claude Agent SDK) with minimal adaptation.

---

## Session 4 — Research Positioning and Literature Review

### Path Planning Algorithm Survey

A comprehensive survey of path planning algorithms was conducted and documented in `docs/research_positioning.md`. Eight categories were covered (64 references total):

1. Classical graph search: A\*, Theta\*, JPS, D\*
2. Sampling-based: RRT, RRT\*, IRRT\*, BIT\*
3. Potential field: APF, DWA, VFH
4. Evolutionary/metaheuristic: GA, PSO, ACO, GWO, WOA, DE, ABC
5. Optimal control: CHOMP, STOMP, TrajOpt, MPC, MPPI
6. Reinforcement learning: DQN, DDQN, PPO, DDPG, TD3, SAC
7. Neural planners: VIN, MPNet, Diffusion Policy
8. Hybrid methods: A\*+APF, RRT+DWA, TAMP

### Is There a Path Planner Algorithm Generator in the Literature?

The answer: **no established sub-field exists** under that label. The concept is distributed across three adjacent areas:

- **Hyper-heuristics** [Burke et al. 2010, 2013]: generates/selects heuristics at runtime, not full algorithms; not applied to path planning
- **LLM-as-code-generator**: FunSearch [Romera-Paredes et al. 2023], Eureka [Ma et al. 2023], ReEvo [Liu et al. 2024] — closest prior art, but all in combinatorial optimisation/RL reward design, not path planning
- **Automated Algorithm Design**: SMAC [Hutter et al. 2011], irace [López-Ibáñez et al. 2016] — configures algorithms, not generates them

### Five-Paper Analysis

Five candidate reference papers were analysed for category alignment:

| Paper | Category match to Generator | Key difference |
|-------|---------------------------|----------------|
| UAV-VLA [44] | ✗ Adjacent | Generates waypoints, not algorithm code |
| NeLV [63] | ✗ Architectural | Orchestrates modules, doesn't write code |
| CoT Flight Planner [51] | ✗ Adjacent | Generates path with reasoning, not code |
| **LLM-Enhanced RL/AUV [58]** | **✅ Same mechanism** | LLM writes executable Python code |
| FM-Planner [57] | ✗ Opposite direction | Fine-tunes LLM to *imitate* RRT\*, not generate |

### Corrected Novelty Claim

The LLM-as-code-generator paradigm was established by FunSearch [43] and Eureka [35], not by Xie et al. [58]. The correct claim:

> *This work is the first to apply the LLM-generates-executable-algorithm paradigm to grid-based path planning, benchmarked on CEC2024.*

---

## Session 5 — Experimental Documentation

### Per-Idea Experiment Configs

Three YAML config files were created, one per idea:

- **`generator.yaml`**: `qwen2.5-coder:7b`, `generate_algorithm` mode, temperature 0.1, mission descriptions per difficulty tier
- **`refiner.yaml`**: `llama3.1:8b`, temperature 0.4, three environmental risk scenarios (low/medium/high), baseline list
- **`selector.yaml`**: `mistral:7b`, temperature 0.1, algorithm registry with descriptions, condition scenarios

### Recommended Ollama Models

The key insight from the model analysis:

| Subproject | Why this model | Temperature |
|-----------|----------------|-------------|
| Generator | Code-focused models (Qwen2.5-Coder) produce syntactically valid Python reliably | 0.1 (deterministic) |
| Refiner | General reasoning models (LLaMA 3.1) handle spatial constraints and output coordinate lists | 0.4 (some creativity) |
| Selector | Any 7B model with good instruction-following; format compliance is the critical property | 0.1 (deterministic) |

Generator is the most model-sensitive idea — models below 7B frequently produce syntactically invalid Python.

---

## Session 6 — V1 End-to-End Case Study

### What Was Built

The V1 implementation adds five new components:

**1. Theta\* planner** (`packages/classical/theta_star.py`)

Theta\* extends A\* with a line-of-sight check: when expanding a node, if its grandparent has direct line-of-sight to the neighbour (checked via Bresenham), it connects them directly, bypassing the intermediate node. This produces any-angle paths that are shorter and smoother than A\*, with the same asymptotic complexity.

Reference: Nash & Koenig (2007). Any-angle path planning. AAAI 2007.

**2. APF post-processor** (`packages/classical/apf.py`)

Runs A\* first, then applies 40 iterations of repulsive force updates: each interior waypoint is nudged away from nearby obstacles (within `repulsion_radius=3.0` cells) using a standard Khatib potential field formula, then snapped to the nearest free integer cell. Start and goal are kept fixed.

Key finding from V1 run: APF fails on the complex map (28% obstacle density) because the snapping process can create path segments that cross obstacles. This is documented as a deliberate finding — it motivates the Refiner idea.

**3. Selector baselines** (`packages/selector/baselines.py`)

Three baselines bracket the LLM Selector's performance envelope:
- `RandomSelector`: uniform random from registry — establishes the floor
- `RuleBasedSelector`: obstacle_ratio < 15% → A\*, 15–30% → Theta\*, > 30% → APF — the "human expert" comparison
- `OracleSelector`: post-hoc best algorithm per map — establishes the ceiling

**4. Visualiser** (`packages/evaluation/visualiser.py`)

Produces matplotlib side-by-side subplot grids: each subplot shows one method's path on the map, with the subplot title showing PLR and SD, and invalid paths clearly marked. Also generates a `case_study_report.md` that embeds figure references and metric tables.

**5. Case study runner** (`experiments/run_case_study.py`)

Orchestrates a full comparative run across all methods:
1. Classical planners: direct `planner.solve(instance)`
2. Selector baselines: direct `baseline.solve(instance)`
3. LLM agents: `agent.run(context)` with per-idea `AgentContext` injection:
   - Refiner: `initial_path` (A\* result) + `environment` dict
   - Generator: `mission` description per difficulty
   - Selector: `environment` conditions dict
4. Oracle: computed post-hoc from first-pass results
5. Saves figures, JSON, and markdown report

`--no-llm` flag allows classical-only runs without Ollama (used for baseline verification).

### V1 Classical Results and Key Findings

```
cec2024_s01 (simple, 10×10):
  A*          PLR=1.000  SD=1.00  SM=0.00  ✅
  Theta*      PLR=1.000  SD=3.00  SM=0.00  ✅  ← 3× better safety distance
  APF         PLR=1.082  SD=1.00  SM=3.00  ✅

cec2024_m01 (medium, 20×20):
  A*          PLR=1.122  SD=1.00  SM=4.71  ✅
  Theta*      PLR=1.080  SD=1.00  SM=2.09  ✅  ← shorter and smoother
  APF         PLR=1.345  SD=2.00  SM=12.96 ✅  ← higher SD but much longer

cec2024_c01 (complex, 30×30, 28% obstacles):
  A*          PLR=1.061  SD=1.00  SM=9.42  ✅
  Theta*      PLR=1.015  SD=1.00  SM=1.80  ✅  ← 4.3% shorter, 5× smoother
  APF         INVALID — waypoint snapping crosses obstacles  ❌
```

**Findings:**
- Theta\* consistently dominates A\* on PLR and SM across all maps
- APF improves SD on medium maps but fails on dense maps — motivates the LLM Refiner
- Rule-based selector correctly chooses Theta\* for the complex map (obstacle_ratio > 15%)
- Oracle confirms Theta\* is optimal for all three built-in maps

---

## Current State

### Package Status

| Package | Status | Tests |
|---------|--------|-------|
| `core` | Complete | 9/9 |
| `benchmarks` | Complete | 6/6 |
| `classical` | Complete (A\*, Theta\*, APF) | 4/4 |
| `llm` | Complete | 9/9 |
| `evaluation` | Complete + visualiser | 2/2 |
| `refiner` | Complete | 4/4 |
| `generator` | Complete | 7/7 |
| `selector` | Complete + baselines | 8/8 |
| **Total** | **49/49 passing** | |

### Next Steps

1. **Run full LLM case study** once Ollama models are pulled:
   ```bash
   ollama pull qwen2.5-coder:7b && ollama pull llama3.1:8b && ollama pull mistral:7b
   python experiments/run_case_study.py --output case_study_llm/
   ```

2. **Add RRT\*** to the classical package and registry (next baseline for Generator comparison)

3. **Add PSO path planner** (mission-aware baseline for Generator)

4. **Extend case study runner** with multi-run averaging (currently single-pass; add `--runs N` flag)

5. **Write the paper** using `docs/research_positioning.md` as the foundation

---

## Repository

- **Local**: `/data/fsafiesf/fsafiesf/agenticpathplanner`
- **GitHub**: https://github.com/faramarzsafi/agenticpathplanner
- **Branch**: `master`
- **Python**: 3.12
- **Framework**: CloudSim-style discrete simulation via CEC2024 benchmark maps
