# Methods and Ideas — Agentic Path Planner

This document comprehensively describes every planning method implemented in this repository: the classical and selector baselines used for comparison, and the three original agentic ideas that form the core research contribution.

---

## Table of Contents

1. [Research Overview](#1-research-overview)
2. [Evaluation Setup](#2-evaluation-setup)
   - 2.1 [Benchmark Maps (CEC2024)](#21-benchmark-maps-cec2024)
   - 2.2 [Metrics](#22-metrics)
3. [Baseline Methods](#3-baseline-methods)
   - 3.1 [A\* (AStarPlanner)](#31-a-astarlanner)
   - 3.2 [Theta\* (ThetaStarPlanner)](#32-theta-thetastarplanner)
   - 3.3 [APF Refiner (APFRefiner)](#33-apf-refiner-apfrefiner)
   - 3.4 [Random Selector](#34-random-selector)
   - 3.5 [Rule-Based Selector](#35-rule-based-selector)
   - 3.6 [Oracle Selector](#36-oracle-selector)
4. [Agentic Framework](#4-agentic-framework)
5. [The Three Agentic Ideas](#5-the-three-agentic-ideas)
   - 5.1 [Idea 1 — LLM as Path Refiner (RefinerAgent)](#51-idea-1--llm-as-path-refiner-refineragent)
   - 5.2 [Idea 2 — LLM as Algorithm Generator (GeneratorAgent)](#52-idea-2--llm-as-algorithm-generator-generatoragent)
   - 5.3 [Idea 3 — LLM as Algorithm Selector (SelectorAgent)](#53-idea-3--llm-as-algorithm-selector-selectoragent)
6. [V1 Classical Results](#6-v1-classical-results)
7. [Method Comparison Summary](#7-method-comparison-summary)

---

## 1. Research Overview

**Research question**: Can large language models (LLMs) play specialised roles in path planning beyond simple waypoint generation?

This project investigates three distinct ways an LLM can augment classical path planning algorithms, all evaluated on the CEC2024 benchmark suite. All LLMs run locally via [Ollama](https://ollama.com/) — no cloud APIs are used.

The core claim is that an LLM is not just a "path producer" but can act as a **refiner** (post-processing an existing path), a **generator** (writing a bespoke algorithm on demand from a mission description), or a **selector** (choosing the best algorithm from a registry given real-time conditions). These three roles are compared against non-LLM baselines that bracket performance from floor (random) to ceiling (oracle).

---

## 2. Evaluation Setup

### 2.1 Benchmark Maps (CEC2024)

Three built-in map instances are provided in `packages/benchmarks/src/agenticpathplanner/benchmarks/cec2024/maps.py`. All maps use `uint8` NumPy arrays where `0 = free` and `1 = obstacle`. Grid coordinates are `(row, col)`.

| Map ID | Size | Start | Goal | Difficulty | Description | Obstacle % |
|---|---|---|---|---|---|---|
| `cec2024_s01` | 10×10 | (1,1) | (8,8) | simple | Open room with two central wall segments | ~5% |
| `cec2024_m01` | 20×20 | (1,1) | (18,18) | medium | Corridor maze with horizontal walls and vertical pillars | ~15% |
| `cec2024_c01` | 30×30 | (1,1) | (28,28) | complex | Randomly cluttered environment (seed=42) | ~28% |

**`cec2024_s01` (simple)** — A 10×10 grid with two short horizontal wall segments at row 5: columns 1–3 and columns 6–8. The optimal path navigates through the gap at column 4–5 or above/below the walls.

**`cec2024_m01` (medium)** — A 20×20 grid with four long horizontal walls (rows 4, 8, 12, 16) and two vertical pillars (columns 4, 14), forming a corridor maze that forces detours.

**`cec2024_c01` (complex)** — A 30×30 grid with each cell independently set to obstacle with 28% probability (NumPy RNG seed 42). Start (1,1) and goal (28,28) are forced clear. This map tests performance under dense, unpredictable clutter and is where APF fails.

---

### 2.2 Metrics

Defined in `packages/benchmarks/src/agenticpathplanner/benchmarks/cec2024/metrics.py`.

All paths are evaluated on four metrics. Lower PLR and SM are better; higher SD is better.

#### Path Length Ratio (PLR)

```
PLR = path_length / straight_line_distance(start, goal)
```

PLR ≥ 1.0 always. PLR = 1.0 means the path is as short as the straight-line distance. Measures route efficiency. A path that doubles back has a high PLR.

#### Safety Distance (SD)

```
SD = min over all waypoints of: min Euclidean distance to any obstacle cell
```

SD is computed in grid-cell units. SD = 1.0 means the path passes within one cell of the nearest obstacle. Higher SD means the path routes further from obstacles. SD is particularly sensitive to APF post-processing and motivates the LLM Refiner.

#### Smoothness (SM)

```
SM = sum of turning angles (radians) at all interior waypoints
```

SM = 0.0 for a perfectly straight path. Each sharp turn adds to the total. Lower SM means the path flows more naturally, which is important for UAV or vehicle kinematics.

#### Success Rate (SR)

```
SR = number of valid runs / total runs
```

A run is valid if: (1) the path starts at `start` and ends at `goal`, (2) no waypoint is in an obstacle cell, and (3) no segment between consecutive waypoints crosses an obstacle (checked via Bresenham line-of-sight). SR is reported over multiple independent runs; a deterministic planner always has SR = 0.0 or 1.0.

#### Validity (single-run)

A boolean flag — `True` if and only if all three validity conditions above are met. Used in single-run case study tables.

---

## 3. Baseline Methods

Baselines are non-LLM methods that provide reference performance. The six baselines divide into two groups: **classical planners** (always produce a path directly) and **selector baselines** (choose among classical planners).

---

### 3.1 A\* (AStarPlanner)

**Package**: `packages/classical` · **Class**: `AStarPlanner` · **Name**: `"astar"`

**Algorithm**: Dijkstra's algorithm extended with a heuristic — the standard A\* search on an 8-connected grid. Uses Euclidean distance to the goal as the admissible heuristic.

**How it works**:

1. Maintains a min-heap (`heapq`) ordered by `f = g + h`, where `g` is the accumulated cost from start and `h = √((r−goal_r)² + (c−goal_c)²)`.
2. Expands the cell with the lowest `f` value.
3. For each of the 8 neighbours `(dr, dc) ∈ {−1,0,1}²\{(0,0)}`, the step cost is `√(dr²+dc²)` (1.0 for cardinal moves, √2 ≈ 1.414 for diagonal moves).
4. Tracks predecessors in `came_from` and reconstructs the path when the goal is popped.

**Properties**:
- **Optimal**: Guaranteed to find the shortest path (minimises PLR) given the Euclidean heuristic is admissible.
- **Deterministic**: Always returns the same path for the same map.
- **No safety margin**: Paths hug obstacle boundaries because the cost function only penalises distance, not proximity to obstacles.
- **Completeness**: Returns `[]` only if no path exists.

**Role in the project**: The primary baseline. Its path also serves as the `initial_path` input to the Refiner agent (Idea 1). All other methods are compared against A\*'s PLR.

**Typical behaviour on CEC2024**:
- `cec2024_s01`: PLR = 1.000, SD = 1.0 — finds the optimal straight-ish path but may graze the wall gap.
- `cec2024_m01`: PLR = 1.122, SD = 1.0 — navigates the corridor maze efficiently.
- `cec2024_c01`: PLR = 1.061, SD = 1.0 — handles dense clutter reliably.

---

### 3.2 Theta\* (ThetaStarPlanner)

**Package**: `packages/classical` · **Class**: `ThetaStarPlanner` · **Name**: `"theta_star"`

**Reference**: Nash & Koenig (2007). *Any-angle path planning*. AAAI 2007.

**Algorithm**: An extension of A\* that relaxes the constraint that path segments must follow grid edges. Instead of connecting `parent → current → neighbour`, Theta\* checks whether the **grandparent** has direct line-of-sight to the neighbour and, if so, connects them directly — bypassing the intermediate cell.

**How it works**:

1. Same heap structure as A\*, with `f = g + h` (Euclidean heuristic).
2. When expanding node `cur` with parent `p`:
   - For each 8-connected neighbour `nb`:
     - **LoS check**: Run Bresenham's line algorithm between `p` and `nb`. If the line does not pass through any obstacle cell, set `came_from[nb] = p` and compute `g[nb] = g[p] + dist(p, nb)`.
     - **Fallback**: If LoS fails, fall back to standard A\* relaxation: `came_from[nb] = cur`.
3. Path reconstruction follows `came_from` back to `start`.

**Bresenham LoS check** (used identically in `metrics._segment_free`):
```
Incrementally traces a raster line from p1 to p2.
Returns False as soon as any cell is out-of-bounds or an obstacle.
```

**Properties**:
- **Any-angle**: Waypoints can be connected at arbitrary angles, not just grid-edge directions (0°, 45°, 90°, etc.).
- **Shorter paths**: By cutting corners, paths are shorter than A\* on cluttered maps (lower PLR).
- **Smoother paths**: Fewer waypoints and fewer sharp turns (lower SM).
- **Higher SD**: Because LoS lines cut across open space rather than hugging walls, the path naturally stays further from obstacle boundaries.
- **Same completeness guarantee as A\***.

**Role in the project**: The strongest classical baseline. Consistently outperforms A\* on PLR and SM. In the V1 results, Theta\* is also the best-performing method overall on two of three maps, providing the ceiling for classical methods.

**Typical behaviour on CEC2024**:
- `cec2024_s01`: PLR = 1.000, SD = 3.0, SM = 0.00 — matches A\* on length but dramatically higher safety distance.
- `cec2024_m01`: PLR = 1.080, SD = 1.0, SM = 2.09 — shorter and smoother than A\*.
- `cec2024_c01`: PLR = 1.015, SD = 1.0, SM = 1.80 — near-optimal on the complex map.

---

### 3.3 APF Refiner (APFRefiner)

**Package**: `packages/classical` · **Class**: `APFRefiner` · **Name**: `"apf_refined"`

**Reference**: Khatib (1986). *Real-time obstacle avoidance for manipulators and mobile robots*. IJRR 5(1), 90–98.

**Algorithm**: Artificial Potential Field (APF) post-processing. First runs A\* to get an initial valid path, then iteratively nudges interior waypoints away from obstacles using repulsive forces. Start and goal are kept fixed.

**How it works**:

1. **Initialise**: Run A\* on the map to get a base path. If A\* fails, return empty.
2. **Float-space relaxation**: Convert all integer waypoints to floating-point coordinates.
3. **Iterative refinement** (default: 40 iterations):
   - For each interior waypoint `i` (index 1 to `n−2`):
     - Find all obstacle cells within `repulsion_radius = 3.0` cells.
     - For each nearby obstacle at distance `d`, compute a repulsive force vector:
       ```
       magnitude = gain × (1/d − 1/r_max) / d²
       direction = unit vector from obstacle to waypoint
       ```
     - Sum all repulsive force vectors.
     - Clamp the total force to unit magnitude if it exceeds 1.0.
     - Nudge the waypoint: `pos += step × force`  (default `step = 0.35`).
     - Clamp to grid bounds.
4. **Snap to grid**: After all iterations, round each float position to the nearest free integer cell using a spiral search (radius 0 to 3).
5. **Remove duplicates**: Skip snapped points that equal the previous point.

**Key parameters**:
| Parameter | Default | Effect |
|---|---|---|
| `iterations` | 40 | More iterations → more separation from obstacles, but higher PLR |
| `step` | 0.35 | Step size per iteration; larger = more aggressive movement |
| `repulsion_radius` | 3.0 | Cells within this radius exert force; larger = more clearance |
| `repulsion_gain` | 1.5 | Scales all repulsive forces |

**Properties**:
- **Improves SD**: By design, pushes waypoints away from obstacles.
- **Increases PLR**: Detours around obstacles make the path longer.
- **Fragile in dense environments**: When obstacle density is high (~28%), repulsive forces from all sides cancel out or push waypoints into other obstacles. The snap-to-free step may fail to find a free cell if all nearby cells are blocked.
- **Fails on `cec2024_c01`**: The 28% obstacle density of the complex map causes snap failures, producing an invalid path.

**Role in the project**: Demonstrates a classical approach to improving safety distance. Its failure on the complex map directly motivates Idea 1 (LLM Refiner): a method with semantic understanding of the environment may handle dense clutter where APF's purely geometric force field breaks down.

**Typical behaviour on CEC2024**:
- `cec2024_s01`: PLR = 1.082, SD = 1.0, SM = 3.00 — longer path but no SD improvement over A\*.
- `cec2024_m01`: PLR = 1.345, SD = 2.0, SM = 12.96 — significantly longer and less smooth; some SD gain.
- `cec2024_c01`: **Invalid** — snap-to-free fails under dense clutter.

---

### 3.4 Random Selector

**Package**: `packages/selector` · **Class**: `RandomSelector` · **Name**: `"random_selector"`

**Algorithm**: Uniformly samples one algorithm name from the `AlgorithmRegistry` at random, then executes it.

**How it works**:
```python
chosen = random.choice(registry.names())
return registry.get(chosen).solve(instance).path
```

Seeded RNG is supported for reproducibility.

**Role in the project**: The **performance floor** for selector comparison. If the LLM Selector (Idea 3) cannot consistently beat random algorithm selection, it adds no value. On each map, the Random Selector's expected performance is the average of all registered algorithms — roughly midway between best and worst.

**Typical behaviour**: Because the registry contains A\*, Theta\*, and APF, and random selection has a 1/3 chance of picking each:
- On `cec2024_s01`: happened to pick APF (PLR = 1.082).
- On `cec2024_m01`: happened to pick A\* (PLR = 1.122).
- On `cec2024_c01`: happened to pick A\* (PLR = 1.061, valid) — avoided the APF failure by chance.

---

### 3.5 Rule-Based Selector

**Package**: `packages/selector` · **Class**: `RuleBasedSelector` · **Name**: `"rule_based_selector"`

**Algorithm**: A hand-coded decision tree over obstacle density (`grid.mean()`), encoding standard expert intuition about which algorithm suits which environment.

**Rules**:
```
obstacle_ratio < 0.15  →  astar       (open grid, A* is near-optimal)
0.15 ≤ ratio  < 0.30   →  theta_star  (moderate clutter, smoother paths matter)
ratio ≥ 0.30           →  apf_refined (dense clutter, clearance priority)
```

The `selection_reason()` method returns a human-readable explanation of which rule fired.

**Role in the project**: The **human expert baseline** for selector comparison. It represents what a knowledgeable engineer would hard-code given domain knowledge. The LLM Selector should match or exceed this in accuracy and additionally provide natural-language reasoning.

**Behaviour on CEC2024**:
- `cec2024_s01`: ratio ≈ 0.05 < 0.15 → selects `astar`. PLR = 1.000, valid.
- `cec2024_m01`: ratio ≈ 0.15 → selects `astar` (threshold is strict `<`). PLR = 1.122, valid. *(Note: Theta\* would have been better — a known limitation of threshold-based rules.)*
- `cec2024_c01`: ratio ≈ 0.28 → selects `theta_star` (< 0.30 threshold). PLR = 1.015, valid. Correctly avoids APF on the complex map.

---

### 3.6 Oracle Selector

**Package**: `packages/selector` · **Class**: `OracleSelector` · **Name**: `"oracle_selector"`

**Algorithm**: Looks up the pre-determined best algorithm for each `map_id` from a hardcoded `oracle_map` dictionary. No computation — pure lookup.

```python
chosen = oracle_map.get(instance.map_id, default_algo)
return registry.get(chosen).solve(instance).path
```

**How the oracle map is built**: After a first-pass evaluation run, the algorithm that achieves the lowest PLR on each map is identified and stored:
```python
oracle_map = {
    "cec2024_s01": "astar",      # A* and Theta* tie; A* chosen
    "cec2024_m01": "theta_star", # Theta* has lower PLR (1.080 vs 1.122)
    "cec2024_c01": "theta_star", # Only A* and Theta* are valid; Theta* wins
}
```

**Role in the project**: The **theoretical performance ceiling** for any selection policy operating over the same registry. No selection method can outperform the Oracle because the Oracle always picks the best available algorithm in hindsight. The LLM Selector is expected to fall between Rule-Based and Oracle, with the gap representing the LLM's added value from natural-language reasoning.

**Behaviour on CEC2024**: Always matches the best valid classical planner on each map. PLR = 1.000 (s01), 1.080 (m01), 1.015 (c01).

---

## 4. Agentic Framework

All three LLM-based ideas inherit from `BaseAgent` (defined in `packages/core/src/agenticpathplanner/core/agent.py`) and implement the same four-step agentic cycle.

### The Perceive → Reason → Act → Reflect Cycle

```
context: AgentContext
    │
    ▼
perceive(context) → observation: dict
    │
    ▼
reason(observation, context) → plan: dict   [LLM call happens here]
    │
    ▼
act(plan, context) → result: AgentResult
    │
    ▼
reflect(result, context) → result: AgentResult   [validation/logging]
```

**`perceive`**: Converts `AgentContext` to a structured observation dictionary. Default implementation calls `context.as_observation()` which serialises map ID, size, difficulty, start/goal, mission, environment, and initial path length.

**`reason`**: The only step that calls the LLM (via `OllamaClient`). Constructs a prompt using the agent's `PromptBuilder`, sends it to Ollama, and returns the raw LLM response wrapped in a dict. Each agent has its own prompt strategy.

**`act`**: Parses the LLM response and executes the corresponding action (refine a path, run generated code, or dispatch a selected algorithm). Returns an `AgentResult`. Falls back gracefully on parse failure.

**`reflect`**: Post-execution quality check or logging. For Refiner: collision validation. For Generator: log generated code length. For Selector: log selection decision and reasoning.

### AgentContext

The single input to every agent:

| Field | Type | Purpose |
|---|---|---|
| `instance` | `MapInstance` | The map, start, and goal |
| `environment` | `dict` | Domain context (wind zones, obstacle density, urgency, etc.) |
| `mission` | `str` | Natural language mission description (used by Generator) |
| `initial_path` | `Path` | Pre-computed A\* path (used by Refiner) |
| `metadata` | `dict` | Arbitrary extra data |

### AgentResult

The single output from every agent:

| Field | Type | Purpose |
|---|---|---|
| `path` | `Path` | The final list of `(row, col)` waypoints |
| `success` | `bool` | Whether the path is non-empty and reaches the goal |
| `agent_name` | `str` | Identifies which agent produced the result |
| `reasoning` | `str` | LLM's reasoning or a fallback message |
| `selected_algorithm` | `str \| None` | Filled by Selector only |
| `generated_code` | `str \| None` | Filled by Generator only |
| `metadata` | `dict` | Model name, elapsed time, fallback flags, etc. |

---

## 5. The Three Agentic Ideas

---

### 5.1 Idea 1 — LLM as Path Refiner (RefinerAgent)

**Package**: `packages/refiner` · **Class**: `RefinerAgent` · **Name**: `"refiner"`

**Recommended model**: `llama3.1:8b` (temperature 0.2–0.4)

#### Motivation

Classical planners (A\*, Theta\*) optimise geometric objectives (path length, smoothness) but are blind to **semantic domain context** — wind zones, electromagnetic interference radii, no-fly areas, surface type, operational safety margins. APF can improve safety distance geometrically, but it fails under high obstacle density (28% on `cec2024_c01`) and has no understanding of domain-specific constraints. An LLM can reason over these environmental factors in natural language and adjust waypoints accordingly.

#### What it does

The Refiner takes an existing path (typically A\* output) plus an `environment` dictionary describing real-world context, and outputs a refined path that better accounts for the domain constraints while remaining collision-free.

#### Agentic Cycle

**Perceive**: Extracts from `AgentContext`:
- The map (`instance`), initial A\* path, and environment context.
- For maps ≤ 900 cells (≤ 30×30), also generates an ASCII text grid with the path marked:
  ```
  S * . . # . .
  . * . . # . .
  . . * . . . .
  . . . * . . G
  ```

**Reason** (LLM call): Sends a prompt to Ollama containing:
- Map ID, dimensions, difficulty, start, goal.
- The initial path as a list of up to 20 waypoints.
- ASCII grid (if small enough).
- Environmental context as key–value pairs.

System prompt instructs the LLM to act as a *UAV path planning advisor*, output only a Python list of `(row, col)` tuples, keep start and goal fixed, and prefer gradual changes over drastic deviations.

**Act**: Parses the LLM response to extract a new path:
1. Strip markdown code fences.
2. Try `ast.literal_eval()` on the full response.
3. Fall back to a regex scan for `(int, int)` pairs.
4. If parsing fails entirely, fall back to `context.initial_path` unchanged.

**Reflect** (collision validation): Iterates over every waypoint in the refined path using `GridWorld.is_free()`. If any waypoint is in an obstacle cell, the entire refinement is discarded and `initial_path` is restored. Sets `metadata["reverted"] = True` to record the fallback.

#### Key Design Points

- **Two-stage fallback**: (1) parse failure → keep initial path; (2) invalid path → revert to initial path. The agent is never worse than A\*.
- **Grid ASCII for small maps**: The text grid gives the LLM spatial context that a coordinate list alone does not convey.
- **Temperature 0.2–0.4**: Lower temperatures produce more conservative refinements; higher temperatures explore more aggressively.
- **Context keys in `environment`**: Examples include `{"wind_zone_rows": [3, 4], "interference_radius_m": 5, "no_fly_cols": [7, 8], "safety_margin_cells": 2}`.

#### What to expect

On maps where the initial A\* path is geometrically fine but operationally unsafe (passes through wind zones, near interference sources), the Refiner should shift waypoints to route around these areas. On `cec2024_c01` where APF fails, the Refiner's semantic reasoning may still produce a valid safer path — the key research question is whether this works in practice.

---

### 5.2 Idea 2 — LLM as Algorithm Generator (GeneratorAgent)

**Package**: `packages/generator` · **Class**: `GeneratorAgent` · **Name**: `"generator"`

**Recommended model**: `qwen2.5-coder:7b` (temperature 0.1–0.3)

#### Motivation

Different missions have fundamentally different path quality priorities that generic algorithms cannot fully capture:
- A **search-and-rescue** mission needs maximum clearance from unstable walls, accepting a longer path.
- A **time-critical delivery** mission needs the shortest path, accepting proximity to obstacles.
- A **stealth surveillance** mission needs to avoid open spaces, preferring to route along edges.

A general-purpose path planner cannot simultaneously optimise for all these priorities. Instead of selecting a pre-existing algorithm, the Generator asks an LLM to **write a new algorithm from scratch** tailored to the specific mission's stated constraints and priorities.

#### What it does

Given a natural language mission description and a map, the Generator produces a Python `plan(grid, start, goal)` function, executes it in a sandboxed environment, and returns the resulting path.

#### Agentic Cycle

**Perceive**: Extracts from `AgentContext`:
- Map instance and ASCII grid (for maps ≤ 30×30).
- The `context.mission` string, e.g.:
  ```
  "Search and rescue in collapsed building.
   Prioritise clearance from unstable walls.
   Accept longer paths to stay safe."
  ```
  Defaults to `"Find the shortest collision-free path."` if not provided.

**Reason** (LLM call): Sends a prompt to Ollama containing:
- Map ID, dimensions, obstacle ratio, start, goal.
- ASCII grid.
- The full mission description.

System prompt instructs the LLM to act as an *algorithm designer*, write a `plan(grid, start, goal) -> list[tuple[int,int]]` function, use only allowed imports (`numpy`, `math`, `heapq`, `collections`), return a collision-free path or `[]` on failure, and use 8-connected movement.

**Act**: Passes the raw LLM response to `CodeExecutor.execute()`:
1. Strips the code.
2. Runs `exec()` in a restricted namespace.
3. Calls `plan_fn(instance.grid, instance.start, instance.goal)`.
4. Converts the return value to `list[tuple[int,int]]`.
5. Returns `(path, code_string)` or `([], None)` on any exception.

`success` is set to `True` only if the path is non-empty **and** starts at `start` **and** ends at `goal`.

**Reflect**: Logs the character count of the generated code and the target map ID for reproducibility.

#### CodeExecutor — Sandboxed Execution

Defined in `packages/generator/src/agenticpathplanner/generator/executor.py`.

The executor runs LLM-generated code with a deliberately restricted `__builtins__`:

**Allowed builtins**: `range`, `len`, `list`, `tuple`, `int`, `float`, `bool`, `min`, `max`, `abs`, `sum`, `sorted`, `enumerate`, `zip`, `print`, `isinstance`, `type`, `hasattr`, `__import__`

**Allowed imports**: `numpy` / `np`, `math`, `heapq`, `collections`

**Blocked**: `open`, `os`, `sys`, `subprocess`, `eval`, `compile`, `exec` (nested), `__import__` for arbitrary modules, and all other standard library access.

This means the LLM can implement any graph search, dynamic programming, or numerical algorithm, but cannot read files, make network requests, or execute arbitrary system commands.

#### Key Design Points

- **Mission-first design**: The same map with different mission descriptions should yield different algorithms. A clearance-first mission should produce an APF-like or safety-weighted A\* variant; a speed-first mission should produce a plain A\* or greedy best-first.
- **Code logging**: The generated code is stored in `AgentResult.generated_code` and logged — enabling inspection, debugging, and reproducibility.
- **Qwen 2.5 Coder**: Chosen because it is a code-specialised model fine-tuned for Python generation, outperforming general instruction models on programming tasks.
- **Low temperature (0.1)**: Code generation benefits from high determinism. Slightly higher (0.3) allows some variation across runs to assess SR.
- **No validation step**: Unlike Refiner, Generator has no `reflect` phase that checks collision-freeness — this is intentional, as the generated `plan()` function is responsible for returning a valid path. The success flag checks start/goal endpoints but not intermediate collisions.

#### What to expect

The quality of results depends heavily on the LLM's ability to write correct path planning code. Expected outcomes:
- Simple missions (shortest path): The LLM typically regenerates a recognisable A\* variant.
- Clearance missions: The LLM may add obstacle-distance penalties to the cost function.
- Novel missions (stay near edges, avoid open spaces): Less predictable — the key research question.

---

### 5.3 Idea 3 — LLM as Algorithm Selector (SelectorAgent)

**Package**: `packages/selector` · **Class**: `SelectorAgent` · **Name**: `"selector"`

**Recommended model**: `mistral:7b` (temperature 0.1)

#### Motivation

In dynamic, real-world deployment, the best algorithm changes depending on conditions that are known only at runtime: current obstacle density (detected by sensors), battery level, mission urgency, whether the target is moving, weather conditions, etc. Hard-coded rule-based selectors (like `RuleBasedSelector`) capture only the conditions the engineer anticipated. An LLM can reason over arbitrary natural-language conditions and justify its selection — enabling more flexible, explainable algorithm selection.

#### What it does

Given a description of available algorithms and current environmental/operational conditions, the Selector asks the LLM which algorithm to use, then executes that algorithm and returns its path. The LLM's reasoning is preserved in the result.

#### Agentic Cycle

**Perceive**: Extracts the standard observation dict from `AgentContext.as_observation()`, which includes map ID, difficulty, size, start/goal, mission, environment dict, and initial path length.

**Reason** (LLM call): Constructs a prompt containing:
1. **Algorithm descriptions**: Name → one-line docstring for each registered algorithm, pulled from `AlgorithmRegistry.descriptions()`.
   ```
   Available algorithms:
     - astar: Classical A* — used as the deterministic optimal baseline.
     - theta_star: Any-angle Theta* — produces shorter, smoother paths than A* on grids.
     - apf_refined: Refines an A* path by repelling waypoints from nearby obstacles.
   ```
2. **Current conditions**: Merged from `context.environment` plus `map_difficulty` and computed `obstacle_ratio`.
   ```
   Current conditions:
     - obstacle_density: high
     - battery: low
     - target_moving: True
     - map_difficulty: complex
     - obstacle_ratio: 27.89%
   ```

System prompt instructs the LLM to output strictly:
```
Selected: <algorithm_name>
Reasoning: <one sentence>
```

**Act**: Parses the LLM response with two regex patterns:
- `Selected:\s*([^\n]+)` → extract algorithm name, lowercased.
- `Reasoning:\s*([^\n]+)` → extract reasoning sentence.

Validates the extracted name against `registry.names()`. If the name is unknown, falls back to `"astar"`. Executes the selected planner and returns its path with the LLM's reasoning attached.

**Reflect**: Logs the selection decision at INFO level:
```
Selector chose 'theta_star' for map 'cec2024_c01' (difficulty=complex):
  Low battery favours the shorter path from Theta*.
```

#### AlgorithmRegistry

The registry (`packages/selector/src/agenticpathplanner/selector/registry.py`) is a simple `dict[str, BasePlanner]` wrapper. `AlgorithmRegistry.default()` pre-registers A\*, Theta\*, and APF.

New algorithms from any subproject can be added:
```python
reg = AlgorithmRegistry.default()
reg.register("rrt_star", RRTStarPlanner())
```

The `descriptions()` method extracts the first line of each planner class's docstring — these descriptions are what the LLM reads when making its selection decision.

#### Key Design Points

- **Explainability**: Every selection comes with a one-sentence natural-language justification. This is the key advantage over rule-based systems from a deployment perspective.
- **Extensibility**: Adding a new algorithm to the registry automatically makes it available for selection without changing the LLM prompt structure.
- **Graceful degradation**: Unknown algorithm names fall back to A\* rather than raising an exception.
- **Low temperature (0.1)**: Selection is a classification task — low temperature maximises consistency.
- **Conditions are open-ended**: Unlike the Rule-Based Selector which hardcodes `obstacle_ratio`, the LLM can reason over any key–value pair in `context.environment`, including ones the system designer did not anticipate.

#### Comparison with Rule-Based Selector

| Aspect | RuleBasedSelector | SelectorAgent |
|---|---|---|
| Condition space | `obstacle_ratio` only | Any key–value pairs |
| Extensibility | Requires code changes | Add to `environment` dict |
| Explainability | Fixed string from `selection_reason()` | LLM-generated sentence |
| Reliability | Deterministic, always correct for known cases | Probabilistic, may err |
| Speed | Microseconds | LLM inference time (~1–10s) |
| New algorithms | Must update rules | Registry update only |

---

## 6. V1 Classical Results

Results from `case_study/case_study_report.md` (classical-only run, `--no-llm`).

### cec2024_s01 (simple, 10×10)

| Method | Valid | PLR ↓ | SD ↑ | SM ↓ | Path len |
|---|:---:|---:|---:|---:|---:|
| A\* | ✅ | **1.000** | 1.00 | **0.00** | 9.9 |
| Theta\* | ✅ | **1.000** | **3.00** | **0.00** | 9.9 |
| APF | ✅ | 1.082 | 1.00 | 3.00 | 10.7 |
| Random Selector | ✅ | 1.082 | 1.00 | 3.00 | 10.7 |
| Rule-Based Selector | ✅ | **1.000** | 1.00 | **0.00** | 9.9 |
| Oracle Selector | ✅ | **1.000** | 1.00 | **0.00** | 9.9 |

Theta\* matches A\* on PLR but achieves 3× higher SD — the path stays further from the wall gap.

### cec2024_m01 (medium, 20×20)

| Method | Valid | PLR ↓ | SD ↑ | SM ↓ | Path len |
|---|:---:|---:|---:|---:|---:|
| A\* | ✅ | 1.122 | 1.00 | 4.71 | 27.0 |
| Theta\* | ✅ | **1.080** | 1.00 | **2.09** | 26.0 |
| APF | ✅ | 1.345 | **2.00** | 12.96 | 32.3 |
| Random Selector | ✅ | 1.122 | 1.00 | 4.71 | 27.0 |
| Rule-Based Selector | ✅ | 1.122 | 1.00 | 4.71 | 27.0 |
| Oracle Selector | ✅ | **1.080** | 1.00 | **2.09** | 26.0 |

Theta\* wins on PLR and SM. APF improves SD (2.00 vs 1.00) at the cost of a 20% longer path and very poor smoothness. The Rule-Based Selector falls short by picking A\* (obstacle ratio is exactly at the 0.15 threshold).

### cec2024_c01 (complex, 30×30)

| Method | Valid | PLR ↓ | SD ↑ | SM ↓ | Path len |
|---|:---:|---:|---:|---:|---:|
| A\* | ✅ | 1.061 | 1.00 | 9.42 | 40.5 |
| Theta\* | ✅ | **1.015** | 1.00 | **1.80** | 38.8 |
| APF | ❌ | — | — | — | — |
| Random Selector | ✅ | 1.061 | 1.00 | 9.42 | 40.5 |
| Rule-Based Selector | ✅ | **1.015** | 1.00 | **1.80** | 38.8 |
| Oracle Selector | ✅ | **1.015** | 1.00 | **1.80** | 38.8 |

**APF fails** — 28% obstacle density causes the snap-to-free step to produce invalid waypoints. Theta\* dominates: near-optimal PLR (1.5% above straight-line), smooth path. Rule-Based Selector correctly routes to Theta\* (ratio 0.28 < 0.30 threshold).

---

## 7. Method Comparison Summary

| Method | Type | PLR optimises | SD optimises | Handles density | Reasoning | Extensible |
|---|---|:---:|:---:|:---:|:---:|:---:|
| A\* | Classical | ✅ Best | ❌ Minimal | ✅ Always valid | None | No |
| Theta\* | Classical | ✅ Best | ✅ Higher than A\* | ✅ Always valid | None | No |
| APF | Classical | ❌ Longer | ✅ Goal | ❌ Fails >25% | None | No |
| Random Selector | Selector baseline | Random | Random | Depends on luck | None | Via registry |
| Rule-Based Selector | Selector baseline | Good (known cases) | Good (known cases) | Good (known cases) | Fixed string | Code change |
| Oracle Selector | Selector baseline | Optimal | Optimal | Optimal | None | Via registry |
| **RefinerAgent** | **LLM Idea 1** | Maintained | **Target: improve** | **Target: handle** | **Natural language** | Via environment dict |
| **GeneratorAgent** | **LLM Idea 2** | Mission-dependent | Mission-dependent | Mission-dependent | **Natural language** | Via mission string |
| **SelectorAgent** | **LLM Idea 3** | Inherits selected | Inherits selected | Inherits selected | **Natural language** | Via registry + conditions |

**Key takeaways from V1**:
- Theta\* is the strongest classical baseline on all three maps.
- APF's safety distance improvement comes at an unacceptable cost on dense maps.
- The LLM Refiner is directly motivated by APF's failure mode: semantic reasoning over dense environments.
- The LLM Selector is directly motivated by the Rule-Based Selector's rigidity: threshold rules miss the optimal choice on `cec2024_m01`.
- The LLM Generator is independently motivated: no classical method can adapt its algorithm to mission semantics at runtime.
