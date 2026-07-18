# Generator Experiment Plan
## Comparing LLM-as-Algorithm-Generator (Idea 2) Against Classical Baselines on CEC2024

---

## 1. Research Questions

This experiment answers four concrete questions about `GeneratorAgent`:

| # | Question | Measured by |
|---|---|---|
| **RQ1** | Does a mission-specific generated algorithm outperform the task-agnostic classical baseline for that mission's primary metric? | Per-mission metric comparison (PLR / SD / SM) |
| **RQ2** | Does providing a rich mission description change what algorithm the LLM writes, compared to a generic "find shortest path" prompt? | Ablation: rich vs generic mission on same map |
| **RQ3** | How reliable is code generation? What fraction of LLM calls produce valid, executable, path-returning code? | Code validity rate and success rate (SR) |
| **RQ4** | Which model and temperature produce the best path quality and highest code validity? | Model and temperature sweep |

---

## 2. Methods Under Comparison

### Classical Baselines (deterministic)

| Method | Primary strength | Role in this study |
|---|---|---|
| `astar` | Minimises PLR (shortest path) | Specialist for efficiency missions |
| `theta_star` | Minimises PLR + SM (any-angle, smooth) | Specialist for smoothness missions |
| `apf_refined` | Maximises SD (safety distance) | Specialist for safety missions |
| `rrt_star` | Sampling diversity, any-angle | Specialist for diversity/exploration |
| `mpc` | Balances SD + length (receding-horizon) | Specialist for multi-objective missions |

### LLM Generator (stochastic)

| Method | Description |
|---|---|
| `generator_rich` | `GeneratorAgent` with a full mission-specific prompt per difficulty/mission type |
| `generator_generic` | `GeneratorAgent` with a generic "find shortest path" prompt (ablation — Exp 2) |

The `generator_rich` condition is the main subject. `generator_generic` serves as the ablation control.

---

## 3. Evaluation Metrics

### Path Quality Metrics (from CEC2024)

| Metric | Direction | Definition |
|---|---|---|
| **PLR** (Path Length Ratio) | lower ↓ | `path_length / straight_line_distance` |
| **SD** (Safety Distance) | higher ↑ | `min Euclidean distance to any obstacle cell` |
| **SM** (Smoothness) | lower ↓ | `sum of interior turning angles (radians)` |
| **SR** (Success Rate) | higher ↑ | `valid_runs / total_runs` (over N repeats) |

### Code Generation Metrics (Generator-specific)

| Metric | Definition |
|---|---|
| **CVR** (Code Validity Rate) | Fraction of LLM calls where `exec()` succeeds without exception |
| **PFR** (Plan Function Rate) | Fraction of calls where a callable `plan` function is found post-exec |
| **PHR** (Path Hit Rate) | Fraction of calls where `plan()` returns a list reaching the goal |
| **elapsed_s** | Wall-clock time per generation + execution cycle |

> PHR ≡ SR for the Generator. CVR ≥ PFR ≥ PHR; the gaps reveal where failures occur.

---

## 4. Mission Taxonomy

The Generator's distinguishing feature is mission-conditioned code generation. Each mission type is designed to elicit a different algorithmic priority and maps to a different primary comparison metric.

### Mission Types (4 types × 3 difficulty levels = 12 conditions)

| Type | Mission string template | Primary metric | Expected behaviour | Specialist baseline |
|---|---|---|---|---|
| **efficiency** | "Find the shortest collision-free path with minimal detours. Time is critical; optimise total distance." | PLR ↓ | A*-like greedy search | `astar` |
| **safety** | "Maximise clearance from all obstacles at all times. Safety is the top priority; longer paths are acceptable." | SD ↑ | APF-like repulsion or safety-weighted cost | `apf_refined`, `mpc` |
| **smoothness** | "Generate a smooth path with minimal sharp turns. The vehicle has a limited turning radius; sharp corners are penalised." | SM ↓ | Any-angle or spline-like search | `theta_star` |
| **multi** | "Balance path length and obstacle clearance. Navigate efficiently while staying away from walls where possible." | PLR + SD | Safety-weighted A* or MPC-like | `mpc`, `theta_star` |

### Mission Strings per Difficulty

The full mission strings used in the experiment (passed as `context.mission`):

**efficiency:**
```
simple:  "Navigate from start to goal in a small open environment. Minimise total path
          distance. There are no special constraints."
medium:  "Navigate a corridor maze quickly. Time is critical. Find the shortest valid
          path. Do not sacrifice length for clearance."
complex: "Navigate a dense cluttered environment under time pressure. Find any valid
          path as quickly as possible; favour direct routes."
```

**safety:**
```
simple:  "Navigate safely with maximum clearance from walls. Never pass closer than
          2 cells to any obstacle. Accept longer paths to stay safe."
medium:  "Navigate a corridor maze with maximum obstacle clearance. The platform is
          fragile; prioritise clearance over path length."
complex: "Navigate a dense cluttered environment with extreme caution. Maximise the
          minimum distance to any obstacle at every waypoint."
```

**smoothness:**
```
simple:  "Generate a smooth path with minimal direction changes. The vehicle cannot
          turn sharply. Prefer wide arcs over tight corners."
medium:  "Navigate the corridor maze with the fewest and gentlest turns. Avoid
          zigzag paths. Smooth continuous movement is required."
complex: "Navigate the cluttered environment as smoothly as possible. Minimise total
          turning angle. Straight or gently curved segments are preferred."
```

**multi:**
```
simple:  "Find a path that is both efficient and safe. Stay reasonably close to the
          straight-line distance while keeping at least 1 cell from obstacles."
medium:  "Navigate the corridor maze balancing speed and safety. Do not hug walls.
          Aim for a path that is both short and clear of obstacles."
complex: "Navigate dense clutter efficiently without sacrificing safety entirely.
          Prioritise finding a valid path, but avoid unnecessary obstacle proximity."
```

---

## 5. Experimental Design

### Experiment 1 — Main Comparison (RQ1)

**Goal**: Determine whether the Generator matches or outperforms the specialist classical baseline for each mission type.

**Design**: Full factorial over mission type × map × method.

| Factor | Levels |
|---|---|
| Mission type | efficiency, safety, smoothness, multi (4) |
| Map | cec2024_s01, cec2024_m01, cec2024_c01 (3) |
| Method | astar, theta_star, apf_refined, rrt_star, mpc, generator_rich (6) |
| Runs | 10 per (map × method) for stochastic methods; 1 for deterministic |

**Total LLM calls**: 4 missions × 3 maps × 10 runs = **120 calls**
**Total classical runs**: 5 methods × 3 maps × 1 run = **15 runs** (deterministic, reused)

**Primary metric per mission**:
- `efficiency` → PLR
- `safety` → SD
- `smoothness` → SM
- `multi` → (PLR + 1/SD) composite or both reported separately

**Hypothesis**:
- H1a: Generator matches A* PLR on `efficiency` missions (PLR within ±0.05 on ≥2/3 maps).
- H1b: Generator achieves SD ≥ APF on `safety` missions on `cec2024_c01` (where APF fails).
- H1c: Generator achieves SM ≤ Theta* on `smoothness` missions on ≥1 map.
- H1d: Generator SR ≥ 0.70 on `efficiency` missions (code generation reliable enough to use).

---

### Experiment 2 — Mission Ablation (RQ2)

**Goal**: Verify that mission context changes the generated algorithm, not just the wording.

**Design**: Paired comparison of rich vs generic mission on same map.

| Factor | Levels |
|---|---|
| Condition | `generator_rich` vs `generator_generic` |
| Mission | safety, smoothness (the two most distinctive) |
| Map | cec2024_m01 (medium — enough structure to see differences) |
| Runs | 10 per condition |

**Total LLM calls**: 2 missions × 2 conditions × 10 runs = **40 calls**

**Metrics**: SD (for safety mission), SM (for smoothness mission), PLR for both.

**Hypothesis**:
- H2a: `generator_rich` achieves higher SD than `generator_generic` on `safety` mission (SD_rich > SD_generic in ≥7/10 runs).
- H2b: `generator_rich` achieves lower SM than `generator_generic` on `smoothness` mission.

**Analysis**: Paired Wilcoxon signed-rank test on the 10 run pairs (non-parametric, small N).

---

### Experiment 3 — Code Reliability (RQ3)

**Goal**: Characterise the code generation failure modes and their frequency.

**Design**: High-repetition run with failure tracking.

| Factor | Levels |
|---|---|
| Map | all 3 (to see if map complexity affects code quality) |
| Mission | efficiency (simplest code expected → baseline reliability) |
| Model | primary model (`qwen2.5-coder:7b`) |
| Runs | **20** per map |

**Total LLM calls**: 3 maps × 20 runs = **60 calls**

**Metrics tracked per call**:
1. `code_generated` (bool): did the LLM return non-empty content?
2. `code_valid` (bool): did `exec()` complete without exception?
3. `plan_fn_found` (bool): is `plan` a callable in the namespace?
4. `path_reached_goal` (bool): does `plan(grid, start, goal)[-1] == goal`?
5. `path_collision_free` (bool): are all segments free?
6. `elapsed_s` (float): wall-clock time for the full agent cycle

**Derived rates**: CVR = mean(code_valid), PFR = mean(plan_fn_found), PHR = SR = mean(path_reached_goal ∧ path_collision_free)

**Hypothesis**:
- H3a: CVR ≥ 0.85 on all maps (code mostly compiles).
- H3b: PHR degrades as map complexity increases (simple > medium > complex).
- H3c: The gap between CVR and PHR is mainly due to runtime errors in `plan()`, not compilation failures.

---

### Experiment 4 — Model and Temperature Sweep (RQ4)

**Goal**: Identify the best model + temperature setting for code generation quality.

**Sub-experiment 4a — Model comparison**:

| Factor | Levels |
|---|---|
| Model | qwen2.5-coder:7b, qwen2.5-coder:3b, codellama:13b (3) |
| Mission | efficiency (controlled for content) |
| Map | cec2024_m01 (controlled, medium difficulty) |
| Runs | 10 per model |

**Total LLM calls**: 3 models × 10 runs = **30 calls**

**Sub-experiment 4b — Temperature sweep**:

| Factor | Levels |
|---|---|
| Temperature | 0.0, 0.1, 0.3, 0.5 (4) |
| Model | qwen2.5-coder:7b (fixed) |
| Mission | efficiency |
| Map | cec2024_m01 |
| Runs | 10 per temperature |

**Total LLM calls**: 4 temps × 10 runs = **40 calls**

**Metrics**: SR, PLR_mean, CVR per configuration.

**Hypothesis**:
- H4a: Larger models (7b > 3b) achieve higher SR and lower PLR.
- H4b: Temperature 0.1 maximises SR (most deterministic code), while 0.3–0.5 introduces variation that occasionally finds better paths but also more failures.

---

## 6. Implementation Plan

### Files to Create / Modify

| File | Action | Purpose |
|---|---|---|
| `experiments/run_generator_experiment.py` | **Create** | Main experiment runner implementing all 4 experiments |
| `experiments/configs/generator_experiment.yaml` | **Create** | Full config with all mission strings and sweep parameters |
| `experiments/results/generator/` | Auto-created | Output directory for CSVs, JSONs, figures |

### Runner Script Architecture

```
run_generator_experiment.py
├── Exp1Runner.run()   → results/generator/exp1_main_comparison/
├── Exp2Runner.run()   → results/generator/exp2_mission_ablation/
├── Exp3Runner.run()   → results/generator/exp3_reliability/
└── Exp4Runner.run()   → results/generator/exp4_model_sweep/
```

Each runner:
1. Loads config for that experiment
2. Instantiates `GeneratorAgent` and classical planners
3. Runs N repetitions, catching all exceptions gracefully
4. Saves per-run raw results (JSON) + aggregated results (CSV)
5. Generates a per-experiment comparison figure

### Output Structure

```
results/generator/
├── exp1_main_comparison/
│   ├── exp1_results.json          ← per-run raw data
│   ├── exp1_summary.csv           ← aggregated by method × map × mission
│   ├── exp1_figures/
│   │   ├── cec2024_s01_efficiency.png
│   │   ├── cec2024_m01_safety.png
│   │   └── ...
│   └── exp1_report.md             ← auto-generated comparison tables
├── exp2_mission_ablation/
│   ├── exp2_results.json
│   ├── exp2_summary.csv
│   └── exp2_report.md
├── exp3_reliability/
│   ├── exp3_results.json
│   ├── exp3_reliability_breakdown.csv   ← CVR / PFR / PHR per map
│   └── exp3_report.md
└── exp4_model_sweep/
    ├── exp4a_model_results.json
    ├── exp4b_temp_results.json
    └── exp4_report.md
```

---

## 7. Execution Instructions

### Prerequisites

```bash
# Pull required Ollama models
ollama pull qwen2.5-coder:7b    # primary
ollama pull qwen2.5-coder:3b    # Exp 4a
ollama pull codellama:13b       # Exp 4a (optional — large download)

# Verify Ollama is running
curl http://localhost:11434/api/tags
```

### Run Commands

```bash
# Run all experiments (full suite — ~250 LLM calls, ~30–90 min depending on hardware)
python experiments/run_generator_experiment.py --all

# Run individual experiments
python experiments/run_generator_experiment.py --exp 1
python experiments/run_generator_experiment.py --exp 2
python experiments/run_generator_experiment.py --exp 3
python experiments/run_generator_experiment.py --exp 4

# Dry-run: classical baselines only (no Ollama required)
python experiments/run_generator_experiment.py --exp 1 --no-llm

# Override model
python experiments/run_generator_experiment.py --exp 1 --model qwen2.5-coder:3b

# Override runs per condition (quick test)
python experiments/run_generator_experiment.py --exp 1 --runs 3
```

---

## 8. Analysis Plan

### Exp 1 — Main Comparison

For each `(mission_type, map)` cell:
1. Report `mean ± std` for PLR, SD, SM across N runs per method.
2. Identify the **specialist classical baseline** for this mission type (see §4).
3. Compute **relative delta**: `(Generator_metric − Specialist_metric) / Specialist_metric × 100%`.
4. A negative delta on PLR (or SM) or positive delta on SD means Generator wins.

**Winner determination**: Generator is considered to "win" a cell if its primary metric is within ±5% of the specialist classical baseline AND SR ≥ 0.70.

**Visualisation**: Heatmap grid (mission_type × map) with cells coloured green (Generator wins), yellow (within 10%), red (Generator loses badly).

### Exp 2 — Mission Ablation

For each metric value pair `(rich_i, generic_i)` over 10 runs:
- Wilcoxon signed-rank test (two-sided, α=0.05).
- Report W statistic, p-value, and effect size (r = Z/√N).
- If p < 0.05: mission text demonstrably changes the generated algorithm.

### Exp 3 — Reliability

Report the failure funnel:
```
100%  LLM called
 CVR% code compiled
 PFR% plan() found
 PHR% path reached goal (SR)
```

Breakdown per map: if CVR drops on the complex map, it suggests the large ASCII grid confuses the LLM. If PFR drops but CVR stays, the LLM generates non-planning code.

### Exp 4 — Model/Temperature

- Sort models by SR (primary) then PLR_mean (secondary).
- Plot SR vs temperature as a line chart with error bars (10 runs each).
- Identify the "elbow" temperature: lowest temperature achieving max SR.

---

## 9. Hypotheses Summary and Pass/Fail Criteria

| Hypothesis | Pass criterion |
|---|---|
| H1a (Generator matches A* efficiency) | PLR within ±0.05 of A* on ≥2/3 maps with SR ≥ 0.70 |
| H1b (Generator beats APF on complex) | SD_generator > SD_apf on c01 (APF fails, any valid SD wins) |
| H1c (Generator approaches Theta* smoothness) | SM within ±2 rad of Theta* on ≥1 map |
| H1d (Acceptable SR on efficiency) | SR ≥ 0.70 on all 3 maps |
| H2a (Mission context changes SD) | Wilcoxon p < 0.05, SD_rich > SD_generic |
| H2b (Mission context changes SM) | Wilcoxon p < 0.05, SM_rich < SM_generic |
| H3a (Code compiles reliably) | CVR ≥ 0.85 on all maps |
| H3b (SR degrades with map complexity) | SR_simple > SR_complex |
| H3c (Main failure is runtime not compile) | PFR/CVR ≥ 0.80 (most compiling code defines plan()) |
| H4a (Larger model wins) | SR_7b > SR_3b by ≥ 0.10 |
| H4b (Low temp maximises SR) | argmax(SR) at temperature ≤ 0.1 |

---

## 10. Known Risks and Mitigations

| Risk | Mitigation |
|---|---|
| LLM generates non-8-connected movement (diagonal skips) | CodeExecutor returns `[]`; fallback path reported as INVALID; counted in SR |
| Generated code times out (infinite loop) | Wrap `plan()` call in `signal.alarm` / thread timeout (2 s limit) in executor |
| Ollama model not pulled | Script checks before each experiment and prints pull command |
| APF fails on c01 (known) | Still included for completeness; noted in report as expected |
| Low SR invalidates statistical comparison | Report SR alongside all metrics; label results with SR < 0.50 as "unreliable" |
| RNG in RRT* makes path vary | RRT* already uses fixed seed; set `seed=run_index` per run for reproducible variation |

---

## 11. Total Experiment Budget

| Experiment | LLM calls | Classical runs | Est. time @ 30s/call |
|---|---|---|---|
| Exp 1 — Main comparison | 120 | 15 | ~60 min |
| Exp 2 — Mission ablation | 40 | 0 | ~20 min |
| Exp 3 — Reliability | 60 | 0 | ~30 min |
| Exp 4a — Model sweep | 30 | 0 | ~15 min |
| Exp 4b — Temp sweep | 40 | 0 | ~20 min |
| **Total** | **290** | **15** | **~145 min** |

*Actual time depends heavily on hardware (GPU vs CPU) and model size.*

---

## 12. Expected Findings

Based on the V1 classical results and known LLM code generation behaviour:

**Likely outcomes:**
- On `efficiency` missions, Generator will produce A*-like code and match A* PLR on the simple and medium maps, but SR will drop on the complex map due to longer prompts.
- On `safety` missions, Generator will sometimes insert clearance penalties into A* cost functions, achieving SD between A* (1.0) and APF/MPC, without APF's dense-map failure.
- On `smoothness` missions, Generator rarely produces any-angle traversal; it more likely produces A* with some post-smoothing, achieving SM between A* and Theta*.
- Mission ablation will show statistically significant difference for `safety` (clearest objective) but weaker signal for `smoothness` (harder to encode in code).
- CVR ≥ 0.90 on simple/medium maps; drops to 0.70–0.85 on complex (large ASCII grid).

**Negative result** (acceptable and publishable): If the Generator does not outperform the relevant specialist classical baseline on any mission type, the conclusion is that current 7B-scale models cannot reliably implement mission-aware algorithms — a meaningful finding that motivates larger models or few-shot examples.
