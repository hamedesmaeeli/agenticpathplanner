# Research Positioning: LLM-Based Path Planning Algorithm Generation

**Project:** Agentic Path Planner (`agenticpathplanner`)  
**Benchmark:** CEC2024 Path Planning Benchmark Suite  
**Date:** 2026-05-29

---

## 1. Core Research Ideas

Three subprojects are implemented in this repository, each corresponding to a distinct role for the LLM in path planning. The ideas originate from `docs/ideas.pptx`.

### Idea 1 — LLM as Refiner (`agenticpathplanner.refiner`)

A classical planner (e.g. A\*) produces a geometrically valid path first. The LLM then receives that path together with semantic environmental context — wind zones, electromagnetic interference radii, structural hazards, safety margins — and refines it into a safer, operationally sounder trajectory, even at the cost of a slightly longer route.

**Agentic cycle:**

```
perceive  →  initial path + environmental context from AgentContext
reason    →  LLM rewrites path given semantic constraints
act       →  parse refined waypoints; fall back to initial path on failure
reflect   →  validate refined path is still collision-free
```

**Example prompt context:**
```
Current path: [(0,0), (10,10), (20,20), (30,25)]
Context: High-voltage power lines at (15,15) with 5m interference radius.
         Wind speed at 20m altitude is 15 knots from North.
Task: Refine coordinates to maximise safety. Maintain ≥10m from power lines.
Output only the new coordinate list.
```

---

### Idea 2 — LLM as Generator (`agenticpathplanner.generator`)

Given a mission description in natural language, the LLM writes a complete Python path planning function — `plan(grid, start, goal)` — tailored to the mission's specific constraints and priorities. The generated code is executed in a sandboxed namespace and its output path is evaluated against CEC2024 metrics.

**Agentic cycle:**

```
perceive  →  mission description + map instance from AgentContext
reason    →  LLM writes plan(grid, start, goal) in Python
act       →  execute generated code in sandbox; extract path
reflect   →  log generated code for reproducibility
```

**Example prompt context:**
```
Mission: Search and rescue in a collapsed warehouse.
Constraints: Limited battery (15 min), narrow passages, moving debris.
Task: Generate a Python plan(grid, start, goal) function.
      Prioritise exploration and human detection (+100 reward).
      Heavy penalty (-50) for proximity to unstable structures.
Output only the function code.
```

---

### Idea 3 — LLM as Selector (`agenticpathplanner.selector`)

An `AlgorithmRegistry` holds named planner instances (A\*, RRT, APF, MPC, …). The LLM reads real-time environmental conditions and algorithm descriptions, then selects the most appropriate algorithm for the current situation and explains its reasoning in one sentence.

**Agentic cycle:**

```
perceive  →  map conditions + environment dict from AgentContext
reason    →  LLM outputs "Selected: <name>\nReasoning: <sentence>"
act       →  retrieve planner from registry; execute; return path
reflect   →  log selection decision for auditability
```

**Example prompt context:**
```
Available algorithms: [A-Star, RRT-Connect, APF, MPC]
Conditions: Dense forest (high obstacle density), drone speed 5 m/s.
Task: Select the best algorithm. Explain in one sentence.

Expected output:
Selected: APF
Reasoning: Reactive algorithms handle real-time dense avoidance better
           than global re-planners under unpredictable obstacle patterns.
```

---

## 2. Shared Agentic Interface

All three subprojects implement `BaseAgent` from `agenticpathplanner.core`, providing a consistent four-step agentic cycle that can be wired into any agentic AI framework (LangChain, CrewAI, Claude Agent SDK, etc.):

```python
class BaseAgent(ABC):
    def run(context: AgentContext) -> AgentResult
    def perceive(context)            -> dict        # read environment
    def reason(observation, context) -> dict        # LLM or logic
    def act(plan, context)           -> AgentResult # execute
    def reflect(result, context)     -> AgentResult # validate / log
```

---

## 3. Path Planning Algorithms from the Literature

The following tables enumerate established path planning algorithms by category — the pool from which the Selector agent chooses and from which the Generator agent draws inspiration when writing mission-tailored code.

### 3.1 Classical Graph Search

| Algorithm | Abbr. | Ref. | Core Idea |
|-----------|-------|------|-----------|
| Dijkstra's Algorithm | — | [6] | Greedy lowest-cost expansion; optimal for non-negative weights |
| A\* | A\* | [17] | f(n)=g(n)+h(n); optimal with admissible heuristic; standard grid baseline |
| Bidirectional A\* | Bi-A\* | [13] | Simultaneous forward/backward search; halves explored space |
| Theta\* | Theta\* | [40] | Any-angle A\* on grids via visibility checks; smoother paths |
| Jump Point Search | JPS | [15] | Symmetry-breaking; skips redundant nodes; >10× faster than A\* on uniform grids |
| JPS+ | JPS+ | [16] | JPS with offline preprocessing of jump-point tables |
| Dynamic A\* | D\* | [49] | Incremental replanning when map changes; suited for dynamic environments |
| D\* Lite | D\* Lite | [28] | Simpler D\* with LPA\* core; efficient re-use of prior search |

### 3.2 Sampling-Based

| Algorithm | Abbr. | Ref. | Core Idea |
|-----------|-------|------|-----------|
| Rapidly-Exploring Random Tree | RRT | [31] | Incremental tree toward random samples; handles high-DOF / non-holonomic constraints |
| RRT-Connect | RRT-C | [30] | Bidirectional RRT; faster connection in cluttered spaces |
| RRT\* | RRT\* | [23] | RRT + rewiring; asymptotically optimal |
| Informed RRT\* | IRRT\* | [10] | Samples inside an ellipsoidal heuristic region; faster convergence |
| Batch Informed Trees | BIT\* | [11] | Alternates approximation and search in batches; anytime optimal |
| Probabilistic Roadmap | PRM | [24] | Offline roadmap for multi-query use; suited to static environments |
| PRM\* | PRM\* | [23] | PRM with connection radius scaled for asymptotic optimality |
| Expansive Space Trees | EST | [48] | Expands toward under-explored regions; better narrow-passage handling than RRT |

### 3.3 Potential Field Methods

| Algorithm | Abbr. | Ref. | Core Idea |
|-----------|-------|------|-----------|
| Artificial Potential Field | APF | [27] | Attractive goal force + repulsive obstacle forces; fast but prone to local minima |
| Dynamic Window Approach | DWA | [8] | Evaluates feasible velocity samples in a local window; real-time reactive avoidance |
| Vector Field Histogram | VFH | [2] | Polar obstacle histogram; efficient for sensor-driven avoidance |
| VFH+ | VFH+ | [54] | VFH with path-width and narrowness awareness |

### 3.4 Evolutionary and Metaheuristic

| Algorithm | Abbr. | Ref. | Core Idea |
|-----------|-------|------|-----------|
| Genetic Algorithm | GA | [18] | Selection, crossover, mutation on path populations; multi-objective capable |
| Particle Swarm Optimisation | PSO | [25] | Swarm moves toward personal/global best; fast convergence on continuous spaces |
| Ant Colony Optimisation | ACO | [7] | Pheromone trails on graph edges; strong on discrete/grid problems |
| Differential Evolution | DE | [50] | Vector mutation/recombination; efficient continuous optimisation |
| Grey Wolf Optimiser | GWO | [37] | Wolf pack hierarchy (α/β/δ/ω) drives search; fewer parameters than PSO |
| Whale Optimisation Algorithm | WOA | [38] | Bubble-net hunting behaviour; strong global exploration |
| Firefly Algorithm | FA | [60] | Bioluminescence-based attraction; handles multi-modal landscapes |
| Bat Algorithm | BA | [61] | Echolocation metaphor; good constraint handling |
| Cuckoo Search | CS | [62] | Lévy-flight parasitic reproduction; simple with good convergence speed |
| Artificial Bee Colony | ABC | [22] | Employed/onlooker/scout bee roles; robust exploration |
| Harmony Search | HS | [12] | Musical improvisation analogy; mixed discrete/continuous problems |
| Dung Beetle Optimiser | DBO | [59] | Ball-rolling behaviour; competitive with PSO/WOA on recent benchmarks |
| Aquila Optimiser | AO | [1] | Aquila hunting strategy; strong exploration/exploitation balance |
| Hybrid PSO-ACO | PSO-ACO | [7, 25] | PSO for exploration, ACO for pheromone-guided exploitation |
| Hybrid GWO-DE | HGWODE | [37, 50] | GWO positions enhanced with DE rank-based mutation; 27–38% improvement over GWO |

### 3.5 Optimal Control and Trajectory Optimisation

| Algorithm | Abbr. | Ref. | Core Idea |
|-----------|-------|------|-----------|
| CHOMP | CHOMP | [64] | Covariant gradient descent on trajectory cost; smooth paths |
| STOMP | STOMP | [21] | Stochastic noisy samples + EM updates; escapes local minima better than CHOMP |
| TrajOpt | TrajOpt | [45] | Sequential convex optimisation; handles joint/collision constraints |
| Model Predictive Control | MPC | [36] | Receding-horizon optimisation with state/input constraints; standard for UAV |
| Model Predictive Path Integral | MPPI | [53] | Stochastic path integral receding horizon; stable for nonlinear dynamics |

### 3.6 Reinforcement Learning Planners

| Algorithm | Abbr. | Ref. | Core Idea |
|-----------|-------|------|-----------|
| Deep Q-Network | DQN | [39] | Neural Q-function for discrete actions |
| Double DQN | DDQN | [55] | Decouples selection/evaluation; reduces overestimation bias |
| Dueling DQN | Dueling-DQN | [56] | Separates value and advantage streams |
| Proximal Policy Optimisation | PPO | [47] | Clipped objective; stable on-policy learning with continuous actions |
| Deep Deterministic Policy Gradient | DDPG | [32] | Off-policy actor-critic for continuous action spaces |
| Twin Delayed DDPG | TD3 | [9] | Dual Q-networks + delayed updates; more stable than DDPG |
| Soft Actor-Critic | SAC | [14] | Maximum-entropy off-policy; excellent exploration via entropy bonus |

### 3.7 Neural and Deep Planners

| Algorithm | Abbr. | Ref. | Core Idea |
|-----------|-------|------|-----------|
| Value Iteration Networks | VIN | [52] | Embeds differentiable value iteration as convolutional layers |
| Motion Planning Networks | MPNet | [41] | Point-cloud encoder + neural planner; generalises to new geometries |
| Neural A\* | Neural-A\* | — | Learned heuristics inside the A\* search structure |
| Diffusion Policy | DP | [5] | Diffusion models over trajectory distributions; multi-modal planning |

### 3.8 Hybrid and Composite Methods

| Method | Ref. | Core Idea |
|--------|------|-----------|
| A\* + APF | [17, 27] | A\* global path, APF local smoothing |
| RRT + DWA | [31, 8] | RRT global plan, DWA local reactive execution |
| Informed RRT\* + DWA | [10, 8] | Informed-RRT\* fused with dynamic window for UAV |
| Task and Motion Planning | [20] | Interleaves symbolic task planning with continuous motion planning |

---

## 4. Is There a Path Planner Algorithm Generator in the Literature?

### 4.1 Short Answer

There is no established sub-field under the label "path planner algorithm generator." The concept is distributed across three adjacent research areas, none of which applies to grid-based path planning or the CEC2024 benchmark.

### 4.2 Adjacent Research Threads

**Hyper-heuristics (closest structural prior art)**

Hyper-heuristics operate on the space of heuristics or algorithms rather than on the problem directly. A *generation* hyper-heuristic synthesises new low-level heuristics from a grammar of primitives [3, 4]. Genetic programming [29] is the classical instantiation: it evolves executable heuristic functions rather than fixing them manually. However, hyper-heuristics have not been applied to grid-based path planning as a code-generation paradigm.

| Work | Ref. | Relevance |
|------|------|-----------|
| Hyper-heuristics: emerging direction | [3] | Survey of algorithm-generating approaches for combinatorial optimisation |
| Hyper-heuristics: state of the art | [4] | Foundational taxonomy: selection vs. generation hyper-heuristics |
| Genetic programming | [29] | Evolves heuristic functions as executable programs |

**LLM-as-algorithm-generator (most directly relevant)**

| Work | Ref. | What is generated | Domain |
|------|------|-------------------|--------|
| FunSearch | [43] | Python functions for combinatorial problems via LLM + evolution | Bin-packing, cap-set |
| Eureka | [35] | RL reward function code via GPT-4 + iterative feedback | Robot locomotion |
| ReEvo / EoH | [33] | Heuristic code for TSP, VRP via LLM hyper-heuristic | Combinatorial optimisation |
| Xie et al. | [58] | RL reward function + controller parameters for AUV | Underwater vehicle control |

**Automated Algorithm Design (AAD)**

| Work | Ref. | Relevance |
|------|------|-----------|
| SMAC | [19] | Configures algorithm hyperparameters; generates tuned variants |
| *irace* | [34] | Iterated racing for algorithm configuration |
| Algorithm Selection (Rice) | [42] | Formal problem of selecting the best solver from a portfolio |
| Algorithm Selection (Kerschke et al.) | [26] | Modern survey of portfolio-based selection methods |

### 4.3 The Gap

None of the above apply LLM-based executable algorithm generation to **grid-based path planning** or to the **CEC2024 benchmark**. The mechanism exists in adjacent domains [33, 35, 43, 58]; the domain application does not.

---

## 5. Related Works: Five-Paper Analysis

The following papers were proposed as potential primary references for the three subprojects. Each is assessed for category alignment.

### Paper 1 — UAV-VLA [44]

> Sautenkov, O., et al. (2025). UAV-VLA: Vision-language-action system for large scale aerial mission generation. *HRI 2025*. IEEE.

**What it does:** A vision-language model interprets a scene image and a natural-language goal, then outputs Python waypoint coordinates directly.

**Assessment:**
- Generates **waypoints** (a path solution), not a **planning algorithm**
- Output is a coordinate list; no algorithmic code is written or executed
- Closest to the `generate_path` mode of `agenticpathplanner.llm`
- No classical planner is involved; there is no refinement step

**Category:** Adjacent — same LLM-to-Python output modality, but the output is a *solution* not an *algorithm*. Relevant as a baseline comparison for the direct path-generation mode.

---

### Paper 2 — NeLV [63]

> Yuan, L., et al. (2025). Next-generation LLM for UAV: From natural language to autonomous flight. *arXiv:2510.21739*.

**What it does:** Defines a five-level hierarchy — Parser, Route Planner, Path Planner, Control, Monitor — where LLMs orchestrate each level of UAV autonomy.

**Assessment:**
- An **architectural framework** for LLM-guided autonomy, not an algorithm generator
- The Path Planner level invokes an existing algorithm; the LLM does not write one
- Closest to the `selector` subproject at a higher system-design level
- Relevant as an architectural reference for multi-agent orchestration

**Category:** Adjacent — architecturally inspirational for the layered agent design, but does not generate algorithms.

---

### Paper 3 — Chain-of-Thought Flight Planner [51]

> Tabrizian, A., Ghazanfari, M., & Wei, P. (2025). Chain-of-thought flight planner: End-to-end LLM routing under wind hazards. *AIAA AVIATION FORUM 2025*. https://doi.org/10.2514/6.2025-3711

**What it does:** LLM uses Chain-of-Thought reasoning to produce a flight route under wind constraints, achieving 98% path validity.

**Assessment:**
- LLM generates a **path with transparent reasoning**, not algorithm code
- Closest to the `refiner` subproject — environmental context shapes the output
- CoT reasoning is directly applicable to explainable Refiner and Generator outputs
- No classical planner runs first; the CoT approach is end-to-end, not a refinement layer

**Category:** Adjacent — important methodological reference for CoT-based explainability. Not in the algorithm-generation category.

---

### Paper 4 — LLM-Enhanced RL Controller for AUVs [58]

> Xie, G., et al. (2025). Never too prim to swim: An LLM-enhanced RL-based adaptive S-surface controller for AUVs under extreme sea conditions. *arXiv:2503.00527*.

**What it does:** An LLM writes and optimises a reward function and controller parameters as executable Python code for an RL agent controlling an underwater vehicle.

**Assessment:**
- LLM generates **executable Python code** (reward function) → code runs → behaviour is evaluated
- This is **the same mechanism** as the Generator subproject: LLM writes code, code executes, output is assessed
- Domain differs (AUV RL controller vs. UAV grid path planning), but the code-generation pattern is identical

**Category:** ✅ Same category as the Generator subproject — the closest paper in the existing literature to this work's core mechanism.

---

### Paper 5 — FM-Planner [57]

> Xiao, J., et al. (2025). FM-Planner: Foundation model guided path planning for autonomous drone navigation. *arXiv:2505.20783*.

**What it does:** Fine-tunes an LLM on paths generated by RRT\*, so the model learns to imitate RRT\* and produce drone trajectories directly at inference time.

**Assessment:**
- **Learns from** a classical algorithm via fine-tuning — the inverse direction of generation
- The LLM approximates an existing fixed algorithm; it does not create a new one
- Output is a trajectory, not algorithmic code

**Category:** Different — the LLM-algorithm relationship is reversed. Useful as a **contrast reference**: FM-Planner imitates a fixed classical algorithm; the Generator subproject creates a new one per mission.

---

### 5.1 Five-Paper Summary

| # | Paper | Ref. | Generates algorithm code? | Same category as Generator? | Most relevant subproject |
|---|-------|------|:-------------------------:|:---------------------------:|--------------------------|
| 1 | UAV-VLA | [44] | ✗ generates waypoints | No — adjacent | `llm` (generate_path mode) |
| 2 | NeLV | [63] | ✗ orchestrates modules | No — architectural | `selector` |
| 3 | CoT Flight Planner | [51] | ✗ generates path with reasoning | No — adjacent | `refiner` |
| **4** | **LLM-Enhanced RL / AUV** | **[58]** | **✅ generates reward function code** | **Yes — same mechanism** | **`generator`** |
| 5 | FM-Planner | [57] | ✗ imitates RRT\* via fine-tuning | No — opposite direction | contrast baseline |

---

## 6. Novelty of Each Idea Relative to the Literature

This section explicitly states what each subproject contributes beyond every directly comparable prior work.

### 6.1 Idea 1 — Refiner: Novelty Statement

**Prior works and their limits:**

| Work | Ref. | What it does | Why it falls short of this idea |
|------|------|--------------|----------------------------------|
| CoT Flight Planner | [51] | LLM generates route end-to-end using CoT | No classical planner runs first; there is no initial path to refine; environmental context is limited to wind |
| UAV-VLA | [44] | VLM generates waypoints from scene images | No refinement loop; no post-hoc safety validation; no classical planner baseline |
| APF + A\* hybrids | [17, 27] | Classical planner + local potential field for smoothing | Refinement is purely geometric (repulsive forces), not semantic; cannot reason about electromagnetic interference, regulations, or mission context |
| FM-Planner | [57] | LLM imitates RRT\* to produce trajectories | No semantic context; no mission-specific reasoning; no refinement of a prior path |

**Novel contribution of Refiner:**
The Refiner is the first approach that (a) runs a classical optimal planner (A\* [17]) to guarantee a geometrically valid baseline, then (b) uses an LLM to inject **semantic and operational intelligence** — hazard zones, regulatory constraints, mission-specific safety margins — that are invisible to geometry-only planners, and (c) validates the refined output for collision safety in a structured `reflect` step. No prior work combines all three elements in a unified agentic loop.

---

### 6.2 Idea 2 — Generator: Novelty Statement

**Prior works and their limits:**

| Work | Ref. | What it generates | Why it falls short of this idea |
|------|------|-------------------|----------------------------------|
| FunSearch | [43] | Python functions for combinatorial problems | Not applied to path planning; requires an evolutionary outer loop; no grid environment |
| Eureka | [35] | RL reward function code | Generates reward for RL training, not a complete path planning algorithm; not path planning domain |
| ReEvo / EoH | [33] | Heuristic code for TSP, VRP | Combinatorial optimisation, not grid-based path planning; no CEC2024 evaluation |
| Xie et al. | [58] | RL reward function + controller for AUV | AUV control, not path planning; generates reward and controller parameters, not a planning algorithm |
| Hyper-heuristics | [3, 4] | Low-level heuristic selection/generation | Grammar-based, not LLM-driven; not applied to path planning; no natural-language mission input |
| Genetic programming | [29] | Evolved heuristic expressions | Slow evolutionary synthesis; no natural-language mission description; no path planning domain |

**Novel contribution of Generator:**
The Generator is the first system to (a) accept a **natural-language mission description** as input, (b) use an LLM to write a **complete, executable `plan(grid, start, goal)` function** tailored to that mission's constraints and priorities, (c) execute the generated code in a sandboxed environment, and (d) evaluate the resulting path against the **CEC2024 standard metrics** (PLR, SD, SM, SR). The mission-to-algorithm pipeline, the path planning domain, and the CEC2024 benchmark application are all absent from prior work.

---

### 6.3 Idea 3 — Selector: Novelty Statement

**Prior works and their limits:**

| Work | Ref. | What it selects | Why it falls short of this idea |
|------|------|-----------------|----------------------------------|
| NeLV | [63] | LLM orchestrates hierarchy levels | Selection is structural (which layer runs), not condition-driven algorithm choice; no extensible registry |
| Rice (Algorithm Selection) | [42] | Formal model of solver selection | Theoretical framework; selection is done by a trained model, not by LLM reasoning; not real-time condition-aware |
| Kerschke et al. | [26] | Portfolio-based algorithm selection | Requires offline feature computation and trained classifier; no natural-language reasoning; not path planning |
| SMAC | [19] | Hyperparameter configuration | Configures a single algorithm's parameters; does not select across fundamentally different algorithms |
| *irace* | [34] | Iterated racing for configuration | Offline configuration of a fixed algorithm; no real-time condition awareness |
| Hyper-heuristics (selection) | [3, 4] | Low-level heuristic selection at runtime | Selects heuristic components, not complete planners; no LLM reasoning; no natural-language explanation |

**Novel contribution of Selector:**
The Selector is the first system to use an LLM as a **reasoning agent** that (a) reads real-time operational conditions expressed in natural language, (b) inspects an **extensible `AlgorithmRegistry`** of named planners, (c) produces a structured selection decision with a **one-sentence human-readable justification**, and (d) executes the chosen planner and returns a CEC2024-evaluated path. Prior algorithm selection methods rely on trained classifiers or configuration spaces; the Selector relies on the LLM's generalisation capacity to reason across arbitrary conditions without retraining.

---

### 6.4 Corrected Novelty Positioning Statement

> The LLM-as-code-generator paradigm was established by FunSearch [43] and Eureka [35] for combinatorial problems and RL reward design respectively, and extended to AUV control by Xie et al. [58]. This work is the first to apply that paradigm to **grid-based path planning**, with three distinct agentic roles evaluated on the **CEC2024 benchmark**: the Generator synthesises complete mission-tailored planning algorithms from natural language; the Refiner injects semantic operational context into classical planner outputs; and the Selector frames algorithm portfolio management as an LLM reasoning task with real-time condition awareness. None of these three contributions — individually or as a system — appear in the reviewed literature.

---

## 7. Comprehensive Comparison Table

The table below compares this work against all directly relevant prior works across nine dimensions.

| Dimension | **Refiner** (this work) | **Generator** (this work) | **Selector** (this work) | UAV-VLA [44] | NeLV [63] | CoT Planner [51] | LLM-AUV [58] | FM-Planner [57] | FunSearch [43] | Eureka [35] | ReEvo/EoH [33] |
|-----------|:-----------------------:|:-------------------------:|:------------------------:|:------------:|:---------:|:----------------:|:------------:|:---------------:|:--------------:|:-----------:|:--------------:|
| **LLM role** | Semantic refiner | Algorithm generator | Algorithm selector | Path generator | Orchestrator | Path reasoner | Code generator | Path imitator | Code generator | Code generator | Code generator |
| **Output type** | Refined waypoints | Executable `plan()` function | Algorithm name + reasoning | Waypoint list | Module invocation | Waypoint list | Reward function | Trajectory | Python function | Reward function | Heuristic function |
| **Executable code generated** | ✗ | ✅ | ✗ | ✗ | ✗ | ✗ | ✅ | ✗ | ✅ | ✅ | ✅ |
| **Classical planner as prerequisite** | ✅ (A\* first) | ✗ | ✅ (registry) | ✗ | ✅ (invoked) | ✗ | ✗ | ✅ (RRT\* for training) | ✗ | ✗ | ✗ |
| **Mission description as input** | ✗ (env. context) | ✅ | ✗ (conditions) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **Environmental/semantic context** | ✅ | ✗ | ✅ | ✗ | ✗ | ✅ (wind) | ✗ | ✗ | ✗ | ✗ | ✗ |
| **Domain** | UAV path planning | UAV path planning | UAV path planning | UAV mission | UAV autonomy | UAV routing | AUV control | UAV navigation | Combinatorial | Robot locomotion | Combinatorial |
| **Evaluated on CEC2024** | ✅ | ✅ | ✅ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **Agentic loop (perceive→reason→act→reflect)** | ✅ | ✅ | ✅ | ✗ | Partial | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **Safety validation step** | ✅ (reflect) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **Human-readable explanation** | ✗ | ✗ | ✅ (one sentence) | ✗ | ✗ | ✅ (CoT) | ✗ | ✗ | ✗ | ✗ | ✗ |
| **Local LLM (Ollama)** | ✅ | ✅ | ✅ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **Open-source / reproducible** | ✅ | ✅ | ✅ | ✗ | ✗ | ✗ | ✗ | ✗ | Partial | ✗ | ✅ |

---

## 8. Repository Structure Reference

```
packages/
├── core/          agenticpathplanner.core         MapInstance · GridWorld · BasePlanner · BaseAgent
├── benchmarks/    agenticpathplanner.benchmarks   CEC2024 maps · loader · PLR/SD/SM/SR metrics
├── classical/     agenticpathplanner.classical    A* (template for RRT, DWA, APF, …)
├── llm/           agenticpathplanner.llm          Ollama client · prompt builder · LLMPlanner
├── evaluation/    agenticpathplanner.evaluation   Evaluator · agentpp CLI
├── refiner/       agenticpathplanner.refiner      Idea 1: RefinerAgent
├── generator/     agenticpathplanner.generator    Idea 2: GeneratorAgent · CodeExecutor
└── selector/      agenticpathplanner.selector     Idea 3: SelectorAgent · AlgorithmRegistry
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

## 9. Recommended Comparison Methods

This section identifies the methods from the literature that should be implemented as experimental baselines for each subproject. Baselines are chosen to be fair (same inputs/outputs/metrics), informative (reveal what the LLM adds over classical approaches), and implementable within the existing repository structure.

---

### 9.1 Baselines for Idea 1 — Refiner

The Refiner's claim is: *"LLM refinement of a classical path improves on geometric-only post-processing."* The baselines must answer: what does the LLM add that classical smoothers cannot?

| Priority | Method | Ref. | Why it belongs here | Implementability |
|----------|--------|------|---------------------|-----------------|
| ⭐⭐⭐ | **Theta\*** | [40] | Produces any-angle smooth paths geometrically — the cleanest classical alternative to LLM refinement. If [40] Theta\* matches Refiner on SD/SM, the LLM adds nothing | Easy — inherits [17] A\* structure already in `classical` |
| ⭐⭐⭐ | **APF post-processing** | [27] | Applies repulsive forces to push an [17] A\* path away from obstacles — classical semantic-free refinement. Direct comparison: [27] APF vs LLM for path safety improvement | Medium — add as a post-processor in `classical` |
| ⭐⭐ | **CHOMP** | [64] | Gradient-based trajectory optimisation; optimises a path jointly for smoothness and obstacle clearance | Medium-hard — requires cost function gradient |
| ⭐ | **Raw A\*** | [17] | Already implemented; serves as the unrefined lower bound | Already in repo |

**Verdict:** Implement [40] **Theta\*** and [27] **APF post-processing**. Together they form the strongest classical challenge to the Refiner: [40] Theta\* tests whether geometric smoothing is enough; [27] APF tests whether obstacle clearance improvement requires an LLM.

---

### 9.2 Baselines for Idea 2 — Generator

The Generator's claim is: *"LLM-generated mission-tailored algorithm outperforms a fixed classical planner."* The baselines must isolate what mission-awareness and code generation each contribute.

| Priority | Method | Ref. | Why it belongs here | Implementability |
|----------|--------|------|---------------------|-----------------|
| ⭐⭐⭐ | **Direct LLM path generation** (`generate_path` mode) | — | Same LLM, same mission prompt, but outputs coordinates instead of code. Isolates the benefit of code generation over direct path answering | Already in repo — wire the ablation |
| ⭐⭐⭐ | **RRT\*** | [23] | Asymptotically optimal sampling-based planner; strong on complex/cluttered CEC2024 maps where [17] A\* finds suboptimal paths | Medium — standard algorithm |
| ⭐⭐ | **PSO path planning** | [25] | Population-based search directly optimises waypoints for a cost function — the closest classical analogue to mission-aware planning | Medium — objective function encodes mission constraints |
| ⭐⭐ | **Fixed A\*** | [17] | Already implemented; the optimal deterministic lower bound for path length | Already in repo |
| ⭐ | **ReEvo / EoH pattern** | [33] | LLM hyper-heuristic with an evolutionary outer loop — tests whether iterative refinement of generated code improves on single-shot generation | Hard — requires evolutionary loop |

**Verdict:** Implement [23] **RRT\*** and [25] **PSO**, and wire the existing **`generate_path` ablation**. The ablation (same LLM, no code) is the single most important comparison because it isolates the value of the code-generation step itself.

---

### 9.3 Baselines for Idea 3 — Selector

The Selector's claim is: *"LLM condition-aware reasoning outperforms fixed or rule-based algorithm selection."* Three non-LLM baselines fully characterise the performance envelope.

| Priority | Method | Ref. | Why it belongs here | Implementability |
|----------|--------|------|---------------------|-----------------|
| ⭐⭐⭐ | **Rule-based Selector** | — | Expert hand-coded if/else: high density → [27] APF, open space → [23] RRT\*, moving target → [36] MPC. Tests whether LLM reasoning beats human expert rules | Easy — a few `if` statements in a new planner |
| ⭐⭐⭐ | **Random Selector** | — | Uniform random choice from the registry. Establishes a lower bound; if the LLM barely beats random, the claim is weak | Trivial |
| ⭐⭐⭐ | **Oracle Selector** | — | Post-hoc best algorithm per scenario (picked by examining results). Establishes the theoretical upper bound for selection quality | Trivial — computed from other results |
| ⭐⭐ | **Always-A\*** | [17] | Fixed single-algorithm baseline — already in repo | Already in repo |
| ⭐ | **SMAC-style offline selector** | [19] | Trained classifier over scenario features selects algorithm. Tests whether the LLM generalises as well as a trained model | Hard — requires training data |

**Verdict:** Implement **Rule-based**, **Random**, and **Oracle** selectors. These three together produce the full performance envelope (floor → human expert → ceiling) against which the LLM Selector is evaluated. All are trivial to add as new agents under `packages/selector/`.

---

### 9.4 Implementation Roadmap

Ordered by impact-to-effort ratio:

| # | Method | Ref. | Subproject | Effort | Impact |
|---|--------|------|-----------|--------|--------|
| 1 | Direct LLM path ablation (`generate_path`) | — | Generator | Zero — already in repo | Highest — isolates code-generation value |
| 2 | Random Selector | — | Selector | 1 hour | High — floor baseline |
| 3 | Oracle Selector | — | Selector | 1 hour | High — ceiling baseline |
| 4 | Rule-based Selector | [27, 23, 36] | Selector | Half day | High — human expert comparison |
| 5 | **Theta\*** | [40] | Refiner | 1 day | High — geometric smoothing baseline |
| 6 | **APF post-processing** | [27] | Refiner | 1 day | High — obstacle clearance baseline |
| 7 | **RRT\*** | [23] | Generator + Selector registry | 2 days | Medium — strong sampling-based baseline |
| 8 | **PSO path planner** | [25] | Generator | 2 days | Medium — mission-cost-aware baseline |
| 9 | CHOMP | [64] | Refiner | 3 days | Medium — trajectory optimisation baseline |

---

### 9.5 Minimum Viable Comparison Set for a Paper

Items 1–7 in the roadmap cover all three ideas with a defensible baseline set:

- **Refiner**: [17] A\* (raw) vs. [40] Theta\* vs. [27] APF-refined vs. LLM-Refiner
- **Generator**: [17] A\* vs. [23] RRT\* vs. [25] PSO vs. LLM direct path vs. LLM-Generator
- **Selector**: Random vs. [17] Always-A\* vs. Rule-based ([27, 23, 36]) vs. Oracle vs. LLM-Selector

All five baselines — [40] Theta\*, [27] APF, [23] RRT\*, Rule-based, and Random/Oracle — can be added under `packages/classical/` or `packages/selector/` without modifying any existing code.

---

## 10. Recommended Ollama LLMs for Implementation

The three ideas have different LLM capability requirements. The table below maps each subproject to its optimal locally-hosted model and explains the rationale.

### 10.1 Capability Requirements

| Subproject | Primary capability | Secondary capability |
|-----------|-------------------|---------------------|
| **Generator** | Python code generation — must write a correct, executable `plan(grid, start, goal)` function | Algorithm knowledge (A\*, BFS, heuristics, heapq) |
| **Refiner** | Spatial reasoning + instruction following — must output valid coordinate lists under constraints | Understanding of safety and environmental context |
| **Selector** | Structured output + algorithmic reasoning — must produce `Selected: <name>\nReasoning: <sentence>` reliably | Knowledge of algorithm trade-offs |

---

### 10.2 Recommended Models by Subproject

#### Generator — code generation is critical

| Rank | Model | Size | Rationale |
|------|-------|------|-----------|
| ⭐⭐⭐ | `qwen2.5-coder:14b` | 14 B | Best code quality among locally-runnable models; strong on Python algorithm generation; reliably follows output format constraints |
| ⭐⭐⭐ | `qwen2.5-coder:7b` | 7 B | Near-identical quality to 14 B for short functions; runs on 8 GB VRAM; best size/quality trade-off |
| ⭐⭐ | `deepseek-coder-v2:16b` | 16 B | Excellent at algorithmic thinking and structured code; good at `heapq`/`numpy` usage needed for path planners |
| ⭐⭐ | `codellama:13b` | 13 B | Well-tested, reliable Python code generation; older but stable |
| ⭐ | `qwen2.5-coder:3b` | 3 B | CPU-friendly fallback; generates plausible but sometimes incomplete functions |

#### Refiner — spatial reasoning and format compliance

| Rank | Model | Size | Rationale |
|------|-------|------|-----------|
| ⭐⭐⭐ | `llama3.1:8b` | 8 B | Strong instruction following and spatial reasoning; reliably outputs coordinate lists in the correct format |
| ⭐⭐⭐ | `qwen2.5:7b` | 7 B | Good at structured output; handles environmental context descriptions well |
| ⭐⭐ | `mistral:7b` | 7 B | Lightweight and highly reliable at format compliance; good for iterative refinement loops |
| ⭐⭐ | `phi3.5:3.8b` | 3.8 B | Strong reasoning for its size; handles constraint-following well on CPU |
| ⭐ | `gemma2:9b` | 9 B | Good contextual understanding; slightly weaker at coordinate output precision |

#### Selector — structured reasoning and reliable output format

| Rank | Model | Size | Rationale |
|------|-------|------|-----------|
| ⭐⭐⭐ | `llama3.1:8b` | 8 B | Best at following the strict `Selected: / Reasoning:` format; strong algorithm knowledge |
| ⭐⭐⭐ | `mistral:7b` | 7 B | Extremely reliable at structured output; low hallucination rate on algorithm names |
| ⭐⭐ | `qwen2.5:7b` | 7 B | Good reasoning; handles condition descriptions in natural language well |
| ⭐⭐ | `phi3:14b` | 14 B | Strong reasoning capability; good at explaining algorithm trade-offs |
| ⭐ | `llama3.2:3b` | 3 B | Lightweight fallback; format compliance is inconsistent at this size |

---

### 10.3 Hardware Tiers

| Tier | VRAM / RAM | Recommended models |
|------|-----------|-------------------|
| **High** (24 GB+ VRAM) | GPU | `qwen2.5-coder:14b`, `deepseek-coder-v2:16b`, `phi3:14b` |
| **Mid** (8–12 GB VRAM) | GPU | `qwen2.5-coder:7b`, `llama3.1:8b`, `mistral:7b`, `gemma2:9b` |
| **Low** (CPU / 16 GB RAM) | CPU | `qwen2.5-coder:3b`, `phi3.5:3.8b`, `llama3.2:3b` |

---

### 10.4 Summary Recommendation Table

| Subproject | Primary (best quality) | Secondary (balanced) | Lightweight (CPU) |
|-----------|----------------------|---------------------|-------------------|
| **Generator** | `qwen2.5-coder:14b` | `qwen2.5-coder:7b` | `qwen2.5-coder:3b` |
| **Refiner** | `llama3.1:8b` | `mistral:7b` | `phi3.5:3.8b` |
| **Selector** | `llama3.1:8b` | `mistral:7b` | `phi3.5:3.8b` |

---

### 10.5 Temperature Settings

Temperature controls randomness in LLM outputs. Different subprojects require different settings:

| Subproject | Recommended temperature | Rationale |
|-----------|------------------------|-----------|
| **Generator** | 0.1 – 0.2 | Low temperature ensures syntactically valid, deterministic Python code |
| **Selector** | 0.1 – 0.2 | Low temperature ensures consistent structured output format |
| **Refiner** | 0.3 – 0.5 | Moderate temperature allows creative alternative route exploration |

---

### 10.6 Notes for a Research Paper

1. **Test multiple models** and report results per model — which model best generates path planning algorithms is itself a research finding.
2. **Generator is the most model-sensitive** idea — small models frequently produce syntactically invalid Python; `qwen2.5-coder:7b` is the practical minimum for reliable results.
3. **Selector is the most model-agnostic** — even `mistral:7b` achieves high format compliance because the output is short and structured.
4. For reproducibility, **pin the model version tag** (e.g. `qwen2.5-coder:7b-instruct-q4_K_M`) in the experiment config.
5. Run each map **at least 5 times per model** and report mean ± std to account for LLM stochasticity.

---

## References

[1] Abualigah, L., Yousri, D., Abd Elaziz, M., Ewees, A. A., Al-qaness, M. A. A., & Gandomi, A. H. (2021). Aquila optimizer: A novel meta-heuristic optimization algorithm. *Computers & Industrial Engineering*, 157, 107250.

[2] Borenstein, J., & Koren, Y. (1991). The vector field histogram — fast obstacle avoidance for mobile robots. *IEEE Transactions on Robotics and Automation*, 7(3), 278–288.

[3] Burke, E. K., Gendreau, M., Hyde, M., Kendall, G., Ochoa, G., Özcan, E., & Qu, R. (2010). Hyper-heuristics: An emerging direction in modern search technology. In *Handbook of Metaheuristics* (pp. 457–474). Springer.

[4] Burke, E. K., Hyde, M., Kendall, G., Ochoa, G., Özcan, E., & Woodward, J. R. (2013). A classification of hyper-heuristics approaches: Revisited. In *Handbook of Metaheuristics* (pp. 453–477). Springer.

[5] Chi, C., Feng, S., Du, Y., Xu, Z., Cousineau, E., Burchfiel, B., & Song, S. (2023). Diffusion policy: Visuomotor policy learning via action diffusion. In *Proceedings of Robotics: Science and Systems (RSS 2023)*.

[6] Dijkstra, E. W. (1959). A note on two problems in connexion with graphs. *Numerische Mathematik*, 1(1), 269–271.

[7] Dorigo, M., & Di Caro, G. (1999). Ant colony optimization: A new meta-heuristic. In *Proceedings of the 1999 Congress on Evolutionary Computation* (Vol. 2, pp. 1470–1477). IEEE.

[8] Fox, D., Burgard, W., & Thrun, S. (1997). The dynamic window approach to collision avoidance. *IEEE Robotics & Automation Magazine*, 4(1), 23–33.

[9] Fujimoto, S., van Hoof, H., & Meger, D. (2018). Addressing function approximation error in actor-critic methods. In *Proceedings of the 35th International Conference on Machine Learning (ICML 2018)* (pp. 1587–1596).

[10] Gammell, J. D., Srinivasa, S. S., & Barfoot, T. D. (2014). Informed RRT\*: Optimal sampling-based path planning focused via direct sampling of an admissible ellipsoidal heuristic. In *IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2014)* (pp. 2997–3004).

[11] Gammell, J. D., Srinivasa, S. S., & Barfoot, T. D. (2015). Batch informed trees (BIT\*): Sampling-based optimal planning via the heuristically guided search of implicit random geometric graphs. In *IEEE International Conference on Robotics and Automation (ICRA 2015)* (pp. 3067–3074).

[12] Geem, Z. W., Kim, J. H., & Loganathan, G. V. (2001). A new heuristic optimization algorithm: Harmony search. *Simulation*, 76(2), 60–68.

[13] Goldberg, A. V., Kaplan, H., & Werneck, R. F. (2006). Reach for A\*: Efficient point-to-point shortest path algorithms. In *ALENEX* (pp. 129–143).

[14] Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In *Proceedings of the 35th International Conference on Machine Learning (ICML 2018)* (pp. 1861–1870).

[15] Harabor, D., & Grastien, A. (2011). Online graph pruning for pathfinding on grid maps. In *Proceedings of the 25th AAAI Conference on Artificial Intelligence (AAAI 2011)* (pp. 1114–1119).

[16] Harabor, D., Grastien, A., Öz, D., & Aksakalli, V. (2016). Optimal any-angle pathfinding in practice. *Journal of Artificial Intelligence Research*, 56, 89–118.

[17] Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). A formal basis for the heuristic determination of minimum cost paths. *IEEE Transactions on Systems Science and Cybernetics*, 4(2), 100–107.

[18] Holland, J. H. (1975). *Adaptation in Natural and Artificial Systems*. University of Michigan Press.

[19] Hutter, F., Hoos, H. H., & Leyton-Brown, K. (2011). Sequential model-based optimization for general algorithm configuration. In *Learning and Intelligent Optimization (LION 5)* (pp. 507–523). Springer.

[20] Kaelbling, L. P., & Lozano-Pérez, T. (2011). Hierarchical task and motion planning in the now. In *IEEE International Conference on Robotics and Automation (ICRA 2011)* (pp. 1470–1477).

[21] Kalakrishnan, M., Chitta, S., Theodorou, E., Pastor, P., & Schaal, S. (2011). STOMP: Stochastic trajectory optimization for motion planning. In *IEEE International Conference on Robotics and Automation (ICRA 2011)* (pp. 4569–4574).

[22] Karaboga, D., & Basturk, B. (2007). A powerful and efficient algorithm for numerical function optimization: Artificial bee colony (ABC) algorithm. *Journal of Global Optimization*, 39(3), 459–471.

[23] Karaman, S., & Frazzoli, E. (2011). Sampling-based algorithms for optimal motion planning. *International Journal of Robotics Research*, 30(7), 846–894.

[24] Kavraki, L. E., Svestka, P., Latombe, J.-C., & Overmars, M. H. (1996). Probabilistic roadmaps for path planning in high-dimensional configuration spaces. *IEEE Transactions on Robotics and Automation*, 12(4), 566–580.

[25] Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization. In *Proceedings of the IEEE International Conference on Neural Networks (ICNN 1995)* (Vol. 4, pp. 1942–1948).

[26] Kerschke, P., Hoos, H. H., Neumann, F., & Trautmann, H. (2019). Automated algorithm selection: Survey and perspectives. *Evolutionary Computation*, 27(1), 3–45.

[27] Khatib, O. (1986). Real-time obstacle avoidance for manipulators and mobile robots. *International Journal of Robotics Research*, 5(1), 90–98.

[28] Koenig, S., & Likhachev, M. (2002). D\* Lite. In *Proceedings of the 18th AAAI Conference on Artificial Intelligence (AAAI 2002)* (pp. 476–483).

[29] Koza, J. R. (1992). *Genetic Programming: On the Programming of Computers by Means of Natural Selection*. MIT Press.

[30] Kuffner, J. J., & LaValle, S. M. (2000). RRT-Connect: An efficient approach to single-query path planning. In *IEEE International Conference on Robotics and Automation (ICRA 2000)* (Vol. 2, pp. 995–1001).

[31] LaValle, S. M. (1998). Rapidly-exploring random trees: A new tool for path planning. *Technical Report TR 98-11*, Iowa State University.

[32] Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., Silver, D., & Wierstra, D. (2016). Continuous control with deep reinforcement learning. In *International Conference on Learning Representations (ICLR 2016)*.

[33] Liu, F., Tong, X., Yuan, M., Lin, X., Luo, F., Wang, Z., Lu, Z., & Cheng, Q. (2024). Evolution of heuristics: Towards efficient automatic algorithm design using large language model. In *Proceedings of the 41st International Conference on Machine Learning (ICML 2024)*.

[34] López-Ibáñez, M., Dubois-Lacoste, J., Cáceres, L. P., Birattari, M., & Stützle, T. (2016). The irace package: Iterated racing for automatic algorithm configuration. *Operations Research Perspectives*, 3, 43–58.

[35] Ma, Y., Liang, W., Wang, G., Huang, D.-A., Bastani, O., Jayaraman, D., Zhu, Y., Fan, L., & Anandkumar, A. (2023). Eureka: Human-level reward design via coding large language models. *arXiv preprint arXiv:2310.12931*.

[36] Mayne, D. Q., Rawlings, J. B., Rao, C. V., & Scokaert, P. O. M. (2000). Constrained model predictive control: Stability and optimality. *Automatica*, 36(6), 789–814.

[37] Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). Grey wolf optimizer. *Advances in Engineering Software*, 69, 46–61.

[38] Mirjalili, S., & Lewis, A. (2016). The whale optimization algorithm. *Advances in Engineering Software*, 95, 51–67.

[39] Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., & Riedmiller, M. (2013). Playing Atari with deep reinforcement learning. *arXiv preprint arXiv:1312.5602*.

[40] Nash, A., Daniel, K., Koenig, S., & Felner, A. (2007). Theta\*: Any-angle path planning on grids. In *Proceedings of the 22nd AAAI Conference on Artificial Intelligence (AAAI 2007)* (pp. 1177–1183).

[41] Qiao, Z., Tyree, Z., Bhatt, P., Mukherjee, J., & Ghaffari, M. (2020). Motion planning networks: Bridging the gap between learning-based and classical motion planners. *IEEE Transactions on Robotics*, 37(1), 48–66.

[42] Rice, J. R. (1976). The algorithm selection problem. *Advances in Computers*, 15, 65–118.

[43] Romera-Paredes, B., Barekatain, M., Novikov, A., Balog, M., Kumar, M. P., Dupont, E., Ruiz, F. J. R., Ellenberg, J. S., Wang, P., Fawzi, O., Kohli, P., & Fawzi, A. (2023). Mathematical discoveries from program search with large language models. *Nature*, 625, 468–475.

[44] Sautenkov, O., Yaqoot, Y., Lykov, A., Mustafa, M. A., Tadevosyan, G., Akhmetkazy, A., Cabrera, M. A., Martynov, M., Karaf, S., & Tsetserukou, D. (2025). UAV-VLA: Vision-language-action system for large scale aerial mission generation. In *2025 20th ACM/IEEE International Conference on Human-Robot Interaction (HRI)*. IEEE.

[45] Schulman, J., Duan, Y., Ho, J., Lee, A., Awwal, I., Bradlow, H., Pan, J., Patil, S., Goldberg, K., & Abbeel, P. (2014). Motion planning with sequential convex optimization and convex collision checking. *International Journal of Robotics Research*, 33(9), 1251–1270.

[46] Schulman, J., Levine, S., Abbeel, P., Jordan, M., & Moritz, P. (2015). Trust region policy optimization. In *Proceedings of the 32nd International Conference on Machine Learning (ICML 2015)* (pp. 1889–1897).

[47] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*.

[48] Simeon, T., Laumond, J.-P., & Nissoux, C. (2000). Visibility-based probabilistic roadmaps for motion planning. *Advanced Robotics*, 14(6), 477–493.

[49] Stentz, A. (1994). Optimal and efficient path planning for partially-known environments. In *Proceedings of the IEEE International Conference on Robotics and Automation (ICRA 1994)* (pp. 3310–3317).

[50] Storn, R., & Price, K. (1997). Differential evolution — A simple and efficient heuristic for global optimization over continuous spaces. *Journal of Global Optimization*, 11(4), 341–359.

[51] Tabrizian, A., Ghazanfari, M., & Wei, P. (2025). Chain-of-thought flight planner: End-to-end LLM routing under wind hazards. In *AIAA AVIATION FORUM AND ASCEND 2025*. https://doi.org/10.2514/6.2025-3711

[52] Tamar, A., Wu, Y., Thomas, G., Levine, S., & Abbeel, P. (2016). Value iteration networks. In *Advances in Neural Information Processing Systems (NeurIPS 2016)* (pp. 2154–2162).

[53] Theodorou, E. A., & Todorov, E. (2012). Relative entropy and free energy dualities: Connections to path integral and KL control. In *Proceedings of the 51st IEEE Conference on Decision and Control (CDC 2012)* (pp. 1466–1473).

[54] Ulrich, I., & Borenstein, J. (1998). VFH+: Reliable obstacle avoidance for fast mobile robots. In *Proceedings of the IEEE International Conference on Robotics and Automation (ICRA 1998)* (Vol. 2, pp. 1572–1577).

[55] van Hasselt, H., Guez, A., & Silver, D. (2016). Deep reinforcement learning with double Q-network. In *Proceedings of the 30th AAAI Conference on Artificial Intelligence (AAAI 2016)* (pp. 2094–2100).

[56] Wang, Z., Schaul, T., Hessel, M., van Hasselt, H., Lanctot, M., & de Freitas, N. (2016). Dueling network architectures for deep reinforcement learning. In *Proceedings of the 33rd International Conference on Machine Learning (ICML 2016)* (pp. 1995–2003).

[57] Xiao, J., Tsao, C. W., Zhang, Y., & Feroskhan, M. (2025). FM-Planner: Foundation model guided path planning for autonomous drone navigation. *arXiv preprint arXiv:2505.20783*.

[58] Xie, G., Xu, J., Ding, Y., Zhang, Z., Zhang, S., & Li, Y. (2025). Never too prim to swim: An LLM-enhanced RL-based adaptive S-surface controller for AUVs under extreme sea conditions. *arXiv preprint arXiv:2503.00527*.

[59] Xue, J., & Shen, B. (2022). Dung beetle optimizer: A new meta-heuristic algorithm for global optimization. *Journal of Supercomputing*, 79(7), 7305–7336.

[60] Yang, X.-S. (2008). *Nature-Inspired Metaheuristic Algorithms*. Luniver Press.

[61] Yang, X.-S. (2010). A new metaheuristic bat-inspired algorithm. In *Nature Inspired Cooperative Strategies for Optimization (NICSO 2010)* (pp. 65–74). Springer.

[62] Yang, X.-S., & Deb, S. (2009). Cuckoo search via Lévy flights. In *World Congress on Nature & Biologically Inspired Computing (NaBIC 2009)* (pp. 210–214). IEEE.

[63] Yuan, L., Deng, C., Han, D. J., Hwang, I., Brunswicker, S., & Brinton, C. G. (2025). Next-generation LLM for UAV: From natural language to autonomous flight. *arXiv preprint arXiv:2510.21739*.

[64] Zucker, M., Ratliff, N., Dragan, A. D., Pivtoraiko, M., Klingensmith, M., Dellin, C. M., Bagnell, J. A., & Srinivasa, S. S. (2013). CHOMP: Covariant Hamiltonian optimization for motion planning. *International Journal of Robotics Research*, 32(9–10), 1164–1193.
