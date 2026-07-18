# Experiments

This directory contains experiment entry-point scripts and YAML configuration files
for evaluating all three agenticpathplanner subprojects on the CEC2024 benchmark.

---

## Directory Structure

```
experiments/
├── run_cec2024.py          General-purpose CEC2024 benchmark runner
└── configs/
    ├── cec2024.yaml        General config (baseline LLM path generation)
    ├── generator.yaml      Idea 2 — LLM as Generator (qwen2.5-coder, generate_algorithm)
    ├── refiner.yaml        Idea 1 — LLM as Refiner (llama3.1, generate_path + env context)
    ├── selector.yaml       Idea 3 — LLM as Selector (mistral, algorithm registry)
    └── models.yaml         Full model registry: all recommended Ollama models + hardware tiers
```

---

## Quick Start

### 1. Install all packages

```bash
make install-dev
```

### 2. Pull the recommended models

```bash
# Generator
ollama pull qwen2.5-coder:7b

# Refiner and Selector
ollama pull llama3.1:8b
ollama pull mistral:7b

# Lightweight CPU fallback
ollama pull phi3.5:3.8b
```

### 3. Run experiments

#### General benchmark (baseline LLM path generation)

```bash
python experiments/run_cec2024.py --config experiments/configs/cec2024.yaml
```

#### Idea 2 — Generator

```bash
python experiments/run_cec2024.py \
    --config experiments/configs/generator.yaml \
    --mode generate_algorithm \
    --model qwen2.5-coder:7b
```

#### Idea 1 — Refiner

```bash
python experiments/run_cec2024.py \
    --config experiments/configs/refiner.yaml \
    --model llama3.1:8b
```

#### Idea 3 — Selector

```bash
python experiments/run_cec2024.py \
    --config experiments/configs/selector.yaml \
    --model mistral:7b
```

#### Override model or mode on the command line

```bash
python experiments/run_cec2024.py \
    --config experiments/configs/generator.yaml \
    --model qwen2.5-coder:14b \
    --output results/generator_14b
```

### 4. View results

```bash
agentpp report --results results/
```

---

## Recommended Models Summary

See `configs/models.yaml` for the full registry. Quick reference:

| Subproject | Primary model | Balanced | CPU fallback |
|-----------|--------------|----------|--------------|
| **Generator** | `qwen2.5-coder:14b` | `qwen2.5-coder:7b` | `qwen2.5-coder:3b` |
| **Refiner** | `llama3.1:8b` | `mistral:7b` | `phi3.5:3.8b` |
| **Selector** | `llama3.1:8b` | `mistral:7b` | `phi3.5:3.8b` |

---

## Comparison Baselines

Each idea has specific comparison baselines. See `docs/research_positioning.md §9` for details.

### Refiner baselines

| Baseline | Ref. | Description |
|----------|------|-------------|
| Raw A\* | [17] | Unrefined geometric path — lower bound |
| Theta\* | [40] | Any-angle geometric smoothing |
| APF post-processing | [27] | Classical obstacle clearance improvement |

### Generator baselines

| Baseline | Ref. | Description |
|----------|------|-------------|
| A\* | [17] | Fixed optimal classical planner |
| RRT\* | [23] | Asymptotically optimal sampling-based planner |
| PSO | [25] | Population-based waypoint optimisation |
| LLM direct path | — | Same LLM, `generate_path` mode (no code generation) |

### Selector baselines

| Baseline | Ref. | Description |
|----------|------|-------------|
| Always-A\* | [17] | Fixed single-algorithm |
| Random Selector | — | Uniform random algorithm choice — floor |
| Rule-based Selector | [27,23,36] | Hand-coded condition → algorithm mapping |
| Oracle Selector | — | Post-hoc optimal choice — ceiling |

---

## Reproducibility Notes

- Each experiment uses **fixed seeds** `{42, 137, 2024}` per map per run.
- Run each map **at least 5 times** (`runs_per_map: 5`) to report mean ± std.
- **Pin the Ollama model tag** (e.g. `qwen2.5-coder:7b-instruct-q4_K_M`) for exact reproducibility.
- Results are saved as both `.csv` and `.json` under `results/`.
