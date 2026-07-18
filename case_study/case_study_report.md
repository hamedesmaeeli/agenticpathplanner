# CEC2024 Case Study — Path Planning Method Comparison

Results from `experiments/run_case_study.py`.
Metrics: **PLR** = Path Length Ratio (lower ↓), **SD** = Safety Distance (higher ↑), **SM** = Smoothness (lower ↓).

## Map: `cec2024_s01` (simple, 10×10)

![cec2024_s01](figures\cec2024_s01.png)

### cec2024_s01

| Method | Valid | PLR | SD | SM | Path len |
|--------|:-----:|----:|---:|---:|---------:|
| astar | ✅ | 1.000 | 1.00 | 0.00 | 9.9 |
| theta_star | ✅ | 1.000 | 3.00 | 0.00 | 9.9 |
| apf_refined | ✅ | 1.082 | 1.00 | 3.00 | 10.7 |
| random_selector | ✅ | 1.000 | 1.00 | 0.00 | 9.9 |
| rule_based_selector | ✅ | 1.000 | 1.00 | 0.00 | 9.9 |
| oracle_selector | ✅ | 1.000 | 1.00 | 0.00 | 9.9 |

## Map: `cec2024_m01` (medium, 20×20)

![cec2024_m01](figures\cec2024_m01.png)

### cec2024_m01

| Method | Valid | PLR | SD | SM | Path len |
|--------|:-----:|----:|---:|---:|---------:|
| astar | ✅ | 1.122 | 1.00 | 4.71 | 27.0 |
| theta_star | ✅ | 1.080 | 1.00 | 2.09 | 26.0 |
| apf_refined | ✅ | 1.345 | 2.00 | 12.96 | 32.3 |
| random_selector | ✅ | 1.122 | 1.00 | 4.71 | 27.0 |
| rule_based_selector | ✅ | 1.122 | 1.00 | 4.71 | 27.0 |
| oracle_selector | ✅ | 1.080 | 1.00 | 2.09 | 26.0 |

## Map: `cec2024_c01` (complex, 30×30)

![cec2024_c01](figures\cec2024_c01.png)

### cec2024_c01

| Method | Valid | PLR | SD | SM | Path len |
|--------|:-----:|----:|---:|---:|---------:|
| astar | ✅ | 1.061 | 1.00 | 9.42 | 40.5 |
| theta_star | ✅ | 1.015 | 1.00 | 1.80 | 38.8 |
| apf_refined | ❌ | — | — | — | — |
| random_selector | ❌ | — | — | — | — |
| rule_based_selector | ✅ | 1.015 | 1.00 | 1.80 | 38.8 |
| oracle_selector | ✅ | 1.015 | 1.00 | 1.80 | 38.8 |
