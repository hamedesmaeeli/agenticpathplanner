# Exp2.1 Pilot Report

- Model: `qwen2.5-coder:7b`
- Temperature: `0.1`
- Runs per mission/map: 1
- Total generated cases: 12
- Valid paths: 4

| Mission | Runs | Valid paths | Valid rate | Mean objective |
|---|---:|---:|---:|---:|
| efficiency | 3 | 3 | 1.000 | 1.0 |
| safety | 3 | 0 | 0.000 | N/A |
| smoothness | 3 | 0 | 0.000 | N/A |
| multi | 3 | 1 | 0.333 | 32.38214979843309 |

Generation time and planner runtime are recorded separately.
A valid path only establishes path feasibility; mission-objective compliance must be interpreted from the independent mission metrics and code review.
