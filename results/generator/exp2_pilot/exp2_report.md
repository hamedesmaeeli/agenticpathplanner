# Exp2 Pilot Report

- Model: `qwen2.5-coder:7b`
- Temperature: `0.1`
- Runs per mission/map: 1
- Total generated cases: 12
- Valid paths: 7

| Mission | Runs | Valid paths | Valid rate | Mean objective |
|---|---:|---:|---:|---:|
| efficiency | 3 | 3 | 1.000 | 1.0136276050522253 |
| safety | 3 | 3 | 1.000 | 1.3333333333333333 |
| smoothness | 3 | 0 | 0.000 | N/A |
| multi | 3 | 1 | 0.333 | 11.724452847995972 |

Generation time and planner runtime are recorded separately.
A valid path only establishes path feasibility; mission-objective compliance must be interpreted from the independent mission metrics and code review.
