# Exp2.2 Pilot Report

- Model: `qwen2.5-coder:7b`
- Temperature: `0.1`
- Runs per mission/map: 1
- Total generated cases: 12
- Valid paths: 8

| Mission | Runs | Valid paths | Valid rate | Mean objective |
|---|---:|---:|---:|---:|
| efficiency | 3 | 3 | 1.000 | 1.0 |
| safety | 3 | 0 | 0.000 | N/A |
| smoothness | 3 | 2 | 0.667 | 11.780972450961727 |
| multi | 3 | 3 | 1.000 | 31.9462382628966 |

All generation/repair attempts are retained under each case.
Total generation time includes all attempts; selected planner runtime is separate.
A valid path only establishes path feasibility; mission-objective compliance must be interpreted from the independent mission metrics and code review.
