# Overall op time summary


| Experiment          | Run Type | Op Time Total (s) | % Total Compute | % Total Non-Compute | Current Speedup | Ideal Speedup (no overhead) |
| ------------------- | -------- | ----------------- | --------------- | ------------------- | --------------- | --------------------------- |
| 1M rows, 10k batch  | GPU      | 3.50              | 29.15%          | 70.85%              | 1.34            | 4.61                        |
|                     | CPU      | 4.7               | 77.26%          | 22.74%              |                 |                             |
| 1M rows, 100k batch | GPU      | 2.80              | 10.93%          | 89.07%              | 1.79            | 16.33                       |
|                     | CPU      | 5.0               | 76.30%          | 23.70%              |                 |                             |
| 5M rows, 10k batch  | GPU      | 13.50             | 36.86%          | 63.14%              | 1.67            | 4.54                        |
|                     | CPU      | 22.6              | 80.33%          | 19.67%              |                 |                             |
| 5M rows, 100k batch | GPU      | 10.90             | 13.62%          | 86.38%              | 2.44            | 17.92                       |
|                     | CPU      | 26.6              | 79.89%          | 20.11%              |                 |                             |

