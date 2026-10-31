# Pooled vs region-solo — evolve_all_science_v1 champion `gen031_00`

Same candidate code, same params (epoch cap 60, budget 3240s), seeds [42, 43, 44]. Scored on each region's identical temporal holdout.

| region | rows (train) | pooled r2_u | solo r2_u | pooled r2_v | solo r2_v | winner (mean r2) |
|---|---|---|---|---|---|---|
| uswc | 1,897,113 | 0.217 | 0.237 | 0.049 | 0.049 | solo (0.133 vs 0.143) |
| usegc | 1,700,731 | 0.455 | 0.445 | 0.532 | 0.523 | pooled (0.494 vs 0.484) |
| prvi | 160,683 | 0.082 | 0.033 | -0.072 | -0.123 | pooled (0.005 vs -0.045) |
| ushi | 34,362 | 0.018 | -0.057 | 0.144 | 0.146 | pooled (0.081 vs 0.045) |
