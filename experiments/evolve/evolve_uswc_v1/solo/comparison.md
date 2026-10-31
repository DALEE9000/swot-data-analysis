# Pooled vs region-solo — evolve_uswc_v1 champion `gen020_02`

Same candidate code, same params (epoch cap 60, budget 1620s), seeds [42, 43, 44]. Scored on each region's identical temporal holdout.

| region | rows (train) | pooled r2_u | solo r2_u | pooled r2_v | solo r2_v | winner (mean r2) |
|---|---|---|---|---|---|---|
| usegc | 1,784,455 | nan | 0.631 | nan | 0.496 | solo (nan vs 0.564) |
| ushi | 13,879 | nan | 0.692 | nan | 0.589 | solo (nan vs 0.641) |
