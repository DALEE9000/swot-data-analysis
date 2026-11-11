# Methods notes: concepts behind the evaluation, written out for review

Plain-prose explanations of the methods the tables and figures rely on, in the
order they came up. Each entry says what the thing is, how the code computes
it, and what it means for the results. Companion to `docs/glossary.md`
(acronyms) and `docs/pipeline-map.md` (where each step sits). Add a section
whenever a method needs explaining twice.

## 1. R² as the project scores it (2026-11-03)

**What it is.** R² compares a model's errors to the errors of one dumb
reference: a single number, the mean of the held-out truth, predicted at every
row. Written out,

    R² = 1 − Σ (truth − prediction)² / Σ (truth − mean of truth)²

where both sums run over the same held-out rows. Zero means "no better than
predicting the average everywhere"; one means perfect; negative means worse
than the constant.

**How the code computes it.** One R² per region per component (u and v
separately), pooled: every held-out radar cell at every held-out SWOT pass goes
into one long list. The predictions are made row by row, each from that row's
own inputs. The score is not row by row; it is one number for the whole list.
The mean in the denominator is the mean of that whole list, so it is a
regional constant. A block bootstrap over (cycle, pass) gives the 95 %
interval. Rows are the headline protocol: 25-h target, SWOT-valid rows,
buffered after the last training pass (30 days science, 7 days calval).

**Why the pooled version.** It is what the HFR validation literature reports
(Kachelein, the DUACS assessments), so the numbers are comparable, and
per-cell R² is noisy with about twelve passes per cell on the science orbit.
The cost is the point of the next section: the pooled denominator mixes fast
cells with slow cells, so most of the "spread" a model is credited for
removing is the difference between cells, not the change from day to day.

## 2. Climatology: the per-cell training mean as a model (2026-11-03)

**What it is.** A lookup table with one (u, v) pair per radar cell: that
cell's average 25-h current over the training passes. Laid out on the radar
grid it is a map of the mean flow. It is a model with zero inputs from the
satellite or the wind; it knows only which cell it is in.

**How the code computes it** (`cell_climatology` in
`scripts/baselines_eval.py`). Group the training rows by cell (latitude and
longitude rounded to three decimals), average u and v within each group. For a
held-out row, look up its cell and predict that pair, the same pair at every
pass. A held-out cell never seen in training (rare) gets the regional training
mean instead. The table is built from training passes only, so it never sees
the held-out months; a map that included the test period would leak the test
mean into the reference.

**How it is scored.** Exactly like every other rung: its per-cell predictions
on the held-out rows go through the same pooled R² and bootstrap. Values are
per cell; the score is per region.

**Worked example.** Two cells, three training passes of u:

| | pass 1 | pass 2 | pass 3 | climatology |
|---|---|---|---|---|
| cell A | 0.7 | 0.9 | 0.8 | 0.8 |
| cell B | 0.0 | 0.2 | -0.2 | 0.0 |

Two held-out passes:

| | pass 4 truth | climatology | pass 5 truth | climatology |
|---|---|---|---|---|
| cell A | 0.9 | 0.8 | 0.6 | 0.8 |
| cell B | 0.1 | 0.0 | -0.1 | 0.0 |

Pool the four truths: 0.9, 0.6, 0.1, -0.1; mean 0.375; spread about the
mean 0.628. Climatology's squared error 0.07. R² = 1 − 0.07/0.628 = 0.89. A
lookup table that never looked at a satellite scores 0.89, because the
difference between cell A and cell B dominates the pooled spread. A model
using SWOT and wind that predicts 0.9, 0.65, 0.1, -0.05 has error 0.005 and
R² 0.99: of that 0.99, 0.89 is the map and 0.10 is tracking the day-to-day
wobble.

**Climatology versus R²'s own reference.** Both are "predict a mean", and the
R² formula is identical for both. They differ in two ways: R²'s reference is
one regional number, climatology is one number per cell; and R²'s reference is
the held-out mean, climatology is the training-period mean. The first
difference is the one that matters: a per-cell mean captures the map of the
mean flow, a regional constant cannot. The second is honesty, so the reference
never sees the test months.

**Why the rung is on the ladder.** The pooled R² credits every model for
knowing the map, and the champion reads location off its static inputs (mean
dynamic topography, stencil pattern) and memorises the local mean. The
climatology rung is the model that knows only the map, so its score is the map
share of everyone else's score. Model R² minus climatology R² is the
day-to-day share, the part only an observation of today can supply, and the
part the paper's claims rest on. The alternative presentation is the skill
score, 1 − MSE(model)/MSE(climatology): zero for climatology, positive only
when the model knows something the map does not. Per-cell R² (each cell scored
over its own time series, reference the cell's own mean) removes the map
credit inside the score instead of by subtraction; climatology lands at zero
there by construction. Same information, different display; the ladder keeps
the pooled version and adds the rung.

**The full ladder with the climatology rung** (headline protocol, R² u/v;
`train_mean` is the regional constant, `cell_mean` the climatology).

Science orbit (21-day, 30-day buffer):

| Model | East Coast | West Coast | Puerto Rico | Hawaii |
|---|---|---|---|---|
| regional constant | -0.01 / 0.00 | -0.01 / -0.84 | -0.02 / -0.01 | -0.04 / -0.05 |
| climatology (per-cell mean) | 0.37 / 0.55 | -0.01 / -0.98 | -0.01 / -0.40 | 0.03 / 0.05 |
| climatology + wind | 0.45 / 0.56 | 0.10 / -0.42 | 0.01 / -0.52 | 0.04 / 0.09 |
| SWOT calibrated + wind, 5 par. | 0.48 / 0.56 | 0.22 / 0.09 | 0.06 / -0.11 | 0.22 / 0.16 |
| per-cell linear | 0.48 / 0.58 | 0.25 / 0.11 | 0.04 / -0.20 | 0.28 / 0.22 |
| DUACS calibrated + wind | 0.53 / 0.58 | 0.20 / 0.15 | 0.19 / 0.12 | 0.30 / 0.50 |
| ridge, 72 cols | 0.54 / 0.59 | 0.22 / 0.11 | 0.08 / -0.20 | 0.06 / 0.07 |
| champion | 0.57 / 0.64 | 0.21 / 0.11 | 0.14 / -0.05 | 0.17 / 0.13 |
| champion minus climatology | 0.20 / 0.09 | 0.22 / 1.09 | 0.15 / 0.35 | 0.14 / 0.08 |

Calval orbit (1-day, 7-day buffer):

| Model | East Coast | West Coast | Hawaii |
|---|---|---|---|
| regional constant | -0.06 / -0.04 | 0.00 / -0.01 | -0.05 / -0.03 |
| climatology (per-cell mean) | 0.41 / 0.59 | 0.20 / 0.06 | 0.90 / 0.73 |
| climatology + wind | 0.52 / 0.61 | 0.17 / 0.18 | 0.90 / 0.73 |
| SWOT calibrated + wind, 5 par. | 0.37 / 0.60 | 0.04 / 0.26 | -0.01 / 0.03 |
| per-cell linear | 0.59 / 0.65 | 0.38 / 0.36 | 0.87 / 0.69 |
| DUACS calibrated + wind | 0.47 / 0.63 | 0.11 / 0.29 | 0.05 / 0.06 |
| ridge, 72 cols | 0.52 / 0.65 | 0.10 / 0.14 | 0.73 / 0.56 |
| champion | 0.62 / 0.62 | 0.43 / 0.56 | 0.72 / 0.60 |
| champion minus climatology | 0.21 / 0.03 | 0.23 / 0.50 | -0.18 / -0.13 |

Reading the ladder against the climatology row: on the East Coast adding wind
to the climatology alone reaches 0.45 to 0.52 in u, so wind is about half of
the champion's u gain and SWOT's share of v there is near zero; in Hawaii
climatology plus wind equals climatology, so wind adds nothing; in Puerto
Rico only DUACS plus wind is positive in both components.

**What the results say** (champion against climatology, R² u/v):

| Region, orbit | Climatology | Champion | Day-to-day share |
|---|---|---|---|
| East Coast, science | 0.37 / 0.55 | 0.57 / 0.64 | 0.20 / 0.09 |
| East Coast, calval | 0.41 / 0.59 | 0.62 / 0.62 | 0.21 / 0.03 |
| West Coast, science | -0.01 / -0.98 | 0.21 / 0.11 | 0.22 / about 1.1 |
| West Coast, calval | 0.20 / 0.06 | 0.43 / 0.56 | 0.23 / 0.50 |
| Hawaii, calval | 0.90 / 0.73 | 0.72 / 0.60 | negative |

East Coast: more than half of the headline is the map, and in v (along the
shelf and the Gulf Stream) the champion adds only 0.03 to 0.09 above it. West
Coast science: no stable map (climatology v of -0.98 means the training mean
predicts the held-out season worse than a constant; the flow reversed at the
spring transition to upwelling), so the 0.2 the models reach is all
day-to-day skill. West Coast calval: the one cell where the network earns a
large day-to-day share (0.50 in v), on the run it was evolved on. Hawaii: the
map beats the network; with 1,240 rows and a steady flow, letting the
satellite perturb the prediction adds noise, and the per-cell linear model
(the map plus a small calibrated correction) is best at 0.87/0.69.

The number to lead the paper with is the day-to-day share: about 0.2 in u
almost everywhere and near zero in v on the East Coast. That is what SWOT and
wind explain, and it is what Phase 2 has to move.

## 3. The ladder, one equation per rung (2026-11-03)

LaTeX version, standalone and `\input`-able into the paper:
`docs/methods-equations/methods-equations.tex`, built PDF beside it
(`pdflatex` twice in that folder; keep the two in step).

Notation, all at one radar cell and one SWOT pass time: truth $y = (y_u, y_v)$
is the 25-h-mean radar current; $g = (g_u, g_v)$ is SWOT L3 geostrophic
velocity at the centre cell; $W = (W_u, W_v)$ is ERA5 10-m wind at the pass
time; $d = (d_u, d_v)$ is DUACS L4 geostrophic velocity interpolated to the
row; $x$ is the full 72-column stencil (3×3 cells × 8 variables; 81 on calval
with SST). $\bar y$ is the regional training mean; $\bar y(c)$ is the
training mean at cell $c$. Every coefficient is fitted by least squares on the
training rows of the region (`fit_linear` in `scripts/baselines_eval.py`), u
and v fitted separately with their own coefficients unless stated, and then
applied unchanged to the held-out rows. "Calibrated" means a slope is fitted
rather than set to one.

| Rung | Prediction for $u$ (same form for $v$ with its own coefficients) | Fitted parameters |
|---|---|---|
| regional constant (`train_mean`) | $\hat u = \bar y_u$ | 0 |
| climatology (`cell_mean`) | $\hat u = \bar y_u(c)$ | one value per cell |
| climatology + wind (`cell_mean+wind_lsq`) | $\hat u = \bar y_u(c) + a + b\,W_u + e\,W_v$, with $(a,b,e)$ fitted to $y_u - \bar y_u(c)$ | 3 per component, regional |
| wind only (`wind_only_lsq`) | $\hat u = a + b\,W_u + e\,W_v$ | 3 |
| SWOT geostrophy, raw (`swot_geos`) | $\hat u = g_u$ | 0 |
| SWOT + Ekman rule (`geos+ekman_3pct45deg`) | $\hat u = g_u + 0.03\,(W_u\cos45° + W_v\sin45°)$, $\hat v = g_v + 0.03\,(-W_u\sin45° + W_v\cos45°)$ | 0 (3 % of wind, 45° right) |
| SWOT + wind, slope one (`geos+wind_lsq`) | $\hat u = g_u + a + b\,W_u + e\,W_v$, with $(a,b,e)$ fitted to $y_u - g_u$ | 3 |
| SWOT calibrated (`geos_calibrated`) | $\hat u = a + b\,g_u$ (and $\hat v = a' + b'\,g_v$; each component on its own geostrophic component) | 2 |
| SWOT calibrated + wind (`geos_cal+wind_lsq`) | $\hat u = a + b\,g_u + c\,g_v + d\,W_u + e\,W_v$ | 5 |
| per-cell linear (`cell_linear_geos_wind`) | $\hat u = a_c + b_c\,g_u + c_c\,g_v + d_c\,W_u + e_c\,W_v$, coefficients per cell $c$, shrunk toward the regional five-parameter fit with a prior worth five training rows (inputs standardised first); cells with under five complete training rows use the regional fit | 5 per cell |
| DUACS raw (`duacs_geos`) | $\hat u = d_u$ | 0 |
| DUACS calibrated (`duacs_calibrated`) | $\hat u = a + b\,d_u$ | 2 |
| DUACS calibrated + wind (`duacs_cal+wind_lsq`) | $\hat u = a + b\,d_u + c\,d_v + d\,W_u + e\,W_v$ | 5 |
| ridge, 72 columns (`ridge72_solo`) | $\hat u = a + \sum_{k=1}^{72} \beta_k\,\tilde x_k$, with $\tilde x$ the standardised stencil (NaN set to 0 after standardising) and the $\beta$ fitted by ridge with penalty $\alpha = 1$ | 73 per component |
| AlphaEvolve champion (`champion_*`) | $(\hat u, \hat v) = f_\theta(\tilde x)$, the candidate's MLP on the same standardised stencil, joint output, masked-MSE loss, early stopping; architecture in `candidate.py` of the run | thousands |

The per-cell linear shrinkage, written out for cell $c$ with $n_c$ training
rows, design matrix $A_c$ (a column of ones and the four standardised inputs)
and regional coefficients $\beta_g$:

$$\beta_c = (A_c^\top A_c + \lambda I)^{-1}\,(A_c^\top y_c + \lambda\,\beta_g), \qquad \lambda = 5.$$

With about 170 training rows per cell on the calval orbit the local data
dominate; with about ten per cell on the science orbit the prior does, which
is why the rung sits close to the regional five-parameter fit there and well
above it on calval.

What the rungs are for, read upward: the first two measure what location
alone gives (the map); the wind rungs measure the instantaneous Ekman-like
response; the SWOT rungs measure geostrophy raw, then rescaled, then with
wind, then with a cell-specific rescaling; the DUACS rungs repeat the SWOT
rungs with the nadir-altimeter map as the height source; ridge is the linear
ceiling on the champion's own inputs; the champion is the only nonlinear
model. Each rung's R² minus the climatology's R² is its day-to-day share.
