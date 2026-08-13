# Decision Doc


## Target variable
Used APPROVAL_CREATE_DATE to APPROVAL_ISSUE_DATE instead of PROJECT dates, 
because PROJECT_* columns were only ~39% populated while APPROVAL_ID was 100% complete.

## Handling negative days_to_approval
Found ~X rows with negative delay (issue date before create date), 
likely data entry errors. Filtered out rather than imputed since it's <1% of data.

## Delay cutoff: 75th percentile
Chose 75th percentile as a data-driven, standard convention for splitting the upper tail 
of a skewed distribution. Considered 90th (flag only extreme cases) and 60th (catch 
problems earlier) as alternatives. Would use an official city SLA/target instead if 
available, since that'd be more directly tied to real-world "delayed" than a statistical 
percentile.

## Binary vs multi-class framing
Started with binary classification (delayed vs not) rather than multi-class categories, 
to establish a working baseline before adding complexity. Class balance came out to 
~75/25, reasonable enough to not require special imbalance handling initially.

## Train/test split strategy
Using 2024 as training data and 2025 as the test set (temporal split) instead of a 
random split across both years, to simulate real deployment (predicting future permits 
from past patterns) and avoid data leakage. Confirmed the delayed/not-delayed class 
balance is similar across both years (~75/25 vs ~77/23), supporting this approach.

## APPROVAL_TYPE encoding
94 unique values, heavily long-tailed. Grouped to top 15 + "Other" (frequency-based, 
covers ~90% of data) rather than encoding all 94 categories or manually merging by 
semantic similarity.

## Submission timing feature
Compared delay rate by month (8pp spread, Aug 21.5% to Dec 29.4%) vs by quarter 
(1.4pp spread, too flat to be useful). Kept month as a one-hot encoded feature, 
dropped quarter since averaging masked the December/February pattern.

## Location feature
Kept GIS_LATITUDE/GIS_LONGITUDE as raw numeric values rather than manually clustering 
into neighborhoods, since gradient boosting can split directly on coordinate values to 
find spatial patterns on its own. Dropped 2,114 rows (~3.8%) missing coordinates.

## Baseline model results
GradientBoostingClassifier, default params, trained on 2024, tested on 2025.
Accuracy: 85.1%, Precision: 65.7%, Recall: 74.9%, ROC-AUC: 0.916.
Beats the naive baseline (77.7% accuracy from always predicting "not delayed"),
confirming the engineered features carry real signal.

## SHAP findings
Permit type dominates all other features by a wide margin. Full construction/plan-review 
types (Combination Building, Building, Electrical/Plumbing/Mechanical Pmt) push toward 
delay; routine no-plan types (Traffic Control, No-Plan Residential Combo) push toward 
on-time. This matches the No-Plan vs. full-review pattern first noticed in early EDA. 
Location shows a modest continuous effect; month has minimal individual impact.




# Readability Pass for Non-ML Readers (August 2026)

The app was written for someone who already knows what a SHAP value is and
what a "good" delay probability looks like. Four changes so the output
stands on its own:

1. **Base-rate anchors.** A bare "81.2%" means nothing without a reference
   point. The result now sits next to the observed 2024 delay rates for all
   permits (26%) and for the selected permit type (73%), computed in
   train.py and saved as base_rates.pkl. The gap between the estimate and
   the type's rate was first shown as a Streamlit metric delta ("+8 pts"),
   which a reader immediately misread — it sits under the 73% so it looks
   like it modifies that number, and a delta conventionally means "change
   since last period." Replaced with a plain sentence: "This permit's 81%
   estimate is 8 points above the historical rate for its type."
2. **Plain-English drivers before the chart.** Two sentences naming the top
   factors that raised and lowered the estimate, with the SHAP chart kept
   below for readers who want the magnitudes.
3. **Model provenance above the fold.** Training/test size and held-out
   accuracy moved out of a collapsed expander into a caption under the
   title, read from metrics.json rather than hardcoded.
4. **Fixed a clipped axis label.** Long permit-type names expanded the
   chart's left margin until the x-axis label ran off the right edge; type
   names are now truncated and the figure is saved with a tight bounding box.

One subtlety worth recording: the first version of the plain-English
summary read the top SHAP features regardless of their value, which
produced sentences like "the Express processing track lowered it" for a
permit on the *Standard* track — one-hot columns that are OFF still carry
an attribution. Phrases are now conditioned on the feature's actual value
("not being on the Express track"), permit-type and month features are
skipped entirely when off, and factors below 10% of the largest
contribution are dropped so near-certain predictions don't list rounding
noise as if it mattered.


# App Defaults Bug: Untouched Inputs Implied the Safest Permit (August 2026)

Found after deploy: leaving "More project details" untouched fed the model
has_project=False, no valuation, no scope, first-time applicant — the
safest possible profile. For plan-review types that combination barely
exists in training (Combination Building Permits are ~100% project-linked),
so the model read an untouched form as strong evidence of a routine permit:
a Combination Building Permit showed 1.7% delay probability when its actual
2025 delay rate is 62.5%. Every plan-review type was affected.

Fix: per-type defaults computed from training data (mode for binary/
categorical fields, median for numeric ones), stored in type_defaults.pkl
and re-seeded whenever the selected type changes. A blank scope box now
imputes the type's median description length instead of zero. Untouched
predictions now track actual per-type delay rates (Combination Building
81.2% predicted vs 62.5% actual; routine no-plan types stay ~1%) — a bit
above the type average by design, since the typical profile (project-linked,
median valuation) is riskier than the within-type mean. Lesson: default
input values are a modeling decision, not a UI detail — "left blank" must
map to "typical," not "best case."


# Scope Reduction (August 2026)

Removed two shipped pieces from the app and pipeline: the deadline risk
curve (the 15-180 day threshold-grid classifiers with interpolation) and
the calibration analysis (isotonic/sigmoid selection, ECE verification).
Both worked and their evaluation numbers were solid, but they are more
sophisticated than I can currently explain and defend end-to-end, and a
project I can fully own matters more than extra features. The entries
below are kept as the record of what was tried, what failed (quantile
regression), and what the analysis found, in case I bring either piece
back after studying the underlying methods properly.

What remains shipped: the v2 feature set, the single binary classifier
(87.3% accuracy / 0.942 AUC on 2025), SHAP explanations, the clickable
map, and the error analysis.


# v2: Feature Mining, Calibration, Custom Deadlines (August 2026)

## Shared feature module (features.py)
Moved feature construction into features.py, imported by both train.py and
app.py. Previously the app hand-built its feature vector, which is exactly
where a silent train/serve mismatch bug would live. One code path now serves
both.

## Reproducible pipeline (train.py) alongside the notebook
eda.ipynb remains the original exploration record; train.py is the
reproducible pipeline that regenerates every model artifact and figure.
Verified it reproduces the documented v1 metrics exactly (85.1% / 65.7% /
74.9% / 0.916) before changing anything.

## New features (v2)
Audited all 54 raw columns for signal and submission-time availability. Added:
- has_project (linked to a larger development project): 1% vs 62% delay rate
- processing track (Standard/Express/Expedite/Not Required): 0.1%-95% range
- declared valuation (log-scaled + has_valuation flag): higher valuation,
  much higher delay rate
- scope description length (log chars)
- permit-holder annual volume (log permits/yr from training year): high-volume
  contractors 5.6% delayed vs 38% for one-timers
Documented assumption: processing code and scope text are set at intake and
not edited mid-review (the data snapshot can't verify this). Skipped
JOB_BC_CODE (25% coverage, overlaps approval type) and project age
(non-monotonic, weak). Post-decision columns (close/expire dates, status)
excluded as leakage.

Result on 2025: accuracy 85.1% -> 87.3%, precision 65.7% -> 69.3%, recall
74.9% -> 80.6%, ROC-AUC 0.916 -> 0.942.

## Top-15 types and cutoff computed before the lat/long drop
Two pipeline-reproduction bugs caught and fixed: the original notebook
computed the top-15 approval types and the 66-day cutoff BEFORE dropping
rows missing coordinates. Computing them after the drop changes both
(cutoff drifts to 73, Transportation Permit falls out of the top 15).
train.py now matches the original order of operations.

## Calibration: checked, deliberately NOT applied
The app exposes probabilities and a sensitivity threshold, so predicted 30%
should mean ~30% observed. Tested isotonic and sigmoid calibration wrappers,
selecting on a held-out Nov-Dec 2024 slice (never the 2025 test set, and
temporal to mimic deployment). Both calibrators were much WORSE than the raw
model (holdout Brier 0.30 vs 0.18) — they overfit 2024 probability quirks
that don't transfer across years. The uncalibrated model is already
well-calibrated on 2025 (ECE ~2.3%), so it ships as-is. Lesson: calibration
wrappers fitted on one year can hurt under temporal shift; check before
applying.

## Custom deadlines: threshold grid, after quantile regression failed
A tester wanted their own day threshold, which the single binary model can't
answer. First attempt: quantile regression (GradientBoostingRegressor,
quantile loss) at the 10th/25th/50th/75th/90th percentiles. It failed in a
specific way: days_to_approval is zero-inflated (~38% of permits approved
almost instantly), so the lower-quantile models collapsed to predicting ~0
for everything (2025 coverage 0.38 at both q10 and q25 — see
figures/quantile_coverage.png). A permit with 93% delay probability still got
q25=0, poisoning any interpolated estimate.

Shipped approach instead: a grid of binary classifiers at 15/30/45/66/90/120/
180 days — the same model class already validated at AUC 0.94 — with
monotonicity enforced across the grid and linear interpolation between
thresholds for arbitrary deadlines. Each grid model is individually
AUC/Brier-checked on 2025. Quantile models kept in the pipeline as a
documented negative result.

## Error analysis (reports/error_analysis.md)
The model is near-perfect on routine no-plan types and catches 91-96% of
delays on classic plan-review types (Building, Electrical, Plumbing,
Mechanical, Combination). Two real blind spots: Right-of-Way permits (38%
delay rate, 11.5% recall) and Fire Alarm permits (22% delay rate, 11%
recall) — delay in those types isn't explained by the current features.
19.4% of delayed permits are missed overall; median actual delay among
misses is 148 days, so misses skew toward shorter delays. Errors are flat
across months and show no geographic clustering (figures/fn_map.png).

## Map input tiles
Clickable folium/Leaflet map replaces raw lat/long number boxes. CartoDB
Voyager tiles: Mapbox-style look without an API token. Swapping in true
Mapbox tiles later is a one-line change reading a token from st.secrets.


# Changes Made After Peer Feedback

## Month slider
st.slider(1, 12, format_func=...) shows "March" instead of "3" while dragging, because this is more intuitive

## Sensitivity Slider
this replaces the hardcoded 50% cutoff with sensitivity = st.slider(...), then prediction = int(probability >= sensitivity).

## Legend as icons
two columns with 🔴/🔵 dots instead of the descriptive paragraph, quick visual scan instead of reading.

## Bar chart replacing the waterfall
np.argsort picks the top 10 features by absolute SHAP impact, ax.barh draws them as a horizontal bar chart, colored red (pushes toward delayed) or blue (pushes toward on-time), with ax.invert_yaxis() so the strongest feature sits at the top, matching natural reading order. No base value, no cumulative flow, no directional confusion, each bar just shows "this feature's size and direction of pull," which is a much cleaner mental model than the waterfall's accounting-style chain.