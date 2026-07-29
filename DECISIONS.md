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