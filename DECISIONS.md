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