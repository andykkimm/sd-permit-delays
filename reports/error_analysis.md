# Where the model is wrong (2025 test set)

## Per permit type

| type                                                  |    n |   delay_rate |   accuracy |   precision |   recall |
|:------------------------------------------------------|-----:|-------------:|-----------:|------------:|---------:|
| Traffic Control Permit                                | 9681 |        0.007 |      0.993 |     nan     |    0     |
| No-Plan - Residential - Combination Mech/Elec/Plum    | 9649 |        0.006 |      0.994 |     nan     |    0     |
| Photovoltaic - SB 379                                 | 5986 |        0.008 |      0.991 |       0     |    0     |
| Other                                                 | 4690 |        0.495 |      0.663 |       0.632 |    0.768 |
| Combination Building Permit                           | 4570 |        0.625 |      0.691 |       0.679 |    0.96  |
| Building Permit                                       | 2848 |        0.632 |      0.68  |       0.685 |    0.911 |
| Approval - Construction - Right Of Way Pmt-Const Plan | 2748 |        0.382 |      0.648 |       0.756 |    0.115 |
| Electrical Pmt                                        | 2305 |        0.733 |      0.727 |       0.748 |    0.946 |
| No-Plan - Nonresidential/Multifamily - Electrical     | 1916 |        0.014 |      0.986 |     nan     |    0     |
| Construction Noise Permit                             | 1787 |        0     |      1     |     nan     |  nan     |
| Plumbing Pmt                                          | 1392 |        0.713 |      0.729 |       0.737 |    0.964 |
| Mechanical Pmt                                        | 1308 |        0.752 |      0.764 |       0.778 |    0.959 |
| No-Plan - Nonresidential/Multifamily - Plumbing       | 1252 |        0.006 |      0.994 |     nan     |    0     |
| No-Plan - Nonresidential/Multifamily - Mechanical     | 1165 |        0.005 |      0.995 |     nan     |    0     |
| Approval - Construction - Fire Pmt - Alarm            | 1154 |        0.221 |      0.75  |       0.311 |    0.11  |
| Transportation Permit                                 |  139 |        0.014 |      0.986 |     nan     |    0     |

## By submission month

|   APPROVAL_CREATE_DATE |    n |   delay_rate |   error_rate |   fn_rate |
|-----------------------:|-----:|-------------:|-------------:|----------:|
|                      1 | 4587 |        0.246 |        0.11  |     0.045 |
|                      2 | 4405 |        0.233 |        0.124 |     0.046 |
|                      3 | 4479 |        0.21  |        0.167 |     0.061 |
|                      4 | 4301 |        0.208 |        0.139 |     0.046 |
|                      5 | 4399 |        0.244 |        0.135 |     0.059 |
|                      6 | 4146 |        0.239 |        0.129 |     0.036 |
|                      7 | 4332 |        0.232 |        0.142 |     0.039 |
|                      8 | 4419 |        0.244 |        0.127 |     0.048 |
|                      9 | 4633 |        0.209 |        0.118 |     0.044 |
|                     10 | 4836 |        0.231 |        0.116 |     0.039 |
|                     11 | 3862 |        0.248 |        0.112 |     0.047 |
|                     12 | 4191 |        0.237 |        0.109 |     0.026 |

## False negatives: how bad were the misses?

- 2358 delayed permits were predicted on-time (19.4% of all delayed permits)
- median actual delay among false negatives: 148 days (vs 192 days for all delayed permits)
- worst miss: 1824 days
