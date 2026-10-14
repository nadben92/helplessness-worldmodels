# E3 campaign: analyses (docs/e3/DECISIONS.md, 2026-10-07 (29))

## Main: b by zone at the budget (300,000 environment steps), campaign rule and aberration control

Final environment steps of the runs read: [300000].

# b by zone: experimental minus reference (docs/e3/DECISIONS.md, 2026-09-30)

b = I(experimental) - I(reference), same config and seed. **I_s is the main measure** (on the kernel as it is); I_s projected on the support and I_z (upper bound) are secondary. Zones are named by their appearance category; their dynamics in each environment are listed in the runs' reports. Class means: mean over the class's seeds, se = sqrt(sum se^2) / n.

## I_s (MAIN): class means

| config / class | seeds | all runs calibrated (C-a, C-b, C-c) | uncontrollable_deterministic | controllable_noisy | uncontrollable_noisy | controllable_deterministic |
|---|---|---|---|---|---|---|
| agent-gpu_campaign/A | [1] | no | -2.2704 [0.0011] | -0.1502 [0.0042] | -1.8663 [0.0029] | -0.2721 [0.0027] |
| agent-gpu_campaign/B | [6] | no | -2.2314 [0.0015] | +0.0729 [0.0041] | -1.8365 [0.0028] | +0.0970 [0.0018] |
| agent-gpu_pilot_closed_loop/A | [0] | no | -2.2572 [0.0014] | +0.1075 [0.0043] | -1.7885 [0.0032] | +0.0009 [0.0013] |
| agent-gpu_pilot_closed_loop/B | [5] | no | -2.2740 [0.0012] | -0.0757 [0.0041] | -1.6239 [0.0033] | +0.0212 [0.0011] |

## I_s (MAIN): per pair

| config | class | seed | uncontrollable_deterministic | controllable_noisy | uncontrollable_noisy | controllable_deterministic |
|---|---|---|---|---|---|---|
| agent-gpu_campaign | A | 1 | -2.2704 [0.0011] (0.026 vs 2.297) | -0.1502 [0.0042] (1.765 vs 1.915) | -1.8663 [0.0029] (0.044 vs 1.910) | -0.2721 [0.0027] (1.999 vs 2.271) |
| agent-gpu_campaign | B | 6 | -2.2314 [0.0015] (0.007 vs 2.239) | +0.0729 [0.0041] (1.936 vs 1.864) | -1.8365 [0.0028] (0.025 vs 1.862) | +0.0970 [0.0018] (2.305 vs 2.208) |
| agent-gpu_pilot_closed_loop | A | 0 | -2.2572 [0.0014] (0.007 vs 2.264) | +0.1075 [0.0043] (1.885 vs 1.778) | -1.7885 [0.0032] (0.027 vs 1.816) | +0.0009 [0.0013] (2.298 vs 2.297) |
| agent-gpu_pilot_closed_loop | B | 5 | -2.2740 [0.0012] (0.008 vs 2.282) | -0.0757 [0.0041] (1.868 vs 1.944) | -1.6239 [0.0033] (0.021 vs 1.645) | +0.0212 [0.0011] (2.313 vs 2.292) |

## I_s projected on the support (secondary): class means

| config / class | seeds | all runs calibrated (C-a, C-b, C-c) | uncontrollable_deterministic | controllable_noisy | uncontrollable_noisy | controllable_deterministic |
|---|---|---|---|---|---|---|
| agent-gpu_campaign/A | [1] | no | -2.2662 [0.0014] | -0.1522 [0.0046] | -1.8022 [0.0032] | -0.2902 [0.0029] |
| agent-gpu_campaign/B | [6] | no | -2.1988 [0.0019] | +0.0887 [0.0043] | -1.7364 [0.0031] | +0.1263 [0.0022] |
| agent-gpu_pilot_closed_loop/A | [0] | no | -2.2529 [0.0015] | +0.0967 [0.0046] | -1.7532 [0.0034] | +0.0109 [0.0016] |
| agent-gpu_pilot_closed_loop/B | [5] | no | -2.2596 [0.0015] | -0.0682 [0.0044] | -1.5982 [0.0034] | +0.0330 [0.0016] |

## I_s projected on the support (secondary): per pair

| config | class | seed | uncontrollable_deterministic | controllable_noisy | uncontrollable_noisy | controllable_deterministic |
|---|---|---|---|---|---|---|
| agent-gpu_campaign | A | 1 | -2.2662 [0.0014] (0.006 vs 2.272) | -0.1522 [0.0046] (1.666 vs 1.819) | -1.8022 [0.0032] (0.018 vs 1.821) | -0.2902 [0.0029] (1.974 vs 2.264) |
| agent-gpu_campaign | B | 6 | -2.1988 [0.0019] (0.002 vs 2.201) | +0.0887 [0.0043] (1.866 vs 1.777) | -1.7364 [0.0031] (0.006 vs 1.743) | +0.1263 [0.0022] (2.299 vs 2.172) |
| agent-gpu_pilot_closed_loop | A | 0 | -2.2529 [0.0015] (0.002 vs 2.255) | +0.0967 [0.0046] (1.777 vs 1.680) | -1.7532 [0.0034] (0.009 vs 1.762) | +0.0109 [0.0016] (2.289 vs 2.278) |
| agent-gpu_pilot_closed_loop | B | 5 | -2.2596 [0.0015] (0.003 vs 2.262) | -0.0682 [0.0044] (1.827 vs 1.895) | -1.5982 [0.0034] (0.008 vs 1.606) | +0.0330 [0.0016] (2.304 vs 2.270) |

## I_z (secondary): class means

| config / class | seeds | all runs calibrated (C-a, C-b, C-c) | uncontrollable_deterministic | controllable_noisy | uncontrollable_noisy | controllable_deterministic |
|---|---|---|---|---|---|---|
| agent-gpu_campaign/A | [1] | no | -2.1773 [0.0032] | -0.2263 [0.0024] | -2.0136 [0.0031] | -0.2170 [0.0019] |
| agent-gpu_campaign/B | [6] | no | -2.2265 [0.0034] | +0.0474 [0.0019] | -1.8978 [0.0038] | +0.0261 [0.0010] |
| agent-gpu_pilot_closed_loop/A | [0] | no | -2.2431 [0.0028] | +0.0518 [0.0022] | -1.9552 [0.0032] | -0.0037 [0.0009] |
| agent-gpu_pilot_closed_loop/B | [5] | no | -2.2050 [0.0047] | -0.0708 [0.0020] | -1.9509 [0.0032] | +0.0031 [0.0009] |

## I_z (secondary): per pair

| config | class | seed | uncontrollable_deterministic | controllable_noisy | uncontrollable_noisy | controllable_deterministic |
|---|---|---|---|---|---|---|
| agent-gpu_campaign | A | 1 | -2.1773 [0.0032] (0.144 vs 2.321) | -0.2263 [0.0024] (1.986 vs 2.212) | -2.0136 [0.0031] (0.201 vs 2.214) | -0.2170 [0.0019] (2.099 vs 2.316) |
| agent-gpu_campaign | B | 6 | -2.2265 [0.0034] (0.082 vs 2.308) | +0.0474 [0.0019] (2.221 vs 2.174) | -1.8978 [0.0038] (0.285 vs 2.182) | +0.0261 [0.0010] (2.322 vs 2.296) |
| agent-gpu_pilot_closed_loop | A | 0 | -2.2431 [0.0028] (0.072 vs 2.315) | +0.0518 [0.0022] (2.206 vs 2.154) | -1.9552 [0.0032] (0.210 vs 2.165) | -0.0037 [0.0009] (2.317 vs 2.320) |
| agent-gpu_pilot_closed_loop | B | 5 | -2.2050 [0.0047] (0.114 vs 2.319) | -0.0708 [0.0020] (2.160 vs 2.231) | -1.9509 [0.0032] (0.197 vs 2.148) | +0.0031 [0.0009] (2.322 vs 2.319) |

## Bias control (DECISIONS 2026-10-02 (16))

Per pair and zone, as measured. Aberration flags: values outside E3-0's ranges (configs/e3/e30_aberration_control.yaml) -- to be explained before interpretation, never excluded.

| config | seed | zone | b (I_s) | b (I_s projected) | difference | off-support env / ref (difference) | aberration flags |
|---|---|---|---|---|---|---|---|
| agent-gpu_campaign | 1 | uncontrollable_deterministic | -2.2704 | -2.2662 | -0.0042 | 0.022 / 0.007 (+0.015) | off-support (env) 0.0224 outside [0.0297, 0.0770] |
| agent-gpu_campaign | 1 | controllable_noisy | -0.1502 | -0.1522 | +0.0019 | 0.065 / 0.050 (+0.015) | none |
| agent-gpu_campaign | 1 | uncontrollable_noisy | -1.8663 | -1.8022 | -0.0640 | 0.059 / 0.054 (+0.005) | off-support (env) 0.0589 outside [0.1207, 0.1787] |
| agent-gpu_campaign | 1 | controllable_deterministic | -0.2721 | -0.2902 | +0.0182 | 0.042 / 0.006 (+0.036) | |off-support env - ref| = 0.0362 > 0.0035 |
| agent-gpu_campaign | 6 | uncontrollable_deterministic | -2.2314 | -2.1988 | -0.0325 | 0.007 / 0.014 (-0.008) | off-support (env) 0.0067 outside [0.0297, 0.0770] |
| agent-gpu_campaign | 6 | controllable_noisy | +0.0729 | +0.0887 | -0.0158 | 0.041 / 0.055 (-0.014) | none |
| agent-gpu_campaign | 6 | uncontrollable_noisy | -1.8365 | -1.7364 | -0.1000 | 0.038 / 0.071 (-0.033) | off-support (env) 0.0382 outside [0.1207, 0.1787] |
| agent-gpu_campaign | 6 | controllable_deterministic | +0.0970 | +0.1263 | -0.0293 | 0.003 / 0.021 (-0.018) | |b(I_s) - b(I_s projected)| = 0.0293 > 0.0285; |off-support env - ref| = 0.0179 > 0.0035 |
| agent-gpu_pilot_closed_loop | 0 | uncontrollable_deterministic | -2.2572 | -2.2529 | -0.0042 | 0.012 / 0.007 (+0.005) | off-support (env) 0.0116 outside [0.0297, 0.0770] |
| agent-gpu_pilot_closed_loop | 0 | controllable_noisy | +0.1075 | +0.0967 | +0.0107 | 0.060 / 0.065 (-0.005) | none |
| agent-gpu_pilot_closed_loop | 0 | uncontrollable_noisy | -1.7885 | -1.7532 | -0.0353 | 0.052 / 0.039 (+0.013) | off-support (env) 0.0516 outside [0.1207, 0.1787] |
| agent-gpu_pilot_closed_loop | 0 | controllable_deterministic | +0.0009 | +0.0109 | -0.0100 | 0.005 / 0.006 (-0.002) | none |
| agent-gpu_pilot_closed_loop | 5 | uncontrollable_deterministic | -2.2740 | -2.2596 | -0.0144 | 0.007 / 0.007 (-0.000) | off-support (env) 0.0069 outside [0.0297, 0.0770] |
| agent-gpu_pilot_closed_loop | 5 | controllable_noisy | -0.0757 | -0.0682 | -0.0075 | 0.033 / 0.045 (-0.012) | none |
| agent-gpu_pilot_closed_loop | 5 | uncontrollable_noisy | -1.6239 | -1.5982 | -0.0258 | 0.050 / 0.065 (-0.015) | off-support (env) 0.0501 outside [0.1207, 0.1787] |
| agent-gpu_pilot_closed_loop | 5 | controllable_deterministic | +0.0212 | +0.0330 | -0.0118 | 0.003 / 0.009 (-0.006) | |off-support env - ref| = 0.0060 > 0.0035 |

## Campaign rule (DECISIONS 2026-10-02 (16), 3)

Status: applied. Pairs: 4, classes {'A': 2, 'B': 2}; bound +- 0.0100.

| zone | b(I_s) - b(I_s projected): mean [95 % CI] | off-support env - ref: mean [95 % CI] | verdict |
|---|---|---|---|
| controllable_noisy | -0.0027 [-0.0209, +0.0156] | -0.0040 [-0.0249, +0.0170] | NOT interpretable |
| controllable_deterministic | -0.0082 [-0.0395, +0.0230] | +0.0026 [-0.0346, +0.0399] | NOT interpretable |

## Calibration and off-support mass of each run

| config | seed | condition | C-a | C-c | C-b by zone (L1_WM vs L1_B + 0.05; floor) | off-support mass by zone |
|---|---|---|---|---|---|---|
| agent-gpu_campaign | 1 | experimental | 0.9980 ok | 0.9764 FAIL | uncontrollable_d 0.020 vs 0.059 (0.000) ok; controllable_noi 0.154 vs 0.090 (0.020) FAIL; uncontrollable_n 0.168 vs 0.099 (0.020) FAIL; controllable_det 0.122 vs 0.057 (0.000) FAIL | uncontrollable_d 0.022; controllable_noi 0.065; uncontrollable_n 0.059; controllable_det 0.042 |
| agent-gpu_campaign | 1 | reference | 0.9961 ok | 0.9992 ok | uncontrollable_d 0.010 vs 0.056 (0.000) ok; controllable_noi 0.153 vs 0.093 (0.020) FAIL; uncontrollable_n 0.157 vs 0.094 (0.020) FAIL; controllable_det 0.008 vs 0.057 (0.000) ok | uncontrollable_d 0.007; controllable_noi 0.050; uncontrollable_n 0.054; controllable_det 0.006 |
| agent-gpu_campaign | 6 | experimental | 1.0000 ok | 0.9989 ok | uncontrollable_d 0.006 vs 0.058 (0.000) ok; controllable_noi 0.154 vs 0.091 (0.020) FAIL; uncontrollable_n 0.171 vs 0.100 (0.020) FAIL; controllable_det 0.006 vs 0.057 (0.000) ok | uncontrollable_d 0.007; controllable_noi 0.041; uncontrollable_n 0.038; controllable_det 0.003 |
| agent-gpu_campaign | 6 | reference | 0.9961 ok | 0.9964 ok | uncontrollable_d 0.023 vs 0.056 (0.000) ok; controllable_noi 0.172 vs 0.089 (0.020) FAIL; uncontrollable_n 0.185 vs 0.088 (0.020) FAIL; controllable_det 0.032 vs 0.055 (0.000) ok | uncontrollable_d 0.014; controllable_noi 0.055; uncontrollable_n 0.071; controllable_det 0.021 |
| agent-gpu_pilot_closed_loop | 0 | experimental | 1.0000 ok | 0.9987 ok | uncontrollable_d 0.011 vs 0.059 (0.000) ok; controllable_noi 0.160 vs 0.096 (0.020) FAIL; uncontrollable_n 0.164 vs 0.105 (0.020) FAIL; controllable_det 0.005 vs 0.056 (0.000) ok | uncontrollable_d 0.012; controllable_noi 0.060; uncontrollable_n 0.052; controllable_det 0.005 |
| agent-gpu_pilot_closed_loop | 0 | reference | 0.9961 ok | 0.9974 ok | uncontrollable_d 0.012 vs 0.056 (0.000) ok; controllable_noi 0.161 vs 0.090 (0.020) FAIL; uncontrollable_n 0.143 vs 0.086 (0.020) FAIL; controllable_det 0.010 vs 0.056 (0.000) ok | uncontrollable_d 0.007; controllable_noi 0.065; uncontrollable_n 0.039; controllable_det 0.006 |
| agent-gpu_pilot_closed_loop | 5 | experimental | 1.0000 ok | 0.9988 ok | uncontrollable_d 0.006 vs 0.059 (0.000) ok; controllable_noi 0.144 vs 0.098 (0.020) FAIL; uncontrollable_n 0.145 vs 0.099 (0.020) FAIL; controllable_det 0.006 vs 0.057 (0.000) ok | uncontrollable_d 0.007; controllable_noi 0.033; uncontrollable_n 0.050; controllable_det 0.003 |
| agent-gpu_pilot_closed_loop | 5 | reference | 0.9980 ok | 0.9989 ok | uncontrollable_d 0.009 vs 0.056 (0.000) ok; controllable_noi 0.157 vs 0.084 (0.020) FAIL; uncontrollable_n 0.187 vs 0.088 (0.020) FAIL; controllable_det 0.010 vs 0.055 (0.000) ok | uncontrollable_d 0.007; controllable_noi 0.045; uncontrollable_n 0.065; controllable_det 0.009 |

## Secondary analyses, by pair and checkpoint

### agent-gpu_campaign, seed 1 (class A)

| checkpoint | quantity | uncontrollable_deterministic | controllable_noisy | uncontrollable_noisy | controllable_deterministic |
|---|---|---|---|---|---|
| 15849 | b (I_s) | -0.0376 | +0.3080 | +0.0216 | +0.3079 |
| 15849 | b (I_s projected) | -0.0748 | +0.2230 | -0.0933 | +0.2664 |
| 15849 | difference of differences (descriptive) | +1.9179 | +0.3043 | +1.4229 | +0.3602 |
| 15849 | its parts: b_twin; WM - true env / ref | -1.9555; +0.2921 / -1.9922 | +0.0037; -1.1212 / -1.4292 | -1.4013; +0.2583 / -1.4162 | -0.0523; -1.7560 / -2.0639 |
| 15849 | alpha env / ref | NaN / +0.9511 | +0.8108 / +0.9611 | NaN / +0.9302 | +0.8017 / +0.8894 |
| 25119 | b (I_s) | -0.5991 | +0.1819 | -0.4450 | -0.5790 |
| 25119 | b (I_s projected) | -0.4708 | +0.1999 | -0.4285 | -0.6391 |
| 25119 | difference of differences (descriptive) | +1.4222 | +0.1770 | +1.0302 | -0.4840 |
| 25119 | its parts: b_twin; WM - true env / ref | -2.0213; +0.2918 / -1.4311 | +0.0049; -0.9714 / -1.1533 | -1.4752; +0.2328 / -0.9752 | -0.0951; -1.6414 / -1.0624 |
| 25119 | alpha env / ref | NaN / +0.7939 | +0.7172 / +0.8773 | NaN / +0.7438 | +0.7379 / +0.4385 |
| 39811 | b (I_s) | -1.3405 | +0.1241 | -1.0383 | -1.0208 |
| 39811 | b (I_s projected) | -1.3086 | +0.1958 | -0.9806 | -1.1211 |
| 39811 | difference of differences (descriptive) | +0.7406 | +0.1495 | +0.4528 | -0.9566 |
| 39811 | its parts: b_twin; WM - true env / ref | -2.0811; +0.2840 / -0.6974 | -0.0254; -0.6834 / -0.8075 | -1.4911; +0.2070 / -0.4077 | -0.0642; -1.5231 / -0.5022 |
| 39811 | alpha env / ref | NaN / +0.3971 | +0.5233 / +0.6682 | NaN / +0.4221 | +0.6857 / +0.1986 |
| 63096 | b (I_s) | -1.8451 | -0.0820 | -1.4343 | -1.1062 |
| 63096 | b (I_s projected) | -1.8858 | -0.0471 | -1.3887 | -1.2343 |
| 63096 | difference of differences (descriptive) | +0.2874 | -0.0674 | +0.0853 | -1.0641 |
| 63096 | its parts: b_twin; WM - true env / ref | -2.1324; +0.2195 / -0.2573 | -0.0146; -0.3701 / -0.2881 | -1.5196; +0.1685 / -0.0501 | -0.0421; -1.3366 / -0.2304 |
| 63096 | alpha env / ref | NaN / +0.1249 | +0.3258 / +0.3273 | NaN / +0.1868 | +0.6041 / +0.0831 |
| 100000 | b (I_s) | -2.0372 | -0.1776 | -1.6428 | -1.0029 |
| 100000 | b (I_s projected) | -2.0872 | -0.1603 | -1.5985 | -1.1202 |
| 100000 | difference of differences (descriptive) | +0.1322 | -0.1503 | -0.0945 | -0.9683 |
| 100000 | its parts: b_twin; WM - true env / ref | -2.1694; +0.1583 / -0.1264 | -0.0274; -0.1854 / -0.0078 | -1.5483; +0.1288 / +0.1186 | -0.0346; -1.1327 / -0.1297 |
| 100000 | alpha env / ref | NaN / +0.0547 | +0.2194 / +0.1428 | NaN / +0.0847 | +0.4945 / +0.0446 |
| 158489 | b (I_s) | -2.1869 | -0.1793 | -1.7741 | -0.7573 |
| 158489 | b (I_s projected) | -2.2084 | -0.1682 | -1.7215 | -0.8206 |
| 158489 | difference of differences (descriptive) | +0.0205 | -0.1523 | -0.2032 | -0.7264 |
| 158489 | its parts: b_twin; WM - true env / ref | -2.2074; +0.0721 / -0.0629 | -0.0270; -0.0379 / +0.1415 | -1.5709; +0.0830 / +0.2042 | -0.0308; -0.8504 / -0.0931 |
| 158489 | alpha env / ref | NaN / +0.0239 | +0.1344 / +0.0593 | NaN / +0.0355 | +0.3459 / +0.0322 |
| 251189 | b (I_s) | -2.2590 | -0.1522 | -1.8361 | -0.4251 |
| 251189 | b (I_s projected) | -2.2619 | -0.1511 | -1.7683 | -0.4539 |
| 251189 | difference of differences (descriptive) | -0.0234 | -0.1234 | -0.2541 | -0.4115 |
| 251189 | its parts: b_twin; WM - true env / ref | -2.2356; +0.0312 / -0.0318 | -0.0288; +0.0893 / +0.2414 | -1.5820; +0.0480 / +0.2311 | -0.0136; -0.4817 / -0.0565 |
| 251189 | alpha env / ref | NaN / +0.0113 | +0.0666 / -0.0001 | NaN / +0.0145 | +0.1727 / +0.0170 |
| 300000 | b (I_s) | -2.2704 | -0.1502 | -1.8663 | -0.2721 |
| 300000 | b (I_s projected) | -2.2662 | -0.1522 | -1.8022 | -0.2902 |
| 300000 | difference of differences (descriptive) | -0.0255 | -0.1209 | -0.2755 | -0.2667 |
| 300000 | its parts: b_twin; WM - true env / ref | -2.2449; +0.0261 / -0.0254 | -0.0293; +0.1121 / +0.2624 | -1.5907; +0.0442 / +0.2575 | -0.0054; -0.3226 / -0.0505 |
| 300000 | alpha env / ref | NaN / +0.0100 | +0.0547 / -0.0102 | NaN / -0.0004 | +0.1054 / +0.0140 |

b by visit class (n(s) = visits of the state's least-visited action; b [states env / ref]):

| checkpoint | class | uncontrollable_deterministic | controllable_noisy | uncontrollable_noisy | controllable_deterministic |
|---|---|---|---|---|---|
| 15849 | 0-9 | +0.0159 [19/4] | +0.0225 [6/9] | NaN [19/0] | -0.0641 [4/1] |
| 15849 | 10-29 | +0.0418 [6/21] | +0.3594 [15/16] | -0.0239 [4/25] | +0.2561 [16/19] |
| 15849 | 30-99 | NaN [0/0] | NaN [4/0] | NaN [2/0] | +0.6779 [5/5] |
| 25119 | 0-9 | -0.3058 [16/2] | -0.2566 [2/6] | NaN [17/0] | NaN [1/0] |
| 25119 | 10-29 | -0.6561 [8/22] | +0.1627 [17/19] | -0.5086 [6/24] | -0.4364 [19/12] |
| 25119 | 30-99 | -0.7495 [1/1] | NaN [6/0] | -0.2104 [2/1] | -0.4143 [5/13] |
| 39811 | 0-9 | -0.8968 [15/2] | NaN [1/0] | NaN [14/0] | NaN [0/0] |
| 39811 | 10-29 | -1.3435 [9/17] | +0.0192 [17/23] | -1.0142 [9/16] | -0.8819 [20/11] |
| 39811 | 30-99 | -1.9021 [1/6] | +0.5684 [7/2] | -1.3071 [2/9] | -0.7632 [5/14] |
| 63096 | 0-9 | NaN [14/0] | NaN [0/0] | NaN [9/0] | NaN [0/0] |
| 63096 | 10-29 | -1.8854 [9/19] | -0.1499 [13/18] | -1.4169 [14/13] | -1.0867 [15/9] |
| 63096 | 30-99 | -2.1608 [2/6] | -0.0934 [12/7] | -1.6211 [2/12] | -0.9618 [10/16] |
| 100000 | 0-9 | NaN [9/0] | NaN [0/0] | NaN [5/0] | NaN [0/0] |
| 100000 | 10-29 | -2.0023 [12/12] | -0.3774 [9/14] | -1.5897 [15/9] | -1.1123 [8/6] |
| 100000 | 30-99 | -2.2151 [4/13] | -0.1160 [16/11] | -1.7482 [5/16] | -0.9205 [17/19] |
| 158489 | 0-9 | NaN [5/0] | NaN [0/0] | NaN [0/0] | NaN [0/0] |
| 158489 | 10-29 | -2.0877 [13/3] | -0.2930 [7/9] | -1.7739 [14/4] | -1.0212 [7/5] |
| 158489 | 30-99 | -2.2535 [7/22] | -0.1578 [18/16] | -1.8076 [11/21] | -0.6261 [18/19] |
| 158489 | 100-299 | NaN [0/0] | NaN [0/0] | NaN [0/0] | NaN [0/1] |
| 251189 | 0-9 | NaN [1/0] | NaN [0/0] | NaN [0/0] | NaN [0/0] |
| 251189 | 10-29 | NaN [8/0] | -0.3501 [4/5] | -1.7361 [6/1] | -0.4569 [1/2] |
| 251189 | 30-99 | -2.2687 [13/23] | -0.1481 [18/20] | -1.8503 [18/24] | -0.4242 [24/18] |
| 251189 | 100-299 | -2.3074 [3/2] | NaN [3/0] | NaN [1/0] | NaN [0/5] |
| 300000 | 10-29 | NaN [5/0] | -0.3480 [3/2] | -1.7334 [3/1] | NaN [0/2] |
| 300000 | 30-99 | -2.2769 [17/23] | -0.1408 [19/23] | -1.8781 [18/23] | -0.2772 [25/17] |
| 300000 | 100-299 | -2.3124 [3/2] | NaN [3/0] | -1.8813 [4/1] | NaN [0/6] |

Zones' separation in the latent (held-out accuracy of the zone probe / controllability probe; centroid distance controllable vs uncontrollable categories, in within-zone RMS units):

| checkpoint | run | latent | zone probe | controllability probe | centroid distance ratio |
|---|---|---|---|---|---|
| 15849 | experimental | h and z (recorded in the run) | 0.980 | 0.961 | 0.284 |
| 15849 | reference | h and z (recorded in the run) | 0.983 | 0.951 | 0.382 |
| 25119 | experimental | h and z (recorded in the run) | 0.989 | 0.982 | 0.372 |
| 25119 | reference | h and z (recorded in the run) | 0.995 | 0.976 | 0.445 |
| 39811 | experimental | h and z (recorded in the run) | 0.995 | 0.986 | 0.371 |
| 39811 | reference | h and z (recorded in the run) | 0.998 | 0.983 | 0.394 |
| 63096 | experimental | h and z (recorded in the run) | 0.993 | 0.984 | 0.349 |
| 63096 | reference | h and z (recorded in the run) | 0.999 | 0.989 | 0.403 |
| 100000 | experimental | h and z (recorded in the run) | 0.996 | 0.985 | 0.342 |
| 100000 | reference | h and z (recorded in the run) | 0.999 | 0.994 | 0.429 |
| 158489 | experimental | h and z (recorded in the run) | 0.995 | 0.990 | 0.346 |
| 158489 | reference | h and z (recorded in the run) | 0.999 | 0.997 | 0.437 |
| 251189 | experimental | h and z (recorded in the run) | 0.998 | 0.990 | 0.349 |
| 251189 | reference | h and z (recorded in the run) | 1.000 | 0.997 | 0.442 |
| 300000 | experimental | h and z (recorded in the run) | 0.996 | 0.990 | 0.345 |
| 300000 | reference | h and z (recorded in the run) | 1.000 | 0.995 | 0.434 |

### agent-gpu_campaign, seed 6 (class B)

| checkpoint | quantity | uncontrollable_deterministic | controllable_noisy | uncontrollable_noisy | controllable_deterministic |
|---|---|---|---|---|---|
| 25119 | b (I_s) | -1.1787 | -0.1419 | -0.9314 | -0.0375 |
| 25119 | b (I_s projected) | -1.1060 | -0.0076 | -0.7031 | +0.0214 |
| 25119 | difference of differences (descriptive) | +0.9071 | -0.2335 | +0.5299 | -0.0744 |
| 25119 | its parts: b_twin; WM - true env / ref | -2.0858; +0.2266 / -0.9167 | +0.0916; -0.7691 / -0.6272 | -1.4613; +0.2066 / -0.5149 | +0.0370; -1.2262 / -1.1888 |
| 25119 | alpha env / ref | NaN / +0.5132 | +0.6105 / +0.6457 | NaN / +0.6435 | +0.5527 / +0.6187 |
| 39811 | b (I_s) | -1.7553 | +0.1140 | -1.3628 | +0.2714 |
| 39811 | b (I_s projected) | -1.7756 | +0.2098 | -1.1712 | +0.3394 |
| 39811 | difference of differences (descriptive) | +0.3639 | +0.0493 | +0.1479 | +0.2120 |
| 39811 | its parts: b_twin; WM - true env / ref | -2.1192; +0.1871 / -0.3795 | +0.0647; -0.1255 / -0.2394 | -1.5106; +0.1592 / -0.1310 | +0.0594; -0.2560 / -0.5274 |
| 39811 | alpha env / ref | NaN / +0.1893 | +0.2412 / +0.3762 | NaN / +0.3374 | +0.1053 / +0.2527 |
| 63096 | b (I_s) | -2.0741 | -0.0165 | -1.7706 | +0.1965 |
| 63096 | b (I_s projected) | -2.0466 | +0.0178 | -1.6330 | +0.2445 |
| 63096 | difference of differences (descriptive) | +0.0709 | -0.0621 | -0.2268 | +0.1368 |
| 63096 | its parts: b_twin; WM - true env / ref | -2.1450; +0.0655 / -0.1824 | +0.0456; +0.0861 / +0.1026 | -1.5438; +0.0807 / +0.1984 | +0.0597; -0.0717 / -0.2682 |
| 63096 | alpha env / ref | NaN / +0.0742 | +0.1005 / +0.1427 | NaN / +0.0978 | +0.0250 / +0.1123 |
| 100000 | b (I_s) | -2.1764 | +0.0128 | -1.8223 | +0.1567 |
| 100000 | b (I_s projected) | -2.1322 | +0.0286 | -1.6912 | +0.1953 |
| 100000 | difference of differences (descriptive) | -0.0047 | -0.0219 | -0.2698 | +0.1042 |
| 100000 | its parts: b_twin; WM - true env / ref | -2.1717; +0.0183 / -0.1272 | +0.0347; +0.1812 / +0.1684 | -1.5526; +0.0466 / +0.2160 | +0.0525; -0.0278 / -0.1845 |
| 100000 | alpha env / ref | NaN / +0.0473 | +0.0406 / +0.0781 | NaN / +0.0601 | +0.0088 / +0.0694 |
| 158489 | b (I_s) | -2.2319 | +0.0136 | -1.8475 | +0.1198 |
| 158489 | b (I_s projected) | -2.1924 | +0.0261 | -1.7279 | +0.1514 |
| 158489 | difference of differences (descriptive) | -0.0316 | +0.0009 | -0.2781 | +0.0746 |
| 158489 | its parts: b_twin; WM - true env / ref | -2.2002; +0.0096 / -0.0804 | +0.0127; +0.2252 / +0.2116 | -1.5694; +0.0303 / +0.2248 | +0.0452; -0.0189 / -0.1387 |
| 158489 | alpha env / ref | NaN / +0.0313 | +0.0126 / +0.0428 | NaN / +0.0403 | +0.0061 / +0.0513 |
| 251189 | b (I_s) | -2.2343 | +0.0633 | -1.8358 | +0.1069 |
| 251189 | b (I_s projected) | -2.2073 | +0.0896 | -1.7270 | +0.1315 |
| 251189 | difference of differences (descriptive) | -0.0092 | +0.0429 | -0.2480 | +0.0729 |
| 251189 | its parts: b_twin; WM - true env / ref | -2.2251; +0.0084 / -0.0792 | +0.0204; +0.2674 / +0.2041 | -1.5878; +0.0256 / +0.2084 | +0.0340; -0.0164 / -0.1232 |
| 251189 | alpha env / ref | NaN / +0.0264 | -0.0106 / +0.0346 | NaN / +0.0397 | +0.0050 / +0.0431 |
| 300000 | b (I_s) | -2.2314 | +0.0729 | -1.8365 | +0.0970 |
| 300000 | b (I_s projected) | -2.1988 | +0.0887 | -1.7364 | +0.1263 |
| 300000 | difference of differences (descriptive) | +0.0029 | +0.0615 | -0.2423 | +0.0664 |
| 300000 | its parts: b_twin; WM - true env / ref | -2.2342; +0.0073 / -0.0833 | +0.0114; +0.2835 / +0.2106 | -1.5942; +0.0254 / +0.2089 | +0.0306; -0.0170 / -0.1141 |
| 300000 | alpha env / ref | NaN / +0.0279 | -0.0222 / +0.0246 | NaN / +0.0369 | +0.0049 / +0.0412 |

b by visit class (n(s) = visits of the state's least-visited action; b [states env / ref]):

| checkpoint | class | uncontrollable_deterministic | controllable_noisy | uncontrollable_noisy | controllable_deterministic |
|---|---|---|---|---|---|
| 25119 | 0-9 | NaN [18/0] | NaN [0/2] | -0.3700 [17/2] | NaN [0/0] |
| 25119 | 10-29 | -1.2901 [4/20] | -0.3127 [16/23] | -1.0422 [6/23] | -0.3326 [12/24] |
| 25119 | 30-99 | -1.6769 [3/5] | NaN [9/0] | NaN [2/0] | -0.3093 [13/1] |
| 39811 | 0-9 | NaN [15/0] | NaN [0/1] | -0.7969 [13/2] | NaN [0/0] |
| 39811 | 10-29 | -1.8491 [5/20] | -0.1481 [14/24] | -1.4616 [7/23] | -0.2095 [4/20] |
| 39811 | 30-99 | -2.1149 [4/5] | NaN [11/0] | NaN [5/0] | +0.2088 [21/5] |
| 39811 | 100-299 | NaN [1/0] | NaN [0/0] | NaN [0/0] | NaN [0/0] |
| 63096 | 0-9 | NaN [9/0] | NaN [0/0] | -1.4155 [7/2] | NaN [0/0] |
| 63096 | 10-29 | -2.0313 [10/19] | -0.1781 [11/25] | -1.8050 [13/22] | +0.1121 [2/16] |
| 63096 | 30-99 | -2.2193 [5/5] | NaN [13/0] | -1.7424 [5/1] | +0.1497 [22/9] |
| 63096 | 100-299 | -2.3056 [1/1] | NaN [1/0] | NaN [0/0] | NaN [1/0] |
| 100000 | 0-9 | NaN [5/0] | NaN [0/0] | -1.3871 [1/1] | NaN [0/0] |
| 100000 | 10-29 | -2.1454 [14/14] | -0.1241 [7/24] | -1.8293 [15/22] | +0.1288 [1/12] |
| 100000 | 30-99 | -2.2214 [5/10] | -0.0023 [16/1] | -1.8018 [9/2] | +0.1402 [23/13] |
| 100000 | 100-299 | -2.2808 [1/1] | NaN [2/0] | NaN [0/0] | NaN [1/0] |
| 158489 | 0-9 | NaN [1/0] | NaN [0/0] | NaN [1/0] | NaN [0/0] |
| 158489 | 10-29 | -2.2070 [15/11] | -0.0690 [4/16] | -1.8285 [8/18] | NaN [0/6] |
| 158489 | 30-99 | -2.2479 [8/12] | -0.0088 [19/9] | -1.8723 [15/7] | +0.1174 [20/19] |
| 158489 | 100-299 | -2.3040 [1/2] | NaN [2/0] | NaN [1/0] | NaN [5/0] |
| 251189 | 10-29 | -2.1757 [6/8] | -0.1904 [1/9] | -1.7486 [7/9] | NaN [0/1] |
| 251189 | 30-99 | -2.2549 [16/14] | +0.0505 [20/16] | -1.8762 [11/16] | +0.1101 [13/21] |
| 251189 | 100-299 | -2.2911 [3/3] | NaN [4/0] | NaN [7/0] | +0.0509 [12/3] |
| 300000 | 10-29 | -2.1560 [2/5] | -0.2554 [1/6] | -1.7076 [6/6] | NaN [0/1] |
| 300000 | 30-99 | -2.2420 [20/15] | +0.0392 [17/19] | -1.8724 [12/19] | +0.0968 [13/21] |
| 300000 | 100-299 | -2.2760 [3/5] | NaN [7/0] | NaN [7/0] | +0.0452 [12/3] |

Zones' separation in the latent (held-out accuracy of the zone probe / controllability probe; centroid distance controllable vs uncontrollable categories, in within-zone RMS units):

| checkpoint | run | latent | zone probe | controllability probe | centroid distance ratio |
|---|---|---|---|---|---|
| 25119 | experimental | h and z (recorded in the run) | 0.998 | 0.980 | 0.357 |
| 25119 | reference | h and z (recorded in the run) | 1.000 | 0.993 | 0.462 |
| 39811 | experimental | h and z (recorded in the run) | 1.000 | 0.995 | 0.388 |
| 39811 | reference | h and z (recorded in the run) | 1.000 | 0.994 | 0.460 |
| 63096 | experimental | h and z (recorded in the run) | 1.000 | 0.995 | 0.419 |
| 63096 | reference | h and z (recorded in the run) | 1.000 | 0.997 | 0.426 |
| 100000 | experimental | h and z (recorded in the run) | 1.000 | 0.998 | 0.433 |
| 100000 | reference | h and z (recorded in the run) | 1.000 | 0.998 | 0.421 |
| 158489 | experimental | h and z (recorded in the run) | 1.000 | 0.997 | 0.448 |
| 158489 | reference | h and z (recorded in the run) | 1.000 | 0.997 | 0.418 |
| 251189 | experimental | h and z (recorded in the run) | 1.000 | 0.999 | 0.465 |
| 251189 | reference | h and z (recorded in the run) | 1.000 | 0.998 | 0.410 |
| 300000 | experimental | h and z (recorded in the run) | 1.000 | 0.999 | 0.470 |
| 300000 | reference | h and z (recorded in the run) | 1.000 | 0.995 | 0.406 |

### agent-gpu_pilot_closed_loop, seed 0 (class A)

| checkpoint | quantity | uncontrollable_deterministic | controllable_noisy | uncontrollable_noisy | controllable_deterministic |
|---|---|---|---|---|---|
| 15849 | b (I_s) | -0.1069 | +0.1320 | -0.1593 | +0.1899 |
| 15849 | b (I_s projected) | -0.0990 | +0.0954 | -0.1223 | +0.2097 |
| 15849 | difference of differences (descriptive) | +1.9224 | +0.1454 | +1.3356 | +0.2282 |
| 15849 | its parts: b_twin; WM - true env / ref | -2.0293; +0.1323 / -2.0828 | -0.0133; -1.2861 / -1.4181 | -1.4949; +0.1275 / -1.3661 | -0.0383; -1.7033 / -1.8932 |
| 15849 | alpha env / ref | NaN / +0.9173 | +0.8933 / +0.9534 | NaN / +0.9095 | +0.8039 / +0.9038 |
| 25119 | b (I_s) | -0.8282 | +0.1882 | -0.4836 | +0.2281 |
| 25119 | b (I_s projected) | -0.7783 | +0.2149 | -0.4166 | +0.2756 |
| 25119 | difference of differences (descriptive) | +1.2984 | +0.1985 | +1.0493 | +0.2478 |
| 25119 | its parts: b_twin; WM - true env / ref | -2.1266; +0.1182 / -1.3755 | -0.0103; -0.8923 / -1.0805 | -1.5329; +0.1663 / -1.0031 | -0.0198; -0.6882 / -0.9163 |
| 25119 | alpha env / ref | NaN / +0.5895 | +0.6779 / +0.8301 | NaN / +0.7435 | +0.3294 / +0.5052 |
| 39811 | b (I_s) | -1.5963 | +0.1764 | -1.0543 | -0.0273 |
| 39811 | b (I_s projected) | -1.5776 | +0.2249 | -0.9678 | +0.0111 |
| 39811 | difference of differences (descriptive) | +0.5650 | +0.1963 | +0.4844 | -0.0233 |
| 39811 | its parts: b_twin; WM - true env / ref | -2.1613; +0.0826 / -0.6430 | -0.0199; -0.4984 / -0.6748 | -1.5387; +0.1260 / -0.4726 | -0.0039; -0.2805 / -0.2533 |
| 39811 | alpha env / ref | NaN / +0.2551 | +0.4529 / +0.5738 | NaN / +0.4144 | +0.1277 / +0.1249 |
| 63096 | b (I_s) | -2.0067 | +0.0904 | -1.4791 | -0.0295 |
| 63096 | b (I_s projected) | -1.9981 | +0.1187 | -1.4240 | -0.0163 |
| 63096 | difference of differences (descriptive) | +0.1750 | +0.0835 | +0.0837 | -0.0342 |
| 63096 | its parts: b_twin; WM - true env / ref | -2.1818; +0.0428 / -0.2724 | +0.0069; -0.1492 / -0.2396 | -1.5629; +0.0871 / -0.0867 | +0.0048; -0.1172 / -0.0877 |
| 63096 | alpha env / ref | NaN / +0.0901 | +0.2370 / +0.2875 | NaN / +0.1706 | +0.0520 / +0.0387 |
| 100000 | b (I_s) | -2.1610 | +0.0546 | -1.6714 | -0.0181 |
| 100000 | b (I_s projected) | -2.1530 | +0.0551 | -1.6334 | -0.0058 |
| 100000 | difference of differences (descriptive) | +0.0388 | +0.0465 | -0.0941 | -0.0239 |
| 100000 | its parts: b_twin; WM - true env / ref | -2.1998; +0.0197 / -0.1412 | +0.0081; +0.0543 / -0.0003 | -1.5773; +0.0597 / +0.0782 | +0.0057; -0.0717 / -0.0536 |
| 100000 | alpha env / ref | NaN / +0.0426 | +0.1171 / +0.1409 | NaN / +0.0717 | +0.0254 / +0.0203 |
| 158489 | b (I_s) | -2.2136 | +0.0755 | -1.7501 | -0.0092 |
| 158489 | b (I_s projected) | -2.2046 | +0.0482 | -1.7144 | +0.0032 |
| 158489 | difference of differences (descriptive) | +0.0056 | +0.0711 | -0.1560 | -0.0125 |
| 158489 | its parts: b_twin; WM - true env / ref | -2.2191; +0.0087 / -0.0997 | +0.0045; +0.1712 / +0.0957 | -1.5941; +0.0368 / +0.1339 | +0.0034; -0.0465 / -0.0374 |
| 158489 | alpha env / ref | NaN / +0.0270 | +0.0550 / +0.0718 | NaN / +0.0326 | +0.0156 / +0.0147 |
| 251189 | b (I_s) | -2.2457 | +0.0931 | -1.7953 | +0.0009 |
| 251189 | b (I_s projected) | -2.2369 | +0.0798 | -1.7586 | +0.0110 |
| 251189 | difference of differences (descriptive) | -0.0073 | +0.0882 | -0.1964 | -0.0040 |
| 251189 | its parts: b_twin; WM - true env / ref | -2.2384; +0.0060 / -0.0702 | +0.0049; +0.2384 / +0.1454 | -1.5989; +0.0288 / +0.1711 | +0.0049; -0.0293 / -0.0301 |
| 251189 | alpha env / ref | NaN / +0.0180 | +0.0062 / +0.0405 | NaN / +0.0086 | +0.0097 / +0.0108 |
| 300000 | b (I_s) | -2.2572 | +0.1075 | -1.7885 | +0.0009 |
| 300000 | b (I_s projected) | -2.2529 | +0.0967 | -1.7532 | +0.0109 |
| 300000 | difference of differences (descriptive) | -0.0117 | +0.1036 | -0.1801 | -0.0054 |
| 300000 | its parts: b_twin; WM - true env / ref | -2.2454; +0.0067 / -0.0581 | +0.0039; +0.2324 / +0.1249 | -1.6084; +0.0274 / +0.1630 | +0.0063; -0.0243 / -0.0252 |
| 300000 | alpha env / ref | NaN / +0.0148 | +0.0041 / +0.0448 | NaN / +0.0108 | +0.0081 / +0.0094 |

b by visit class (n(s) = visits of the state's least-visited action; b [states env / ref]):

| checkpoint | class | uncontrollable_deterministic | controllable_noisy | uncontrollable_noisy | controllable_deterministic |
|---|---|---|---|---|---|
| 15849 | 0-9 | NaN [19/0] | +0.1239 [4/2] | -0.1814 [18/1] | +0.1059 [8/1] |
| 15849 | 10-29 | -0.1390 [5/22] | +0.0944 [17/23] | -0.1531 [6/24] | +0.1191 [9/22] |
| 15849 | 30-99 | -0.1233 [1/3] | NaN [4/0] | NaN [1/0] | +0.4443 [8/2] |
| 25119 | 0-9 | NaN [16/0] | +0.0290 [4/1] | NaN [17/0] | NaN [5/0] |
| 25119 | 10-29 | -0.7365 [7/16] | -0.0090 [15/24] | -0.5138 [6/22] | +0.3616 [9/17] |
| 25119 | 30-99 | -1.1724 [2/9] | NaN [6/0] | -0.6300 [2/3] | +0.3287 [11/8] |
| 39811 | 0-9 | NaN [13/0] | NaN [2/0] | NaN [14/0] | NaN [4/0] |
| 39811 | 10-29 | -1.4350 [10/12] | -0.0279 [17/24] | -0.9953 [9/20] | +0.0207 [6/15] |
| 39811 | 30-99 | -1.8637 [2/13] | +0.7242 [6/1] | -1.5473 [2/5] | +0.0943 [15/10] |
| 63096 | 0-9 | NaN [8/0] | NaN [1/0] | NaN [7/0] | NaN [2/0] |
| 63096 | 10-29 | -1.8872 [14/9] | -0.0925 [16/23] | -1.4137 [13/19] | -0.2230 [5/11] |
| 63096 | 30-99 | -2.1217 [3/16] | +0.3572 [8/2] | -1.7666 [5/6] | +0.0557 [18/14] |
| 100000 | 0-9 | NaN [6/0] | NaN [1/0] | NaN [3/0] | NaN [1/0] |
| 100000 | 10-29 | -2.0785 [13/6] | -0.0331 [14/19] | -1.6153 [11/15] | -0.1469 [5/4] |
| 100000 | 30-99 | -2.2024 [6/19] | +0.1453 [10/6] | -1.7759 [11/10] | +0.0270 [18/21] |
| 100000 | 100-299 | NaN [0/0] | NaN [0/0] | NaN [0/0] | NaN [1/0] |
| 158489 | 0-9 | NaN [2/0] | NaN [0/0] | NaN [0/0] | NaN [0/0] |
| 158489 | 10-29 | -2.0478 [9/4] | +0.0067 [10/14] | -1.6579 [6/7] | -0.1674 [4/2] |
| 158489 | 30-99 | -2.2440 [14/19] | +0.1084 [15/11] | -1.7837 [19/18] | +0.0180 [16/23] |
| 158489 | 100-299 | NaN [0/2] | NaN [0/0] | NaN [0/0] | NaN [5/0] |
| 251189 | 10-29 | -2.1231 [6/3] | +0.1512 [4/5] | -1.6758 [2/4] | -0.1369 [3/1] |
| 251189 | 30-99 | -2.2602 [19/19] | +0.0808 [21/20] | -1.8089 [16/21] | +0.0132 [14/21] |
| 251189 | 100-299 | NaN [0/3] | NaN [0/0] | NaN [7/0] | +0.0323 [8/3] |
| 300000 | 10-29 | -2.1868 [5/2] | +0.1518 [4/3] | -1.7809 [1/3] | -0.0714 [3/1] |
| 300000 | 30-99 | -2.2566 [18/19] | +0.0988 [21/22] | -1.7787 [13/22] | +0.0122 [11/20] |
| 300000 | 100-299 | -2.3028 [2/4] | NaN [0/0] | NaN [11/0] | +0.0082 [11/4] |

Zones' separation in the latent (held-out accuracy of the zone probe / controllability probe; centroid distance controllable vs uncontrollable categories, in within-zone RMS units):

| checkpoint | run | latent | zone probe | controllability probe | centroid distance ratio |
|---|---|---|---|---|---|
| 15849 | experimental | h and z (recorded in the run) | 0.982 | 0.977 | 0.438 |
| 15849 | reference | h and z (recorded in the run) | 0.991 | 0.950 | 0.389 |
| 25119 | experimental | h and z (recorded in the run) | 0.997 | 0.994 | 0.500 |
| 25119 | reference | h and z (recorded in the run) | 0.999 | 0.974 | 0.474 |
| 39811 | experimental | h and z (recorded in the run) | 0.999 | 0.995 | 0.516 |
| 39811 | reference | h and z (recorded in the run) | 1.000 | 0.990 | 0.407 |
| 63096 | experimental | h and z (recorded in the run) | 1.000 | 0.998 | 0.531 |
| 63096 | reference | h and z (recorded in the run) | 1.000 | 0.993 | 0.398 |
| 100000 | experimental | h and z (recorded in the run) | 0.999 | 0.997 | 0.556 |
| 100000 | reference | h and z (recorded in the run) | 1.000 | 0.996 | 0.389 |
| 158489 | experimental | h and z (recorded in the run) | 0.999 | 0.998 | 0.577 |
| 158489 | reference | h and z (recorded in the run) | 0.999 | 0.995 | 0.375 |
| 251189 | experimental | h and z (recorded in the run) | 1.000 | 0.999 | 0.586 |
| 251189 | reference | h and z (recorded in the run) | 1.000 | 0.995 | 0.365 |
| 300000 | experimental | h and z (recorded in the run) | 1.000 | 0.999 | 0.590 |
| 300000 | reference | h and z (recorded in the run) | 1.000 | 0.995 | 0.359 |

### agent-gpu_pilot_closed_loop, seed 5 (class B)

| checkpoint | quantity | uncontrollable_deterministic | controllable_noisy | uncontrollable_noisy | controllable_deterministic |
|---|---|---|---|---|---|
| 15849 | b (I_s) | -0.0621 | +0.3746 | -0.1638 | +0.6549 |
| 15849 | b (I_s projected) | -0.0538 | +0.3982 | -0.1313 | +0.6948 |
| 15849 | difference of differences (descriptive) | +1.9091 | +0.3930 | +1.2831 | +0.6852 |
| 15849 | its parts: b_twin; WM - true env / ref | -1.9713; +0.1807 / -2.0791 | -0.0184; -1.0439 / -1.4184 | -1.4470; +0.0885 / -1.4006 | -0.0303; -1.4211 / -2.0760 |
| 15849 | alpha env / ref | NaN / +0.9644 | +0.7447 / +0.9417 | NaN / +0.8800 | +0.6404 / +0.9723 |
| 25119 | b (I_s) | -0.5863 | +0.2641 | -0.5022 | +0.4464 |
| 25119 | b (I_s projected) | -0.5163 | +0.2873 | -0.4232 | +0.5631 |
| 25119 | difference of differences (descriptive) | +1.4337 | +0.3229 | +0.9879 | +0.4903 |
| 25119 | its parts: b_twin; WM - true env / ref | -2.0199; +0.1719 / -1.5638 | -0.0589; -0.6753 / -0.9394 | -1.4901; +0.1003 / -1.0505 | -0.0439; -0.7395 / -1.1860 |
| 25119 | alpha env / ref | NaN / +0.7690 | +0.5388 / +0.7079 | NaN / +0.7128 | +0.3412 / +0.6224 |
| 39811 | b (I_s) | -1.5321 | +0.1274 | -0.7911 | -0.0024 |
| 39811 | b (I_s projected) | -1.4703 | +0.1315 | -0.7029 | +0.0574 |
| 39811 | difference of differences (descriptive) | +0.6010 | +0.1858 | +0.7466 | -0.0054 |
| 39811 | its parts: b_twin; WM - true env / ref | -2.1331; +0.1391 / -0.6507 | -0.0584; -0.2058 / -0.3332 | -1.5377; +0.0924 / -0.7694 | +0.0030; -0.3744 / -0.3720 |
| 39811 | alpha env / ref | NaN / +0.3041 | +0.2631 / +0.3611 | NaN / +0.5443 | +0.1688 / +0.1847 |
| 63096 | b (I_s) | -2.0677 | +0.0122 | -1.1816 | -0.0406 |
| 63096 | b (I_s projected) | -2.0558 | -0.0015 | -1.1303 | -0.0214 |
| 63096 | difference of differences (descriptive) | +0.1036 | +0.0394 | +0.3953 | -0.0530 |
| 63096 | its parts: b_twin; WM - true env / ref | -2.1713; +0.0922 / -0.1620 | -0.0272; +0.0726 / +0.0604 | -1.5769; +0.0714 / -0.3999 | +0.0124; -0.1485 / -0.1079 |
| 63096 | alpha env / ref | NaN / +0.0661 | +0.0975 / +0.1191 | NaN / +0.3284 | +0.0606 / +0.0501 |
| 100000 | b (I_s) | -2.1956 | -0.0565 | -1.4800 | +0.0035 |
| 100000 | b (I_s projected) | -2.1847 | -0.0714 | -1.4489 | +0.0046 |
| 100000 | difference of differences (descriptive) | +0.0003 | -0.0290 | +0.1107 | -0.0202 |
| 100000 | its parts: b_twin; WM - true env / ref | -2.1959; +0.0441 / -0.0822 | -0.0274; +0.1320 / +0.1885 | -1.5907; +0.0481 / -0.1248 | +0.0237; -0.0607 / -0.0642 |
| 100000 | alpha env / ref | NaN / +0.0306 | +0.0515 / +0.0392 | NaN / +0.1613 | +0.0283 / +0.0266 |
| 158489 | b (I_s) | -2.2428 | -0.0414 | -1.5588 | +0.0249 |
| 158489 | b (I_s projected) | -2.2316 | -0.0526 | -1.5232 | +0.0375 |
| 158489 | difference of differences (descriptive) | -0.0262 | -0.0153 | +0.0488 | -0.0090 |
| 158489 | its parts: b_twin; WM - true env / ref | -2.2166; +0.0238 / -0.0553 | -0.0261; +0.1696 / +0.2110 | -1.6076; +0.0344 / -0.0597 | +0.0339; -0.0209 / -0.0458 |
| 158489 | alpha env / ref | NaN / +0.0188 | +0.0252 / +0.0218 | NaN / +0.1255 | +0.0090 / +0.0188 |
| 251189 | b (I_s) | -2.2648 | -0.0815 | -1.6035 | +0.0244 |
| 251189 | b (I_s projected) | -2.2458 | -0.0844 | -1.5749 | +0.0337 |
| 251189 | difference of differences (descriptive) | -0.0301 | -0.0585 | +0.0217 | -0.0057 |
| 251189 | its parts: b_twin; WM - true env / ref | -2.2347; +0.0128 / -0.0443 | -0.0230; +0.1963 / +0.2778 | -1.6253; +0.0241 / -0.0253 | +0.0301; -0.0109 / -0.0353 |
| 251189 | alpha env / ref | NaN / +0.0143 | +0.0047 / -0.0109 | NaN / +0.0989 | +0.0047 / +0.0127 |
| 300000 | b (I_s) | -2.2740 | -0.0757 | -1.6239 | +0.0212 |
| 300000 | b (I_s projected) | -2.2596 | -0.0682 | -1.5982 | +0.0330 |
| 300000 | difference of differences (descriptive) | -0.0325 | -0.0417 | +0.0073 | -0.0067 |
| 300000 | its parts: b_twin; WM - true env / ref | -2.2415; +0.0081 / -0.0398 | -0.0340; +0.2151 / +0.2907 | -1.6312; +0.0211 / -0.0079 | +0.0280; -0.0091 / -0.0303 |
| 300000 | alpha env / ref | NaN / +0.0123 | -0.0075 / -0.0185 | NaN / +0.0896 | +0.0041 / +0.0124 |

b by visit class (n(s) = visits of the state's least-visited action; b [states env / ref]):

| checkpoint | class | uncontrollable_deterministic | controllable_noisy | uncontrollable_noisy | controllable_deterministic |
|---|---|---|---|---|---|
| 15849 | 0-9 | -0.0996 [21/2] | NaN [5/0] | -0.1026 [19/3] | +0.0275 [7/3] |
| 15849 | 10-29 | -0.1759 [4/23] | +0.3184 [17/25] | -0.1712 [6/22] | +0.5699 [11/22] |
| 15849 | 30-99 | NaN [0/0] | NaN [3/0] | NaN [0/0] | NaN [7/0] |
| 25119 | 0-9 | -0.8264 [20/1] | NaN [4/0] | NaN [18/0] | NaN [6/0] |
| 25119 | 10-29 | -0.6763 [2/23] | +0.0101 [14/23] | -0.5276 [5/25] | +0.7138 [9/16] |
| 25119 | 30-99 | -1.9175 [3/1] | +1.2693 [7/2] | NaN [2/0] | +0.6977 [9/8] |
| 25119 | 100-299 | NaN [0/0] | NaN [0/0] | NaN [0/0] | +0.0470 [1/1] |
| 39811 | 0-9 | NaN [16/0] | NaN [0/0] | NaN [16/0] | NaN [0/0] |
| 39811 | 10-29 | -1.4921 [4/17] | -0.3666 [8/16] | -0.8173 [4/24] | -0.4306 [8/13] |
| 39811 | 30-99 | -1.9368 [5/8] | +0.1979 [16/9] | -0.9353 [5/1] | +0.0963 [14/10] |
| 39811 | 100-299 | NaN [0/0] | NaN [1/0] | NaN [0/0] | +0.0266 [3/2] |
| 63096 | 0-9 | NaN [13/0] | NaN [0/0] | NaN [8/0] | NaN [0/0] |
| 63096 | 10-29 | -2.1011 [7/11] | -0.2621 [5/12] | -1.1365 [11/16] | -0.5427 [3/12] |
| 63096 | 30-99 | -2.1771 [5/14] | -0.0116 [17/13] | -1.3136 [6/9] | -0.0335 [17/11] |
| 63096 | 100-299 | NaN [0/0] | NaN [3/0] | NaN [0/0] | +0.0203 [5/2] |
| 100000 | 0-9 | NaN [7/0] | NaN [0/0] | NaN [3/0] | NaN [0/0] |
| 100000 | 10-29 | -2.2206 [13/6] | -0.2752 [5/5] | -1.4210 [9/11] | -0.1651 [2/10] |
| 100000 | 30-99 | -2.2337 [5/19] | -0.0704 [15/20] | -1.5332 [11/14] | -0.0056 [18/13] |
| 100000 | 100-299 | NaN [0/0] | NaN [5/0] | NaN [2/0] | +0.0034 [5/2] |
| 158489 | 0-9 | NaN [3/0] | NaN [0/0] | NaN [1/0] | NaN [0/0] |
| 158489 | 10-29 | -2.2554 [14/3] | +0.1023 [3/1] | -1.4995 [7/8] | NaN [0/6] |
| 158489 | 30-99 | -2.2603 [8/22] | -0.1036 [15/24] | -1.5805 [11/17] | +0.0097 [17/17] |
| 158489 | 100-299 | NaN [0/0] | NaN [7/0] | NaN [6/0] | +0.0018 [7/2] |
| 158489 | 300-999 | NaN [0/0] | NaN [0/0] | NaN [0/0] | NaN [1/0] |
| 251189 | 0-9 | NaN [1/0] | NaN [0/0] | NaN [0/0] | NaN [0/0] |
| 251189 | 10-29 | -2.2163 [13/1] | NaN [0/0] | -1.5022 [4/1] | NaN [0/2] |
| 251189 | 30-99 | -2.2733 [10/22] | -0.1359 [18/24] | -1.6100 [15/24] | +0.0242 [13/19] |
| 251189 | 100-299 | -2.2915 [1/2] | -0.0118 [7/1] | NaN [6/0] | -0.0021 [11/4] |
| 251189 | 300-999 | NaN [0/0] | NaN [0/0] | NaN [0/0] | NaN [1/0] |
| 300000 | 0-9 | NaN [1/0] | NaN [0/0] | NaN [0/0] | NaN [0/0] |
| 300000 | 10-29 | -2.2582 [9/1] | NaN [0/0] | NaN [2/0] | NaN [0/0] |
| 300000 | 30-99 | -2.2732 [12/21] | -0.1412 [17/24] | -1.6227 [15/25] | +0.0180 [9/21] |
| 300000 | 100-299 | -2.3128 [3/3] | -0.0552 [8/1] | NaN [7/0] | +0.0017 [15/4] |
| 300000 | 300-999 | NaN [0/0] | NaN [0/0] | NaN [1/0] | NaN [1/0] |

Zones' separation in the latent (held-out accuracy of the zone probe / controllability probe; centroid distance controllable vs uncontrollable categories, in within-zone RMS units):

| checkpoint | run | latent | zone probe | controllability probe | centroid distance ratio |
|---|---|---|---|---|---|
| 15849 | experimental | h and z (recorded in the run) | 0.997 | 0.988 | 0.379 |
| 15849 | reference | h and z (recorded in the run) | 0.989 | 0.946 | 0.372 |
| 25119 | experimental | h and z (recorded in the run) | 1.000 | 0.989 | 0.392 |
| 25119 | reference | h and z (recorded in the run) | 0.998 | 0.977 | 0.423 |
| 39811 | experimental | h and z (recorded in the run) | 1.000 | 0.987 | 0.381 |
| 39811 | reference | h and z (recorded in the run) | 1.000 | 0.989 | 0.416 |
| 63096 | experimental | h and z (recorded in the run) | 1.000 | 0.995 | 0.386 |
| 63096 | reference | h and z (recorded in the run) | 1.000 | 0.995 | 0.419 |
| 100000 | experimental | h and z (recorded in the run) | 1.000 | 0.992 | 0.406 |
| 100000 | reference | h and z (recorded in the run) | 1.000 | 0.997 | 0.422 |
| 158489 | experimental | h and z (recorded in the run) | 1.000 | 0.994 | 0.420 |
| 158489 | reference | h and z (recorded in the run) | 1.000 | 0.997 | 0.416 |
| 251189 | experimental | h and z (recorded in the run) | 1.000 | 0.997 | 0.432 |
| 251189 | reference | h and z (recorded in the run) | 1.000 | 0.998 | 0.393 |
| 300000 | experimental | h and z (recorded in the run) | 1.000 | 0.996 | 0.426 |
| 300000 | reference | h and z (recorded in the run) | 1.000 | 0.998 | 0.386 |

