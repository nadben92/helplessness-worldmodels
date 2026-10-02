# b by zone: experimental minus reference (docs/e3/DECISIONS.md, 2026-09-30)

b = I(experimental) - I(reference), same config and seed. **I_s is the main measure** (on the kernel as it is); I_s projected on the support and I_z (upper bound) are secondary. Zones are named by their appearance category; their dynamics in each environment are listed in the runs' reports. Class means: mean over the class's seeds, se = sqrt(sum se^2) / n.

## I_s (MAIN): class means

| config / class | seeds | all runs calibrated (C-a, C-b, C-c) | uncontrollable_deterministic | controllable_noisy | uncontrollable_noisy | controllable_deterministic |
|---|---|---|---|---|---|---|
| gpu_e30b_win64/A | [0] | no | -2.2394 [0.0015] | +0.0084 [0.0039] | -1.7921 [0.0031] | -0.0028 [0.0014] |
| gpu_e30b_win64/B | [5] | no | -2.2681 [0.0011] | -0.0040 [0.0040] | -1.7504 [0.0032] | -0.0090 [0.0012] |

## I_s (MAIN): per pair

| config | class | seed | uncontrollable_deterministic | controllable_noisy | uncontrollable_noisy | controllable_deterministic |
|---|---|---|---|---|---|---|
| gpu_e30b_win64 | A | 0 | -2.2394 [0.0015] (0.020 vs 2.259) | +0.0084 [0.0039] (1.945 vs 1.937) | -1.7921 [0.0031] (0.050 vs 1.842) | -0.0028 [0.0014] (2.290 vs 2.293) |
| gpu_e30b_win64 | B | 5 | -2.2681 [0.0011] (0.029 vs 2.297) | -0.0040 [0.0040] (1.942 vs 1.945) | -1.7504 [0.0032] (0.035 vs 1.785) | -0.0090 [0.0012] (2.292 vs 2.301) |

## I_s projected on the support (secondary): class means

| config / class | seeds | all runs calibrated (C-a, C-b, C-c) | uncontrollable_deterministic | controllable_noisy | uncontrollable_noisy | controllable_deterministic |
|---|---|---|---|---|---|---|
| gpu_e30b_win64/A | [0] | no | -2.2432 [0.0015] | -0.0075 [0.0043] | -1.7777 [0.0032] | +0.0093 [0.0017] |
| gpu_e30b_win64/B | [5] | no | -2.2689 [0.0013] | -0.0058 [0.0043] | -1.7375 [0.0033] | -0.0170 [0.0017] |

## I_s projected on the support (secondary): per pair

| config | class | seed | uncontrollable_deterministic | controllable_noisy | uncontrollable_noisy | controllable_deterministic |
|---|---|---|---|---|---|---|
| gpu_e30b_win64 | A | 0 | -2.2432 [0.0015] (0.006 vs 2.249) | -0.0075 [0.0043] (1.866 vs 1.874) | -1.7777 [0.0032] (0.016 vs 1.794) | +0.0093 [0.0017] (2.281 vs 2.272) |
| gpu_e30b_win64 | B | 5 | -2.2689 [0.0013] (0.010 vs 2.279) | -0.0058 [0.0043] (1.895 vs 1.901) | -1.7375 [0.0033] (0.015 vs 1.752) | -0.0170 [0.0017] (2.269 vs 2.286) |

## I_z (secondary): class means

| config / class | seeds | all runs calibrated (C-a, C-b, C-c) | uncontrollable_deterministic | controllable_noisy | uncontrollable_noisy | controllable_deterministic |
|---|---|---|---|---|---|---|
| gpu_e30b_win64/A | [0] | no | -2.1355 [0.0051] | +0.0145 [0.0036] | -1.8034 [0.0070] | -0.0016 [0.0021] |
| gpu_e30b_win64/B | [5] | no | -1.9922 [0.0074] | -0.0064 [0.0035] | -1.8485 [0.0070] | -0.0055 [0.0020] |

## I_z (secondary): per pair

| config | class | seed | uncontrollable_deterministic | controllable_noisy | uncontrollable_noisy | controllable_deterministic |
|---|---|---|---|---|---|---|
| gpu_e30b_win64 | A | 0 | -2.1355 [0.0051] (0.176 vs 2.312) | +0.0145 [0.0036] (2.203 vs 2.188) | -1.8034 [0.0070] (0.333 vs 2.136) | -0.0016 [0.0021] (2.314 vs 2.316) |
| gpu_e30b_win64 | B | 5 | -1.9922 [0.0074] (0.325 vs 2.317) | -0.0064 [0.0035] (2.215 vs 2.222) | -1.8485 [0.0070] (0.298 vs 2.147) | -0.0055 [0.0020] (2.316 vs 2.322) |

## Bias control (reading rule, DECISIONS 2026-09-30 (9))

Controllable zones: b is interpretable only if |b(I_s) - b(I_s projected)| <= tau_b and |off-support(env) - off-support(ref)| <= tau_off (per zone). Uncontrollable zones: the off-support condition fails by construction (the reference is a controllable zone) and is not applied; b is read against the E3-0 floor provided the environment-side off-support mass stays in the interval observed on the E3-0 environment runs +- tau_off_floor, else diagnostic only.

| zone | tau_b | tau_off | tau_off_floor |
|---|---|---|---|
| uncontrollable_deterministic | - | - | 0.017 |
| controllable_noisy | 0.020 | 0.015 | - |
| uncontrollable_noisy | - | - | 0.021 |
| controllable_deterministic | 0.028 | 0.003 | - |

| config | seed | zone | b (I_s) | b (I_s projected) | off-support env / ref (difference) | vs E3-0 floor | verdict |
|---|---|---|---|---|---|---|---|
| gpu_e30b_win64 | 0 | uncontrollable_deterministic | -2.2394 | -2.2432 | 0.047 / 0.009 (+0.039) | floor -2.2538, b - floor +0.0144, interval [0.030, 0.077] | read against the floor |
| gpu_e30b_win64 | 0 | controllable_noisy | +0.0084 | -0.0075 | 0.050 / 0.037 (+0.012) | - | interpretable |
| gpu_e30b_win64 | 0 | uncontrollable_noisy | -1.7921 | -1.7777 | 0.157 / 0.046 (+0.112) | floor -1.7713, b - floor -0.0208, interval [0.121, 0.179] | read against the floor |
| gpu_e30b_win64 | 0 | controllable_deterministic | -0.0028 | +0.0093 | 0.005 / 0.006 (-0.002) | - | interpretable |
| gpu_e30b_win64 | 5 | uncontrollable_deterministic | -2.2681 | -2.2689 | 0.060 / 0.005 (+0.054) | floor -2.2538, b - floor -0.0144, interval [0.030, 0.077] | read against the floor |
| gpu_e30b_win64 | 5 | controllable_noisy | -0.0040 | -0.0058 | 0.043 / 0.042 (+0.002) | - | interpretable |
| gpu_e30b_win64 | 5 | uncontrollable_noisy | -1.7504 | -1.7375 | 0.142 / 0.053 (+0.089) | floor -1.7713, b - floor +0.0208, interval [0.121, 0.179] | read against the floor |
| gpu_e30b_win64 | 5 | controllable_deterministic | -0.0090 | -0.0170 | 0.008 / 0.007 (+0.001) | - | interpretable |

## Calibration and off-support mass of each run

| config | seed | condition | C-a | C-c | C-b by zone (L1_WM vs L1_B + 0.05; floor) | off-support mass by zone |
|---|---|---|---|---|---|---|
| gpu_e30b_win64 | 0 | experimental | 0.9961 ok | 0.9975 ok | uncontrollable_d 0.001 vs 0.061 (0.000) ok; controllable_noi 0.163 vs 0.104 (0.020) FAIL; uncontrollable_n 0.185 vs 0.107 (0.020) FAIL; controllable_det 0.006 vs 0.059 (0.000) ok | uncontrollable_d 0.047; controllable_noi 0.050; uncontrollable_n 0.157; controllable_det 0.005 |
| gpu_e30b_win64 | 0 | reference | 1.0000 ok | 0.9952 ok | uncontrollable_d 0.033 vs 0.061 (0.000) ok; controllable_noi 0.156 vs 0.109 (0.020) FAIL; uncontrollable_n 0.186 vs 0.111 (0.020) FAIL; controllable_det 0.019 vs 0.062 (0.000) ok | uncontrollable_d 0.009; controllable_noi 0.037; uncontrollable_n 0.046; controllable_det 0.006 |
| gpu_e30b_win64 | 5 | experimental | 0.9980 ok | 0.9987 ok | uncontrollable_d 0.010 vs 0.061 (0.000) ok; controllable_noi 0.150 vs 0.103 (0.020) FAIL; uncontrollable_n 0.157 vs 0.110 (0.020) FAIL; controllable_det 0.010 vs 0.059 (0.000) ok | uncontrollable_d 0.060; controllable_noi 0.043; uncontrollable_n 0.142; controllable_det 0.008 |
| gpu_e30b_win64 | 5 | reference | 0.9980 ok | 0.9981 ok | uncontrollable_d 0.016 vs 0.061 (0.000) ok; controllable_noi 0.170 vs 0.111 (0.020) FAIL; uncontrollable_n 0.208 vs 0.113 (0.020) FAIL; controllable_det 0.018 vs 0.062 (0.000) ok | uncontrollable_d 0.005; controllable_noi 0.042; uncontrollable_n 0.053; controllable_det 0.007 |
