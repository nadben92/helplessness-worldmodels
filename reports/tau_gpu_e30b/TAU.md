# gpu_e30b re-measured: thresholds of the reading rule and its verdict

Definitions fixed in docs/e3/DECISIONS.md 2026-09-30 (9), B, before the values were computed; not changed here.

## 1. Batch check (one platform, one code version, the expected weights)

- passed

| job | weights SHA-256 | code source_sha256 | device / GPU | TF32 matmul / cudnn | K |
|---|---|---|---|---|---|
| win64_seed0_experimental | f60377e0e7a8... | dae7ff2e302b... | cuda / NVIDIA A100-SXM4-80GB | False / False | 1358 |
| win64_seed0_reference | fa60ea69d694... | dae7ff2e302b... | cuda / NVIDIA A100-SXM4-80GB | False / False | 1358 |
| win64_seed5_experimental | 9cc40755cc61... | dae7ff2e302b... | cuda / NVIDIA A100-SXM4-80GB | False / False | 1358 |
| win64_seed5_reference | 690ea27d22a4... | dae7ff2e302b... | cuda / NVIDIA A100-SXM4-80GB | False / False | 1358 |

## 2. Consistency with the original gpu_e30b measurements (tolerance 1e-06)

0 of 16 (run, zone) rows differ.

| job | zone | I_s original / re-measured | I_z original / re-measured | off-support original / re-measured | max diff | flag |
|---|---|---|---|---|---|---|
| win64_seed0_experimental | uncontrollable_deterministic | 0.020002 / 0.020002 | 0.176121 / 0.176121 | 0.047195 / 0.047195 | 0.00e+00 | same |
| win64_seed0_experimental | controllable_noisy | 1.945278 / 1.945278 | 2.202963 / 2.202963 | 0.049773 / 0.049773 | 0.00e+00 | same |
| win64_seed0_experimental | uncontrollable_noisy | 0.049560 / 0.049560 | 0.332964 / 0.332964 | 0.157309 / 0.157309 | 0.00e+00 | same |
| win64_seed0_experimental | controllable_deterministic | 2.290078 / 2.290078 | 2.314369 / 2.314369 | 0.004760 / 0.004760 | 0.00e+00 | same |
| win64_seed0_reference | uncontrollable_deterministic | 2.259428 / 2.259428 | 2.311623 / 2.311623 | 0.008537 / 0.008537 | 0.00e+00 | same |
| win64_seed0_reference | controllable_noisy | 1.936840 / 1.936840 | 2.188414 / 2.188414 | 0.037365 / 0.037365 | 0.00e+00 | same |
| win64_seed0_reference | uncontrollable_noisy | 1.841651 / 1.841651 | 2.136390 / 2.136390 | 0.045662 / 0.045662 | 0.00e+00 | same |
| win64_seed0_reference | controllable_deterministic | 2.292907 / 2.292907 | 2.315930 / 2.315930 | 0.006266 / 0.006266 | 0.00e+00 | same |
| win64_seed5_experimental | uncontrollable_deterministic | 0.028715 / 0.028715 | 0.325177 / 0.325177 | 0.059551 / 0.059551 | 0.00e+00 | same |
| win64_seed5_experimental | controllable_noisy | 1.941509 / 1.941509 | 2.215305 / 2.215305 | 0.043094 / 0.043094 | 0.00e+00 | same |
| win64_seed5_experimental | uncontrollable_noisy | 0.034592 / 0.034592 | 0.298024 / 0.298024 | 0.142149 / 0.142149 | 0.00e+00 | same |
| win64_seed5_experimental | controllable_deterministic | 2.291860 / 2.291860 | 2.316488 / 2.316488 | 0.007902 / 0.007902 | 0.00e+00 | same |
| win64_seed5_reference | uncontrollable_deterministic | 2.296861 / 2.296861 | 2.317363 / 2.317363 | 0.005087 / 0.005087 | 0.00e+00 | same |
| win64_seed5_reference | controllable_noisy | 1.945471 / 1.945471 | 2.221713 / 2.221713 | 0.041572 / 0.041572 | 0.00e+00 | same |
| win64_seed5_reference | uncontrollable_noisy | 1.785037 / 1.785037 | 2.146538 / 2.146538 | 0.052708 / 0.052708 | 0.00e+00 | same |
| win64_seed5_reference | controllable_deterministic | 2.300861 / 2.300861 | 2.321942 / 2.321942 | 0.006954 / 0.006954 | 0.00e+00 | same |

Per-state arrays, largest absolute difference: win64_seed0_experimental: real_I_s 0.00e+00, real_I_z 0.00e+00, real_p_hat 0.00e+00; win64_seed0_reference: real_I_s 0.00e+00, real_I_z 0.00e+00, real_p_hat 0.00e+00; win64_seed5_experimental: real_I_s 0.00e+00, real_I_z 0.00e+00, real_p_hat 0.00e+00; win64_seed5_reference: real_I_s 0.00e+00, real_I_z 0.00e+00, real_p_hat 0.00e+00

## 3. Thresholds (two pairs: seeds 0 and 5, different block classes -- an sd over n = 2, one degree of freedom)

| zone | values over the two pairs | threshold |
|---|---|---|
| uncontrollable_deterministic | off-support (environment): 0.0472, 0.0596 | tau_off_floor = 0.0175 (implementer's reading, to be confirmed) |
| controllable_noisy | b(I_s) - b(I_s projected): +0.0160, +0.0018 | tau_b = 0.0200 |
| controllable_noisy | off-support env - ref: +0.0124, +0.0015 | tau_off = 0.0154 |
| uncontrollable_noisy | off-support (environment): 0.1573, 0.1421 | tau_off_floor = 0.0214 (implementer's reading, to be confirmed) |
| controllable_deterministic | b(I_s) - b(I_s projected): -0.0121, +0.0080 | tau_b = 0.0285 |
| controllable_deterministic | off-support env - ref: -0.0015, +0.0009 | tau_off = 0.0035 |

## 4. Verdict of the rule on gpu_e30b

Applied to the same two pairs that set the thresholds (see DECISIONS 2026-10-01 (10) on what this verdict can mean).

| seed | zone | b (I_s) | b (I_s projected) | off-support env / ref | verdict |
|---|---|---|---|---|---|
| 0 | uncontrollable_deterministic | -2.2394 | -2.2432 | 0.0472 / 0.0085 | read against the floor |
| 0 | controllable_noisy | +0.0084 | -0.0075 | 0.0498 / 0.0374 | interpretable |
| 0 | uncontrollable_noisy | -1.7921 | -1.7777 | 0.1573 / 0.0457 | read against the floor |
| 0 | controllable_deterministic | -0.0028 | +0.0093 | 0.0048 / 0.0063 | interpretable |
| 5 | uncontrollable_deterministic | -2.2681 | -2.2689 | 0.0596 / 0.0051 | read against the floor |
| 5 | controllable_noisy | -0.0040 | -0.0058 | 0.0431 / 0.0416 | interpretable |
| 5 | uncontrollable_noisy | -1.7504 | -1.7375 | 0.1421 / 0.0527 | read against the floor |
| 5 | controllable_deterministic | -0.0090 | -0.0170 | 0.0079 / 0.0070 | interpretable |
