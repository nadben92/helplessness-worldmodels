# Setup visuals (p1-b-d1)

Scripts: `make_trajectory.py` (GIF + JSON), `make_pseudocode.py` (PNG + SVG). Config values: `configs/e4/e4.yaml`.

## Environment (12x12 torus, goal (0,0), `envs/layouts/two_zones.py`, diagonal 1)
- Corridor: rows 0 and 6, columns 0 and 6 (neutral, controllable-deterministic). Goal: (0,0) (yellow, terminal).
- UD (blue (22,0,244) in the environment config, swapped on the slide: UD drawn red, CD drawn blue, slot nw): rows 1-5, cols 1-5; conveyor drift (+1,0) = down (away from the goal), whatever the action.
- CD (red (227,17,15), slot se): rows 7-11, cols 7-11; controllable-deterministic.
- ne and sw: corridor-like grey (128,128,128), controllable-deterministic, not visited. Agent: brown (100,45,0).
- Dynamics and images come from `e4_setup.world(cfg, conf, "experimental")` (env.step, `templates_np`); nothing re-implemented.

## Trajectory (38 images, 37 steps, 10.25 s: 250 ms/image, 1 s on the last; `E4_trajectory.json`)
Start (6,11), corridor. Action sequence (up/right/down/left):
- steps 1-2: left, down -> enters CD at step 2 (image 2).
- steps 3-7 (CD): right, down, down, left, up -> agent follows the actions: (7,11),(8,11),(9,11),(9,10),(8,10).
- steps 8-9: up, up -> back in the corridor at step 9 (image 9).
- steps 10-23: left x4, up x6, left x3 along the corridor to (0,3); step 24: down -> enters UD at step 23's image (image 23, cell (1,3)).
- steps 24-28 (UD): the SAME five actions right, down, down, left, up -> the agent slides down (2,3)...(6,3) regardless; leaves UD at image 28.
- steps 29-37: left x3, up x6 -> goal (0,0), terminated.
Region per image is in the JSON (`regions`).

## Pseudo-code line -> code
| Block / line | File: function |
|---|---|
| 1 filtered posterior h, z | `agents/worldmodel/policy.py`: `LatentFilter` (`rssm.observation_step`) |
| 1 lookahead h'_a, z'_k (K=16) | `policy.py`: `lookahead` (`rssm.imagination_step`); `k_act` |
| 1 Q(a) = r̂ + γ c V | `policy.py`: `q_values`; `heads.py`: `reward_from_goal`, `continuation_from_goal` |
| 1 ε-greedy, random ties | `policy.py`: `act`, `tie_set`, `tie_tolerance` |
| 1 env.step + replay | `agents/worldmodel/closed_loop.py`: collector `step`; `replay.py` |
| 2 loop 0.6 update/step, batch 16x64 | `pipeline/agent_run.py`: `run_one` (credit += replay_ratio) |
| 2 GRU / encoder / posterior / prior / decoder | `agents/worldmodel/rssm.py`: `RSSM`, `Encoder`, `Decoder`; `loop.py`: `unroll_observation` |
| 2 weighted reconstruction w = 1 + 9 M | `losses.py`: `reconstruction_mask`, `weighted_reconstruction_loss` |
| 2 KL dyn / rep | `losses.py`: `dyn_rep_kl` |
| 2 probabilities -> straight-through, lr 1e-3 -> 1e-4 | `loop.py`: `SeparationSwitch`, `phase_learning_rate` |
| 3 goal head on sg(h, z), r̂ = 50p - (1-p), c = 1-p | `imagination.py`: `goal_head_loss`, `train_heads_step`; `heads.py` |
| 3 imagination H=15, ε-greedy K=4 | `imagination.py`: `imagine` |
| 3 λ-returns, weights w_k | `imagination.py`: `lambda_returns`, `critic_loss` |
| 3 Adam lr 3e-5, slow critic EMA 0.02 | `train_heads_step`, `heads.update_slow_critic` |

## Simplifications in the pseudo-code
- Tie tolerance in the greedy choice shown as "ties drawn at random" (the real set is Q >= max Q - 0.0751).
- Unimix (0.01) on the categorical probabilities, layer norms, the mask's dilation, prefill (256 steps) omitted.
- Latent: a 8x8 categorical (8 variables x 8 classes); KL summed over variables; reconstruction summed per image, mean over the batch (not written).
- Critic and goal head start only after the separation switch (stated in the header of block 3); before it, only the world model trains.
- Goal head: BCE on 1[r>0] (loss not written); gradient clipping (critic, 100) omitted; z in imagination is detached.
- "o' , r" shown as one environment step; episode resets (max 36 steps, uniform non-terminal starts) and truncation bootstrap omitted.
- Block 2 shows the unroll as a per-step loop; in the code it is a scan over the 64-step window with is_first resets.
- Notation: ĥ r̂ = reward read from p(goal), c = continuation = 1 - p(goal); V_lent = slow critic.
