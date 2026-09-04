Scientific context

Computational neuroscience project. Broad goal: train a world model to act in an environment, analyse its trajectories in a healthy environment, and extract quantifiable behavioural phenotypes. Eventually, manipulate the world model's latent to identify what causes those behavioural alterations.

First axis (the current code): a gridworld with tunable controllability. The target phenotype is learned helplessness — an agent trained in a poorly controllable world (high p) retains passive or miscalibrated behaviour when tested in a perfectly controllable one (p = 0).

Conceptual reference: Maier & Seligman (2016). Passivity is the default; what is learned is the detection of control. Hence the interest in a model-based agent, which structurally separates the transition belief T̂ from the reward belief R̂.

Protocol structure

Phase 1 — training. N agents per value of p, one seed per agent. Everything is identical across conditions except p: same transition budget, same exploration policy, same planner.

Phase 2 — testing in the healthy environment. All agents are evaluated at p = 0, with learning disabled and the planner frozen. Phenotypes are measured here.

Phase 3 — aetiology. Component swaps between agents (T̂ from one with R̂ from another, true T with learned R̂, and so on) to localise the origin of the phenotype.

Invariants that must never be broken
GridWorld stays Markovian in (position, explicit augmented state). Any mechanic that makes reward depend on history must either be removed or be reflected in the state space. No silent exceptions.
transition_tensor() and reward_vector() are the ground truth. They must stay exactly consistent with step(). Any change to the dynamics must be propagated to both, and verified by an empirical agreement test.
terminated and truncated are never conflated. The MDP is infinite-horizon discounted. The step limit is not part of the MDP: on truncation we bootstrap on V(s') rather than treating the state as terminal. Conflating the two would teach the agent that the world ends at max_steps, producing an artefact indistinguishable from the phenotype under study.
RNGs are kept separate. One for environment dynamics, one for agent exploration, one for initialisation. It must be possible to hold one fixed while varying another.
Metrics apply to an arbitrary tensor, not just to the environment. Every measure (controllability, value, etc.) must be a free function taking P as an argument, so it can be computed on the ground truth and on an agent's learned T̂ alike. This is the core of the analysis.
Code conventions
Python 3.10+, numpy. No heavy dependencies in phase 1.
Gymnasium-compatible API without a mandatory gymnasium import.
Type hints on public signatures.
The existing code uses short French docstrings and unaccented ASCII comments. Keep that style for consistency unless the whole file is converted at once.
No external RL framework for now. The tabular agent must stay readable line by line.
Analysis functions must have no side effects on the environment.
Design choices already settled
Parameterisation of p: with probability p, the action is drawn uniformly over the 4 directions. The probability of executing the intended action is (1-p) + p/4. Controllability decreases linearly and reaches exactly zero at p = 1.
Central metric: mutual information I(a ; s' | s) in bits, under a uniform prior over actions, averaged over non-terminal states. The actual ceiling on the default layout is ~1.78 bits (not 2), because of walls and borders where several actions lead to the same cell. This is the normalisation reference.
Default layout: two paths of length 8 to the goal, one running alongside a trap. The detour costs nothing in distance, which cleanly isolates risk aversion from movement cost. Do not modify without a reason.
Budgets are counted in transitions, never in episodes. Episode length varies strongly with p; counting in episodes would introduce a direct confound on data volume.
Phenotypes measured in phase 2

Instrumental: return suboptimality vs the oracle, trajectory length relative to the shortest path, latency to first reward.

Behavioural: policy entropy, perseveration (revisits, in-place oscillations), mean distance maintained from traps.

Belief: perceived controllability I(a;s'|s) computed under T̂, and its gap to the value under the true T. Value gap max_a Q - mean_a Q.

Recovery: with learning re-enabled at p = 0, how fast the agent returns to normal performance. This is what discriminates a merely biased prior from a genuine learning deficit.

Interpretation caveat

An agent trained at moderate p that gives traps a wide berth is not "helpless": it is applying a policy that was optimal in its training world. That is maladaptive transfer, not helplessness.

The helplessness regime is the one where differentiation between actions collapses. Sampling p towards the high end (0.85, 0.95) is therefore required to reach it. p = 1 is the negative control: all policies are equivalent there.

Roadmap
Step 1 (current): tabular, single grid, sweep over p, baseline phenotypes, component swaps.
Step 2: distribution of worlds (varied p, layouts and trap positions), recurrent neural world model over symbolic states. The goal is for a controllability axis to emerge in the latent and be decodable.
Step 3: causal manipulation of the latent, steering along that axis, verifying that the induced behaviour reproduces the step 1 phenotype.

Gating criterion: do not move to the next step until the previous one produces clean results. Otherwise two problems get debugged at once.

Note: a successful linear decode from the latent does not prove the information is used. Only intervention proves that.

What not to do
Do not add a game mechanic without being able to state precisely what it contributes to measuring controllability. The richer the game, the more the phenotype drowns in noise.
Do not replace the tabular model with a network while T̂ still fits in a readable array. That would trade away interpretability for nothing.
Do not vary the planner across agents. Otherwise behavioural differences are no longer attributable to the world model.