# PPO-Cutting-Stock-2D

Deep Reinforcement Learning for **2D Cutting Stock** using **Proximal Policy Optimization (PPO)**.

This project explores how an agent can learn to place rectangular parts onto a sheet (stock material) to reduce waste / maximize utilization. Instead of relying on hand-designed heuristics, the agent is trained via reinforcement learning to make a sequence of placement decisions.

---

## Problem: 2D Cutting Stock (Rectangular Nesting)

In the 2D cutting stock problem, you are given:

- A fixed-size sheet (or a sequence of sheets) with width/height constraints
- A set of rectangular items to place (often with quantities, and sometimes allowing rotation)
- The objective to minimize waste, minimize the number of sheets used, or maximize filled area

The environment is **sequential**: each action places one item (or selects an item and a position), which changes the remaining free space for future placements.

---

## RL Formulation

### State / Observation
The observation typically encodes the current cutting context, for example:

- representation of the stock sheet(s) and occupied/free regions
- the remaining pieces (sizes and counts)
- optional: current step index, remaining area, or other summary features

(Exact state encoding depends on the implementation details in this repository.)

### Action Space
Actions usually represent a placement decision, such as:

- selecting which item to place next
- selecting a location (x, y) on the sheet
- optionally selecting rotation (0/90 degrees)

Some implementations use a restricted action set (e.g., candidate placement points from a skyline/guillotine structure) to make learning tractable.

### Reward
Reward is shaped to encourage efficient packing and feasible placements, commonly using signals like:

- positive reward proportional to placed area
- penalties for invalid/overlapping/out-of-bounds placements
- end-of-episode reward tied to utilization / number of sheets used / leftover waste

Good reward design is crucial: PPO will optimize whatever signal you give it, so the reward must align with the packing objective.

### Episode Termination
Episodes generally end when:

- all required items are placed, or
- no valid placements remain / max steps reached

---

## Project Structure (Typical)

While this repo is Python-only, most PPO cutting-stock projects are structured around:

- **Environment**: implements reset/step and the packing logic
- **Policy / Value Networks**: neural nets that map observations to action distributions and state values
- **Training Loop**: collects rollouts, computes returns/advantages, and performs PPO updates
- **Evaluation / Rendering**: measures utilization and optionally visualizes placements

If you’re new to this domain, start by locating:
- the environment class (often Gym-like),
- the PPO training script,
- and any config/hyperparameter definitions.

---

## Key PPO Concepts Used Here

### Clipped Policy Objective
PPO uses a ratio between new and old action probabilities and clips it to limit the update size:

- prevents destructive policy updates
- improves stability for long-horizon sequential tasks like packing

### Value Function Baseline
A separate network (or head) estimates the state value:

- reduces variance of policy-gradient updates
- enables advantage estimation (often GAE)

### Generalized Advantage Estimation (GAE)
GAE trades bias vs variance to produce smoother advantage estimates, which helps learning stability in environments where reward is delayed (common in packing).

---

## Metrics to Track

When training PPO for 2D cutting stock, common metrics include:

- **Utilization / Fill ratio** (placed area / sheet area)
- **Number of sheets used**
- **Invalid action rate**
- **Episode length** (steps to completion)
- PPO training metrics: policy loss, value loss, entropy, KL divergence
