# Learning Guide: FrozenLake Deep Q-Learning

This project teaches how a DQN (Deep Q-Network) agent learns to navigate Gymnasium's **FrozenLake** environment.

## FrozenLake in simple terms

- The agent starts at the **S** tile.
- **F** tiles are safe frozen ground.
- **H** tiles are holes.
- **G** is the goal.
- The agent can move **left, down, right, or up**.
- A normal lake moves exactly as commanded.
- A slippery lake can move sideways because the ice is stochastic.
- A successful episode reaches the goal and gets reward `1`.

## Main DQN ideas

### 1. State

A state is the current tile number. The environment has 16 states on a 4x4 lake and 64 states on an 8x8 lake.

### 2. Action

There are four actions:

- `0`: Left
- `1`: Down
- `2`: Right
- `3`: Up

### 3. Reward

The reward tells the agent whether its latest action was useful:

- Goal: positive reward.
- Hole: usually zero or a negative shaped reward.
- Frozen tile: usually zero or a small negative step penalty.

### 4. Q-value

A Q-value estimates how much future reward the agent can receive after choosing an action from a state.

The network receives a state and predicts a Q-value for every action:

```text
State -> Neural network -> [Q(left), Q(down), Q(right), Q(up)]
```

The largest Q-value suggests the best action.

### 5. One-hot encoding

Each tile number is converted into a vector containing mostly zeros and one `1`.

Example:

```text
State 3 -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

This stops tile numbers from being interpreted as continuous numerical values.

### 6. Experience replay

Every move creates an experience:

```text
(state, action, reward, next_state, done)
```

The replay buffer stores these experiences and randomly samples batches for training. Random sampling makes learning less dependent on one sequence of moves.

### 7. Epsilon-greedy exploration

At the beginning, epsilon is high and the agent explores randomly.

- Random action: probability `epsilon`
- Best predicted action: probability `1 - epsilon`

Epsilon slowly decreases so the trained agent eventually relies more on learned knowledge.

### 8. Target network

The policy network learns continuously. The target network provides more stable target values and is copied from the policy network periodically.

This prevents the target values and the network's predictions from changing at exactly the same time.

### 9. Bellman update

DQN tries to move the predicted Q-value toward this target:

```text
reward + gamma * max(next-state Q-values)
```

For a finished episode:

```text
reward
```

For a normal step:

```text
reward + gamma * max(next-state Q-values)
```

The parameter `gamma` is a discount factor that decides how much future rewards matter.

## Notebook progression

### Exercise 1: Deterministic vs. slippery ice

[exercise-1.ipynb](notebooks_easy/exercise-1.ipynb) compares:

- `is_slippery=False`: the intended action always happens.
- `is_slippery=True`: ice can cause unexpected sideways movement.

Key components:

- DQN neural network.
- Policy and target networks.
- Replay buffer.
- Epsilon-greedy actions.
- 50-episode success curves.
- 4x4 learned Q-value heatmap.

Main lessons:

- Slippery ice needs a safer policy.
- A tile near a hole may require avoiding the apparently shortest route.
- Success is less guaranteed, so Q-values depend on probability and path length.
- Reaching the goal in fewer steps does not guarantee it is easier on slippery ice.

### Exercise 2: Target network stabilization

[exercise-2.ipynb](notebooks_easy/exercise-2.ipynb) compares training:

1. With a target network.
2. Without a target network.

The agent updates the target network every 20 episodes.

Main lesson:

- A frozen or periodically updated target creates stable learning targets.
- Updating every target from the same continuously changing network creates a moving-target problem.
- This can cause oscillating Q-values, exploding losses, and catastrophic forgetting.
- The target-network version learns much more reliably.

### Exercise 3: Reward shaping and side effects

[exercise-3.ipynb](notebooks_easy/exercise-3.ipynb) changes the reward for ordinary moves and holes.

The mild setting is:

```text
Normal frozen tile: -0.01
Hole: -0.2
Goal: +1
```

The harsh setting is:

```text
Normal frozen tile: -0.25
Hole: -0.1
Goal: +1
```

Main lessons:

- The mild reward setting allowed learning a successful policy.
- An overly harsh step penalty makes living expensive.
- The agent may prefer a short path into a hole rather than a longer successful path.
- Reward shaping changes the behavior you actually teach the agent.
- Rewards must support the true goal rather than accidentally rewarding shortcuts or bad behavior.

### Exercise 4: Standard DQN vs. Double DQN

[exercise-4.ipynb](notebooks_easy/exercise-4.ipynb) compares standard DQN with Double DQN.

Standard DQN:

1. Uses the target network to find the maximum next-state Q-value.
2. Uses that selected action and value together.

Double DQN:

1. Uses the policy network to select the best next action.
2. Uses the target network to evaluate that selected action.

The notebook trains both agents on slippery 4x4 ice.

Main lesson:

- Taking a maximum from noisy estimates can create systematic overestimation.
- Double DQN separates action selection from action evaluation.
- DDQN usually produces lower, more realistic Q-values and cleaner behavior.

### Exercise 5: Scaling to 8x8 FrozenLake

[exercise-5.ipynb](notebooks_easy/exercise-5.ipynb) changes the lake to 8x8.

The 8x8 lake contains:

- 64 states instead of 16.
- 10 holes.
- A goal farther from the start.

It also changes:

- Replay-buffer capacity.
- Number of episodes.
- Epsilon decay.
- Heatmap size.

Main lessons:

- The network and display code must use the actual number of states.
- `sqrt(num_states)` can calculate the grid size for a square map.
- A larger map needs more exploration and more training episodes.
- Slower epsilon decay gives the agent more chances to discover the goal.
- The tuned settings achieved at least 90% rolling success, with a reported 100% final 50-episode success rate.

## Completed notebook

[frozen_lake_dqn_colab_completed.ipynb](notebooks_easy/frozen_lake_dqn_colab_completed.ipynb) combines the complete basic DQN implementation and then repeats the exercises.

Its parts are:

1. **Dependency setup:** Installs Gymnasium for Colab.
2. **Hyperparameters:** Configures learning rate, gamma, epsilon, replay size, and episode count.
3. **One-hot encoding:** Converts discrete tile indices into vectors.
4. **DQN model:** Uses 16 inputs, two 64-unit hidden layers, and 4 outputs.
5. **Replay buffer:** Stores and randomly samples past experiences.
6. **Agent:** Implements epsilon-greedy selection and Bellman updates.
7. **Training loop:** Runs episodes, trains the network, decays epsilon, and updates the target network.
8. **Visualization:** Displays success, epsilon, Q-values, and greedy policy arrows.
9. **Greedy evaluation:** Uses the trained policy without random exploration.
10. **Exercises:** Covers slippery physics, target networks, reward shaping, Double DQN, and 8x8 scaling.

## Standard training flow

```text
1. Create FrozenLake environment
2. Create policy network and target network
3. Create replay buffer
4. Reset environment
5. Choose an epsilon-greedy action
6. Execute the action
7. Store the experience
8. Sample a random batch
9. Calculate current Q-values
10. Calculate Bellman target Q-values
11. Minimize the TD loss
12. Repeat until the episode finishes
13. Decay epsilon
14. Periodically copy policy weights to target network
15. Repeat for every episode
16. Display graphs and heatmaps
17. Run a greedy evaluation
```

## Heatmap interpretation

- **Lighter or higher-value tiles:** The agent predicts higher future rewards from that tile.
- **Darker or lower-value tiles:** The agent predicts lower future rewards.
- **Arrow:** The greedily selected best action for that tile.
- **H:** Hole.
- **G:** Goal.
- **S:** Starting tile.

An arrow means the action with the largest predicted Q-value at that state, not a guaranteed move in slippery mode.

## 50-episode rolling success

The rolling success rate is the percentage of successful episodes across the latest 50 episodes.

Example:

```text
35 successful episodes out of 50 -> 70% rolling success
```

It smooths one-episode noise and makes learning progress easier to understand.

## Important hyperparameters

### Learning rate

Controls how strongly the network updates each time.

- Too high: learning may oscillate or fail.
- Too low: learning may be extremely slow.

### Gamma

Controls the importance of future rewards.

- `0.99` makes distant rewards matter strongly.
- Lower values make immediate rewards more important.

### Epsilon

Controls exploration.

- High epsilon: more random actions.
- Low epsilon: more actions chosen from learned Q-values.

### Batch size

The number of stored experiences used in one learning update.

### Target update frequency

The number of episodes before the target network copies the policy network.

### Replay capacity

How many recent experiences are kept in memory.

## Common mistakes

- Not collecting enough random experiences before updating.
- Updating the policy without a sufficiently stable target.
- Giving normal moves too harsh a penalty.
- Decaying epsilon too quickly in a large map.
- Forgetting to mark terminal states correctly.
- Using tile numbers directly instead of one-hot encoding.
- Using hardcoded 4x4 assumptions on an 8x8 map.
- Comparing raw episode rewards instead of rolling success rates.
- Evaluating with epsilon enabled when a pure learned-policy test is desired.

## Final summary

The project demonstrates that DQN learning depends on all of these working together:

- The environment.
- State representation.
- Exploration.
- Replay.
- Reward design.
- Stable targets.
- Double-DQN action/value separation.
- Enough episodes and exploration.
- Clear result visualization.
