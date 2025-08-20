# 📘 Monte Carlo Control with Constant-Alpha — Detailed Explanation and Implementation

This README explains a detailed walkthrough of implementing **Monte Carlo Control** with constant-α learning to estimate the optimal policy and action-value function (`Q`) in an OpenAI Gym environment (e.g., Blackjack).

---

## 🎯 Objective

We want to solve a Reinforcement Learning problem using **Monte Carlo Control**. Specifically, we want to:

1. Learn the **action-value function** `Q(s, a)` — how good an action is in a state.
2. Derive the **optimal policy** `π(s)` — the best action to take in any given state.

This is done **without knowledge of environment dynamics**. Instead, the agent learns purely from experience.

---

## 💡 Key Concepts

### 🔁 Monte Carlo (MC) Control
- **Monte Carlo** methods learn from *complete episodes* (from start to terminal state).
- MC Control finds optimal policies by improving estimates of `Q(s,a)` through simulated episodes.
- We use **First-Visit** MC: update `Q` only the first time `(s, a)` appears in an episode.

### ⚙️ Constant-α Update
Instead of storing every return and computing an average, we use:
```python
Q[s][a] += alpha * (G - Q[s][a])
```
This update converges faster and is memory-efficient, suitable for large or continuous environments.

### 🎲 Epsilon-Greedy Policy
We balance **exploration** (trying new actions) and **exploitation** (choosing best-known actions) using:
- Probability `ε` to explore.
- Probability `1-ε` to choose the best action.

---

## 🧱 Code Structure

### 1. `epsilon_greedy_policy()`

This function builds the ε-greedy policy distribution for action selection.

```python
def epsilon_greedy_policy(action_values, epsilon=0.1):
    nA = len(action_values)
    policy = np.ones(nA) * epsilon / nA
    best_action = np.argmax(action_values)
    policy[best_action] += (1.0 - epsilon)
    return policy
```

---

### 2. `mc_control()`

This is the main function where:
- Episodes are simulated.
- Q-values are updated.
- Final policy is extracted.

```python
from collections import defaultdict
import numpy as np
import sys

def mc_control(env, num_episodes, alpha, gamma=1.0):
    nA = env.action_space.n
    Q = defaultdict(lambda: np.zeros(nA))  # Q[state][action]

    for i_episode in range(1, num_episodes + 1):
        if i_episode % 10000 == 0:
            print(f"Episode {i_episode}/{num_episodes}.", end="")
            sys.stdout.flush()

        episode = []
        state = env.reset()
        done = False

        while not done:
            probs = epsilon_greedy_policy(Q[state], epsilon=0.1)
            action = np.random.choice(np.arange(nA), p=probs)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state

        G = 0
        visited = set()
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = gamma * G + reward
            if (state, action) not in visited:
                visited.add((state, action))
                Q[state][action] += alpha * (G - Q[state][action])

    policy = {state: np.argmax(actions) for state, actions in Q.items()}
    return policy, Q
```

---

### 3. Run and Visualize

To estimate the optimal policy:

```python
policy, Q = mc_control(env, num_episodes=500000, alpha=0.02)
```

Then compute state-value function from Q:

```python
V = {state: np.max(actions) for state, actions in Q.items()}
```

Plot results using helper functions:

```python
plot_blackjack_values(V)
plot_policy(policy)
```

---

## 📊 Output

- `Q[state][action]`: Estimated return for state-action pairs.
- `policy[state]`: Optimal action in each state under ε-greedy learning.
- `V[state]`: Maximum value of actions from each state.

---

## 📚 Why This Approach?

| Technique | Why We Use It |
|----------|----------------|
| Constant-α | Faster convergence without storing every return |
| ε-greedy | Balances exploration and exploitation |
| First-Visit MC | Simpler and avoids repeated updates in one episode |
| `defaultdict` | Avoids manual initialization of Q-table |

---

## ✅ Benefits

- Does not require knowledge of transition probabilities.
- Works well for episodic tasks (e.g., Blackjack).
- Easy to implement and extend to control algorithms like SARSA, Q-learning.

---

## 🧠 Notes

- Always use a large number of episodes for convergence (e.g., `500,000`).
- Tuning `α`, `γ`, and `ε` affects convergence rate.
- For continuing tasks (non-episodic), use Temporal-Difference (TD) methods instead.

---

## 🔗 References

- Sutton & Barto, *Reinforcement Learning: An Introduction*
- OpenAI Gym: [https://gym.openai.com](https://gym.openai.com)
