---
### `agent.py`

This file contains DDPG agent defined as a class. It contains methods that takes actions given state, update the neural network from stored experiences and resetting, storing the data in experience replay buffer.
---

### `buffer.py`

Defines experience replay buffer which is the part of our DDPG agent. Currently, we have only implemented vanilla experience replay buffer.

---

### `env.py`

The environment `StockTradingEnv` class is implemented with openAI's `gym` framework. This will be used across all stages of training process, assessment. We only implemented Accounting P&L scheme as a reward function.

---

### `hyperparam_tuning.py`

This script performs hyperparameter tuning of DDPG agent. Currently, it optimizes actor, critic learning rate, number of agent update for each episode. Hyperparameter tuning is done using optuna and hyperparameters that gives smalles objective function is saved automatically in `model/hypparams.json` which is used in the trainig of the agent.

---

### `network.py`

Definitions of the neural networks that represent the agent. We only used Multi-Layer Perceptron in this project. MLP contains two Linear hidden layers with Layernorm and ReLU activation. User can choose specific activation function at end, we currently support `Tanh, Sigmoid, ReLU`.

---

### `simulation.py`

All functions necessary for the generation and handling of data. See **notebook/simulation.ipynb** for implementations of these functions.

---
