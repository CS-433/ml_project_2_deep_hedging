---
### `agent.py`

This file contains DDPG agent defined as a class. It contains methods that takes actions given state, update the neural network from stored experiences and resetting, storing the data in experience replay buffer.
---

### `buffer.py`

Defines experience replay buffer which is the part of our DDPG agent. Currently, we have only implemented vanilla experience replay buffer.

---

### `env.py`

The environment class implemented with openAI's `gym` framework.

---

### `network.py`

Definitions of the neural networks that represent the agent.

---

### `simulation.py`

All functions necessary for the generation and handling of data. See **notebook/simulation.ipynb** for implementations of these functions.

---
