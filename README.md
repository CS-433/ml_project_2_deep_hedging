# EPFL CS-433: Project 2 "Deep hedging" (reproducibility challenge)


This is an implementation of the Deep Deterministic Policy Gradient reinforcement learning algorithm to perform hedging for a call option, as described in the underlying paper *Deep Hedging of Derivatives Using Reinforcement Learning* (Jay Cao, Jacky Chen, John Hull, Zissis Poulos, 2019). The code mainly uses PyTorch and NumPy. All required librairies are specified in `requirements.pip`. 



## Code description

### `model`

A folder containing different versions of the DDPG implementation (denoted by v1, v2, etc...) defined by the weights of the neural-networks and a text file quickly describing the changes for each version.

---

### `src`

A folder with code for the implementation of DDPG. Refer to the README file inside the folder for more detailed descriptions.

---

### `main.py`

Use `python main.py` or `python3 main.py` to run the training of the agent and save obtained parameters (neural network weights) in a `models/model_name` folder (careful, this takes a long time to run).

---

### `hyperparam_tuning.py`

Hyperparameter tuning using optuna. 

---

### `requirements.pip`

Use `pip install requirements` in your terminal to install the necessary librairies for running code in this repository .

---
## Authors

- Kim Ki Beom

- Marcell Jordan

- Alexei Ermochkine
