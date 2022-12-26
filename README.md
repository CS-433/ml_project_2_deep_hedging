# EPFL CS-433: Project 2 "Deep hedging" (reproducibility challenge)

This is an implementation of the Deep Deterministic Policy Gradient reinforcement learning algorithm to perform hedging for a call option, as described in the underlying paper _Deep Hedging of Derivatives Using Reinforcement Learning_ (Jay Cao, Jacky Chen, John Hull, Zissis Poulos, 2019). All librairies are specified under `requirements.pip` below.

## Getting Started

This is an example of how you may give instructions on setting up your project locally. To get a local copy up and running follow these simple example steps.

### Setting up the virtual environment

Before running our repository, you need to set up the virtual environment first and run the `requirements.pip` to install necessary packages. Below example is based on python, you can also use conda environment.

```
python -m venv venv
(venv) pip install -r requirements.pip
```

### Data generation

After setting up the virtual environment, we need to download training data as current repository does not contain these. First, you need to make `data` folder and run the `notebook/dataGen.ipynb`.

In `notebook/dataGen.ipynb`, you can change the parameters of stock/option data generating process. Especially, you will need to change `T`, `freq` parameters to run our code. Our code supports `T = {1,3}`, `freq = {1,2,5}` cases. Once you set up these parameters and run the notebook, it will automatically save csv files for you.

### Performance Evaluation

After obtainig train/test data from simulation, you can run `performance_evaluation.ipynb` to check our pretrained agent's hedgin performance. 


## Repository Structure

```
├── README.md
├── main.py
├── requirements.pip
├── data # contains train/test data for every maturity and frequency
│   ├── 1month
│   │   ├── 1d
│   │   │   ├── asset_price_GBM_sim.csv
│   │   │   ├── asset_price_mixed_sim.csv
│   │   │   ├── asset_price_price_sim.csv
│   │   │   ├── option_price_GBM_sim.csv
│   │   │   ├── option_price_mixed_sim.csv
│   │   │   └── option_price_SABR_sim.csv
│   │   ├── 2d
│   │   └── 5d
│   └── 3month
│       └── ...
│── model # pretrained model parameters
│   ├── v1
│   ├── ...
│   ├── v8
│   │   ├── actor_weight.pt
│   │   ├── critic_1_weight.csv
│   │   └── critic_2_weight.csv
│   ├── hypparams.json
│   └── report.txt
│
│── notebooks # contains notebooks to reproduce results and generate data
│   ├── dataGen.ipynb
│   ├── simulation.ipynb
│   ├── performance_evaluation.ipynb
│   └── README.md
│
└── src # DDPG agent, StockTradingEnv source code
    ├── README.md
    ├── agent.py
    ├── buffer.py
    ├── env.py
    ├── hyperparam_tuning.py
    ├── network.py
    └── simulation.py
```

## Code description

### `requirements.pip`

Use `pip install requirements` in your terminal to install the necessary librairies for running code in this repository. The following librairies are used:

- black
- gym
- gym[classic_control]
- ipykernel
- matplotlib
- numpy
- optuna
- pandas
- pyglet (1.5.27)
- python-dateutil (2.8.2)
- pytz (2021.1)
- scipy
- scikit-learn
- statsmodels
- torch
- tqdm

---

### `model`

A folder containing different versions of the DDPG implementation (denoted by v1, v2, etc...) defined by the weights of the neural-networks and a text file quickly describing the changes for each version.

---

### `notebook`

A folder containing ipython notebooks, which can be executed instead of running .py file using command lines. If you are new to our repository, and would like to see the quick results, we advise you to run .ipynb scripts in this folder. We have scripts that generate the training data and saves in the `Data` folder and the assessment of the agent by making a comparison with classic delta hedging.

---

### `src`

A folder with code for the implementation of DDPG. Refer to the README file inside the folder for more detailed descriptions.

---

### `main.py`

Use `python main.py` or `python3 main.py` to run the training of the agent and save obtained parameters (neural network weights) in a `models/model_name` folder (careful, this takes a long time to run).

---

## Authors

- Ki Beom Kim

- Marcell Jordan

- Alexei Ermochkine
