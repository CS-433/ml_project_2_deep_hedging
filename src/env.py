import gym
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gym import spaces


class StockTradingEnv(gym.Env):
    """Environment for agent, consists of __init__, step, and reset functions"""

    def __init__(self, reset_path=False, data_type="mixed"):
        self.asset_price = pd.read_csv(f"data/asset_price_{data_type}_1_sim.csv").values
        self.option_price = pd.read_csv(
            f"data/option_price_{data_type}_1_sim.csv"
        ).values
        self.nPaths = self.option_price.shape[0]
        self.nSteps = self.option_price.shape[1]

        # user-defined options (path)
        self.reset_path = reset_path
        self.path_choice = int(random.uniform(0, self.nPaths))
        self.path_idx = self.path_choice

        # initialize price memory for normalization
        self.price_memory = []
        self.price_stat = []
        self.window_len = 200

        # transaction cost (for rewards)
        self.kappa = 0.0001

        # initializing underlying amount
        self.holdings = 0
        self.curr_step = 0

        # Actions of the format hold amount [0,1]
        self.action_space = spaces.Box(low=-1, high=100, dtype=np.float16)

        # agent is given previous action + current asset price and time to maturity (H_i-1, S_i, tau_i)
        self.observation_space = spaces.Box(
            low=np.array([-1, 0, 0]),
            high=np.array([1, np.inf, self.nSteps]),
            shape=(3,),
            dtype=np.float16,
        )

    def step(self, action: float):
        # Execute one time step within the environment
        self.curr_step += 1

        # next call price, call price now, next asset price, asset price now.
        c_next, c_now, s_next, s_now = (
            self.option_price[self.path_idx, self.curr_step],
            self.option_price[self.path_idx, self.curr_step - 1],
            self.asset_price[self.path_idx, self.curr_step],
            self.asset_price[self.path_idx, self.curr_step - 1],
        )

        # R_{t} is Acc PnL
        reward = self.holdings * (s_next - s_now) - self.kappa * s_next * np.abs(
            action - self.holdings
        )

        # A_{t}: update the holding info.
        self.holdings = action

        # S_{t+1}: previous action, current asset price and time to maturity (H_i-1, S_i, tau_i)
        next_state = [
            self.holdings.item(),
            s_next,
            self.nSteps - self.curr_step,
        ]
        # done: whether the episode is ended or not
        done = True if self.curr_step + 1 >= self.nSteps else False

        # if terminal subtract option price difference, assumed next option price is just a call payoff and cost for exiting delta hedge position
        if done:
            reward = (
                reward
                - (max(s_next - 100, 0) - c_now) * 100
                - self.kappa * s_next * self.holdings
            )
        else:  # if not terminal, substract option price difference.
            reward = reward - (c_next - c_now) * 100
        return next_state, reward, done

    def reset(self):
        if self.reset_path:  # if user chose True, sets to his choice, else, random
            self.path_idx = self.path_choice

        # when resetting the env, set current_step and previous holdings equal to 0.
        self.curr_step = 0
        self.holdings = 0
        return [
            self.holdings,
            self.asset_price[self.path_idx, self.curr_step],
            self.nSteps,
        ]  # state0 of new path

    def normalize(self, state):
        # store price data first
        self.price_memory.append(state[1])

        if len(self.price_memory) == 1:
            mu_, std_ = 100, 1
        else:
            mu_, std_ = np.mean(self.price_memory[-self.window_len :]), np.std(
                self.price_memory[-self.window_len :]
            )

        self.price_stat = [mu_, std_]
        norm_price = (state[1] - mu_) / std_
        norm_tau = state[2] / 60
        norm_action = state[0] / 100
        return [norm_action, norm_price, norm_tau]
