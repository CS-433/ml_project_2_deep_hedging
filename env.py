import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np
from simulation import *
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm

asset_price = pd.read_csv('sim_data/asset_price_sim.csv')
option_price = pd.read_csv('sim_data/option_price_sim.csv')
asset_price = asset_price.to_numpy()
option_price = option_price.to_numpy()
option_price = -option_price
N, P = np.shape(option_price)
MAX_STEPS = P

class StockTradingEnv(gym.Env):
    """Environment for agent, consists of __init__, step, and reset functions"""
    metadata = {'render.modes': ['human']} # don't know what this is, stack overflow has long explanations
    def __init__(self, df, option_reset=False, path_choice=int(random.uniform(0,N)), reward_type="PnL"):
        super(StockTradingEnv, self).__init__() 
        # user-defined options
        self.option_reset = option_reset
        self.path_choice = path_choice
        self.reward_type = reward_type
        # initializing underlying amount
        self.underlying = 0 

        self.reward_range = (0, np.inf)
        # Actions of the format hold amount [0,1]
        self.action_space = spaces.Box(
            low=0, high=1, dtype=np.float16)  
        # agent is given previous action + current asset price and time to maturity (H_i-1, S_i, tau_i)
        self.observation_space = spaces.Box(
            low=np.array([0,0,0]), high=np.array([1,np.inf,60]), shape=(1, 3), dtype=np.float16)


    def step(self, action):
        
        # Execute one time step within the environment
        current_price = asset_price[self.pathnumb, self.current_step]
        self.underlying = action
        self.current_step += 1
        if self.current_step > MAX_STEPS:  #  add reset and switch to new path?
            self.current_step = 0
        
        if self.reward_type == "CF":
            reward = APL_process(option_price[self.pathnumb,self.current_step], asset_price[self.pathnumb,self.current_step], action)
        else: # needs a cash flow reward function, left APL for now
            reward = APL_process(option_price[self.pathnumb,self.current_step], asset_price[self.pathnumb,self.current_step], action)
        #previous action + current asset price and time to maturity (H_i-1, S_i, tau_i)
        obs = [action, asset_price[self.pathnumb,self.current_step], 60-self.current_step] 
        return obs, reward

    def reset(self): 
        if self.option_reset: # if user chose True, sets to his choice, else, random
            self.pathnumb = self.path_choice
        return asset_price[self.pathnumb,0], 60 # state0 of new path
