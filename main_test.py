import gym
import os
import argparse
import torch

from gym import envs
from src.agent import DDPG
from src.network import MLP
from src.utils import NetkwargAction

###############################################
############# Parameter Setting ###############
###############################################

BATCH_SIZE = 100
DISC_RATE  = 0.99
TRG_UPDATE = 10
N_EPISODE  = 50

env = gym.make('Pendulum-v1')

nState, nAction = env.observation_space.shape[0], env.action_space.shape[0]
critic = MLP(nState+nAction, 12, nAction)
actor  = MLP(nState, 12, nAction)
agent = DDPG(actor, critic, 0.001, DISC_RATE, 0.5, BATCH_SIZE)

if __name__ == '__main__':

    batch_reward_average = 0

    for episode in range(N_EPISODE):
        # reset state
        state = env.reset()
        total_reward = 0

        while True:
            # take action given state
            action = agent.act(state)

            # take next step of the environment
            next_state, reward, done, _ = env.step(action)
            
            # record interaction between environment and the agent
            agent.store(state, action, reward, next_state, done)
            env.render()
            total_reward += reward
            state = next_state

            if done:
                break

        agent.update()
        batch_reward_average += total_reward
        total_reward = 0
        
        if episode % 100 == 0:
            batch_reward_average = batch_reward_average/100
            solved = batch_reward_average > 195.0
            print(f'Episode Batch {episode//100}, total_reward: {batch_reward_average}, solved: {solved}')
            
            if solved:
                print(f'solved!! at Episode Batch {episode//100}, total_reward: {batch_reward_average}')
                break
            batch_reward_average = 0


