import sys
import json
import numpy as np

from src.env import StockTradingEnv
from src.agent import DDPG_Hedger
from src.network import MLP

sys.path.insert(1, "ml_project_2_deep_hedging/src")


if __name__ == "__main__":
    BATCH_SIZE = 16
    N_EPISODE = 2000
    DISC_RATE = 1

    with open("model/hypparams.json", "r") as file:
        hyp_params = json.load(file)

    env = StockTradingEnv(reset_path=True)

    nHidden = hyp_params["hidden_dim"]
    actor_lr = 10 ** hyp_params["actor_lr"]
    critic_lr = 10 ** hyp_params["critic_lr"]
    trg_update = hyp_params["polyak_update_freq"]

    nState, nAction = env.observation_space.shape[0], env.action_space.shape[0]  # 3, 1
    actor = MLP(nState, nHidden, nAction, "ReLU", "Sigmoid")
    critic = MLP(nState + nAction, nHidden, nAction)

    agent = DDPG_Hedger(actor, critic, actor_lr, critic_lr, DISC_RATE, BATCH_SIZE)

    target_rewards = []
    noise_std = 0.5

    for episode in range(N_EPISODE):
        # reset state
        state = env.reset()  # s_0
        ep_tot_reward = 0

        if episode > N_EPISODE - 30:
            noise_std = 0.0001

        while True:
            # take action given state
            action = agent.act(state, noise_std)

            # take next step of the environment
            next_state, reward, done = env.step(action)

            # record interaction between environment and the agent
            agent.store(state, action, reward, next_state, done)

            ep_tot_reward -= reward
            state = next_state
            agent.update()

            if done:
                break
            
        print(f"Episode {episode} Reward: {ep_tot_reward}")
        # store total rewards after some training is done
        # we only consider alst 10 total rewards as a quantity to minimize
        if episode > N_EPISODE - 30:
            target_rewards.append(ep_tot_reward)

        if episode % trg_update == 0:  # update target network
            agent.polyak_update()

    print(np.mean(target_rewards))
