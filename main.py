import os
import sys
import json
import numpy as np
import pandas as pd

from src.env import StockTradingEnv
from src.agent import DDPG_Hedger
from src.network import MLP

sys.path.insert(1, "ml_project_2_deep_hedging/src")


if __name__ == "__main__":

    # make experiment results folder
    experiment_name = "v7"
    result_folder_path = f"model/{experiment_name}"
    os.makedirs(result_folder_path, exist_ok=True)

    BATCH_SIZE = 32
    N_EPISODE = 100000

    with open("model/hypparams.json", "r") as file:
        hyp_params = json.load(file)
    hyp_params = {"critic_lr": -5.491386792760453, "actor_lr": -5.80149679060888}
    env = StockTradingEnv(reset_path=True)

    actor_lr = 10 ** hyp_params["actor_lr"]
    critic_lr = 10 ** hyp_params["critic_lr"]

    # actor_lr, critic_lr = 10**-4, 10**-4
    nState, nAction = env.observation_space.shape[0], env.action_space.shape[0]  # 3, 1

    # we use hidden layer size of 32, 64 as the author used.
    actor = MLP(nState, 32, nAction, "Sigmoid")
    qnet_1 = MLP(nState + nAction, 32, nAction, "")
    qnet_2 = MLP(nState + nAction, 32, nAction, "")
    agent = DDPG_Hedger(actor, qnet_1, qnet_2, actor_lr, critic_lr, 1, BATCH_SIZE)
    noise_std = 1

    total_rewards = []
    for episode in range(N_EPISODE):
        # reset state
        state = env.reset()  # s_0
        actions = []
        ep_tot_reward = 0

        if episode % 100 == 0 and episode > 0:
            isPrint = True
        else:
            isPrint = False

        while True:
            # take action given state
            action = agent.act(state, noise_std)

            # take next step of the environment
            next_state, reward, done = env.step(action)

            # record interaction between environment and the agent
            agent.store(state, action, reward, next_state, done)

            ep_tot_reward += reward
            state = next_state

            if isPrint == True:
                q1_loss, q2_loss, actor_loss = agent.update(isPrint)
            else:
                agent.update()

            agent.polyak_update()
            actions.append(np.round(action, 2))
            if done:
                break

        noise_std *= 0.9999

        if episode % 100 == 0 and episode > 0:
            print(f"Episode {episode} Total Reward: {ep_tot_reward}")
            print(f"Episode {episode} Action taken: {actions}")
            print(
                f"Episode {episode} Q1 Loss: {q1_loss} Q2 Loss: {q2_loss} Actor loss: {actor_loss}"
            )
            total_rewards.append(
                [episode, ep_tot_reward, q1_loss, q2_loss, actor_loss] + actions
            )

    # At the end of episodes,
    # save training results as csv
    total_rewards = pd.DataFrame(
        total_rewards,
        columns=["Episode", "Episode Total Reward", "Q1 Loss", "Q2 Loss", "Actor Loss"]
        + [f"action_step{i}" for i in range(59)],
    )
    total_rewards.to_csv(result_folder_path + "/results.csv")

    # save trained weight for the later use.
    agent.save(experiment_name)
