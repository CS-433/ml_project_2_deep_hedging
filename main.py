import os
import sys

sys.path.insert(0, "D:/work/Personal/ml_project_2_deep_hedging/src")

import json
import numpy as np
import pandas as pd

from env import StockTradingEnv
from agent import DDPG_Hedger
from network import MLP


if __name__ == "__main__":

    # make experiment results folder
    experiment_name = "v9_no_rand_samp"
    result_folder_path = f"model/{experiment_name}"
    os.makedirs(result_folder_path, exist_ok=True)

    BATCH_SIZE = 1024
    N_EPISODE = 20000
    epsilon = 0.5

    with open("model/hypparams.json", "r") as file:
        hyp_params = json.load(file)
    # hyp_params = {"critic_lr": -5.491386792760453, "actor_lr": -5.80149679060888}
    # actor_lr = 10 ** hyp_params["actor_lr"]
    # critic_lr = 10 ** hyp_params["critic_lr"]

    env = StockTradingEnv(reset_path=True, tc=0.0001)

    try:
        with open(result_folder_path + "/price_stat.json", "r") as f:
            env.price_stat = json.load(f)
    except:
        print("no price stats.")

    actor_lr, critic_lr = 10**-4, 10**-4
    nState, nAction = env.observation_space.shape[0], env.action_space.shape[0]  # 3, 1

    # we use hidden layer size of 32, 64 as the author used.
    actor = MLP(nState, 16, nAction, "Sigmoid")
    qnet_1 = MLP(nState + nAction, 16, nAction, "")
    qnet_2 = MLP(nState + nAction, 16, nAction, "")

    # load agent settings
    agent = DDPG_Hedger(actor, qnet_1, qnet_2, actor_lr, critic_lr, 1, BATCH_SIZE)
    # agent.load("v9_2nd")

    total_rewards = []
    min_actor_loss = 0
    for episode in range(N_EPISODE):
        # reset state
        state = env.reset()  # s_0
        actions = []
        ep_tot_reward = 0

        while True:
            # normalize the state
            normalized_state = env.normalize(state)

            # take action given state
            action = agent.act(normalized_state, epsilon)

            # take next step of the environment
            next_state, reward, done = env.step(action)

            # record interaction between environment and the agent
            agent.store(state, action, -reward, next_state, done)

            ep_tot_reward += reward
            state = next_state

            actions.append(np.round(action, 2))
            if done:
                break

        for i in range(1):
            q1_loss, q2_loss, actor_loss = agent.update(env.price_stat, True)

        agent.polyak_update()
        epsilon *= 0.9999

        if episode % 1000 == 0 and episode > 0:
            print(f"Episode {episode} Total Reward: {ep_tot_reward}")
            print(f"Episode {episode} Action taken: {actions}")
            print(f"Episode {episode} Epsilon     : {epsilon}")

            print(
                f"Episode {episode} Q1 Loss: {round(q1_loss.item())} Q2 Loss: {round(q2_loss.item())} Actor loss: {round(actor_loss,3)} \n\n\n"
            )
            total_rewards.append(
                [episode, ep_tot_reward, epsilon, q1_loss, q2_loss, actor_loss]
                + actions
            )

            # if actor loss is lower than historical minimum
            if actor_loss < min_actor_loss:
                # save trained weight that gives smallest actor loss
                print("save weight")
                agent.save(experiment_name)
                min_actor_loss = actor_loss

    # At the end of episodes,
    # save training results as csv
    total_rewards = pd.DataFrame(
        total_rewards,
        columns=[
            "Episode",
            "Episode Total Cost",
            "epsilon",
            "Q1 Loss",
            "Q2 Loss",
            "Actor Loss",
        ]
        + [f"action_step{i}" for i in range(59)],
    )
    total_rewards.to_csv(result_folder_path + "/results.csv")

    # save trained weight for the later use.
    # agent.save(experiment_name)

    # save running stats of price variable
    with open(result_folder_path + "/price_stat.json", "w") as f:
        json.dump(env.price_stat, f)
