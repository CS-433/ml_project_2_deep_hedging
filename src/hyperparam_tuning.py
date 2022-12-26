import optuna
import numpy as np
import json

from env import StockTradingEnv
from agent import DDPG_Hedger
from network import MLP

BATCH_SIZE = 1024
N_EPISODE = 3000


def objective(trial):

    # set optuna param range
    critic_lr = 10 ** trial.suggest_float("critic_lr", -6, -1)
    actor_lr = 10 ** trial.suggest_float("actor_lr", -6, -1)
    update_freq = trial.suggest_int("update_freq", 1, 20)

    # define environment and the agent
    env = StockTradingEnv(reset_path=True)
    nState, nAction = env.observation_space.shape[0], env.action_space.shape[0]  # 3, 1

    actor = MLP(nState, 16, nAction, "Sigmoid")
    qnet_1 = MLP(nState + nAction, 16, nAction, "")
    qnet_2 = MLP(nState + nAction, 16, nAction, "")
    agent = DDPG_Hedger(actor, qnet_1, qnet_2, actor_lr, critic_lr, 1, BATCH_SIZE)

    target_rewards = []
    epsilon = 0.5

    for episode in range(N_EPISODE):
        # reset state
        state = env.reset()  # s_0
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

            if done:
                break

        for i in range(update_freq):
            agent.update(env.price_stat)
        agent.polyak_update()

        epsilon *= 0.9999
        # store total rewards after some training is done
        # we only consider alst 10 total rewards as a quantity to minimize
        if episode > int(N_EPISODE * 0.95):
            target_rewards.append(ep_tot_reward)

    return np.mean(target_rewards)


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    # pruned_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED]
    # complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    # print("  Number of pruned trials: ", len(pruned_trials))
    # print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print(" {}: {}".format(key, value))

    # store trained hyper parms
    with open("model/hypparams.json", "w") as file:
        json.dump(trial.params, file)
