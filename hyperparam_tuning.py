import optuna
import numpy as np
import json

from src.env import StockTradingEnv
from src.agent import DDPG_Hedger
from src.network import MLP

BATCH_SIZE = 32
N_EPISODE = 500


def objective(trial):

    # set optuna param range
    critic_lr = 10 ** trial.suggest_float("critic_lr", -6, -1)
    actor_lr = 10 ** trial.suggest_float("actor_lr", -6, -1)
    nHidden = trial.suggest_int("hidden_dim", 4, 32)
    trg_update = trial.suggest_int("polyak_update_freq", 5, 20)

    # define environment and the agent
    env = StockTradingEnv(reset_path=True)
    nState, nAction = env.observation_space.shape[0], env.action_space.shape[0]  # 3, 1

    actor = MLP(nState, nHidden, nAction, "Sigmoid")
    qnet_1 = MLP(nState + nAction, nHidden, nAction, "")
    qnet_2 = MLP(nState + nAction, nHidden, nAction, "")
    agent = DDPG_Hedger(actor, qnet_1, qnet_2, actor_lr, critic_lr, 1, BATCH_SIZE)

    target_rewards = []
    noise_std = 1

    for episode in range(N_EPISODE):
        # reset state
        state = env.reset()  # s_0
        ep_tot_reward = 0

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

        # store total rewards after some training is done
        # we only consider alst 10 total rewards as a quantity to minimize
        if episode > N_EPISODE - 20:
            target_rewards.append(ep_tot_reward)

        if episode % trg_update == 0:  # update target network
            agent.polyak_update()

        noise_std -= 0.0001

    return np.mean(target_rewards)


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)

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
