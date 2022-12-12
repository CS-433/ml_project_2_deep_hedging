import optuna

from env import StockTradingEnv
from agent import DDPG_Hedger
from network import MLP

BATCH_SIZE = 16
N_EPISODE = 100

def objective(trial):
    
    # set optuna param range
    critic_lr = 10 ** trial.suggest_float('critic_lr', -6, -1)
    actor_lr  = 10 ** trial.suggest_float('actor_lr', -6, -1)
    nHidden = trial.suggest_int('hidden_dim',2,32)
    trg_update = trial.suggest_int('polyak_update_freq',5,40)
    
    # define environment and the agent
    env = StockTradingEnv(reset_path=False)
    
    nState, nAction = env.observation_space.shape[0], env.action_space.shape[0]  # 3, 1
    actor = MLP(nState, nHidden, nAction, "ReLU", "Tanh")
    critic = MLP(nState + nAction, nHidden, nAction)

    agent = DDPG_Hedger(actor, critic, actor_lr, critic_lr, 1, BATCH_SIZE)
    

    for episode in range(N_EPISODE):
        # reset state
        state = env.reset()  # s_0
        total_reward = 0
        
        while True:
            # take action given state
            action = agent.act(state, 0.5)
            
            # take next step of the environment
            next_state, reward, done = env.step(action)

            # record interaction between environment and the agent
            agent.store(state, action, reward, next_state, done)
            
            total_reward -= reward
            state = next_state
            agent.update()
        
            if done:
                break
        
        if episode % trg_update == 0:  # update target network
            agent.polyak_update()
    
    return total_reward

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=1000)

    pruned_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print(" {}: {}".format(key, value))