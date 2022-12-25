import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from copy import deepcopy
from src.buffer import ExpReplay
from collections import namedtuple
from torch.distributions import Normal


class DDPG_Hedger:
    def __init__(
        self,
        Actor: nn.Module,
        Critic_1: nn.Module,
        Critic_2: nn.Module,
        actor_lr: float,
        critic_lr: float,
        disc_rate: float = 1,
        batch_size: int = 32,
    ):

        # params
        self.gamma = disc_rate
        self.tau = 0.0001
        self.batch_size = batch_size

        # experience replay related
        self.transition = namedtuple(
            "Transition",
            ("state", "action", "reward", "next_state", "done"),
        )
        self.buffer = ExpReplay(600000, self.transition)

        # define actor and critic ANN.
        self.actor = Actor
        self.critic_1 = Critic_1  # mean(cost)
        self.critic_2 = Critic_2  # std(cost)

        # loss function for critic
        self.critic_loss = nn.MSELoss()

        # define optimizer for Actor and Critic network
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=critic_lr)

        # define target network needed for DDPG optimization
        self.actor_target = deepcopy(self.actor)
        self.critic_1_target = deepcopy(self.critic_1)
        self.critic_2_target = deepcopy(self.critic_2)

        # temporary
        self.actor_target.isPrint = False
        self.critic_1_target.isPrint = False
        self.critic_2_target.isPrint = False

    def reset(self):
        self.buffer.clear()

    def store(self, *args):
        self.buffer.store(*args)

    def act(self, state: list, epsilon: float = 0.05, isPrint=False):
        """
        We use policy function to find the deterministic action instead of distributions
        which is parametrized by distribution parameters learned from the policy.

        Here, state input prompts policy network to output a single or multiple-dim
        actions.
        :param state:
        :return:
        """
        x = torch.tensor(state).to(torch.float64)
        if np.random.rand() <= epsilon:
            action = np.random.uniform(0, 1)
        else:
            action = self.actor(x).detach().item()
        return np.clip(action * 100, 0, 100.0)

    def update(self, price_stat, output=False):
        # calculate return of all times in the episode
        if self.buffer.len() < self.batch_size:
            return

        transitions = self.buffer.sample(self.batch_size)
        batch = self.transition(*zip(*transitions))

        # get rolling stats for price
        mu_, std_ = torch.Tensor([0, price_stat[0], 0]), torch.Tensor(
            [100, price_stat[1], 60]
        )

        # extract variables from sampled batch.
        states = torch.tensor(batch.state)
        actions = torch.tensor(batch.action)
        rewards = torch.tensor(batch.reward)
        dones = torch.tensor(batch.done).float()
        next_states = torch.tensor(batch.next_state)

        # normalize the price in state vector
        states = (states - mu_) / std_
        next_states = (next_states - mu_) / std_

        # define stateactions
        stateaction = torch.hstack([states, actions])
        next_stateaction = torch.hstack(
            [next_states, torch.clip(self.actor_target(next_states) * 100, 0, 100)]
        ).detach()

        # compute Q_1 loss
        Q_1 = self.critic_1(stateaction)
        y_1 = rewards + self.gamma * (1 - dones) * self.critic_1_target(
            next_stateaction
        )
        critic_loss_1 = self.critic_loss(Q_1, y_1)

        # Optimize the critic Q_1
        self.critic_1_optimizer.zero_grad()
        critic_loss_1.backward()
        self.critic_1_optimizer.step()

        # compute Q_2 loss
        Q_2 = self.critic_2(stateaction)
        y_2 = (
            rewards**2
            + (self.gamma**2) * (1 - dones) * self.critic_2_target(next_stateaction)
            + 2 * rewards * self.gamma * self.critic_1_target(next_stateaction)
        )

        critic_loss_2 = self.critic_loss(Q_2, y_2)

        # Optimize the critic Q_2
        self.critic_2_optimizer.zero_grad()
        critic_loss_2.backward()
        self.critic_2_optimizer.step()

        # Get actor loss
        state_action = torch.hstack(
            [states, torch.clip(self.actor(states) * 100, 0, 100)]
        )
        q_1, q_2 = self.critic_1(state_action), self.critic_2(state_action)
        cost_std = torch.sqrt(torch.where(q_2 - q_1.pow(2) < 0, 0, q_2 - q_1.pow(2)))
        actor_loss = (self.critic_1(state_action) + 1.5 * cost_std).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if output:
            return critic_loss_1, critic_loss_2, actor_loss.detach().item()

    def polyak_update(self):
        # Update the frozen target models
        for trg_param, src_param in zip(
            list(self.critic_1_target.parameters()), list(self.critic_1.parameters())
        ):
            trg_param = trg_param * (1.0 - self.tau) + src_param * self.tau

        # Update the frozen target models
        for trg_param, src_param in zip(
            list(self.critic_2_target.parameters()), list(self.critic_2.parameters())
        ):
            trg_param = trg_param * (1.0 - self.tau) + src_param * self.tau

        for trg_param, src_param in zip(
            list(self.actor_target.parameters()), list(self.actor.parameters())
        ):
            trg_param = trg_param * (1.0 - self.tau) + src_param * self.tau

    def save(self, name):
        torch.save(self.critic_1.state_dict(), f"model/{name}/critic_1_weight.pt")
        torch.save(self.critic_2.state_dict(), f"model/{name}/critic_2_weight.pt")
        torch.save(self.actor.state_dict(), f"model/{name}/actor_weight.pt")

    def load(self, name):
        # load trained weights to Q_1, Q_2, Actor
        self.critic_1.load_state_dict(torch.load(f"model/{name}/critic_1_weight.pt"))
        self.critic_2.load_state_dict(torch.load(f"model/{name}/critic_2_weight.pt"))
        self.actor.load_state_dict(torch.load(f"model/{name}/actor_weight.pt"))

        # Copy above 3 to target networks.
        self.critic_1_target = deepcopy(self.critic_1)
        self.critic_2_target = deepcopy(self.critic_2)
        self.actor_target = deepcopy(self.actor)
