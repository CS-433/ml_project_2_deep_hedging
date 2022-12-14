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
        self.tau = 0.01
        self.batch_size = batch_size

        # experience replay related
        self.transition = namedtuple(
            "Transition",
            ("state", "action", "reward", "next_state", "done"),
        )
        self.buffer = ExpReplay(10000, self.transition)

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

    def reset(self):
        self.buffer.clear()

    def store(self, *args):
        self.buffer.store(*args)

    def act(self, state: list, sigma: float = 0.2):
        """
        We use policy function to find the deterministic action instead of distributions
        which is parametrized by distribution parameters learned from the policy.

        Here, state input prompts policy network to output a single or multiple-dim
        actions.
        :param state:
        :return:
        """
        x = torch.tensor(state).to(torch.float64)
        action = self.actor.forward(x)
        noise = Normal(torch.tensor([0.0]), torch.tensor([sigma])).sample().item()
        return (
            torch.clip((action - 0.5) * 2 + noise, -state[0], 1.0 - state[0])
            .detach()
            .numpy()
        )

    def update(self, output=False):
        # calculate return of all times in the episode
        if self.buffer.len() < self.batch_size:
            return

        transitions = self.buffer.sample(self.batch_size)
        batch = self.transition(*zip(*transitions))

        # extract variables from sampled batch.
        states = torch.tensor(batch.state)
        actions = torch.tensor(batch.action)
        rewards = torch.tensor(batch.reward)
        dones = torch.tensor(batch.done).float()
        next_states = torch.tensor(batch.next_state)

        # compute Q_1 loss
        Q_1 = self.critic_1(torch.hstack([states, actions]))
        y_1 = rewards + self.gamma * (1 - dones) * self.critic_1_target(
            torch.hstack([next_states, self.actor_target(next_states)]).detach()
        )

        critic_loss_1 = self.critic_loss(Q_1, y_1)

        # Optimize the critic Q_1
        self.critic_1_optimizer.zero_grad()
        critic_loss_1.backward()
        self.critic_1_optimizer.step()

        # compute Q_2 loss
        Q_2 = self.critic_2(torch.hstack([states, actions]))
        y_2 = (
            rewards**2
            + (self.gamma**2)
            * (1 - dones)
            * self.critic_2_target(
                torch.hstack([next_states, self.actor_target(next_states)]).detach()
            )
            + 2
            * self.gamma
            * rewards
            * self.critic_1_target(
                torch.hstack([next_states, self.actor_target(next_states)]).detach()
            )
        )

        critic_loss_2 = self.critic_loss(Q_2, y_2)

        # Optimize the critic Q_2
        self.critic_2_optimizer.zero_grad()
        critic_loss_2.backward()
        self.critic_2_optimizer.step()

        # Get actor loss
        state_action = torch.hstack([states, self.actor(states)])
        cost_variance = self.critic_2(state_action) - self.critic_1(state_action) ** 2
        # print(self.critic_1(state_action)[:3], self.critic_2(state_action)[:3])
        actor_loss = (
            self.critic_1(state_action)
            + 1.5 * torch.sqrt(torch.where(cost_variance < 0, 0, cost_variance))
        ).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # print(critic_loss_1, critic_loss_2, actor_loss)

        if output:
            return actor_loss.detach().item()

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

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.actor.state_dict(), filename + "_actor")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.actor.load_state_dict(torch.load(filename + "_actor"))

        # define target network needed for DDPG optimization
        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)
