import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from buffer import ExpReplay
from copy import deepcopy
from collections import namedtuple
from torch.distributions import Normal


class DDPG:
    def __init__(
        self,
        Actor: nn.Module,
        Critic: nn.Module,
        actor_lr: float,
        critic_lr: float,
        disc_rate: float = 1.00,
        batch_size: int = 64,
    ):

        """
        Author: Kibeom

        Need to write docstring
        :param env: openai gym environment.
        :param Net: neural network class from pytorch module.
        :param learning_rate: learning rate of optimizer.
        :param disc_rate: discount rate used to calculate return G
        """

        self.gamma = disc_rate
        self.tau = 0.05
        self.batch_size = batch_size

        # experience replay related
        self.transition = namedtuple(
            "Transition",
            ("state", "action", "reward", "next_state", "done"),
        )
        self.buffer = ExpReplay(10000, self.transition)

        # define actor and critic ANN.
        self.actor = Actor
        self.critic = Critic

        # loss function for critic
        self.critic_loss = nn.MSELoss()

        # define optimizer for Actor and Critic network
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # define target network needed for DDPG optimization
        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)

    def reset(self):
        self.buffer.clear()

    def store(self, *args):
        self.buffer.store(*args)

    def act(self, state: list, sigma: float = 0.5):
        """
        We use policy function to find the deterministic action instead of distributions
        which is parametrized by distribution parameters learned from the policy.

        Here, state input prompts policy network to output a single or multiple-dim
        actions.
        :param state:
        :return:
        """
        x = torch.tensor(state.astype(np.float32))
        action = self.actor.forward(x)
        noise = Normal(torch.tensor([0.0]), torch.tensor([sigma])).sample()
        return torch.clip(action + noise, -2.0, 2.0).detach().numpy()

    def update(self):
        # calculate return of all times in the episode
        if self.buffer.len() < self.batch_size:
            return

        transitions = self.buffer.sample(self.batch_size)
        batch = self.transition(*zip(*transitions))

        # extract variables from sampled batch.
        states = torch.tensor(batch.state)
        actions = torch.tensor(batch.action)
        rewards = torch.tensor(batch.reward).float()
        dones = torch.tensor(batch.done).float()
        next_states = torch.tensor(batch.next_state)

        # compute critic loss
        Q = self.critic(torch.hstack([states, actions]))
        y = (
            rewards
            + self.gamma
            * (1 - dones)
            * self.critic_target(
                torch.hstack([next_states, self.actor_target(next_states)])
            )
        ).detach()
        critic_loss = self.critic_loss(Q, y)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Get actor loss
        actor_loss = -self.critic(torch.hstack([states, self.actor(states)])).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update target net
        self.polyak_update()

    def polyak_update(self):
        # Update the frozen target models
        for trg_param, src_param in zip(
            list(self.critic_target.parameters()), list(self.critic.parameters())
        ):
            trg_param = trg_param * (1.0 - self.tau) + src_param * self.tau

        for trg_param, src_param in zip(
            list(self.actor_target.parameters()), list(self.actor.parameters())
        ):
            trg_param = trg_param * (1.0 - self.tau) + src_param * self.tau

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_hyp_params")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_hyp_params")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(
            torch.load(filename + "_critic_hyp_params")
        )

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_hyp_params"))

        # define target network needed for DDPG optimization
        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)


class DDPG_Hedger(DDPG):
    def __init__(
        self,
        Actor: nn.Module,
        Critic: nn.Module,
        actor_lr: float,
        critic_lr: float,
        disc_rate: float = 1,
        batch_size: int = 32,
    ):
        super().__init__(Actor, Critic, actor_lr, critic_lr, disc_rate, batch_size)

        # define actor and critic ANN.
        self.actor = Actor
        self.critic_1 = Critic  # mean(cost)
        self.critic_2 = Critic  # std(cost)

        # define optimizer for Actor and Critic network
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=critic_lr)

        # define target network needed for DDPG optimization
        self.actor_target = deepcopy(self.actor)
        self.critic_1_target = deepcopy(self.critic_1)
        self.critic_2_target = deepcopy(self.critic_2)

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
        cost_variance = (
            self.critic_2(torch.hstack([states, self.actor(states)]))
            - self.critic_1(torch.hstack([states, self.actor(states)])) ** 2
        )
        actor_loss = (
            self.critic_1(torch.hstack([states, self.actor(states)]))
            + 1.5 * torch.sqrt(torch.where(cost_variance < 0, 0, cost_variance))
        ).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if output:
            return actor_loss.detach().item()
