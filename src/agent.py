import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from copy import deepcopy
from collections import namedtuple
from .buffer import ExpReplay


class DDPG:
    def __init__(
        self, 
        Actor:nn.Module, 
        Critic:nn.Module, 
        actor_lr:float, 
        critic_lr:float, 
        disc_rate:float=0.99, 
        batch_size:int=64
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
        self.actor  = Actor
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

    def act(self, state:list, sigma:float=0.5):
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
        rewards = torch.tensor(batch.reward)
        dones = torch.tensor(batch.done).long()
        next_states = torch.tensor(batch.next_state)
        
        # compute critic loss
        Q = self.critic(torch.hstack([states, actions]))
        y = rewards + self.gamma * (1 - dones) * self.critic_target(
            torch.hstack((next_states, self.actor_target(next_states)).detach())
        )
        critic_loss = self.critic_loss(Q, y)

        # Get actor loss
        actor_loss = -self.critic(torch.hstack([states, self.actor(states)])).mean()

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

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
