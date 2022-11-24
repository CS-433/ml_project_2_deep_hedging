import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Normal
from abc import ABCMeta, abstractmethod
from buffer import ExpReplay
from copy import deepcopy
from collections import namedtuple


class Algorithm(metaclass=ABCMeta):
    """
    Author: Kibeom

    Constructs an abstract based class Algorithm.
    This will be the main bone of all Reinforcement Learning algorithms.
    This class gives a clear structure of what needs to be implemented for all reinforcement algorithm in general.
    If there is other extra methods to be specified, child class will inherit the existing methods as well as
    add new methods to it.
    """

    def __init__(self, env, disc_rate, learning_rate):
        self.env = env
        self.gamma = disc_rate
        self.lr = learning_rate

    @abstractmethod
    def act(self):
        pass

    @abstractmethod
    def store(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update(self):
        pass


class DDPG(Algorithm):
    def __init__(self, env, Actor, Critic, learning_rate, disc_rate, sigma, batch_size):
        """
        Author: Kibeom

        Need to write docstring
        :param env: openai gym environment.
        :param Net: neural network class from pytorch module.
        :param learning_rate: learning rate of optimizer.
        :param disc_rate: discount rate used to calculate return G
        """

        # define parameters of DDPG
        self.dim_in = env.observation_space.shape[0]
        try:
            self.dim_out = env.action_space.n
        except:
            self.dim_out = 1

        self.tau = 0.05
        self.gamma = disc_rate
        self.batch_size = batch_size
        self.buffer = ExpReplay(10000, self.transition)
        self.transition = namedtuple(
            "Transition",
            ("state", "action", "logprobs", "reward", "next_state", "dones"),
        )
        self.noise_dist = Normal(torch.tensor([0.0]), torch.tensor([sigma]))

        # define actor and critic ANN.
        self.actor = Actor(self.dim_in, self.dim_out)
        self.critic = Critic(self.dim_in)

        # define optimizer for Actor and Critic network
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        # define target network needed for DDPG optimization
        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)

    def reset(self):
        self.buffer.clear()

    def store(self, *args):
        self.buffer.store(*args)

    def act(self, state):
        """
        We use policy function to find the deterministic action instead of distributions
        which is parametrized by distribution parameters learned from the policy.

        Here, state input prompts policy network to output a single or multiple-dim
        actions.
        :param state:
        :return:
        """
        # we
        x = torch.tensor(state.astype(np.float32))
        action = self.actor.forward(x)
        return torch.clip(action + self.noise_dist.sample(), -2.0, 2.0).detach().numpy()

    def soft_update(self, target, source):
        for target_param, param in zip(
            list(target.parameters()), list(source.parameters())
        ):
            target_param = target_param * (1.0 - self.tau) + param * self.tau

    def update(self):
        # calculate return of all times in the episode
        if self.buffer.len() < self.batch_size:
            return

        transitions = self.buffer.sample(self.batch_size)
        batch = self.transition(*zip(*transitions))

        # extract variables from sampled batch.
        states = torch.tensor(batch.state).view(self.batch_size, self.dim_in)
        actions = torch.tensor(batch.action).view(self.batch_size, 1)
        rewards = torch.tensor(batch.reward).view(self.batch_size, 1)
        dones = torch.tensor(batch.dones).view(self.batch_size, 1).long()
        next_states = torch.tensor(batch.next_state).view(self.batch_size, self.dim_in)
        next_actions = self.actor_target(next_states)

        # compute target
        y = rewards + self.gamma * (1 - dones) * self.critic_target(
            torch.hstack((next_states, next_actions))
        )
        advantage = self.critic(torch.hstack([states, actions])) - y.detach()
        critic_loss = advantage.pow(2).mean()

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

        # Update the frozen target models
        self.soft_update(self.critic_target, self.critic)
        self.soft_update(self.actor_target, self.actor)

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
