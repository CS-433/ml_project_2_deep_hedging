import random
import numpy as np
from collections import deque

class ExpReplay:
    def __init__(self, capacity, transition):
        self.memory = deque([], maxlen=capacity)
        self.record = transition

    def len(self):
        return len(self.memory)

    def store(self, *args):
        """Save a transition"""
        record = self.record(*[item.tolist() if type(item) == np.ndarray else [item] for item in list(args)])
        self.memory.append(record)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def all(self):
        return list(self.memory)

    def clear(self):
        return self.memory.clear()


class PrioritizedExpReplay(ExpReplay):
    """
    Need to implement this (Kibeom).

    """

    def __init__(self, capacity, transition, alpha):
        super().__init__(capacity, transition)
        self.alpha = alpha
        self.priority_sum = [0 for _ in range(2 * self.capacity)]
        self.priority_min = [float("inf") for _ in range(2 * self.capacity)]
        self.max_priority = 1.0
        self.next_idx = 0
        self.size = 0

    def store(self, *args):
        """Save a transition"""
        idx = self.next_idx

        self.memory.append(self.record(*args))
        self.next_idx = (idx + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)

        priority_alpha = self.max_priority ** self.alpha
