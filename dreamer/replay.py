"""Flat numpy replay buffer; samples contiguous sequences.

Row i holds image_i, reward_i (received on reaching image_i), flags, and action_i (chosen after image_i).
Sequences may cross episode boundaries; the world model resets its state at `is_first`.
"""
import numpy as np


class Replay:
    def __init__(self, capacity, size=64, seed=0):
        self.capacity, self.n = capacity, 0
        self.image = np.zeros((capacity, size, size, 3), np.uint8)
        self.action = np.zeros(capacity, np.int64)
        self.reward = np.zeros(capacity, np.float32)
        self.is_first = np.zeros(capacity, bool)
        self.is_terminal = np.zeros(capacity, bool)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.n

    def add(self, obs, action):
        assert self.n < self.capacity, "Replay is full: increase capacity (>= total steps)."
        i = self.n
        self.image[i], self.reward[i] = obs["image"], obs["reward"]
        self.is_first[i], self.is_terminal[i], self.action[i] = obs["is_first"], obs["is_terminal"], action
        self.n += 1

    def sample(self, batch, length):
        """Return arrays [batch, length, ...] of uniformly sampled contiguous segments."""
        starts = self.rng.integers(0, self.n - length + 1, size=batch)
        idx = starts[:, None] + np.arange(length)[None]
        return dict(image=self.image[idx], action=self.action[idx], reward=self.reward[idx],
                    is_first=self.is_first[idx], is_terminal=self.is_terminal[idx])

    def save(self, path):
        n = self.n
        np.savez(path, n=n, image=self.image[:n], action=self.action[:n], reward=self.reward[:n],
                 is_first=self.is_first[:n], is_terminal=self.is_terminal[:n])

    def load(self, path):
        d = np.load(path)
        n = int(d["n"])
        for k in ("image", "action", "reward", "is_first", "is_terminal"):
            getattr(self, k)[:n] = d[k]
        self.n = n
