"""Learning check on a toy env: a square appears on the left or right and the action guesses the side.

Episodes last 20 steps (random ~ 10, optimal = 20). The reward arrives with the NEXT observation, so
the world model must remember the previous square through h_t.

    python -m tests.test_learning        (a few minutes on CPU)
"""
import time

import numpy as np
import torch

from dreamer.agent import Dreamer
from dreamer.config import Config
from dreamer.replay import Replay

LEN = 20


class ToyEnv:
    num_actions = 2

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def _obs(self, reward, first=False, last=False):
        img = np.zeros((64, 64, 3), np.uint8)
        x = 8 if self.side == 0 else 40
        img[24:40, x:x + 16] = 255
        return dict(image=img, reward=np.float32(reward), is_first=first, is_last=last, is_terminal=last)

    def reset(self):
        self.t, self.side = 0, int(self.rng.integers(2))
        return self._obs(0.0, first=True)

    def step(self, action):
        reward = float(action == self.side)
        self.t, self.side = self.t + 1, int(self.rng.integers(2))
        return self._obs(reward, last=self.t >= LEN)


def main(steps=3000, seed=0):
    torch.manual_seed(seed)
    cfg = Config(deter=64, hidden=64, units=64, depth=8, stoch=8, classes=8, batch=8, seq=16, horizon=8,
                 train_ratio=64, lr_model=3e-4, lr_actor=1e-4, lr_critic=1e-4, device="cpu")
    env, replay = ToyEnv(seed), Replay(steps + 10)
    agent = Dreamer(2, cfg)
    obs, scores, ep, due, t0, rows = env.reset(), [], 0.0, 0.0, time.time(), []
    state = agent.initial_state(1)
    for step in range(1, steps + 1):
        if step <= 200:
            action = int(np.random.randint(2))
        else:
            a, state = agent.act(torch.from_numpy(obs["image"])[None], torch.tensor([bool(obs["is_first"])]), state)
            action = int(a)
        replay.add(obs, action)
        ep += float(obs["reward"])
        if obs["is_last"]:
            scores.append(ep); ep = 0.0
            obs, state = env.reset(), agent.initial_state(1)
        else:
            obs = env.step(action)
        if step > 200:
            due += cfg.train_ratio / (cfg.batch * cfg.seq)
            while due >= 1:
                due -= 1
                m = agent.train_step({k: torch.from_numpy(v) for k, v in replay.sample(cfg.batch, cfg.seq).items()})
        if step % 500 == 0:
            recent = float(np.mean(scores[-20:]))
            rows.append(recent)
            print(f"step={step} return(last20)={recent:.1f}  rec={m['loss/rec']:.1f} rew={m['loss/reward']:.3f} "
                  f"ent={m['actor/entropy']:.2f} t={time.time() - t0:.0f}s", flush=True)
    return rows


if __name__ == "__main__":
    rows = main()
    assert rows[-1] >= 16, f"Agent did not learn the toy task: {rows}"
    print("OK: final return >= 16 (random ~ 10, optimal = 20)")
