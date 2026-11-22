"""Quick checks of the core components:  python -m tests.test_core"""
import numpy as np
import torch

from dreamer.agent import Dreamer
from dreamer.config import Config
from dreamer.env import Atari
from dreamer.replay import Replay
from dreamer.utils import TwoHot, symexp, symlog


def test_symlog_roundtrip():
    x = torch.tensor([-1000.0, -3.0, 0.0, 0.5, 20.0, 1e5])
    assert torch.allclose(symexp(symlog(x)), x, rtol=1e-4, atol=1e-4)


def test_twohot_roundtrip():
    th = TwoHot(255)
    x = torch.tensor([-1.0, 0.0, 1.0, 3.7, -250.0])
    enc = th.encode(x)
    assert torch.allclose(enc.sum(-1), torch.ones(5))
    assert (enc > 0).sum(-1).max() <= 2  # at most two nonzero bins
    back = th.mean(torch.log(enc + 1e-12))  # logits = log(prob), so softmax returns enc
    assert torch.allclose(back, x, rtol=1e-3, atol=1e-3), back


def test_env():
    env = Atari("pong", seed=0)
    obs = env.reset()
    assert obs["image"].shape == (64, 64, 3) and obs["image"].dtype == np.uint8 and obs["is_first"]
    assert env.num_actions == 6
    total, n = 0.0, 0
    while True:
        obs = env.step(np.random.randint(6))
        total += obs["reward"]; n += 1
        assert obs["reward"] in (-1.0, 0.0, 1.0)
        if obs["is_last"]:
            break
    assert obs["is_terminal"] and (total <= -21 + 21) and n > 100
    # frames = agent steps x repeat (the last step may be shorter)
    assert 4 * (n - 1) < env.frames <= 4 * n, (n, env.frames)


def test_lambda_return():
    """lam=1, value=0, reward=1, disc=0.9, H=3  ->  R_0 = 1 + .9 + .81 = 2.71; R_2 = 1."""
    agent = Dreamer(6, Config(deter=32, hidden=32, units=32, depth=8, stoch=4, classes=4, lam=1.0))
    r, d, v = torch.ones(1, 3), torch.full((1, 3), 0.9), torch.zeros(1, 4)
    ret = agent.lambda_return(r, d, v)
    assert torch.allclose(ret[0], torch.tensor([2.71, 1.9, 1.0])), ret
    # lam=0 -> one-step return: R_t = r + d * v_{t+1}
    agent.cfg.lam = 0.0
    v = torch.tensor([[0.0, 5.0, 5.0, 5.0]])
    assert torch.allclose(agent.lambda_return(r, d, v)[0], torch.full((3,), 1 + 0.9 * 5.0))


def test_hard_action_is_exact_onehot():
    from dreamer.utils import sample_hard
    x = sample_hard(torch.softmax(torch.randn(500, 6), -1))
    assert ((x == 0) | (x == 1)).all() and (x.sum(-1) == 1).all()


def test_replay_and_train_step():
    cfg = Config(deter=32, hidden=32, units=32, depth=8, stoch=4, classes=4, batch=3, seq=8, horizon=4)
    env, rep = Atari("pong", seed=1), Replay(200)
    obs = env.reset()
    for _ in range(100):
        a = np.random.randint(6); rep.add(obs, a)
        obs = env.reset() if obs["is_last"] else env.step(a)
    b = rep.sample(3, 8)
    assert b["image"].shape == (3, 8, 64, 64, 3) and b["action"].shape == (3, 8)
    agent = Dreamer(6, cfg)
    mets = agent.train_step({k: torch.from_numpy(v) for k, v in b.items()})
    assert all(np.isfinite(v) for v in mets.values())
    for k in ("loss/rec", "loss/reward", "loss/actor", "loss/critic", "imag/return"):
        assert k in mets, k


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("OK", name)
