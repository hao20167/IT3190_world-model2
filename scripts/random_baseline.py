"""Random-policy baseline, same environment and protocol as the Dreamer evaluation.

    python -m scripts.random_baseline --episodes 30 --seed 123 --out results/p0_random.json
"""
import argparse
import json
from pathlib import Path

from dreamer.config import Config
from dreamer.env import Atari
from dreamer.evaluate import RandomPolicy, run_episodes, summarize


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out", default="results/p0_random.json")
    a = p.parse_args()
    c = Config()
    env = Atari(c.game, c.size, c.repeat, c.noops, c.max_frames, seed=a.seed)
    res = run_episodes(env, RandomPolicy(env.num_actions, a.seed), a.episodes)
    out = dict(policy="random", seed=a.seed, **summarize(res), results=res)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"random: {out['mean']:.2f} ± {out['std']:.2f} ({a.episodes} episodes)")


if __name__ == "__main__":
    main()
