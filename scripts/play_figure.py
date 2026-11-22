"""Figures: snapshots of one full Pong game by the agent, and the per-episode eval returns.

    python -m scripts.play_figure --ckpt <logdir>/ckpt/model_0100000.pt --seed 2024 --out results/figures

The game is played on a fresh environment with a fixed seed (not cherry-picked).
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dreamer.agent import Dreamer
from dreamer.config import Config
from dreamer.env import Atari
from dreamer.evaluate import AgentPolicy


def play(ckpt_path, seed):
    """Play one game; returns the raw 210x160 screens and (agent, opponent) scores after each step."""
    cfg = Config(**json.loads((ckpt_path.parent.parent / "config.json").read_text()))
    torch.manual_seed(seed)
    env = Atari(cfg.game, cfg.size, cfg.repeat, cfg.noops, cfg.max_frames, seed=seed)
    agent = Dreamer(env.num_actions, cfg)
    agent.restore(torch.load(ckpt_path, map_location="cpu", weights_only=False), optimizers=False)
    agent.eval()
    policy = AgentPolicy(agent, "cpu")
    obs = env.reset()
    policy.reset()
    screens, scores, me, cpu = [env._screen()], [(0, 0)], 0, 0
    while not obs["is_last"]:
        obs = env.step(policy(obs))
        me += obs["reward"] > 0
        cpu += obs["reward"] < 0
        screens.append(env._screen())
        scores.append((int(me), int(cpu)))
    return screens, scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--random", default="results/p0_random.json")
    p.add_argument("--final", default="results/p3_final.json")
    p.add_argument("--out", default="results/figures")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})

    screens, scores = play(Path(a.ckpt), a.seed)
    n = len(screens)
    # six evenly spaced moments; the last is the end of the game
    idx = np.linspace(0, n - 1, 6).astype(int)
    fig, axes = plt.subplots(1, 6, figsize=(7.0, 1.75))
    for ax, i in zip(axes, idx):
        ax.imshow(screens[i])
        ax.set_xticks([]); ax.set_yticks([])
        me, cpu = scores[i]
        ax.set_title(f"step {i}\nopp. {cpu} – agent {me}", fontsize=6.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(out / "game_snapshots.pdf", bbox_inches="tight", pad_inches=0.03); fig.savefig(out / "game_snapshots.png", dpi=200)
    plt.close(fig)

    rnd = [r["return"] for r in json.loads(Path(a.random).read_text())["results"]]
    fin = [r["return"] for r in json.loads(Path(a.final).read_text())["results"]]
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    bins = np.arange(-21.5, 22, 1)
    ax.hist(rnd, bins=bins, color="C3", alpha=0.7, label="random policy")
    ax.hist(fin, bins=bins, color="C0", alpha=0.7, label="Dreamer (100k steps)")
    ax.axvline(0, color="0.4", lw=0.8, ls=":")
    ax.set_xlabel("episode return = agent score − opponent score")
    ax.set_ylabel("episodes")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "episode_returns.pdf"); fig.savefig(out / "episode_returns.png", dpi=200)

    info = dict(ckpt=a.ckpt, seed=a.seed, steps=n - 1, final_score_agent=scores[-1][0], final_score_cpu=scores[-1][1],
                snapshot_steps=[int(i) for i in idx], snapshot_scores=[scores[i] for i in idx],
                wins_dreamer=int(sum(r > 0 for r in fin)), episodes_dreamer=len(fin))
    (out / "game_snapshots.json").write_text(json.dumps(info, indent=2))
    print(info)


if __name__ == "__main__":
    main()
