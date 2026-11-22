"""Illustration: the original Pong screen and the 64x64 image the agent receives.

    python -m scripts.pong_figure --out results/figures
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dreamer.env import Atari


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/figures")
    a = p.parse_args()
    env = Atari("pong", seed=3)
    rng = np.random.default_rng(3)
    obs = env.reset()
    # find a frame where the ball and both paddles are visible (colors from the Pong palette)
    def locate(screen, color, cols):
        area = screen[34:194, cols[0]:cols[1]]
        ys, xs = np.nonzero((np.abs(area.astype(int) - color).sum(-1) < 30))
        return (xs.mean() + cols[0], ys.mean() + 34) if len(ys) else None
    while True:
        obs = env.step(int(rng.integers(env.num_actions)))
        screen = env._screen()
        ball = locate(screen, (236, 236, 236), (20, 140))
        cpu = locate(screen, (213, 130, 74), (14, 22))
        me = locate(screen, (92, 186, 92), (138, 146))
        if ball and cpu and me and 50 < ball[0] < 110:
            break
    fig, axes = plt.subplots(1, 2, figsize=(3.3, 2.1), gridspec_kw=dict(width_ratios=[160, 210]))
    axes[0].imshow(screen)
    axes[0].set_title("original frame 210×160", fontsize=7)
    axes[1].imshow(obs["image"])
    axes[1].set_title("agent input 64×64", fontsize=7)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    arrow = dict(arrowstyle="->", color="white", lw=0.8)
    axes[0].annotate("agent", xy=me, xytext=(112, 185), color="white", fontsize=7, arrowprops=arrow)
    axes[0].annotate("opponent", xy=cpu, xytext=(30, 185), color="white", fontsize=7, arrowprops=arrow)
    axes[0].annotate("ball", xy=ball, xytext=(ball[0] - 10, 60), color="white", fontsize=7, arrowprops=arrow)
    fig.tight_layout(pad=0.3)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(a.out) / "pong_input.pdf"); fig.savefig(Path(a.out) / "pong_input.png", dpi=200)


if __name__ == "__main__":
    main()
