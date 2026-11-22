"""Decoder reconstructions of the same real frames across training checkpoints.

    python -m scripts.recon_figure --ckpts runs/p1/ckpt/model_0000000.pt ... --out results/figures

Frames come from one random-action episode with a fixed seed, shared by all checkpoints;
the model sees real frames from the episode start (posterior) and the decoder reconstructs them.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from dreamer.agent import Dreamer
from dreamer.config import Config
from dreamer.env import Atari


def collect(seed, steps):
    env = Atari("pong", seed=seed)
    rng = np.random.default_rng(seed)
    obs, frames, actions = env.reset(), [], []
    for _ in range(steps):
        a = int(rng.integers(env.num_actions))
        frames.append(obs["image"]); actions.append(a)
        obs = env.step(a)
    return np.stack(frames), actions, env.num_actions


@torch.no_grad()
def reconstruct(ckpt_path, cfg, frames, actions, num_actions):
    agent = Dreamer(num_actions, cfg)
    agent.restore(torch.load(ckpt_path, map_location="cpu", weights_only=False), optimizers=False)
    agent.eval()
    state, recons = agent.initial_state(1), []
    for t in range(len(frames)):
        prev = actions[t - 1] if t > 0 else 0
        state["prevact"] = F.one_hot(torch.tensor([prev]), num_actions).float() * (t > 0)
        embed = agent.encoder(torch.from_numpy(frames[t].copy())[None])
        state, _, _ = agent.rssm.observe_step(state, embed, torch.tensor([t == 0]))
        img = agent.decoder(agent.rssm.feat(state))[0]
        recons.append(np.clip((img.numpy() + 0.5) * 255, 0, 255).astype(np.uint8))
    return np.stack(recons)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--config", default="results/runs/p1_pong_s0/config.json")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--show", type=int, nargs="+", default=[300])
    p.add_argument("--out", default="results/figures")
    a = p.parse_args()
    cfg = Config(**json.loads(Path(a.config).read_text()))
    frames, actions, n = collect(a.seed, a.steps)
    rows, info = [("ground truth", frames, None)], []
    for ck in a.ckpts:
        step = int(Path(ck).stem.split("_")[1])
        rec = reconstruct(ck, cfg, frames, actions, n)
        mse = float(((rec / 255.0 - frames / 255.0) ** 2).mean())
        rows.append((f"{step // 1000}k steps", rec, mse))
        info.append(dict(ckpt=ck, step=step, mse=mse))

    plt.rcParams.update({"font.size": 7})
    # one column per model (first = ground truth), one row per time index
    fig, axes = plt.subplots(len(a.show), len(rows), figsize=(0.98 * len(rows), 1.02 * len(a.show) + 0.35))
    for c, (name, imgs, mse) in enumerate(rows):
        for r, t in enumerate(a.show):
            ax = axes[c] if len(a.show) == 1 else axes[r, c]
            ax.imshow(imgs[t]); ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(name if mse is None else f"{name}\nMSE {mse:.1e}", fontsize=6.5)
            if c == 0:
                ax.set_ylabel(f"t = {t}", fontsize=6.5)
    fig.subplots_adjust(left=0.04, right=0.995, top=0.78, bottom=0.01, wspace=0.05, hspace=0.05)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "recon_over_time.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out / "recon_over_time.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
    (out / "recon_over_time.json").write_text(json.dumps(dict(seed=a.seed, show=a.show, rows=info), indent=2))
    print(info)


if __name__ == "__main__":
    main()
