"""Agent GIF and reconstruction / open-loop prediction figures.

    python -m scripts.visualize --ckpt runs/p1_pong_s0/ckpt/model_0100000.pt --out results/figures

openloop.png rows (a segment of a new episode): real frames; first K steps reconstructed from the
posterior, later steps predicted open-loop from the prior using only real actions; |real - predicted|.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from dreamer.agent import Dreamer
from dreamer.config import Config
from dreamer.env import Atari
from dreamer.evaluate import AgentPolicy


def to_img(x):  # tensor [-0.5, 0.5] -> uint8
    return np.clip((x.float().cpu().numpy() + 0.5) * 255, 0, 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="results/figures")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--context", type=int, default=5)   # steps with observations
    p.add_argument("--horizon", type=int, default=15)  # open-loop prediction steps
    p.add_argument("--start", type=int, default=100)   # segment starts at this agent step
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(a.ckpt)
    cfg = Config(**json.loads((ckpt_path.parent.parent / "config.json").read_text()))
    torch.manual_seed(a.seed)
    env = Atari(cfg.game, cfg.size, cfg.repeat, cfg.noops, cfg.max_frames, seed=a.seed)
    agent = Dreamer(env.num_actions, cfg)
    agent.restore(torch.load(ckpt_path, map_location="cpu", weights_only=False), optimizers=False)
    agent.eval()

    policy = AgentPolicy(agent, "cpu")
    obs, frames, actions, score = env.reset(), [], [], 0.0
    policy.reset()
    while True:
        act = policy(obs)
        frames.append(obs["image"]); actions.append(act)
        obs = env.step(act); score += float(obs["reward"])
        if obs["is_last"]:
            frames.append(obs["image"]); break
    big = [Image.fromarray(f).resize((192, 192), Image.NEAREST) for f in frames[::2]]
    big[0].save(out / "agent_episode.gif", save_all=True, append_images=big[1:], duration=40, loop=0)

    s, K, H = a.start, a.context, a.horizon
    imgs = torch.from_numpy(np.stack(frames[s:s + K + H]))
    acts = torch.nn.functional.one_hot(torch.tensor(actions[s:s + K + H]), env.num_actions).float()
    with torch.no_grad():
        # run through the earlier steps to get the right latent state
        state, preds = agent.initial_state(1), []
        for t in range(s + K):
            state["prevact"] = (torch.nn.functional.one_hot(torch.tensor([actions[t - 1]]), env.num_actions).float()
                                if t > 0 else torch.zeros(1, env.num_actions))
            embed = agent.encoder(torch.from_numpy(frames[t])[None])
            state, _, _ = agent.rssm.observe_step(state, embed, torch.tensor([t == 0]))
            if t >= s:
                preds.append(agent.decoder(agent.rssm.feat(state))[0])
        for t in range(K, K + H):  # open-loop: real actions only, no images
            state = agent.rssm.imagine_step(state, acts[t - 1][None])
            preds.append(agent.decoder(agent.rssm.feat(state))[0])
    true = imgs.numpy(); pred = np.stack([to_img(x) for x in preds])
    err = np.abs(true.astype(int) - pred.astype(int)).astype(np.uint8)
    rows = [np.concatenate(list(r), 1) for r in (true, pred, err)]
    grid = Image.fromarray(np.concatenate(rows, 0)).resize((64 * (K + H) * 2, 64 * 3 * 2), Image.NEAREST)
    d = ImageDraw.Draw(grid)
    x = K * 128
    d.line([(x, 0), (x, grid.height)], fill=(255, 0, 0), width=3)
    grid.save(out / "openloop.png")
    info = dict(ckpt=str(ckpt_path), episode_return=score, episode_steps=len(actions), segment_start=s,
                context_steps=K, openloop_steps=H,
                mse_posterior=float(((true[:K] / 255.0 - pred[:K] / 255.0) ** 2).mean()),
                mse_openloop=float(((true[K:] / 255.0 - pred[K:] / 255.0) ** 2).mean()))
    (out / "visualize.json").write_text(json.dumps(info, indent=2))

    # figure: last 2 posterior steps + 6 prior steps; top row real, bottom row world model
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = [K - 2, K - 1] + [K + k for k in (0, 2, 4, 7, 10, H - 1) if k < H]
    titles = ["posterior", "posterior"] + [f"prior +{c - K + 1}" for c in cols[2:]]
    fig, axes = plt.subplots(2, len(cols), figsize=(7.0, 2.05))
    for j, (c, title) in enumerate(zip(cols, titles)):
        for i, img in enumerate((true[c], pred[c])):
            ax = axes[i, j]
            ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
            if j == 2:  # red edge marks where imagination starts
                for side in ("left",):
                    ax.spines[side].set_color("red"); ax.spines[side].set_linewidth(2)
        axes[0, j].set_title(title, fontsize=6.5)
    axes[0, 0].set_ylabel("ground truth", fontsize=6.5)
    axes[1, 0].set_ylabel("world model", fontsize=6.5)
    fig.subplots_adjust(left=0.04, right=0.995, top=0.9, bottom=0.01, wspace=0.05, hspace=0.05)
    fig.savefig(out / "openloop_figure.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out / "openloop_figure.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
    print(info)


if __name__ == "__main__":
    main()
