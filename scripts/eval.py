"""Evaluate a checkpoint on a fresh environment; model config is read from <logdir>/config.json.

    python -m scripts.eval --ckpt runs/p1/ckpt/latest.pt --episodes 30 --seed 123 --out results/p3_final.json
"""
import argparse
import json
from pathlib import Path

import torch

from dreamer.agent import Dreamer
from dreamer.config import Config
from dreamer.env import Atari
from dreamer.evaluate import AgentPolicy, run_episodes, summarize
from scripts.train import pick_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    ckpt_path = Path(a.ckpt)
    cfg = Config(**json.loads((ckpt_path.parent.parent / "config.json").read_text()))
    device = pick_device(a.device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    env = Atari(cfg.game, cfg.size, cfg.repeat, cfg.noops, cfg.max_frames, seed=a.seed)
    agent = Dreamer(env.num_actions, cfg).to(device)
    agent.restore(ckpt, optimizers=False)
    agent.eval()
    torch.manual_seed(a.seed)
    res = run_episodes(env, AgentPolicy(agent, device), a.episodes)
    # latest.pt stores the step; model_XXXXXXX.pt carries it in the file name
    train_step = ckpt.get("step")
    if train_step is None and ckpt_path.stem.startswith("model_"):
        train_step = int(ckpt_path.stem.split("_")[1])
    out = dict(policy="dreamer", ckpt=str(ckpt_path), train_step=train_step, seed=a.seed,
               **summarize(res), results=res)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"dreamer: {out['mean']:.2f} ± {out['std']:.2f} ({a.episodes} episodes)")


if __name__ == "__main__":
    main()
