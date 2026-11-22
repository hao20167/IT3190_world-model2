"""Train Dreamer. Re-running with the same --logdir resumes from the latest checkpoint.

    python -m scripts.train --logdir runs/p1_pong_s0 --steps 100000

Units: agent step (one observation + action, the unit of --steps), frame (= agent step x repeat),
gradient update (train_ratio / (batch * seq) per agent step).
"""
import dataclasses
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from dreamer.agent import Dreamer
from dreamer.config import parse_args
from dreamer.env import Atari
from dreamer.evaluate import AgentPolicy, run_episodes, summarize
from dreamer.replay import Replay


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    # auto never picks MPS: it crashed in nn.Linear (MPSNDArray error) on PyTorch 2.15 / macOS 14;
    # pass --device mps explicitly to try it.
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def append_jsonl(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def atomic_save(obj, path):
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)  # atomic rename: a crash mid-save leaves the old checkpoint intact


def to_device(batch, device):
    return {k: torch.from_numpy(v).to(device) for k, v in batch.items()}


def main():
    cfg = parse_args()
    logdir = Path(cfg.logdir)
    (logdir / "ckpt").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = pick_device(cfg.device)

    env_kw = dict(game=cfg.game, size=cfg.size, repeat=cfg.repeat, noops=cfg.noops, max_frames=cfg.max_frames)
    env = Atari(seed=cfg.seed, **env_kw)
    eval_env = Atari(seed=cfg.seed + 1000, **env_kw)  # separate eval env, never written to replay
    agent = Dreamer(env.num_actions, cfg).to(device)
    replay = Replay(cfg.steps + 1000, cfg.size, seed=cfg.seed)
    rng = np.random.default_rng(cfg.seed + 1)

    step, updates = 0, 0
    ckpt_path = logdir / "ckpt" / "latest.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        agent.restore(ckpt)
        step, updates = ckpt["step"], ckpt["updates"]
        replay.load(logdir / "ckpt" / "replay.npz")
        assert len(replay) >= step, f"replay ({len(replay)}) is shorter than checkpoint ({step})"
        replay.n = step  # one row per agent step; drop anything newer than the checkpoint
        torch.manual_seed(cfg.seed + step)  # avoid replaying the same random stream as the first run
        replay.rng = np.random.default_rng(cfg.seed + step)
        rng = np.random.default_rng(cfg.seed + 1 + step)
        print(f"[resume] step={step} updates={updates} replay={len(replay)}", flush=True)
    else:
        (logdir / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))
    n_params = sum(p.numel() for p in agent.parameters())
    print(f"device={device} params={n_params / 1e6:.2f}M actions={env.num_actions}", flush=True)

    def save():
        # save replay BEFORE latest.pt: latest.pt marks a complete save point
        replay.save(logdir / "ckpt" / "replay_tmp.npz")
        os.replace(logdir / "ckpt" / "replay_tmp.npz", logdir / "ckpt" / "replay.npz")
        atomic_save(dict(**agent.checkpoint(), step=step, updates=updates), ckpt_path)

    def evaluate():
        t0 = time.time()
        res = run_episodes(eval_env, AgentPolicy(agent, device), cfg.eval_episodes)
        row = dict(step=step, frames=step * cfg.repeat, updates=updates, eval_seconds=time.time() - t0,
                   returns=[r["return"] for r in res], lengths=[r["length"] for r in res], **summarize(res))
        append_jsonl(logdir / "eval.jsonl", row)
        atomic_save(agent.checkpoint(optimizers=False), logdir / "ckpt" / f"model_{step:07d}.pt")  # for offline re-evaluation
        print(f"[eval] step={step} return={row['mean']:.2f}±{row['std']:.2f}", flush=True)

    policy = AgentPolicy(agent, device)
    obs = env.reset()  # the episode in progress before a resume is lost: start a new one
    policy.reset()
    ep_return, ep_len, agg, t_start, step_start = 0.0, 0, {}, time.time(), step
    updates_due, per_step = 0.0, cfg.train_ratio / (cfg.batch * cfg.seq)
    next_eval = (step // cfg.eval_every + 1) * cfg.eval_every
    next_save = (step // cfg.save_every + 1) * cfg.save_every
    next_log = (step // cfg.log_every + 1) * cfg.log_every
    if step == 0:
        evaluate()  # step-0 evaluation (untrained agent)

    while step < cfg.steps:
        action = int(rng.integers(env.num_actions)) if step < cfg.prefill else policy(obs)
        replay.add(obs, action)
        step += 1
        ep_len += 1
        ep_return += float(obs["reward"])
        if obs["is_last"]:  # log the finished episode, then reset
            append_jsonl(logdir / "scores.jsonl", dict(step=step, frames=step * cfg.repeat,
                                                       **{"return": ep_return}, length=ep_len))
            obs, ep_return, ep_len = env.reset(), 0.0, 0
            policy.reset()
        else:
            obs = env.step(action)

        if step >= cfg.prefill:
            updates_due += per_step
            while updates_due >= 1:
                updates_due -= 1
                mets = agent.train_step(to_device(replay.sample(cfg.batch, cfg.seq), device))
                updates += 1
                for k, v in mets.items():
                    agg.setdefault(k, []).append(v)
                if not all(np.isfinite(v) for v in mets.values()):
                    raise RuntimeError(f"Non-finite metric at update {updates}: {mets}")

        if step >= next_log:
            next_log += cfg.log_every
            elapsed = time.time() - t_start
            row = dict(step=step, frames=step * cfg.repeat, updates=updates,
                       sps=(step - step_start) / elapsed, seconds=elapsed)
            row.update({k: float(np.mean(v)) for k, v in agg.items()})
            append_jsonl(logdir / "metrics.jsonl", row)
            print(f"step={step} upd={updates} sps={row['sps']:.1f} "
                  f"rec={row.get('loss/rec', 0):.1f} dyn={row.get('loss/dyn', 0):.2f}", flush=True)
            agg = {}
        if step >= next_eval:
            next_eval += cfg.eval_every
            evaluate()
        if step >= next_save:
            next_save += cfg.save_every
            save()

    save()
    (logdir / "DONE").write_text(f"step={step} updates={updates}\n")
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
