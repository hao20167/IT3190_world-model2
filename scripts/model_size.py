"""Count parameters per model component -> results/model_size.json.

    python -m scripts.model_size --run results/runs/p1_pong_s0
"""
import argparse
import json
from pathlib import Path

from dreamer.agent import Dreamer
from dreamer.config import Config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="results/runs/p1_pong_s0")
    p.add_argument("--out", default="results/model_size.json")
    a = p.parse_args()
    cfg = Config(**json.loads((Path(a.run) / "config.json").read_text()))
    agent = Dreamer(6, cfg)
    count = lambda m: sum(x.numel() for x in m.parameters())
    parts = {k: count(getattr(agent, k)) for k in
             ("encoder", "rssm", "decoder", "reward_head", "cont_head", "actor", "critic")}
    out = dict(parts=parts, trainable=sum(parts.values()), slow_critic=count(agent.slow_critic),
               world_model=sum(parts[k] for k in ("encoder", "rssm", "decoder", "reward_head", "cont_head")))
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(out)


if __name__ == "__main__":
    main()
