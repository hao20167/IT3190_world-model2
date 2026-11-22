"""All hyperparameters in one flat dataclass; override with `--name value`.

    python -m scripts.train --logdir runs/p1 --steps 100000 --deter 256
"""
import argparse
import dataclasses
from dataclasses import dataclass


@dataclass
class Config:
    game: str = "pong"
    size: int = 64              # observation is size x size RGB
    repeat: int = 4             # action repeat (frames per action)
    noops: int = 30             # max random no-ops at reset
    max_frames: int = 108000    # episode length cap in emulator frames
    seed: int = 0

    steps: int = 100_000        # total agent steps
    prefill: int = 1000         # initial random-action steps (counted in `steps`)
    train_ratio: int = 256      # replayed timesteps trained per agent step
    batch: int = 16             # sequences per batch
    seq: int = 64               # sequence length
    eval_every: int = 10_000    # evaluate every N agent steps
    eval_episodes: int = 10
    save_every: int = 10_000    # save checkpoint (with replay) every N agent steps
    log_every: int = 1_000
    logdir: str = "runs/debug"
    device: str = "auto"        # auto (cuda if available, else cpu) | cuda | cpu
    amp: bool = False           # bf16 autocast (CUDA only)

    depth: int = 32             # first conv channels of encoder/decoder
    deter: int = 512            # GRU deterministic state size
    hidden: int = 512           # RSSM hidden size
    stoch: int = 32             # number of categorical latents
    classes: int = 32           # classes per latent
    units: int = 512            # hidden size of heads / actor / critic
    bins: int = 255             # twohot bins for reward and critic
    unimix: float = 0.01        # mix in 1% uniform to avoid zero probabilities

    scale_dyn: float = 1.0
    scale_rep: float = 0.1
    free_nats: float = 1.0

    horizon: int = 15           # imagination steps
    gamma: float = 0.997
    lam: float = 0.95
    entropy: float = 3e-4
    slow_rate: float = 0.02     # EMA rate of the slow critic
    slow_reg: float = 1.0
    ret_rate: float = 0.01      # EMA rate of return normalization (5-95 percentile)

    lr_model: float = 1e-4
    lr_actor: float = 3e-5
    lr_critic: float = 3e-5
    grad_clip: float = 100.0


def parse_args(argv=None) -> Config:
    """Build a Config, allowing each field to be overridden via --name value."""
    parser = argparse.ArgumentParser()
    for f in dataclasses.fields(Config):
        if f.type is bool:
            parser.add_argument(f"--{f.name}", type=lambda s: s.lower() in ("1", "true", "yes"),
                                default=f.default)
        else:
            parser.add_argument(f"--{f.name}", type=f.type, default=f.default)
    return Config(**vars(parser.parse_args(argv)))
