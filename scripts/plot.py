"""Plot learning curve and losses from a run's logs (eval data is not smoothed).

    python -m scripts.plot --run results/runs/p1_pong_s0 --random results/p0_random.json --out results/figures
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

THOUSANDS = FuncFormatter(lambda v, _: f"{v / 1000:.0f}")


def read_jsonl(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--random", default="results/p0_random.json")
    p.add_argument("--out", default="results/figures")
    a = p.parse_args()
    run, out = Path(a.run), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    evals, scores, mets = read_jsonl(run / "eval.jsonl"), read_jsonl(run / "scores.jsonl"), read_jsonl(run / "metrics.jsonl")
    rnd = json.loads(Path(a.random).read_text()) if Path(a.random).exists() else None
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})

    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    if scores:
        ax.scatter([s["step"] for s in scores], [s["return"] for s in scores], s=4, color="0.75",
                   label="training episodes", zorder=1)
    if evals:
        x = np.array([e["step"] for e in evals]); m = np.array([e["mean"] for e in evals]); s = np.array([e["std"] for e in evals])
        ax.fill_between(x, m - s, m + s, color="C0", alpha=0.2, lw=0)
        ax.plot(x, m, "o-", color="C0", ms=3, label=f"eval ({evals[-1]['episodes']} episodes)", zorder=3)
    if rnd:
        ax.axhline(rnd["mean"], color="C3", ls="--", lw=1, label="random policy")
    ax.set_xlabel("training steps (k)")
    ax.xaxis.set_major_formatter(THOUSANDS)
    ax.set_ylabel("episode return")
    ax.set_ylim(-21.5, max(-15, ax.get_ylim()[1]))
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "learning_curve.pdf"); fig.savefig(out / "learning_curve.png", dpi=200)

    if mets:
        x = [r["step"] for r in mets]
        keys = [("loss/rec", "reconstruction loss"), ("loss/reward", "reward loss"),
                ("loss/dyn", "dynamics KL"), ("actor/entropy", "actor entropy")]
        fig, axes = plt.subplots(2, 2, figsize=(3.4, 2.7))
        for ax, (k, name) in zip(axes.flat, keys):
            y = [r.get(k, np.nan) for r in mets]
            ax.plot(x, y, lw=1)
            ax.set_title(name, fontsize=7)
            ax.set_xlabel("training steps (k)", fontsize=6.5)
            ax.xaxis.set_major_formatter(THOUSANDS)
            ax.tick_params(labelsize=6.5)
            if k == "loss/rec":
                ax.set_yscale("log")
        fig.tight_layout()
        fig.savefig(out / "losses.pdf", bbox_inches="tight", pad_inches=0.02); fig.savefig(out / "losses.png", dpi=200)
    print("Saved figures to", out)


if __name__ == "__main__":
    main()
