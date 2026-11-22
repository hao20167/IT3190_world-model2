"""Random vs Dreamer summary table from the eval JSON files and run logs.

    python -m scripts.make_tables --random results/p0_random.json --final results/p3_final.json \
        --run results/runs/p1_pong_s0 --out results/summary.json

Writes JSON and prints a Markdown table. std is over eval episodes of a single training seed.
"""
import argparse
import json
from pathlib import Path


def read_jsonl(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--random", default="results/p0_random.json")
    p.add_argument("--final", default="results/p3_final.json")
    p.add_argument("--run", default="results/runs/p1_pong_s0")
    p.add_argument("--out", default="results/summary.json")
    a = p.parse_args()
    rnd = json.loads(Path(a.random).read_text())
    run = Path(a.run)
    cfg = json.loads((run / "config.json").read_text())
    evals, scores, mets = read_jsonl(run / "eval.jsonl"), read_jsonl(run / "scores.jsonl"), read_jsonl(run / "metrics.jsonl")
    final = json.loads(Path(a.final).read_text()) if Path(a.final).exists() else None

    rows = [dict(name="Random", train_steps=0, seeds="-", episodes=rnd["episodes"], mean=rnd["mean"], std=rnd["std"],
                 min=rnd["min"], max=rnd["max"])]
    if final:
        rows.append(dict(name="Dreamer (final checkpoint)", train_steps=final["train_step"], seeds=1,
                         episodes=final["episodes"], mean=final["mean"], std=final["std"], min=final["min"], max=final["max"]))
    last = mets[-1] if mets else {}
    summary = dict(
        table=rows,
        eval_curve=[dict(step=e["step"], mean=e["mean"], std=e["std"], episodes=e["episodes"]) for e in evals],
        best_eval=max(evals, key=lambda e: e["mean"]) if evals else None,
        train=dict(agent_steps=last.get("step"), frames=last.get("frames"), updates=last.get("updates"),
                   train_episodes=len(scores), wall_seconds=last.get("seconds"),
                   eval_seconds_total=sum(e.get("eval_seconds", 0) for e in evals)),
        config=cfg,
    )
    Path(a.out).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("| Agent | Training agent steps | Seed | Episodes | Return (mean ± std) | min / max |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['name']} | {r['train_steps']} | {r['seeds']} | {r['episodes']} | "
              f"{r['mean']:.2f} ± {r['std']:.2f} | {r['min']:.0f} / {r['max']:.0f} |")
    print("Training:", summary["train"])


if __name__ == "__main__":
    main()
