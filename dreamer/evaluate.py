"""Evaluation episodes, shared by the random baseline, in-training eval and checkpoint eval."""
import numpy as np
import torch


class RandomPolicy:
    """Uniform random actions (baseline)."""

    def __init__(self, num_actions, seed=0):
        self.n, self.rng = num_actions, np.random.default_rng(seed)

    def reset(self):
        pass

    def __call__(self, obs):
        return int(self.rng.integers(self.n))


class AgentPolicy:
    """Wraps Dreamer: keeps the latent state across steps and samples actions from the policy."""

    def __init__(self, agent, device):
        self.agent, self.device, self.state = agent, device, None

    def reset(self):
        self.state = self.agent.initial_state(1)

    def __call__(self, obs):
        image = torch.from_numpy(obs["image"].copy())[None].to(self.device)
        is_first = torch.tensor([bool(obs["is_first"])], device=self.device)
        action, self.state = self.agent.act(image, is_first, self.state)
        return int(action.item())


def run_episodes(env, policy, episodes):
    """Run full episodes; returns a list of {return, length (agent steps), frames}. Nothing goes to replay."""
    results = []
    for _ in range(episodes):
        obs, policy_steps, score = env.reset(), 0, 0.0
        policy.reset()
        while True:
            action = policy(obs)
            obs = env.step(action)
            score += float(obs["reward"])
            policy_steps += 1
            if obs["is_last"]:
                break
        results.append(dict(**{"return": score}, length=policy_steps, frames=env.frames))
    return results


def summarize(results):
    returns = np.array([r["return"] for r in results])
    return dict(episodes=len(returns), mean=float(returns.mean()), std=float(returns.std()),
                min=float(returns.min()), max=float(returns.max()))
