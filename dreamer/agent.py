"""Dreamer agent: world model loss, imagination, and actor-critic updates."""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .networks import MLP, Decoder, Encoder, RSSM
from .utils import (ReturnNorm, TwoHot, entropy_categorical, kl_categorical,
                    probs_from_logits, sample_hard)


class Dreamer(nn.Module):
    def __init__(self, act_dim, cfg):
        super().__init__()
        self.cfg, self.act_dim = cfg, act_dim
        self.encoder = Encoder(cfg.depth)
        self.rssm = RSSM(act_dim, self.encoder.out_dim, cfg.deter, cfg.hidden, cfg.stoch, cfg.classes, cfg.unimix)
        feat = self.rssm.feat_dim
        self.decoder = Decoder(feat, cfg.depth)
        self.reward_head = MLP(feat, cfg.bins, cfg.units, layers=1, out_scale=0.0)
        self.cont_head = MLP(feat, 1, cfg.units, layers=1)
        self.actor = MLP(feat, act_dim, cfg.units, layers=3, out_scale=0.01)
        self.critic = MLP(feat, cfg.bins, cfg.units, layers=3, out_scale=0.0)
        self.slow_critic = copy.deepcopy(self.critic).requires_grad_(False)

        self.twohot = TwoHot(cfg.bins)
        self.ret_norm = ReturnNorm(cfg.ret_rate)
        world = [self.encoder, self.rssm, self.decoder, self.reward_head, self.cont_head]
        self.world_params = [p for m in world for p in m.parameters()]
        self.opt_model = torch.optim.Adam(self.world_params, lr=cfg.lr_model)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr_actor)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr_critic)

    def initial_state(self, batch=1):
        return self.rssm.initial(batch, next(self.parameters()).device)

    def _policy_probs(self, feat):
        return probs_from_logits(self.actor(feat), self.cfg.unimix)

    @torch.no_grad()
    def act(self, image, is_first, state):
        """Pick an action for one real observation. Returns (action, new state)."""
        embed = self.encoder(image)
        state, _, _ = self.rssm.observe_step(state, embed, is_first)
        probs = self._policy_probs(self.rssm.feat(state))
        action = torch.distributions.Categorical(probs=probs, validate_args=False).sample()
        state["prevact"] = F.one_hot(action, self.act_dim).float()
        return action, state

    def world_model_loss(self, batch):
        """batch: image [B,T,64,64,3] u8, action [B,T] int, reward [B,T], is_first/is_terminal [B,T] bool."""
        cfg = self.cfg
        B, T = batch["reward"].shape
        embed = self.encoder(batch["image"])
        # previous action a_{t-1} for each step t (zeros at the first step)
        action = F.one_hot(batch["action"], self.act_dim).float()
        prevact = torch.cat([torch.zeros_like(action[:, :1]), action[:, :-1]], 1)

        state = self.initial_state(B)
        states, posts, priors = [], [], []
        for t in range(T):
            state = {**state, "prevact": prevact[:, t]}
            state, post, prior = self.rssm.observe_step(state, embed[:, t], batch["is_first"][:, t])
            states.append(state), posts.append(post), priors.append(prior)
        feat = torch.stack([self.rssm.feat(s) for s in states], 1)          # [B,T,F]
        post, prior = torch.stack(posts, 1), torch.stack(priors, 1)         # [B,T,stoch,classes]

        recon = self.decoder(feat)
        target = batch["image"].float() / 255 - 0.5
        rec_loss = ((recon.float() - target) ** 2).sum((-3, -2, -1))
        rew_loss = self.twohot.loss(self.reward_head(feat), batch["reward"])
        cont_target = 1.0 - batch["is_terminal"].float()
        cont_loss = F.binary_cross_entropy_with_logits(
            self.cont_head(feat).squeeze(-1).float(), cont_target, reduction="none")

        # free nats: KL below the threshold gets no gradient
        dyn = kl_categorical(post.detach(), prior).sum(-1)    # trains the prior
        rep = kl_categorical(post, prior.detach()).sum(-1)    # trains the posterior (weighted lower)
        dyn_loss = dyn.clamp(min=cfg.free_nats)
        rep_loss = rep.clamp(min=cfg.free_nats)

        loss = (rec_loss + rew_loss + cont_loss + cfg.scale_dyn * dyn_loss + cfg.scale_rep * rep_loss).mean()
        metrics = {
            "loss/rec": rec_loss.mean(), "loss/reward": rew_loss.mean(), "loss/cont": cont_loss.mean(),
            "loss/dyn": dyn.mean(), "loss/rep": rep.mean(),
            "latent/post_entropy": entropy_categorical(post).sum(-1).mean(),
            "latent/prior_entropy": entropy_categorical(prior).sum(-1).mean(),
        }
        # detached real latent states are the starting points for imagination
        starts = {k: torch.stack([s[k] for s in states], 1).flatten(0, 1).detach() for k in ("deter", "stoch", "prevact")}
        return loss, metrics, starts

    @torch.no_grad()
    def imagine(self, start):
        """Roll out `horizon` imagined steps. No gradients: the discrete actor uses REINFORCE,
        so log-probs of the chosen actions are recomputed later."""
        feats, acts, state = [], [], start
        for _ in range(self.cfg.horizon):
            feat = self.rssm.feat(state)
            action = sample_hard(self._policy_probs(feat))
            feats.append(feat), acts.append(action)
            state = self.rssm.imagine_step(state, action)
        feats.append(self.rssm.feat(state))
        return torch.stack(feats, 1), torch.stack(acts, 1)  # [N,H+1,F], [N,H,A]

    def lambda_return(self, reward, disc, value):
        """R_t = r_{t+1} + disc_{t+1} * ((1-lam) * v_{t+1} + lam * R_{t+1}),  R_H = v_H.
        reward, disc: [N,H] (steps 1..H);  value: [N,H+1]."""
        lam = self.cfg.lam
        ret = value[:, -1]
        out = []
        for t in reversed(range(reward.shape[1])):
            ret = reward[:, t] + disc[:, t] * ((1 - lam) * value[:, t + 1] + lam * ret)
            out.append(ret)
        return torch.stack(out[::-1], 1)  # [N,H]

    def behavior_loss(self, starts):
        cfg = self.cfg
        feats, acts = self.imagine(starts)
        with torch.no_grad():
            reward = self.twohot.mean(self.reward_head(feats[:, 1:]))
            cont = torch.sigmoid(self.cont_head(feats[:, 1:]).squeeze(-1).float())
            disc = cfg.gamma * cont
            value = self.twohot.mean(self.critic(feats))
            ret = self.lambda_return(reward, disc, value)                         # [N,H]
            # down-weight later steps in case the episode already ended
            weight = torch.cumprod(torch.cat([torch.ones_like(disc[:, :1]), disc[:, :-1]], 1), 1)
            self.ret_norm.update(ret)
            scale = self.ret_norm.scale()

        # actor: REINFORCE with normalized advantage + entropy bonus
        probs = self._policy_probs(feats[:, :-1])
        logp = (acts * probs.log()).sum(-1)
        adv = ((ret - value[:, :-1]) / scale).detach()
        actor_loss = -(weight * (logp * adv + cfg.entropy * entropy_categorical(probs))).mean()

        # critic: twohot regression to the return, regularized toward the slow (EMA) critic
        logits = self.critic(feats[:, :-1])
        slow = F.softmax(self.slow_critic(feats[:, :-1]).float(), -1)
        critic_loss = self.twohot.loss(logits, ret) - cfg.slow_reg * (slow * F.log_softmax(logits.float(), -1)).sum(-1)
        critic_loss = (weight * critic_loss).mean()

        metrics = {
            "loss/actor": actor_loss, "loss/critic": critic_loss,
            "imag/return": ret.mean(), "imag/value": value.mean(), "imag/reward": reward.mean(),
            "imag/ret_scale": scale, "actor/entropy": entropy_categorical(probs).mean(),
        }
        return actor_loss, critic_loss, metrics

    def train_step(self, batch):
        autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.cfg.amp)
        with autocast:
            loss, metrics, starts = self.world_model_loss(batch)
        self.opt_model.zero_grad(set_to_none=True)
        loss.backward()
        metrics["grad/model"] = nn.utils.clip_grad_norm_(self.world_params, self.cfg.grad_clip)
        self.opt_model.step()

        with autocast:
            actor_loss, critic_loss, mets = self.behavior_loss(starts)
        self.opt_actor.zero_grad(set_to_none=True)
        self.opt_critic.zero_grad(set_to_none=True)
        (actor_loss + critic_loss).backward()  # parameter sets are disjoint, so one backward is fine
        metrics["grad/actor"] = nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip)
        metrics["grad/critic"] = nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.grad_clip)
        self.opt_actor.step()
        self.opt_critic.step()
        self._update_slow_critic()

        metrics.update(mets)
        metrics["loss/world"] = loss
        return {k: float(v.detach()) if torch.is_tensor(v) else float(v) for k, v in metrics.items()}

    @torch.no_grad()
    def _update_slow_critic(self):
        r = self.cfg.slow_rate
        for slow, fast in zip(self.slow_critic.parameters(), self.critic.parameters()):
            slow.mul_(1 - r).add_(fast, alpha=r)

    def checkpoint(self, optimizers=True):
        """optimizers=False saves only weights and return stats (enough for evaluation, ~3x smaller)."""
        ckpt = {"model": self.state_dict(), "ret_norm": self.ret_norm.state_dict()}
        if optimizers:
            ckpt.update(opt_model=self.opt_model.state_dict(), opt_actor=self.opt_actor.state_dict(),
                        opt_critic=self.opt_critic.state_dict())
        return ckpt

    def restore(self, ckpt, optimizers=True):
        self.load_state_dict(ckpt["model"])
        if ckpt["ret_norm"]["low"] is not None:
            self.ret_norm.load_state_dict(ckpt["ret_norm"])
        if optimizers:
            self.opt_model.load_state_dict(ckpt["opt_model"])
            self.opt_actor.load_state_dict(ckpt["opt_actor"])
            self.opt_critic.load_state_dict(ckpt["opt_critic"])
