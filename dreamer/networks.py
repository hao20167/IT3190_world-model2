"""Network blocks: Encoder, Decoder, MLP heads and the RSSM."""
import torch
import torch.nn as nn

from .utils import probs_from_logits, sample_onehot


class MLP(nn.Module):
    """(Linear -> LayerNorm -> SiLU) x layers, then a Linear output."""

    def __init__(self, inp, out, units=512, layers=2, out_scale=1.0):
        super().__init__()
        mods, d = [], inp
        for _ in range(layers):
            mods += [nn.Linear(d, units), nn.LayerNorm(units), nn.SiLU()]
            d = units
        last = nn.Linear(d, out)
        with torch.no_grad():  # out_scale=0 zero-inits the output (reward/critic start at 0)
            last.weight.mul_(out_scale)
            last.bias.zero_()
        self.net = nn.Sequential(*mods, last)

    def forward(self, x):
        return self.net(x)


class Encoder(nn.Module):
    """64x64x3 image -> embedding; four stride-2 convs (64 -> 4)."""

    def __init__(self, depth=32):
        super().__init__()
        ch = [3, depth, depth * 2, depth * 4, depth * 8]
        layers = []
        for i in range(4):
            layers += [nn.Conv2d(ch[i], ch[i + 1], 4, 2, 1), nn.GroupNorm(1, ch[i + 1]), nn.SiLU()]
        self.net = nn.Sequential(*layers)
        self.out_dim = ch[-1] * 4 * 4

    def forward(self, image):  # image: [..., 64, 64, 3] uint8
        lead = image.shape[:-3]
        x = image.reshape(-1, *image.shape[-3:]).permute(0, 3, 1, 2).float() / 255 - 0.5
        return self.net(x).reshape(*lead, -1)


class Decoder(nn.Module):
    """Features (h_t, z_t) -> reconstructed image; mirrors the Encoder with transposed convs."""

    def __init__(self, feat_dim, depth=32):
        super().__init__()
        self.top = depth * 8
        self.fc = nn.Linear(feat_dim, self.top * 4 * 4)
        ch = [self.top, depth * 4, depth * 2, depth, 3]
        layers = [nn.GroupNorm(1, self.top), nn.SiLU()]
        for i in range(4):
            layers.append(nn.ConvTranspose2d(ch[i], ch[i + 1], 4, 2, 1))
            if i < 3:
                layers += [nn.GroupNorm(1, ch[i + 1]), nn.SiLU()]
        self.net = nn.Sequential(*layers)

    def forward(self, feat):  # returns image in [-0.5, 0.5], shape [..., 64, 64, 3]
        lead = feat.shape[:-1]
        x = self.fc(feat.reshape(-1, feat.shape[-1])).reshape(-1, self.top, 4, 4)
        x = self.net(x).permute(0, 2, 3, 1)
        return x.reshape(*lead, *x.shape[1:])


class RSSM(nn.Module):
    """Recurrent State-Space Model.

    State = h_t (deter, GRU over (h_{t-1}, z_{t-1}, a_{t-1})) and z_t (stoch categoricals, one-hot).
    z_t comes from the prior p(z_t | h_t) when imagining, or the posterior q(z_t | h_t, x_t) with a real observation.
    """

    def __init__(self, act_dim, embed_dim, deter=512, hidden=512, stoch=32, classes=32, unimix=0.01):
        super().__init__()
        self.act_dim, self.deter, self.stoch, self.classes, self.unimix = act_dim, deter, stoch, classes, unimix
        self.pre_gru = nn.Sequential(nn.Linear(stoch * classes + act_dim, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.gru = nn.GRUCell(hidden, deter)
        self.prior_net = nn.Sequential(
            nn.Linear(deter, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, stoch * classes))
        self.post_net = nn.Sequential(
            nn.Linear(deter + embed_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, stoch * classes))

    @property
    def feat_dim(self):
        return self.deter + self.stoch * self.classes

    def initial(self, batch, device):
        """All-zero initial state (h, z and previous action)."""
        return dict(
            deter=torch.zeros(batch, self.deter, device=device),
            stoch=torch.zeros(batch, self.stoch, self.classes, device=device),
            prevact=torch.zeros(batch, self.act_dim, device=device))

    def feat(self, state):
        """Features [h_t, z_t] fed to decoder, heads, actor and critic."""
        return torch.cat([state["deter"], state["stoch"].flatten(-2)], -1)

    def _advance(self, deter, stoch, prevact):
        x = self.pre_gru(torch.cat([stoch.flatten(-2), prevact], -1))
        return self.gru(x, deter)

    def _probs(self, net, x):
        logits = net(x).reshape(*x.shape[:-1], self.stoch, self.classes)
        return probs_from_logits(logits, self.unimix)

    def observe_step(self, state, embed, is_first):
        """One step with a real observation; state['prevact'] is the previous action.
        The state is zeroed first when is_first is True."""
        keep = (~is_first).float()[:, None]
        deter, prevact = state["deter"] * keep, state["prevact"] * keep
        stoch = state["stoch"] * keep[..., None]
        deter = self._advance(deter, stoch, prevact)
        prior = self._probs(self.prior_net, deter)
        post = self._probs(self.post_net, torch.cat([deter, embed], -1))
        stoch = sample_onehot(post)
        return dict(deter=deter, stoch=stoch, prevact=prevact), post, prior

    def imagine_step(self, state, action):
        """One imagined step: no observation, z_t is sampled from the prior."""
        deter = self._advance(state["deter"], state["stoch"], action)
        prior = self._probs(self.prior_net, deter)
        return dict(deter=deter, stoch=sample_onehot(prior), prevact=action)
