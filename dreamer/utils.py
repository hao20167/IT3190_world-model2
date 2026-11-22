"""Small shared helpers: symlog, twohot, categorical distributions, return normalization."""
import torch
import torch.nn.functional as F


def symlog(x):
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x):
    return torch.sign(x) * torch.expm1(x.abs())


class TwoHot:
    """Encodes a scalar as a distribution over evenly spaced bins in symlog space.

    x between two neighbouring bins puts its weight on those two bins. The head predicts bin logits,
    the loss is cross-entropy, and the prediction is symexp of the expected bin value.
    """

    def __init__(self, bins=255, low=-20.0, high=20.0):
        self.bins = torch.linspace(low, high, bins)

    def encode(self, x):
        bins = self.bins.to(x.device)
        x = symlog(x).clamp(bins[0], bins[-1])
        below = ((x[..., None] >= bins).sum(-1) - 1).clamp(0, len(bins) - 2)
        above = below + 1
        w_above = (x - bins[below]) / (bins[above] - bins[below])
        target = F.one_hot(below, len(bins)).float() * (1 - w_above[..., None])
        target = target + F.one_hot(above, len(bins)).float() * w_above[..., None]
        return target

    def loss(self, logits, x):
        """Cross-entropy between logits and the twohot encoding of x."""
        return -(self.encode(x) * F.log_softmax(logits.float(), -1)).sum(-1)

    def mean(self, logits):
        bins = self.bins.to(logits.device)
        return symexp((F.softmax(logits.float(), -1) * bins).sum(-1))


def probs_from_logits(logits, unimix=0.01):
    """Softmax mixed with a uniform distribution (unimix) so probabilities are never 0 and KL stays finite."""
    p = F.softmax(logits.float(), -1)
    return (1 - unimix) * p + unimix / p.shape[-1]


def sample_onehot(probs):
    """Sample a one-hot with straight-through gradients to `probs`."""
    idx = torch.distributions.Categorical(probs=probs, validate_args=False).sample()
    onehot = F.one_hot(idx, probs.shape[-1]).float()
    return onehot + probs - probs.detach()


def sample_hard(probs):
    """Sample a one-hot without gradients (actor actions)."""
    idx = torch.distributions.Categorical(probs=probs, validate_args=False).sample()
    return F.one_hot(idx, probs.shape[-1]).float()


def kl_categorical(p, q):
    """KL(p || q) for categoricals, summed over the last (class) dimension."""
    return (p * (p.log() - q.log())).sum(-1)


def entropy_categorical(p):
    return -(p * p.log()).sum(-1)


class ReturnNorm:
    """EMA estimate of the 5th-95th percentile range of returns.

    Advantages are divided by max(1, range): small rewards are not amplified, large ones are shrunk.
    """

    def __init__(self, rate=0.01):
        self.rate = rate
        self.low = None
        self.high = None

    @torch.no_grad()
    def update(self, returns):
        flat = returns.detach().float().flatten()
        low, high = torch.quantile(flat, 0.05), torch.quantile(flat, 0.95)
        if self.low is None:
            self.low, self.high = low, high
        else:
            self.low = (1 - self.rate) * self.low + self.rate * low
            self.high = (1 - self.rate) * self.high + self.rate * high

    def scale(self):
        return torch.clamp(self.high - self.low, min=1.0)

    def state_dict(self):
        return {"low": self.low, "high": self.high}

    def load_state_dict(self, d):
        self.low, self.high = d["low"], d["high"]
