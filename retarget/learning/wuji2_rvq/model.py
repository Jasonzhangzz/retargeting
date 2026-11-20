"""A pose-only MLP autoencoder with independent EMA residual codebooks."""
import torch
from torch import nn
from torch.nn import functional as F


def mlp(sizes):
    layers = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(sizes) - 2:
            layers.extend([nn.LayerNorm(b), nn.GELU()])
    return nn.Sequential(*layers)


class ResidualQuantizer(nn.Module):
    def __init__(self, dimension=64, levels=4, entries=256, decay=.99):
        super().__init__()
        self.dimension, self.levels, self.entries, self.decay = dimension, levels, entries, decay
        self.register_buffer('books', torch.zeros(levels, entries, dimension))
        self.register_buffer('counts', torch.ones(levels, entries))
        self.register_buffer('sums', torch.zeros(levels, entries, dimension))
        self.register_buffer('initialized', torch.tensor(False))
        self.register_buffer('updates', torch.zeros((), dtype=torch.long))

    @staticmethod
    def nearest(x, book):
        return (x.square().sum(-1, keepdim=True) + book.square().sum(-1)[None] - 2*x@book.T).argmin(-1)

    @torch.no_grad()
    def initialize(self, vectors, seed=20261118, iterations=20):
        x = vectors.detach().float()
        if len(x) < self.entries or not torch.isfinite(x).all():
            raise ValueError('Codebook initialization requires enough finite training vectors')
        generator = torch.Generator(device=x.device).manual_seed(seed)
        residual = x.clone()
        for layer in range(self.levels):
            book = residual[torch.randperm(len(x), generator=generator, device=x.device)[:self.entries]].clone()
            for _ in range(iterations):
                count = torch.zeros(self.entries, device=x.device)
                sums = torch.zeros_like(book)
                for chunk in residual.split(8192):
                    ids = self.nearest(chunk, book)
                    count.add_(torch.bincount(ids, minlength=self.entries))
                    sums.index_add_(0, ids, chunk)
                updated = sums / count.clamp_min(1)[:, None]
                empty = count == 0
                if empty.any():
                    updated[empty] = residual[torch.randint(len(x), (int(empty.sum()),), generator=generator, device=x.device)]
                book = updated
            self.books[layer].copy_(book)
            self.counts[layer].fill_(1)
            self.sums[layer].copy_(book)
            residual = residual - book[self.nearest(residual, book)]
        self.initialized.fill_(True)

    def lookup(self, tokens):
        if tokens.ndim != 2 or tokens.shape[1] != self.levels:
            raise ValueError(f'Expected tokens[B,{self.levels}]')
        if tokens.dtype not in (torch.int32, torch.int64, torch.uint8):
            raise ValueError('Tokens must contain integers')
        tokens = tokens.long()
        if tokens.numel() and ((tokens < 0).any() or (tokens >= self.entries).any()):
            raise ValueError('Token outside codebook')
        return torch.stack([self.books[i][tokens[:, i].long()] for i in range(self.levels)], 1)

    def forward(self, z, update=True):
        if not bool(self.initialized):
            raise RuntimeError('Initialize RVQ on training data first')
        residual = z.float()
        quantized, indices, commitments = [], [], []
        # Clone before EMA updates: autograd needs the books used for this forward.
        books = self.books.detach().clone()
        with torch.autocast(device_type=z.device.type, enabled=False):
            for layer in range(self.levels):
                ids = self.nearest(residual.detach(), books[layer])
                q = books[layer][ids]
                commitments.append(F.mse_loss(residual, q.detach()))
                quantized.append(q)
                indices.append(ids)
                if self.training and update:
                    with torch.no_grad():
                        count = torch.bincount(ids, minlength=self.entries).float()
                        sums = torch.zeros_like(books[layer]).index_add_(0, ids, residual.detach())
                        if torch.distributed.is_initialized():
                            torch.distributed.all_reduce(count)
                            torch.distributed.all_reduce(sums)
                        self.counts[layer].mul_(self.decay).add_(count, alpha=1-self.decay)
                        self.sums[layer].mul_(self.decay).add_(sums, alpha=1-self.decay)
                        n = self.counts[layer].sum()
                        smooth = (self.counts[layer] + 1e-5)/(n+self.entries*1e-5)*n
                        self.books[layer].copy_(self.sums[layer] / smooth[:, None])
                residual = residual - q.detach()
            if self.training and update:
                self.updates.add_(1)
        stack = torch.stack(quantized, 1)
        total = stack.sum(1)
        prefix = stack[:, :2].sum(1)
        return dict(quantized=z+(total-z).detach(), prefix=z+(prefix-z).detach(),
                    tokens=torch.stack(indices, 1), commitment=torch.stack(commitments).mean())


class PoseRVQ(nn.Module):
    def __init__(self, lower, upper, joint_names, dimension=64, levels=4, entries=256, decay=.99):
        super().__init__()
        low, high = torch.tensor(lower, dtype=torch.float32), torch.tensor(upper, dtype=torch.float32)
        if low.shape != (20,) or high.shape != (20,) or not torch.isfinite(torch.stack([low, high])).all() or not (high > low).all():
            raise ValueError('Expected 20 finite native joint bounds')
        if len(joint_names) != 20 or len(set(joint_names)) != 20:
            raise ValueError('Expected 20 unique joint names')
        self.config = dict(lower=list(lower), upper=list(upper), joint_names=list(joint_names),
                           dimension=dimension, levels=levels, entries=entries, decay=decay)
        self.register_buffer('lower', low)
        self.register_buffer('upper', high)
        self.encoder = mlp([20, 256, 256, dimension])
        self.quantizer = ResidualQuantizer(dimension, levels, entries, decay)
        self.decoder = mlp([dimension, 256, 256, 20])

    def normalize(self, q):
        if q.ndim != 2 or q.shape[1] != 20 or not torch.isfinite(q).all():
            raise ValueError('Expected finite qpos[B,20] in radians')
        return 2*(q-self.lower)/(self.upper-self.lower)-1

    def denormalize(self, x):
        return self.lower+(x+1)*.5*(self.upper-self.lower)

    def forward(self, q, continuous=False, update=True):
        x = self.normalize(q)
        z = self.encoder(x)
        out = dict(latent=z) if continuous else self.quantizer(z, update=update)
        out['latent'] = z
        out['normalized'] = self.decoder(z if continuous else out['quantized']).tanh()
        out['qpos'] = self.denormalize(out['normalized'])
        return out

    @torch.no_grad()
    def encode(self, q):
        return self.quantizer(self.encoder(self.normalize(q)), update=False)['tokens']

    def decode(self, tokens):
        z = self.quantizer.lookup(tokens).sum(1)
        return self.denormalize(self.decoder(z).tanh())


class PoseDecoder(nn.Module):
    """An independent token decoder: no encoder or class labels are required."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.register_buffer('books', torch.zeros(config['levels'], config['entries'], config['dimension']))
        self.register_buffer('lower', torch.tensor(config['lower'], dtype=torch.float32))
        self.register_buffer('upper', torch.tensor(config['upper'], dtype=torch.float32))
        self.decoder = mlp([config['dimension'], 256, 256, 20])

    @classmethod
    def from_model(cls, model):
        result = cls(model.config).to(model.lower.device)
        result.books.copy_(model.quantizer.books)
        result.decoder.load_state_dict(model.decoder.state_dict())
        return result.eval()

    def forward(self, tokens):
        if tokens.ndim != 2 or tokens.shape[1] != self.config['levels'] or tokens.dtype not in (torch.int32, torch.int64, torch.uint8):
            raise ValueError('Expected integer tokens[B,levels]')
        tokens = tokens.long()
        if tokens.numel() and ((tokens < 0).any() or (tokens >= self.config['entries']).any()):
            raise ValueError('Token outside codebook')
        z = torch.stack([self.books[i][tokens[:, i].long()] for i in range(self.config['levels'])]).sum(0)
        return self.lower+(self.decoder(z).tanh()+1)*.5*(self.upper-self.lower)


def supervised_contrastive(prefix, labels, temperature=.1):
    z = F.normalize(prefix.float(), dim=-1)
    logits = z@z.T/temperature
    diagonal = torch.eye(len(z), dtype=torch.bool, device=z.device)
    positive = (labels[:, None] == labels[None, :]) & ~diagonal
    denominator = torch.logsumexp(logits.masked_fill(diagonal, -torch.inf), dim=1)
    count = positive.sum(1)
    value = -((logits-denominator[:, None]).masked_fill(~positive, 0).sum(1)/count.clamp_min(1))
    return value[count > 0].mean() if (count > 0).any() else prefix.sum()*0
