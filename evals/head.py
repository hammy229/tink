"""A small head trained with prototypical-network episodes: learn what separates one person's sounds from each other.

The frozen encoder knows speech in general. The hope is that a projection trained on many people's few-shot tasks
ignores what varies across takes of a sound and keeps what varies across sounds. Each task has its own labels, so
nothing about a person's meanings needs to carry over, only how to compare sounds.

Leakage rule: the person being tested is never in the training data, and neither is the person used for early stopping.
"""
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from evals.fewshot import EpisodeFn


@dataclass(frozen=True)
class HeadConfig:
    dim: int = 128
    hidden: int = 0  # 0 means a single linear projection
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-2
    temperature: float = 0.1
    episodes: int = 1500
    eval_every: int = 100
    patience: int = 5
    val_episodes: int = 40
    seed: int = 0


@dataclass(frozen=True)
class Task:
    """One person's clips, ready to sample episodes from."""

    X: torch.Tensor
    labels: np.ndarray
    sessions: np.ndarray
    episode: EpisodeFn


class Head(nn.Module):
    def __init__(self, in_dim: int, cfg: HeadConfig):
        super().__init__()
        layers: list[nn.Module] = [nn.Dropout(cfg.dropout)]
        if cfg.hidden:
            layers += [nn.Linear(in_dim, cfg.hidden), nn.GELU(), nn.Dropout(cfg.dropout)]
            in_dim = cfg.hidden
        layers.append(nn.Linear(in_dim, cfg.dim, bias=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.net(x), dim=-1)


def _episode_loss(head: Head, task: Task, rng: np.random.Generator, temperature: float) -> tuple[torch.Tensor, float]:
    ep = task.episode(task.labels, task.sessions, rng)
    ways = len(ep.enroll)
    support = head(task.X[torch.from_numpy(np.concatenate(ep.enroll))])
    y_s = torch.from_numpy(np.concatenate([np.full(len(i), k) for k, i in enumerate(ep.enroll)]))
    query = head(task.X[torch.from_numpy(np.concatenate(ep.test))])
    y_q = torch.from_numpy(np.concatenate([np.full(len(i), k) for k, i in enumerate(ep.test)]))
    mu = support.mean(dim=0, keepdim=True)
    protos = torch.stack([(support[y_s == k] - mu).mean(dim=0) for k in range(ways)])
    logits = nn.functional.normalize(query - mu, dim=-1) @ nn.functional.normalize(protos, dim=-1).T / temperature
    # Weight classes equally, matching the balanced accuracy we report.
    counts = torch.bincount(y_q, minlength=ways).float()
    loss = nn.functional.cross_entropy(logits, y_q, weight=counts.sum() / (ways * counts.clamp(min=1)))
    return loss, float((logits.argmax(dim=1) == y_q).float().mean())


def train_head(train: Sequence[Task], val: Sequence[Task], cfg: HeadConfig = HeadConfig()) -> tuple[Head, dict]:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    head = Head(train[0].X.shape[1], cfg)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best, best_state, stale, history = -np.inf, None, 0, []
    for step in range(1, cfg.episodes + 1):
        head.train()
        task = train[int(rng.integers(len(train)))]
        loss, _ = _episode_loss(head, task, rng, cfg.temperature)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % cfg.eval_every == 0:
            head.eval()
            val_rng = np.random.default_rng(cfg.seed + 1)  # same episodes each time, so the curve isn't noise
            with torch.no_grad():
                acc = float(np.mean([_episode_loss(head, t, val_rng, cfg.temperature)[1]
                                     for t in val for _ in range(cfg.val_episodes)]))
            history.append((step, acc))
            if acc > best:
                best, stale = acc, 0
                best_state = {k: v.clone() for k, v in head.state_dict().items()}
            else:
                stale += 1
                if stale >= cfg.patience:
                    break
    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()
    return head, {"val_accuracy": best, "steps": history[-1][0] if history else 0}


def apply(head: Head, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return head(torch.from_numpy(X)).numpy()
