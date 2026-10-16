"""
joint-align's methods as alignbench plugins (entry point group `alignbench.methods`):

  joint              fit_joint on every harvested anchor; one warp per batch into a shared frame
  star               fit_joint on the anchors that touch one hub batch
  composed-pairwise  fit_pairwise per batch pair with anchors, composed along shortest anchor paths
  neural-joint       the neural refinement on top of the joint model (needs torch; per sample)

Defaults follow script 30: λ = 'cv' picks the smoothness by RT-blocked held-out anchor error over
alignbench.metrics.anchors.LAMBDAS (TIC r is never used), and gauge = 'most_anchors' fixes the
batch with the most anchors. Every method takes lam= and gauge= to override.

    alignbench run --methods joint,composed-pairwise,pooled-pchip --dataset synthetic:k5 --seeds 0-4
"""
from __future__ import annotations

from collections.abc import Callable
from itertools import combinations
from typing import Any

import numpy as np
from alignbench.anchors import Anchors
from alignbench.data import Study
from alignbench.maps import compose_paths, edge_counts, from_latent, shortest_paths
from alignbench.methods import Fit
from alignbench.metrics.anchors import LAMBDAS, select_lambda

from jointalign.multibatch import JointWarp, fit_joint, fit_pairwise


def most_anchors(anchors: Anchors, K: int) -> int:
    """The batch with the most anchors to all others (script 30's gauge reference)."""
    n_pair = np.zeros((K, K), int)
    for a, b in combinations(range(K), 2):
        n_pair[a, b] = n_pair[b, a] = len(anchors.pair(a, b))
    return int(np.argmax(n_pair.sum(1)))


def _require(anchors: Anchors | None, name: str) -> Anchors:
    if anchors is None:
        raise ValueError(f'{name} needs anchors (alignbench.anchors.harvest(study))')
    return anchors


def _lam(lam: float | str, fit, anchors: Anchors, lambdas) -> tuple[float, dict]:
    if lam == 'cv':
        chosen, scores = select_lambda(fit, anchors, lambdas)
        return chosen, {'cv_median_abs_err_bins': scores}
    return float(lam), {}


def _gauge(gauge: int | str, anchors: Anchors, K: int) -> int | str:
    return most_anchors(anchors, K) if gauge == 'most_anchors' else gauge


class Joint:
    name = 'joint'
    needs_anchors = True

    def fit(self, study: Study, anchors: Anchors | None, lam: float | str = 'cv',
            gauge: int | str = 'most_anchors', solver: str = 'bvls', lambdas=LAMBDAS, **_: Any) -> Fit:
        anchors, K = _require(anchors, self.name), len(study)
        g = _gauge(gauge, anchors, K)
        lam_, extras = _lam(lam, lambda a, l: fit_joint(a, K, lam=l, gauge=g, solver=solver), anchors, lambdas)
        m = fit_joint(anchors, K, lam=lam_, gauge=g, solver=solver)
        return Fit(from_latent(m), self.name, {'lam': lam_, 'gauge': g, 'solver': solver},
                   used_anchors=np.ones(len(anchors), bool), model=m, extras=extras)


class Star:
    """The joint fit on the anchors touching the hub (hub='most_anchors', or a batch index);
    batches with no anchor to the hub stay at identity."""

    name = 'star'
    needs_anchors = True

    def fit(self, study: Study, anchors: Anchors | None, hub: int | str = 'most_anchors', lam: float | str = 'cv',
            gauge: int | str = 'hub', solver: str = 'bvls', lambdas=LAMBDAS, **_: Any) -> Fit:
        anchors, K = _require(anchors, self.name), len(study)
        h = most_anchors(anchors, K) if hub == 'most_anchors' else int(hub)
        g = h if gauge == 'hub' else _gauge(gauge, anchors, K)
        touch = (anchors.ba == h) | (anchors.bb == h)
        sub = anchors.subset(touch)
        lam_, extras = _lam(lam, lambda a, l: fit_joint(a, K, lam=l, gauge=g, solver=solver), sub, lambdas)
        m = fit_joint(sub, K, lam=lam_, gauge=g, solver=solver)
        return Fit(from_latent(m), self.name, {'hub': h, 'lam': lam_, 'gauge': g, 'solver': solver},
                   used_anchors=touch, model=m, extras=extras)


def _edge_map(m: JointWarp, k: int, l: int) -> Callable[[np.ndarray], np.ndarray]:
    return lambda t: m.map(k, l, t)


class ComposedPairwise:
    """fit_pairwise per batch pair with anchors (gauge 0, as script 30 and trustalign), composed
    along the shortest anchor paths (ties: most anchors). edges: the batch pairs to use."""

    name = 'composed-pairwise'
    needs_anchors = True

    def fit(self, study: Study, anchors: Anchors | None, lam: float | str = 'cv',
            edges: set[tuple[int, int]] | None = None, solver: str = 'bvls', lambdas=LAMBDAS, **_: Any) -> Fit:
        anchors, K = _require(anchors, self.name), len(study)
        if edges is None:
            edges = {(a, b) for a, b in combinations(range(K), 2) if len(anchors.pair(a, b))}
        edges = {(int(e[0]), int(e[1])) for e in edges}

        def build(anc: Anchors, l: float):
            maps: dict[tuple[int, int], Callable[[np.ndarray], np.ndarray]] = {}
            for a, b in edges:
                m = fit_pairwise(anc, a, b, K, lam=l, gauge=0, solver=solver)
                maps[(a, b)], maps[(b, a)] = _edge_map(m, a, b), _edge_map(m, b, a)
            paths = shortest_paths(K, edges, edge_counts(anc, edges))
            return Fit(compose_paths(maps, paths), self.name, extras={'paths': paths})

        lam_, extras = _lam(lam, build, anchors, lambdas)
        fit = build(anchors, lam_)
        fit.params = {'lam': lam_, 'edges': sorted(edges), 'solver': solver}
        fit.extras.update(extras)
        return fit


def _strict(w: np.ndarray) -> np.ndarray:
    """script 32's guard on the neural warps: non-decreasing, then strictly increasing."""
    return np.maximum.accumulate(w) + 1e-6 * np.arange(len(w))


class NeuralJoint:
    """The neural joint warp on top of Joint (script 32): one latent → sample map per sample, trained
    on anchors and dense groupwise similarity. Per sample; needs torch (pip install joint-align[neural])."""

    name = 'neural-joint'
    needs_anchors = True

    def fit(self, study: Study, anchors: Anchors | None, steps: int = 1500, seed: int = 0, lam: float | str = 'cv',
            gauge: int | str = 'most_anchors', solver: str = 'bvls', lambdas=LAMBDAS, **_: Any) -> Fit:
        try:
            import torch

            from jointalign import neural_joint as nj
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise ImportError('neural-joint needs torch: pip install "joint-align[neural]"') from e
        anchors = _require(anchors, self.name)
        joint = Joint().fit(study, anchors, lam=lam, gauge=gauge, solver=solver, lambdas=lambdas)
        model_j: JointWarp = joint.model  # type: ignore[assignment]
        K, ref = len(study), int(joint.params['gauge'])
        sizes = [len(b) for b in study.batches]
        offs = np.cumsum([0] + sizes[:-1])
        taus, g0 = nj.latent_grid_from_joint(model_j, K)
        batch_of = np.concatenate([[k] * n for k, n in enumerate(sizes)])
        sidx = np.full((K, max(sizes)), -1)
        for k in range(K):
            sidx[k, :sizes[k]] = offs[k] + np.arange(sizes[k])
        torch.manual_seed(seed)
        net = nj.NeuralJointWarp(taus, g0, batch_of, ref, sample_level=True)
        X = torch.tensor(np.concatenate([b.chromas for b in study.batches]))
        nj.train(net, X, anchors, model_j.weights, sidx, nj.TrainConfig(steps=steps))
        with torch.no_grad():
            G = np.stack([_strict(g) for g in net()[0].numpy()])

        def g(k: int, l: int, t: np.ndarray, *, i: int | None = None, j: int | None = None) -> np.ndarray:
            if i is None or j is None:
                raise ValueError('neural-joint is per sample: pass i (in batch k) and j (in batch l)')
            return np.interp(np.interp(t, G[offs[k] + i], taus), taus, G[offs[l] + j])

        return Fit(g, self.name, {**joint.params, 'steps': steps, 'seed': seed}, used_anchors=joint.used_anchors,
                   model=net, extras={'joint': joint, 'taus': taus, 'G': G}, per_sample=True)
