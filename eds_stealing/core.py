"""
eds_stealing.core
=================

A geometric / exterior-differential-systems (EDS) lens on the last-layer
model-stealing attack of Carlini et al. (2024), "Stealing Part of a Production
Language Model" (arXiv:2403.06634), together with an explicit identifiability
analysis of the layer *beneath* the unembedding.

Everything here runs on a small, fully controlled toy "production" model so the
ground truth is known. The functions are deliberately plain NumPy so each claim
in the accompanying note can be reproduced and checked line by line.

Notation (matching the note):
    l : vocabulary size           h : residual / hidden width
    W : (l, h) unembedding matrix (the "embedding projection layer")
    g : post-normalization hidden state, lies on a sphere in R^h
    z = W g : logits returned by the (idealized full-logit) oracle
"""

from __future__ import annotations
import numpy as np


# --------------------------------------------------------------------------- #
#  Normalization
# --------------------------------------------------------------------------- #
def rmsnorm(x: np.ndarray) -> np.ndarray:
    """RMSNorm: map rows of x onto the sphere of radius sqrt(h)."""
    return x / np.linalg.norm(x, axis=-1, keepdims=True) * np.sqrt(x.shape[-1])


def layernorm(x: np.ndarray) -> np.ndarray:
    """LayerNorm: center, then RMSNorm. Image lives on a sphere inside {1^T x = 0}."""
    x = x - x.mean(axis=-1, keepdims=True)
    return x / np.linalg.norm(x, axis=-1, keepdims=True) * np.sqrt(x.shape[-1])


# --------------------------------------------------------------------------- #
#  Toy "production" models
# --------------------------------------------------------------------------- #
class OneLayerModel:
    """norm(.) -> W.  The activations fill the sphere; this is the regime in
    which Carlini et al.'s last-layer attack is exact."""

    def __init__(self, l: int = 2000, h: int = 64, norm="rms", seed: int = 0):
        self.l, self.h = l, h
        self.norm = rmsnorm if norm == "rms" else layernorm
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((l, h)) / np.sqrt(h)
        self.rng = rng

    def query(self, n: int):
        """Return (Z, g) with Z = logits (n, l) and g the hidden states (n, h)."""
        g = self.norm(self.rng.standard_normal((n, self.h)))
        return g @ self.W.T, g


class TwoLayerModel:
    """k-dim content -> MLP block (W2 . phi(W1 .)) + residual -> norm -> W.

    The attainable hidden states now lie on a *curved* k-dimensional submanifold
    of the sphere whose linear span can be much larger than k. This is the
    structure that the last-layer attack reports as a featureless linear layer.
    """

    def __init__(self, l=2000, h=64, k=8, m=32, norm="rms", phi=np.tanh, seed=1):
        self.l, self.h, self.k, self.m = l, h, k, m
        self.norm = rmsnorm if norm == "rms" else layernorm
        self.phi = phi
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((l, h)) / np.sqrt(h)
        self.B = rng.standard_normal((h, k)) / np.sqrt(k)      # input subspace
        self.W1 = rng.standard_normal((m, h)) / np.sqrt(h)
        self.W2 = rng.standard_normal((h, m)) / np.sqrt(m)
        self.rng = rng

    def hidden(self, s: np.ndarray) -> np.ndarray:
        x = s @ self.B.T
        gpre = x + self.phi(x @ self.W1.T) @ self.W2.T
        return self.norm(gpre)

    def query(self, n: int):
        s = self.rng.standard_normal((n, self.k))
        g = self.hidden(s)
        return g @ self.W.T, g


# --------------------------------------------------------------------------- #
#  Step 1 -- the ideal, degree-1 part: hidden dimension from the spectral gap
# --------------------------------------------------------------------------- #
def recover_dimension(Z: np.ndarray):
    """Return (h_hat, singular_values). h_hat is the index of the largest
    multiplicative gap in the singular spectrum of Q = Z^T (the rank/SVD
    signal; degree-1 generators of the ideal)."""
    sv = np.linalg.svd(Z.T, compute_uv=False)
    logs = np.log(sv + 1e-300)
    gaps = logs[:-1] - logs[1:]
    return int(np.argmax(gaps)) + 1, sv


# --------------------------------------------------------------------------- #
#  Step 2 -- the ideal, degree-2 part: the quadric (ellipsoid) the logits lie on
# --------------------------------------------------------------------------- #
def fit_quadric(Z: np.ndarray, h: int):
    """Recover U (orthonormal basis of the recovered subspace V) and the
    symmetric form A with x^T A x = 1 for x = U^T z. Returns (U, A, diagnostics)."""
    U = np.linalg.svd(Z.T, full_matrices=False)[0][:, :h]
    X = U.T @ Z.T                                   # (h, n) projected logits
    iu = np.triu_indices(h)

    def design_row(x):
        M = np.outer(x, x)
        M = M + M.T - np.diag(np.diag(M))           # symmetric vectorization
        return M[iu]

    D = np.stack([design_row(X[:, i]) for i in range(X.shape[1])])
    a, *_ = np.linalg.lstsq(D, np.ones(X.shape[1]), rcond=None)
    A = np.zeros((h, h)); A[iu] = a; A = (A + A.T) - np.diag(np.diag(A))

    quad = np.einsum("in,ij,jn->n", X, A, X)        # should be ~1
    eig = np.linalg.eigvalsh(A)
    design_rank = int((np.linalg.svd(D, compute_uv=False) > 1e-10 *
                       np.linalg.svd(D, compute_uv=False)[0]).sum())
    diag = dict(quad_residual=np.abs(quad - 1).max(),
                eig_min=eig.min(), eig_max=eig.max(),
                pd=bool(eig.min() > 0),
                design_rank=design_rank, n_unknowns=len(iu[0]))
    return U, A, diag


# --------------------------------------------------------------------------- #
#  Step 3a -- polar space == tangent space of the output manifold
# --------------------------------------------------------------------------- #
def polar_vs_tangent(model: OneLayerModel, U: np.ndarray, A: np.ndarray, eps=1e-6):
    """Cross-check: the polar space (A z0)^perp of the recovered quadric equals
    the manifold's tangent space, computed independently by finite differences.
    Returns |cos(angle)| between the data-tangent and the recovered polar normal
    (should be ~0)."""
    h = model.h
    g0 = model.norm(model.rng.standard_normal((1, h)))[0]
    d = model.rng.standard_normal(h); d -= (d @ g0) / (g0 @ g0) * g0
    g1 = g0 + eps * d; g1 = g1 / np.linalg.norm(g1) * np.sqrt(h)
    z0, z1 = model.W @ g0, model.W @ g1
    x0 = U.T @ z0
    tangent = U.T @ (z1 - z0); tangent /= np.linalg.norm(tangent)
    polar_normal = A @ x0; polar_normal /= np.linalg.norm(polar_normal)
    return abs(tangent @ polar_normal)


# --------------------------------------------------------------------------- #
#  Step 3b -- recover W up to an orthogonal matrix (the O(h) gauge)
# --------------------------------------------------------------------------- #
def recover_W(Z: np.ndarray, W_true: np.ndarray, h: int):
    """Cholesky of the quadric -> W_hat = U M^{-1}; align to W_true by both a
    full affine map and a scaled-orthogonal map. Returns a dict of RMS errors."""
    U, A, _ = fit_quadric(Z, h)
    M = np.linalg.cholesky(A).T                      # A = M^T M
    W_hat = U @ np.linalg.inv(M)

    rms_raw = np.sqrt(np.mean((W_hat - W_true) ** 2))
    G_aff, *_ = np.linalg.lstsq(W_hat, W_true, rcond=None)
    rms_affine = np.sqrt(np.mean((W_hat @ G_aff - W_true) ** 2))

    Mm = W_hat.T @ W_true
    P, S, Qt = np.linalg.svd(Mm)
    Omega = P @ Qt
    s = S.sum() / np.sum(W_hat * W_hat)
    rms_orth = np.sqrt(np.mean((W_hat @ (s * Omega) - W_true) ** 2))
    return dict(rms_raw=rms_raw, rms_affine=rms_affine, rms_orth=rms_orth,
                rotation_norm=np.linalg.norm(Omega - np.eye(h)))


# --------------------------------------------------------------------------- #
#  Step 4 -- regularity diagnostics (R1: spectral gap, R2/R3: quadric)
# --------------------------------------------------------------------------- #
def noise_sweep(model: OneLayerModel, sigmas, n=3000):
    """Add i.i.d. Gaussian noise to the logits and report rank detectability and
    W-recovery error vs noise level."""
    Zc, _ = model.query(n)
    rows = []
    for sigma in sigmas:
        Q = Zc + sigma * model.rng.standard_normal(Zc.shape)
        h_hat, sv = recover_dimension(Q)
        gap = sv[model.h - 1] / sv[model.h]
        try:
            rms = recover_W(Q, model.W, model.h)["rms_orth"]
        except np.linalg.LinAlgError:
            rms = np.nan
        rows.append(dict(sigma=sigma, h_hat=h_hat, gap=gap, orth_rms=rms))
    return rows


def rank_deficient_dimension(model: OneLayerModel, effective_ranks, n=3000):
    """Confine activations to an effective-rank subspace before normalization;
    the recovered dimension is the *effective* rank (the GPT-2-Small effect)."""
    rows = []
    for r in effective_ranks:
        Braw = model.rng.standard_normal((model.h, r))
        G = model.norm(model.rng.standard_normal((n, r)) @ Braw.T)
        h_hat, _ = recover_dimension(G @ model.W.T)
        rows.append(dict(effective_rank=r, h_hat=h_hat))
    return rows


# --------------------------------------------------------------------------- #
#  Step 5 -- manifold geometry: intrinsic dimension beats linear span
# --------------------------------------------------------------------------- #
def intrinsic_dimension(Z, h, n_anchor=40, knn=200, rng=None):
    """Local-PCA estimate of the intrinsic dimension of the recoverable
    hidden-state manifold (the first Cartan character). Far below the linear
    span when a low-rank nonlinear sublayer is present."""
    rng = rng or np.random.default_rng(0)
    U = np.linalg.svd(Z.T, full_matrices=False)[0][:, :h]
    X = (U.T @ Z.T).T                                # cloud up to a linear map
    dims = []
    for a in rng.choice(len(X), n_anchor, replace=False):
        dist = np.linalg.norm(X - X[a], axis=1)
        loc = X[np.argsort(dist)[:knn]]
        loc = loc - loc.mean(0)
        sv = np.linalg.svd(loc, compute_uv=False)
        logs = np.log(sv + 1e-12)
        dims.append(int(np.argmax(logs[:-1] - logs[1:])) + 1)
    return int(np.median(dims))


# --------------------------------------------------------------------------- #
#  Step 6 -- explicit non-identifiability fibers for the sublayer
# --------------------------------------------------------------------------- #
def fiber_offsupport(model: TwoLayerModel, n=1000):
    """W1 acting off the input subspace col(B) is invisible. Returns the max
    logit difference (should be ~0) and the number of invisible parameters."""
    s = model.rng.standard_normal((n, model.k)); x = s @ model.B.T
    base = model.norm(x + model.phi(x @ model.W1.T) @ model.W2.T) @ model.W.T
    P_X = model.B @ np.linalg.pinv(model.B)
    Delta = model.rng.standard_normal((model.m, model.h)) @ (np.eye(model.h) - P_X)
    W1_alt = model.W1 + Delta
    alt = model.norm(x + model.phi(x @ W1_alt.T) @ model.W2.T) @ model.W.T
    return dict(max_logit_diff=np.abs(base - alt).max(),
                delta_norm=np.linalg.norm(Delta),
                invisible_params=model.m * (model.h - model.k),
                total_params=model.m * model.h)


def fiber_width(model: TwoLayerModel, n=1000):
    """Width m is not identifiable: a cancelling neuron pair gives m -> m+2 with
    identical outputs."""
    s = model.rng.standard_normal((n, model.k)); x = s @ model.B.T
    base = model.norm(x + model.phi(x @ model.W1.T) @ model.W2.T) @ model.W.T
    w = model.rng.standard_normal((1, model.h)); c = model.rng.standard_normal((model.h, 1))
    W1b = np.vstack([model.W1, w, w]); W2b = np.hstack([model.W2, c, -c])
    big = model.norm(x + model.phi(x @ W1b.T) @ W2b.T) @ model.W.T
    return dict(max_logit_diff=np.abs(base - big).max(),
                m=model.m, m_new=model.m + 2)
