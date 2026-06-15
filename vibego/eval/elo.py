"""Bayesian Elo from pairwise game results.

Fits a Bradley-Terry model P(i beats j) = sigmoid(theta_i - theta_j) by MAP under a Gaussian
prior on the ratings (this is the "Bayesian" part: it regularizes, keeps undefeated players
finite, and gives a posterior we can take credible intervals from). Ratings are reported in
Elo, mean-centered to 0, with 95% intervals from the posterior curvature at the MAP.
"""
from __future__ import annotations

import numpy as np

SCALE = 400.0 / np.log(10.0)  # Elo points per natural logit


def bayes_elo(wins, prior_elo_std: float = 800.0, iters: int = 500, tol: float = 1e-10):
    """wins[i][j] = (possibly fractional) number of times i beat j.

    Returns (elo, stderr) arrays (Elo points), mean-centered. 95% CI = elo ± 1.96*stderr.
    """
    W = np.asarray(wins, dtype=float)
    n = W.shape[0]
    N = W + W.T  # games played between each pair
    sigma = prior_elo_std / SCALE  # prior std in logit units
    theta = np.zeros(n)
    Hess = -np.eye(n)
    for _ in range(iters):
        d = theta[:, None] - theta[None, :]
        P = 1.0 / (1.0 + np.exp(-d))
        grad = (W.sum(axis=1) - (N * P).sum(axis=1)) - theta / sigma**2
        w = N * P * (1.0 - P)
        Hess = np.diag(-w.sum(axis=1) - 1.0 / sigma**2) + w
        step = np.linalg.solve(-Hess, grad)  # Newton (maximize concave log-posterior)
        theta += step
        if np.max(np.abs(step)) < tol:
            break
    theta -= theta.mean()
    cov = np.linalg.inv(-Hess)
    stderr = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    return theta * SCALE, stderr * SCALE
