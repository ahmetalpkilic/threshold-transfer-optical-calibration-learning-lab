"""REML helpers reused from estimated-repeat-correlation-learning-lab (MIT)."""
import numpy as np
from scipy.optimize import minimize_scalar

def reml_score(z, rho):
    """Profile matrix-normal REML: unknown 2-vector mean and 2x2 covariance.

    AR precision is tridiagonal, so applying it costs O(n), not a dense inverse.
    Ordinary centering improves numerical stability; Q removes any constant.
    Additive constants independent of rho are omitted from minus twice log L.
    """
    z = np.asarray(z, float)
    if z.ndim != 2 or z.shape[1] != 2 or len(z) < 6 or not np.all(np.isfinite(z)):
        raise ValueError('Require finite (n>=6,2) characterization sequence')
    if not np.isfinite(rho) or abs(rho) >= 1:
        raise ValueError('Require |rho|<1')
    n = len(z); z = z-z.mean(axis=0)
    diagonal = np.full(n, 1+rho*rho, dtype=float); diagonal[[0,-1]] = 1
    wz = diagonal[:,None]*z
    wz[1:] -= rho*z[:-1]; wz[:-1] -= rho*z[1:]
    wz /= 1-rho*rho
    w1 = diagonal.copy(); w1[1:] -= rho; w1[:-1] -= rho
    w1 /= 1-rho*rho
    v = z.T@w1
    scatter = z.T@wz - np.outer(v,v)/w1.sum()
    sign, logdet = np.linalg.slogdet(scatter/(n-1))
    if sign <= 0: raise ValueError('Characterization scatter must be full rank')
    return 2*(n-1)*np.log1p(-rho*rho)+2*np.log(w1.sum())+(n-1)*logdet


def fit_rho(z):
    """Bounded profile fit; retain endpoint solutions and report their frequency.

    The fixed [-.95,.95] interval is part of the estimator, not a CI. Both
    classes are estimated; knowing the even class is iid would give free info.
    """
    grid = np.linspace(-.95,.95,39)
    scores = np.array([reml_score(z,x) for x in grid])
    candidates = [(scores[0],grid[0]),(scores[-1],grid[-1])]
    for i in range(1,len(grid)-1):
        if scores[i] <= min(scores[i-1],scores[i+1]):
            fit = minimize_scalar(lambda x: reml_score(z,x), bounds=(grid[i-1],grid[i+1]),
                                  method='bounded', options={'xatol':1e-9})
            if not fit.success: raise RuntimeError('Correlation fit failed')
            candidates.append((fit.fun,fit.x))
    return float(min(candidates)[1])


def ar_sequence(normal, rho, cov, mean):
    """Independent stationary AR recursion, with marginal covariance cov."""
    e = np.empty_like(normal); e[0] = normal[0]
    for j in range(1,len(e)):
        e[j] = rho*e[j-1]+np.sqrt(1-rho*rho)*normal[j]
    return mean + e@np.linalg.cholesky(cov).T
