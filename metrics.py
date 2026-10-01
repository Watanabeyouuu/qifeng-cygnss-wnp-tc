"""Deterministic and ensemble metrics, evaluated over a sea mask."""

import numpy as np
from statistics import NormalDist


def _pair(pred, ref, mask=None):
    pred = np.asarray(pred, float).ravel()
    ref = np.asarray(ref, float).ravel()
    ok = np.isfinite(pred) & np.isfinite(ref)
    if mask is not None:
        ok &= np.asarray(mask, bool).ravel()
    return pred[ok], ref[ok]


def bias(pred, ref, mask=None):
    p, r = _pair(pred, ref, mask)
    return float(np.mean(p - r))


def rmse(pred, ref, mask=None):
    p, r = _pair(pred, ref, mask)
    return float(np.sqrt(np.mean((p - r) ** 2)))


def mae(pred, ref, mask=None):
    p, r = _pair(pred, ref, mask)
    return float(np.mean(np.abs(p - r)))


def correlation(pred, ref, mask=None):
    p, r = _pair(pred, ref, mask)
    if p.size < 2:
        return np.nan
    return float(np.corrcoef(p, r)[0, 1])


def deterministic_scores(pred, ref, mask=None):
    return {
        "bias": bias(pred, ref, mask),
        "rmse": rmse(pred, ref, mask),
        "mae": mae(pred, ref, mask),
        "corr": correlation(pred, ref, mask),
    }


def _ens_valid(ensemble, ref, mask=None):
    ens = np.asarray(ensemble, float)
    ref = np.asarray(ref, float)
    ok = np.isfinite(ref) & np.all(np.isfinite(ens), axis=0)
    if mask is not None:
        ok &= np.asarray(mask, bool)
    return ens[:, ok], ref[ok]


def spread(ensemble, ref=None, mask=None):
    if ref is None:
        ens = np.asarray(ensemble, float)
        values = ens.std(axis=0)
        valid = np.isfinite(values)
        if mask is not None:
            valid &= np.asarray(mask, bool)
        return float(np.mean(values[valid])) if valid.any() else np.nan
    ens, _ = _ens_valid(ensemble, ref, mask)
    return float(np.mean(ens.std(axis=0)))


def coverage(obs_mask, sea_mask):
    obs_mask = np.asarray(obs_mask, bool)
    sea_mask = np.asarray(sea_mask, bool)
    n = sea_mask.sum()
    return float((obs_mask & sea_mask).sum() / n) if n else np.nan


def picp_gaussian(ensemble, ref, level=0.9, mask=None):
    """Gaussian interval coverage, as used in the final manuscript."""
    if level not in (0.50, 0.80, 0.90, 0.95):
        raise ValueError("Paper levels are 0.50, 0.80, 0.90 and 0.95")
    ens, r = _ens_valid(ensemble, ref, mask)
    if r.size == 0:
        return np.nan
    m, s = ens.mean(axis=0), ens.std(axis=0)
    q = NormalDist().inv_cdf((1 + level) / 2)
    return float(np.mean((r >= m - q * s) & (r <= m + q * s)))


def paper_zscore(ensemble, ref, mask=None, eps=1e-6):
    """(reference - mean)/(spread + eps), matching the production analysis."""
    ens, r = _ens_valid(ensemble, ref, mask)
    if r.size == 0:
        return np.nan, np.nan
    z = (r - ens.mean(axis=0)) / (ens.std(axis=0) + eps)
    return float(np.mean(z)), float(np.std(z))


def gaussian_nll(ensemble, ref, mask=None, eps=1e-6):
    """Gaussian NLL with the final evaluation's additive spread stabilizer."""
    ens, r = _ens_valid(ensemble, ref, mask)
    s = ens.std(axis=0) + eps
    z = (r - ens.mean(axis=0)) / s
    return float(np.mean(0.5 * np.log(2 * np.pi * s ** 2) + 0.5 * z ** 2))


def gaussian_crps(ensemble, ref, mask=None, eps=1e-6):
    """CRPS of the Gaussian defined by ensemble mean and population spread."""
    from scipy.special import ndtr
    ens, r = _ens_valid(ensemble, ref, mask)
    s = ens.std(axis=0) + eps
    z = (r - ens.mean(axis=0)) / s
    density = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
    return float(np.mean(s * (z * (2 * ndtr(z) - 1) + 2 * density - 1 / np.sqrt(np.pi))))
