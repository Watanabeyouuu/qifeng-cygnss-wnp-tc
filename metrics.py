import numpy as np


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
