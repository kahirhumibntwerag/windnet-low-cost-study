"""Convex point and adjacent-fitting-window difference regression."""
import numpy as np
from sklearn.linear_model import Ridge


def adjacent_pairs(block, offset):
    block = np.asarray(block)
    offset = np.asarray(offset)
    if block.ndim != 1 or offset.shape != block.shape:
        raise ValueError('Block and offset must be aligned vectors')
    if not np.issubdtype(block.dtype, np.integer) or not np.issubdtype(offset.dtype, np.integer):
        raise ValueError('Block and offset must be integer identifiers')
    keys = list(zip(block.tolist(), offset.tolist()))
    lookup = {key: i for i, key in enumerate(keys)}
    if len(lookup) != len(keys):
        raise ValueError('Duplicate fitting window')
    left, right = [], []
    for i, (b, o) in enumerate(keys):
        j = lookup.get((b, o + 1))
        if j is not None:
            left.append(i)
            right.append(j)
    return np.asarray(left, dtype=int), np.asarray(right, dtype=int)


def fit_difference_ridge(x, y, weights, block, offset, alpha, strength):
    x, y, weights = [np.asarray(a, dtype=float) for a in (x, y, weights)]
    if x.ndim != 2 or y.shape != (len(x),) or weights.shape != y.shape:
        raise ValueError('Mismatched fitting shapes')
    if not all(np.isfinite(a).all() for a in (x, y, weights)) or np.any(weights <= 0):
        raise ValueError('Fitting inputs must be finite; weights positive')
    if not np.isfinite(alpha) or alpha <= 0 or not np.isfinite(strength) or strength < 0:
        raise ValueError('Invalid regularization')
    left, right = adjacent_pairs(block, offset)
    if len(block) != len(y) or not len(left):
        raise ValueError('Missing aligned adjacent fitting pairs')
    pair_weights = (weights[left] + weights[right]) / 2
    pair_weights /= pair_weights.mean()
    x_mean = np.average(x, axis=0, weights=weights)
    y_mean = float(np.average(y, weights=weights))
    root_w = np.sqrt(weights)
    design = (x - x_mean) * root_w[:, None]
    target = (y - y_mean) * root_w
    if strength > 0:
        root_pair = np.sqrt(strength * pair_weights)
        design = np.concatenate([design, (x[right] - x[left]) * root_pair[:, None]])
        target = np.concatenate([target, (y[right] - y[left]) * root_pair])
    ridge = Ridge(alpha=alpha, fit_intercept=False, solver='cholesky').fit(design, target)
    return {'ridge': ridge, 'x_mean': x_mean, 'y_mean': y_mean,
            'pair_left': left, 'pair_right': right, 'pair_weights': pair_weights,
            'alpha': alpha, 'strength': strength}


def predict_difference_ridge(model, x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(model['x_mean']) or not np.isfinite(x).all():
        raise ValueError('Invalid inference features')
    return model['y_mean'] + model['ridge'].predict(x - model['x_mean'])
