"""Fitting-weight-centered kernels with an unpenalized intercept."""
import numpy as np
from sklearn.kernel_ridge import KernelRidge


def squared_distances(x, z):
    return np.maximum(np.sum(x*x, axis=1)[:, None] + np.sum(z*z, axis=1)[None, :] - 2*x@z.T, 0)


def prepare_kernel(x, weights, kind, factor=None):
    x = np.asarray(x, dtype=float);weights = np.asarray(weights, dtype=float)
    if x.ndim != 2 or not x.shape[1] or weights.shape != (len(x),) or len(x) < 2:
        raise ValueError('Invalid fitting shapes')
    if not np.isfinite(x).all() or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError('Fitting features and positive weights must be finite')
    a = weights / weights.sum();state = {'x_fit': x, 'a': a, 'kind': kind, 'features': x.shape[1]}
    if kind == 'linear':
        if factor is not None:raise ValueError('Linear kernel has no gamma factor')
        k = x@x.T/x.shape[1]
    elif kind == 'cubic':
        if factor not in (0.5, 1, 2): raise ValueError('Invalid cubic offset')
        state['offset'] = float(factor)
        k = (x@x.T/x.shape[1] + factor)**3
    elif kind == 'poly':
        if factor not in (0.5, 1, 2): raise ValueError('Invalid polynomial offset')
        state['offset'] = float(factor)
        k = (x@x.T/x.shape[1] + factor)**2
    elif kind == 'rbf':
        if factor is None or not np.isfinite(factor) or factor <= 0:raise ValueError('Invalid RBF factor')
        distances = squared_distances(x, x);off = distances[np.triu_indices(len(x), 1)];positive = off[off > 0]
        if not len(positive):raise ValueError('No positive fitting off-diagonal distance')
        median = float(np.median(positive))
        if not np.isfinite(median) or median <= 0:raise ValueError('Invalid fitting distance median')
        state.update(distance_median=median, gamma=float(factor/median), gamma_factor=factor)
        k = np.exp(-state['gamma']*distances)
    else:raise ValueError('Unknown kernel')
    colmean = a@k;grand = float(a@k@a)
    centered = k-(k@a)[:, None]-colmean[None, :]+grand
    state.update(column_mean=colmean, grand_mean=grand)
    if not np.isfinite(centered).all():raise ValueError('Nonfinite centered kernel')
    return state, centered


def fit_centered_kernel(state, centered, y, weights, alpha):
    y = np.asarray(y, dtype=float);weights = np.asarray(weights, dtype=float)
    if y.shape != (len(state['x_fit']),) or weights.shape != y.shape or centered.shape != (len(y), len(y)):
        raise ValueError('Unaligned fitting kernel or targets')
    if not np.isfinite(y).all() or not np.isfinite(weights).all() or np.any(weights <= 0):raise ValueError('Invalid fitting targets or weights')
    if not np.isfinite(alpha) or alpha <= 0:raise ValueError('Invalid ridge alpha')
    np.testing.assert_allclose(weights/weights.sum(), state['a'], rtol=0, atol=1e-15)
    center = float(np.average(y, weights=weights))
    fitted = KernelRidge(alpha=alpha, kernel='precomputed').fit(centered, y-center, sample_weight=weights)
    # Save only coefficients and intercept, never redundant fitted Gram matrices.
    return {'dual_coef': fitted.dual_coef_.copy(), 'target_center': center, 'alpha': alpha}


def predict_centered_kernel(model, state, x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != state['features'] or not np.isfinite(x).all():raise ValueError('Invalid inference features')
    if state['kind'] == 'linear':k = x@state['x_fit'].T/state['features']
    elif state['kind'] == 'poly':k = (x@state['x_fit'].T/state['features'] + state['offset'])**2
    elif state['kind'] == 'cubic':k = (x@state['x_fit'].T/state['features'] + state['offset'])**3
    else:k = np.exp(-state['gamma']*squared_distances(x, state['x_fit']))
    centered = k-(k@state['a'])[:, None]-state['column_mean'][None, :]+state['grand_mean']
    return model['target_center']+centered@model['dual_coef']
