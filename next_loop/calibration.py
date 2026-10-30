"""Monotone empirical quantile correction; fitting arrays only."""
import numpy as np

KNOTS = [.025, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .975]

def fit_map(prediction, target):
    prediction, target = np.asarray(prediction), np.asarray(target)
    assert prediction.ndim == target.ndim == 1 and prediction.shape == target.shape
    assert np.isfinite(prediction).all() and np.isfinite(target).all()
    x, y = np.quantile(prediction, KNOTS), np.quantile(target, KNOTS)
    # Collapse tied predictor knots to preserve a well-defined monotone map.
    unique, inverse = np.unique(x, return_inverse=True)
    y = np.array([y[inverse == i].mean() for i in range(len(unique))])
    return {'x': unique.tolist(), 'y': y.tolist(), 'knots': KNOTS}

def predict_map(mapping, prediction):
    return np.interp(np.asarray(prediction), mapping['x'], mapping['y'])
