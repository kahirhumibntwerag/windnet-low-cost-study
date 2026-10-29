"""Defensive selection checks on synthetic metrics; never fit or load data."""
from copy import deepcopy
from fast_loss import select_candidate


def report(all_rmse, fast_rmse):
    return {'metrics': {'overall': {'rmse': all_rmse, 'mse': None if all_rmse is None else all_rmse**2},
                        'fast_wind': {'ge_500': {'continuous_observed_fast':
                                                {'rmse': fast_rmse, 'mse': None if fast_rmse is None else fast_rmse**2}},
                                      'ge_600': {'continuous_observed_fast': {'rmse': 900, 'mse': 810000}}}}}


r = {'h8_incumbent': report(80, 130), 'wind_only__loss_weight_2': report(80.2, 125)}
selected, _ = select_candidate(r, ['wind_only__loss_weight_2'])
assert selected == 'wind_only__loss_weight_2'
changed = deepcopy(r)
changed['wind_only__loss_weight_2']['metrics']['fast_wind']['ge_600']['continuous_observed_fast']['rmse'] = 1e20
assert select_candidate(changed, ['wind_only__loss_weight_2'])[0] == selected
for bad in [report(None, 125), report(80.2, None)]:
    r['wind_only__loss_weight_2'] = bad
    assert select_candidate(r, ['wind_only__loss_weight_2'])[0] == 'h8_incumbent'
print('Four synthetic selection checks passed, including undefined overall MSE.')
