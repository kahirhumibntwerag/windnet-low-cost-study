"""Meaningful synthetic verification checks; no study data or holdout access."""
from __future__ import annotations

import json
import math
import unittest

import numpy as np

try:
    from .metrics import (continuous_metrics, deduplicate_issue_target_pairs,
                          detection_metrics, evaluate_forecast,
                          fast_wind_metrics, persistence_skill)
except ImportError:
    from metrics import (continuous_metrics, deduplicate_issue_target_pairs,
                         detection_metrics, evaluate_forecast,
                         fast_wind_metrics, persistence_skill)


class SyntheticVerificationChecks(unittest.TestCase):
    def test_continuous_scores_have_correct_units_sign_and_correlation(self):
        result = continuous_metrics([300, 400, 500], [320, 380, 530])
        self.assertEqual(result["n"], 3)
        self.assertAlmostEqual(result["mse"], 1700 / 3)
        self.assertAlmostEqual(result["rmse"], math.sqrt(1700 / 3))
        self.assertAlmostEqual(result["mae"], 70 / 3)
        self.assertAlmostEqual(result["bias"], 10)
        self.assertAlmostEqual(result["r"], 21000 / math.sqrt(468000000))

    def test_nonfinite_pairs_use_one_observation_forecast_mask(self):
        result = continuous_metrics([300, np.nan, 500, 400], [320, 400, np.inf, 380])
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["n_excluded_nonfinite"], 2)
        self.assertAlmostEqual(result["rmse"], 20)
        self.assertAlmostEqual(result["bias"], 0)

    def test_inclusive_threshold_and_hand_counted_contingency(self):
        result = detection_metrics([500, 600, 450, 300], [500, 400, 550, 200])
        self.assertEqual([result[k] for k in ("tp", "fp", "fn", "tn")], [1, 1, 1, 1])
        for key in ("pod", "far", "fpr", "precision"):
            self.assertAlmostEqual(result[key], .5)
        self.assertAlmostEqual(result["csi"], 1 / 3)
        self.assertAlmostEqual(result["tss"], 0)
        self.assertAlmostEqual(result["hss"], 0)
        self.assertAlmostEqual(result["frequency_bias"], 1)

    def test_fast_errors_include_missed_observed_events(self):
        result = fast_wind_metrics([500, 600, 450, 300], [500, 400, 550, 200])
        conditional = result["continuous_observed_fast"]
        self.assertEqual(conditional["n"], 2)
        self.assertAlmostEqual(conditional["rmse"], math.sqrt(20000))
        self.assertAlmostEqual(conditional["mae"], 100)
        self.assertAlmostEqual(conditional["bias"], -100)
        self.assertAlmostEqual(conditional["r"], -1)
        secondary = fast_wind_metrics([500, 600, 450, 300], [500, 400, 550, 200], 600)
        self.assertEqual(secondary["continuous_observed_fast"]["n"], 1)
        self.assertAlmostEqual(secondary["continuous_observed_fast"]["rmse"], 200)
        self.assertIsNone(secondary["continuous_observed_fast"]["r"])

    def test_nontrivial_heidke_and_true_skill(self):
        # TP=3, FP=1, FN=1, TN=5: independently specified contingency table.
        y = [600] * 4 + [300] * 6
        p = [600, 600, 600, 300, 600, 300, 300, 300, 300, 300]
        result = detection_metrics(y, p)
        self.assertAlmostEqual(result["hss"], 28 / 48)
        self.assertAlmostEqual(result["tss"], .75 - 1 / 6)
        self.assertAlmostEqual(result["csi"], 3 / 5)

    def test_skill_uses_mean_square_error_not_rmse_ratio(self):
        result = persistence_skill([500, 600, 300, 400], [500, 550, 350, 400],
                                   [400, 500, 400, 500])
        self.assertEqual(result["n"], 4)
        self.assertAlmostEqual(result["mse_skill_score"], .875)
        self.assertAlmostEqual(result["delta_rmse"], math.sqrt(1250) - 100)

    def test_reference_comparison_uses_common_finite_cases(self):
        result = persistence_skill([500, 600, 300], [490, 650, 300], [500, np.nan, 350])
        self.assertEqual(result["n_common_finite"], 2)
        self.assertEqual(result["model"]["n"], result["reference"]["n"])
        self.assertAlmostEqual(result["mse_skill_score"], .96)
        fast = persistence_skill([500, 600, 300], [490, 650, 300], [500, np.nan, 350], 500)
        self.assertEqual(fast["n"], 1)
        self.assertAlmostEqual(fast["delta_rmse"], 10)
        self.assertIsNone(fast["mse_skill_score"])

    def test_perfect_reference_zero_denominator_is_not_fake_skill(self):
        result = persistence_skill([300, 600], [300, 600], [300, 600])
        self.assertIsNone(result["mse_skill_score"])
        self.assertAlmostEqual(result["delta_rmse"], 0)

    def test_degenerate_and_empty_cases_are_explicitly_undefined(self):
        result = fast_wind_metrics([300, 400], [300, 400])
        self.assertEqual(result["continuous_observed_fast"]["n"], 0)
        for key in ("rmse", "mae", "bias", "r"):
            self.assertIsNone(result["continuous_observed_fast"][key])
        for key in ("pod", "far", "csi", "tss", "hss", "frequency_bias"):
            self.assertIsNone(result["detection"][key])
        self.assertEqual(result["detection"]["fpr"], 0)
        self.assertIsNone(continuous_metrics([], [])["rmse"])
        self.assertIsNone(continuous_metrics([500, 500], [500, 500])["r"])

    def test_fixed_duplicate_pair_aggregation_and_inconsistent_truth_rejection(self):
        reduced = deduplicate_issue_target_pairs(
            ["a", "a", "b", "c"], ["x", "x", "y", "z"], [500, 500, 300, 600],
            {"model": [480, 520, 320, 590], "reference": [500, 500, 350, 600]})
        self.assertEqual(reduced["n_unique_pairs"], 3)
        np.testing.assert_array_equal(reduced["duplicate_counts"], [2, 1, 1])
        np.testing.assert_allclose(reduced["forecasts"]["model"], [500, 320, 590])
        self.assertAlmostEqual(continuous_metrics(reduced["observed"], reduced["forecasts"]["model"])["rmse"], math.sqrt(500 / 3))
        separate = deduplicate_issue_target_pairs(["a", "b"], ["x", "x"], [500, 500], {"p": [480, 520]})
        self.assertEqual(separate["n_unique_pairs"], 2)
        with self.assertRaises(ValueError):
            deduplicate_issue_target_pairs(["a", "a"], ["x", "x"], [500, 501], {"p": [500, 500]})

    def test_shape_threshold_and_strict_json_contracts(self):
        with self.assertRaises(ValueError):
            continuous_metrics([300], [300, 400])
        with self.assertRaises(ValueError):
            continuous_metrics([[300]], [[300]])
        with self.assertRaises(ValueError):
            detection_metrics([300], [300], np.nan)
        with self.assertRaises(ValueError):
            evaluate_forecast([300], [300], thresholds=[500, 500])
        result = evaluate_forecast([300, 500, 600, np.nan], [300, 490, 550, 300],
                                   {"latest": [310, 500, np.nan, 300],
                                    "rotation_27d": [300, 450, 500, 300]})
        self.assertEqual(set(result["fast_wind"]), {"ge_500", "ge_600"})
        self.assertEqual(result["persistence_skill"]["latest"]["ge_500"]["n"], 1)
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
