"""Synthetic tests for the separate quantitative-refinement BO interface."""

from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from xrd_bo_optimization_tf.bayesian_opt import (
    prepare_refined_optimization_data,
    suggest_next_conditions,
)
from xrd_bo_optimization_tf.config import (
    BO_PARAMETER_COLUMNS,
    BO_PARAMETER_GRID_SPECS,
    SE_PULSE_DUTY_CYCLE_BOUNDS,
    SE_PULSE_DUTY_CYCLE_COLUMN,
    SE_PULSE_PERIOD_MS,
)


def examples(count=10):
    metrics = pd.DataFrame([
        {
            "sample_id": f"sample-{i}",
            "status": "converged",
            "fit_mode": "Rietveld",
            "rietveld_validated": True,
            "size_calibration_validated": True,
            "rietveld_validation_provenance": f"archive:refinement-{i}",
            "size_calibration_provenance": "archive:instrument-calibration",
            "alpha_weight_fraction": 0.1 + i * 0.05,
            "alpha_crystallite_size_nm": 10.0 + i,
            "rwp": 0.1 + i * 0.01,
        }
        for i in range(count)
    ])
    synthesis = pd.DataFrame([
        {
            "sample_id": f"sample-{i}",
            "hard_bake_temp_c": 225 if i % 2 == 0 else 250,
            "selenization_temp_c": 520 + (i % 5) * 10,
            "selenization_time_s": 600 + (i % 5) * 300,
            "cbc_ratio": 0.63 + (i % 9) * 0.01,
            "se_pulse_on_ms": 300 + i * 50,
            "se_pulse_off_ms": 2900 - i * 50,
        }
        for i in range(count)
    ])
    return metrics, synthesis


class RefinedOptimizationTests(unittest.TestCase):
    def test_objective_and_parameter_mapping(self):
        metrics, synthesis = examples()
        # Heuristic columns must not influence or leak into the refined objective.
        metrics["phase_purity"] = 0.99
        metrics["crystallinity"] = 9999
        original = metrics.copy(deep=True)
        result = prepare_refined_optimization_data(metrics, synthesis=synthesis, include_pulse=True)
        pd.testing.assert_frame_equal(metrics, original)
        np.testing.assert_allclose(result["alpha_crystallite_size_norm"], np.arange(10) / 9)
        expected = (0.7 * metrics["alpha_weight_fraction"] + 0.3 * np.arange(10) / 9) * (1 - 0.25 * metrics["rwp"])
        np.testing.assert_allclose(result["objective"], expected)
        self.assertNotIn("phase_purity", result)
        self.assertNotIn("crystallinity", result)
        self.assertEqual(result.attrs["objective_rwp_beta"], 0.25)
        self.assertEqual(result.attrs["alpha_crystallite_size_nm_range"], (10.0, 19.0))
        self.assertEqual(result.attrs["parameter_columns"], (*BO_PARAMETER_COLUMNS, SE_PULSE_DUTY_CYCLE_COLUMN))
        np.testing.assert_allclose(result[SE_PULSE_DUTY_CYCLE_COLUMN], synthesis["se_pulse_on_ms"] / SE_PULSE_PERIOD_MS)

    def test_ineligible_rows_excluded_before_normalization(self):
        metrics, synthesis = examples(4)
        metrics.loc[1, "fit_mode"] = "Le Bail (exploratory; fixed cells)"
        metrics.loc[2, "size_calibration_validated"] = False
        metrics.loc[3, "alpha_crystallite_size_nm"] = np.inf
        result = prepare_refined_optimization_data(metrics, synthesis=synthesis)
        self.assertEqual(result["sample_id"].tolist(), ["sample-0"])
        self.assertEqual(result["alpha_crystallite_size_norm"].tolist(), [0.0])
        self.assertEqual(result.attrs["refined_exclusion_counts"]["not_rietveld"], 1)
        self.assertEqual(result.attrs["refined_exclusion_counts"]["size_unvalidated"], 1)
        self.assertEqual(result.attrs["refined_exclusion_counts"]["invalid_size_nm"], 1)

    def test_invalid_or_unvalidated_measurements_are_rejected(self):
        cases = [
            ("status", "max_iter"), ("fit_mode", "Le Bail"),
            ("rietveld_validated", False), ("rietveld_validated", "True"),
            ("size_calibration_validated", False),
            ("rietveld_validation_provenance", " "),
            ("size_calibration_provenance", "unknown"),
            ("alpha_weight_fraction", -0.01), ("alpha_weight_fraction", 1.01),
            ("alpha_weight_fraction", float("nan")),
            ("alpha_crystallite_size_nm", 0), ("alpha_crystallite_size_nm", -1),
            ("alpha_crystallite_size_nm", float("inf")),
            ("rwp", -0.01), ("rwp", float("inf")), ("rwp", "bad"),
        ]
        for column, value in cases:
            with self.subTest(column=column, value=value):
                metrics, synthesis = examples(1)
                metrics[column] = pd.Series([value], dtype=object)
                with self.assertRaisesRegex(ValueError, "No eligible refined BO rows.*Le Bail"):
                    prepare_refined_optimization_data(metrics, synthesis=synthesis)

    def test_nullable_numbers_and_incomplete_recipe_are_ineligible(self):
        metrics, synthesis = examples(2)
        metrics["rwp"] = pd.Series([pd.NA, 0.1], dtype="Float64")
        synthesis.loc[1, "cbc_ratio"] = np.inf
        with self.assertRaisesRegex(ValueError, "No eligible refined BO rows"):
            prepare_refined_optimization_data(metrics, synthesis=synthesis)

    def test_missing_provenance_and_current_lebail_export(self):
        metrics, synthesis = examples(1)
        with self.assertRaisesRegex(ValueError, "missing columns.*Le Bail"):
            prepare_refined_optimization_data(metrics.drop(columns="rietveld_validated"), synthesis=synthesis)
        path = Path(__file__).resolve().parents[1] / "data/results/bacuse2_rietx_refined_metrics.csv"
        if path.exists():
            lebail = pd.read_csv(path)
            with self.assertRaisesRegex(ValueError, "Le Bail export is not eligible"):
                prepare_refined_optimization_data(lebail, synthesis=synthesis)
            # Even explicit attestations and fabricated values cannot make a
            # Le Bail fit into a Rietveld fit.
            lebail = lebail.assign(
                rietveld_validated=True, size_calibration_validated=True,
                rietveld_validation_provenance="test", size_calibration_provenance="test",
                alpha_weight_fraction=0.5, alpha_crystallite_size_nm=20,
            )
            with self.assertRaisesRegex(ValueError, "No eligible refined BO rows.*Le Bail"):
                prepare_refined_optimization_data(lebail, synthesis=synthesis)

    def test_beta_and_nonnegative_penalty(self):
        metrics, synthesis = examples(2)
        metrics["rwp"] = [0.0, 2.0]
        result = prepare_refined_optimization_data(metrics, synthesis=synthesis, beta=0.5)
        self.assertEqual(result["objective_penalty_factor"].tolist(), [1.0, 0.0])
        self.assertEqual(result["objective"].iloc[1], 0.0)
        metrics.loc[1, "rwp"] = 5.0
        with self.assertRaisesRegex(ValueError, "negative penalty factor"):
            prepare_refined_optimization_data(metrics, synthesis=synthesis, beta=0.5)
        for beta in (-1, float("nan"), float("inf"), "bad"):
            with self.subTest(beta=beta), self.assertRaisesRegex(ValueError, "beta must be"):
                prepare_refined_optimization_data(metrics, synthesis=synthesis, beta=beta)

    def test_explicit_pulse_dimension_and_grid_snapping(self):
        metrics, synthesis = examples(2)
        metrics.attrs["objective_penalty_lambda"] = 0.8
        synthesis.loc[0, "cbc_ratio"] = 0.634
        prepared = prepare_refined_optimization_data(
            metrics, synthesis=synthesis, include_pulse=True,
            parameter_columns=(*BO_PARAMETER_COLUMNS, SE_PULSE_DUTY_CYCLE_COLUMN),
        )
        self.assertEqual(prepared["cbc_ratio"].iloc[0], 0.63)
        self.assertAlmostEqual(prepared.attrs["parameter_snap_deltas"]["cbc_ratio"], 0.004)
        self.assertNotIn("objective_penalty_lambda", prepared.attrs)
        synthesis.loc[0, "hard_bake_temp_c"] = 300
        with self.assertRaisesRegex(ValueError, "half-step grid snapping tolerance"):
            prepare_refined_optimization_data(metrics, synthesis=synthesis)

    def test_suggestions_use_recipe_grids_and_fixed_period_pulse(self):
        metrics, synthesis = examples()
        prepared = prepare_refined_optimization_data(metrics, synthesis=synthesis, include_pulse=True)
        result = suggest_next_conditions(prepared, n_suggestions=3, random_state=4)
        self.assertEqual(len(result), 3)
        self.assertTrue(set((*BO_PARAMETER_COLUMNS, SE_PULSE_DUTY_CYCLE_COLUMN, "predicted_objective", "predicted_std")).issubset(result.columns))
        for name, (lower, upper, step, digits) in BO_PARAMETER_GRID_SPECS.items():
            grid = {round(lower + i * step, digits) for i in range(round((upper - lower) / step) + 1)}
            self.assertTrue(set(result[name]).issubset(grid))
        self.assertTrue(result[SE_PULSE_DUTY_CYCLE_COLUMN].between(*SE_PULSE_DUTY_CYCLE_BOUNDS).all())
        np.testing.assert_allclose(result["se_pulse_on_ms"] + result["se_pulse_off_ms"], SE_PULSE_PERIOD_MS)
        np.testing.assert_allclose(result[SE_PULSE_DUTY_CYCLE_COLUMN], result["se_pulse_on_ms"] / SE_PULSE_PERIOD_MS)
        self.assertTrue(np.isfinite(result[["predicted_objective", "predicted_std"]]).all().all())


if __name__ == "__main__":
    unittest.main()