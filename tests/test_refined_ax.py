"""Synthetic tests for the separate refined-data Ax workflow."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from xrd_bo_optimization_tf.refined_ax import (
    OBJECTIVES, PARAMETERS, _new_client, run_refined_ax,
)
from xrd_bo_optimization_tf.config import SE_PULSE_PERIOD_MS


def examples():
    metrics = pd.DataFrame([
        {"sample_id": f"s{i}", "status": "converged", "fit_mode": "Rietveld",
         "rietveld_validated": True, "size_calibration_validated": True,
         "rietveld_validation_provenance": "refinement-record",
         "size_calibration_provenance": "calibration-record",
         "alpha_weight_fraction": fraction, "alpha_crystallite_size_nm": size, "rwp": rwp}
        for i, (fraction, size, rwp) in enumerate([(0.2, 10, 0.1), (0.8, 20, 0.3), (0.5, 15, 0.2)])
    ])
    synthesis = pd.DataFrame([
        {"sample_id": f"s{i}", "hard_bake_temp_c": 225.0,
         "selenization_temp_c": 520.0 + 10 * i, "selenization_time_s": 600.0,
         "cbc_ratio": 0.63, "se_pulse_on_ms": 300 + 100 * i,
         "se_pulse_off_ms": 2900 - 100 * i}
        for i in range(3)
    ])
    return metrics, synthesis


class FakeClient:
    def __init__(self):
        self.trials = []
        self.completed = []
        self.proposed = []
        self.prediction_input = None
        self.bad_prediction = False
        self.duty = 500.4 / SE_PULSE_PERIOD_MS

    def attach_trial(self, parameters):
        self.trials.append(parameters)
        return parameters, len(self.trials) - 1

    def complete_trial(self, trial_index, raw_data):
        self.completed.append((trial_index, raw_data))

    def get_next_trial(self):
        params = dict(zip(PARAMETERS, (250.0, 560.0, 1800.0, 0.71, self.duty)))
        self.proposed.append(params)
        return params, 100 + len(self.proposed)

    def abandon_trial(self, trial_index, reason):
        pass

    def get_model_predictions_for_parameterizations(self, parameterizations, metric_names):
        self.prediction_input = parameterizations
        return [{name: (float("nan") if self.bad_prediction else recipe[PARAMETERS[-1]], 0.1)
                 for name in metric_names} for recipe in parameterizations]


class RefinedAxTests(unittest.TestCase):
    def setUp(self):
        self.metrics, self.synthesis = examples()
        self.client = FakeClient()
        self.factory = patch("xrd_bo_optimization_tf.refined_ax._new_client", return_value=self.client)
        self.new_client = self.factory.start()
        self.addCleanup(self.factory.stop)

    def run_model(self, **kwargs):
        return run_refined_ax(self.metrics, self.synthesis, n_suggestions=1, **kwargs)

    def test_raw_objectives_and_predictions_use_quantized_recipe(self):
        result = self.run_model(random_seed=17)
        self.assertEqual(self.new_client.call_args.args[0], 17)
        self.assertEqual(self.client.completed[0][1]["alpha_weight_fraction"][0], 0.2)
        self.assertEqual(self.client.completed[1][1]["alpha_crystallite_size_norm"][0], 1.0)
        self.assertEqual(self.client.completed[1][1]["rwp"][0], 0.3)
        self.assertEqual(set(self.client.completed[0][1]), set(OBJECTIVES))
        self.assertTrue(np.isnan(self.client.completed[0][1]["rwp"][1]))
        self.assertEqual(self.client.prediction_input[0][PARAMETERS[-1]], 500 / SE_PULSE_PERIOD_MS)
        row = result.suggestions.iloc[0]
        self.assertEqual(row["predicted_rwp_mean"], 500 / SE_PULSE_PERIOD_MS)
        self.assertEqual(row["se_pulse_on_ms"] + row["se_pulse_off_ms"], SE_PULSE_PERIOD_MS)
        self.assertEqual(result.metadata["alpha_crystallite_size_nm_range"], [10.0, 20.0])
        self.assertEqual(result.observed_nondominated["sample_id"].tolist(), ["s0", "s1", "s2"])
        self.assertNotIn("objective", result.observations.columns)

    def test_missing_and_ineligible_data_do_not_construct_or_export(self):
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "out"
            with self.assertRaisesRegex(ValueError, "missing columns.*Le Bail"):
                run_refined_ax(self.metrics.drop(columns="rietveld_validated"), self.synthesis, output_dir=target)
            invalid = self.metrics.assign(fit_mode="Le Bail")
            with self.assertRaisesRegex(ValueError, "No eligible refined BO rows"):
                run_refined_ax(invalid, self.synthesis, output_dir=target)
            self.assertFalse(target.exists())
        self.new_client.assert_not_called()

    def test_actual_lebail_export_rejected(self):
        path = Path(__file__).resolve().parents[1] / "data/results/bacuse2_rietx_refined_metrics.csv"
        if not path.exists():
            self.skipTest("No local Le Bail export")
        with self.assertRaises(ValueError):
            run_refined_ax(pd.read_csv(path), self.synthesis)
        self.new_client.assert_not_called()

    def test_threshold_validation_and_defaults(self):
        result = self.run_model()
        self.assertEqual([result.metadata["objectives"][name]["reference_threshold"] for name in OBJECTIVES],
                         [-0.05, -0.05, 0.35])
        for thresholds in ({}, dict(zip(OBJECTIVES, [0, 0, np.inf])),
                           dict(zip(OBJECTIVES, [0.9, 0.9, 0.01]))):
            with self.subTest(thresholds=thresholds), self.assertRaises(ValueError):
                self.run_model(reference_thresholds=thresholds)
        refs = dict(zip(OBJECTIVES, [-0.1, -0.2, 0.5]))
        self.run_model(reference_thresholds=refs)
        self.assertEqual(self.new_client.call_args.args[1], refs)

    def test_predictions_and_duplicate_delivered_recipes_fail_before_export(self):
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "out"
            self.client.bad_prediction = True
            with self.assertRaisesRegex(ValueError, "Invalid Ax predictions"):
                self.run_model(output_dir=target)
            self.assertFalse(target.exists())
            self.client.bad_prediction = False
            with self.assertRaisesRegex(ValueError, "duplicate recipes"):
                run_refined_ax(self.metrics, self.synthesis, n_suggestions=2, output_dir=target)
            self.assertFalse(target.exists())

    def test_out_of_bounds_and_off_grid_proposals_fail(self):
        for recipe in (
            dict(zip(PARAMETERS, (250.0, 555.0, 1800.0, 0.71, 0.2))),
            dict(zip(PARAMETERS, (250.0, 560.0, 1800.0, 0.71, 1.0))),
            dict(zip(PARAMETERS, (250.0, 560.0, 1800.0, 0.71, float("nan")))),
        ):
            with self.subTest(recipe=recipe), patch.object(self.client, "get_next_trial", return_value=(recipe, 5)):
                with self.assertRaises(ValueError):
                    self.run_model()

    def test_export_is_distinct_and_never_overwrites(self):
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "out"
            result = self.run_model(output_dir=target)
            self.assertEqual(len(list(target.iterdir())), 4)
            with (target / "refined_ax_metadata.json").open() as stream:
                metadata = json.load(stream)
            self.assertEqual(metadata["n_observations"], 3)
            self.assertEqual(len(pd.read_csv(target / "refined_ax_suggestions.csv")), len(result.suggestions))
            with self.assertRaises(FileExistsError):
                self.run_model(output_dir=target)
            self.assertEqual(len(list(target.iterdir())), 4)


class RealAxSetupTests(unittest.TestCase):
    def test_real_ax_objective_directions_thresholds_and_raw_trial(self):
        metrics, synthesis = examples()
        thresholds = dict(zip(OBJECTIVES, [-0.05, -0.05, 0.35]))
        client = _new_client(0, thresholds)
        experiment = client.experiment
        objective = experiment.optimization_config.objective
        self.assertEqual(dict(objective.metric_weights),
                 {OBJECTIVES[0]: 1.0, OBJECTIVES[1]: 1.0, OBJECTIVES[2]: -1.0})
        refs = {constraint.metric_names[0]: constraint.bound for constraint in experiment.optimization_config.objective_thresholds}
        self.assertEqual(refs, thresholds)
        row = synthesis.iloc[0]
        recipe = {name: float(row[name]) for name in PARAMETERS[:-1]}
        recipe[PARAMETERS[-1]] = float(row["se_pulse_on_ms"] / SE_PULSE_PERIOD_MS)
        _, index = client.attach_trial(parameters=recipe)
        client.complete_trial(trial_index=index, raw_data={name: (value, float("nan"))
                              for name, value in zip(OBJECTIVES, [0.2, 0.0, 0.1])})
        data = experiment.lookup_data().df.set_index("metric_name")
        self.assertEqual(data.loc["alpha_weight_fraction", "mean"], metrics.iloc[0]["alpha_weight_fraction"])
        self.assertEqual(data.loc["alpha_crystallite_size_norm", "mean"], 0.0)
        self.assertEqual(data.loc["rwp", "mean"], 0.1)

    def test_real_model_generation_and_predictions(self):
        import torch

        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            metrics, synthesis = examples()
            result = run_refined_ax(metrics, synthesis, n_suggestions=1, random_seed=4)
        finally:
            torch.set_num_threads(previous_threads)
        self.assertEqual(type(result.client.generation_strategy.adapter).__name__, "TorchAdapter")
        predictions = result.suggestions.filter(like="predicted_").to_numpy(dtype=float)
        self.assertEqual(predictions.shape, (1, 6))
        self.assertTrue(np.isfinite(predictions).all())
        self.assertEqual(result.suggestions.iloc[0]["se_pulse_on_ms"]
                         + result.suggestions.iloc[0]["se_pulse_off_ms"], SE_PULSE_PERIOD_MS)


if __name__ == "__main__":
    unittest.main()