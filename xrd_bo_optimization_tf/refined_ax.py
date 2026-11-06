"""Experimental Ax optimization of validated Rietveld results.

The three independent objectives are alpha weight fraction and min-max
normalized calibrated alpha size (maximize), and fractional Rwp (minimize).
Rwp is a fit residual, not an impurity measurement. Size normalization is
relative to the eligible observations, not an absolute physical scale.

Default hypervolume reference thresholds are -0.05, -0.05, and the maximum
observed Rwp plus max(0.05, 5% of that maximum), respectively. These are
computational reference points, not acceptance criteria. Callers may instead
provide all three finite thresholds; at least one observation must strictly
dominate the reference. This experimental model makes no physical claims.
"""

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Mapping

import numpy as np
import pandas as pd

from .bayesian_opt import _grid_values, prepare_refined_optimization_data
from .config import (
    BO_PARAMETER_COLUMNS,
    BO_PARAMETER_GRID_SPECS,
    SE_PULSE_DUTY_CYCLE_BOUNDS,
    SE_PULSE_DUTY_CYCLE_COLUMN,
    SE_PULSE_FREQUENCY_HZ,
    SE_PULSE_ON_MS_BOUNDS,
    SE_PULSE_PERIOD_MS,
)

LOGGER = logging.getLogger(__name__)
OBJECTIVES = ("alpha_weight_fraction", "alpha_crystallite_size_norm", "rwp")
PARAMETERS = (*BO_PARAMETER_COLUMNS, SE_PULSE_DUTY_CYCLE_COLUMN)


@dataclass
class RefinedAxResult:
    """Independent experiment, eligible observations, and model suggestions."""

    client: object
    observations: pd.DataFrame
    observed_nondominated: pd.DataFrame
    suggestions: pd.DataFrame
    metadata: dict


def _reference_thresholds(observations: pd.DataFrame, supplied: Mapping[str, float] | None) -> dict[str, float]:
    if supplied is None:
        max_rwp = float(observations["rwp"].max())
        thresholds = dict(zip(OBJECTIVES, (-0.05, -0.05, max_rwp + max(0.05, 0.05 * max_rwp))))
    else:
        if set(supplied) != set(OBJECTIVES):
            raise ValueError(f"Reference thresholds must contain exactly {OBJECTIVES}")
        try:
            thresholds = {name: float(supplied[name]) for name in OBJECTIVES}
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Reference thresholds must be finite numbers") from exc
    if not all(np.isfinite(value) for value in thresholds.values()):
        raise ValueError("Reference thresholds must be finite numbers")
    dominates = (
        (observations[OBJECTIVES[0]] > thresholds[OBJECTIVES[0]])
        & (observations[OBJECTIVES[1]] > thresholds[OBJECTIVES[1]])
        & (observations[OBJECTIVES[2]] < thresholds[OBJECTIVES[2]])
    )
    if not dominates.any():
        raise ValueError("At least one eligible observation must strictly dominate all reference thresholds")
    return thresholds


def _new_client(seed: int, thresholds: Mapping[str, float]):
    # A model-only strategy prevents an unmodeled Sobol fallback on small datasets.
    from ax.adapter.registry import Generators
    from ax.generation_strategy.generation_node import GenerationStep
    from ax.generation_strategy.generation_strategy import GenerationStrategy
    from ax.service.ax_client import AxClient, ObjectiveProperties

    strategy = GenerationStrategy(nodes=[GenerationStep(generator=Generators.BOTORCH_MODULAR, num_trials=-1)])
    client = AxClient(generation_strategy=strategy, random_seed=seed, verbose_logging=False)
    parameters = [
        {"name": name, "type": "choice", "values": _grid_values(BO_PARAMETER_GRID_SPECS[name]),
         "value_type": "float", "is_ordered": True}
        for name in BO_PARAMETER_COLUMNS
    ]
    parameters.append({"name": SE_PULSE_DUTY_CYCLE_COLUMN, "type": "range",
                       "bounds": list(SE_PULSE_DUTY_CYCLE_BOUNDS), "value_type": "float"})
    client.create_experiment(
        name="bacuse2_refined_rietveld_experimental_three_objective",
        parameters=parameters,
        objectives={name: ObjectiveProperties(minimize=(name == "rwp"), threshold=thresholds[name])
                    for name in OBJECTIVES},
    )
    return client


def _nondominated(observations: pd.DataFrame) -> pd.DataFrame:
    values = observations.loc[:, OBJECTIVES].to_numpy(dtype=float) * np.array([1.0, 1.0, -1.0])
    dominated = np.array([np.any(np.all(values >= value, axis=1) & np.any(values > value, axis=1))
                          for value in values])
    return observations.loc[~dominated].copy()


def _quantize(parameters: Mapping[str, float]) -> dict[str, float]:
    if set(parameters) != set(PARAMETERS):
        raise ValueError("Ax proposal does not contain exactly the five recipe parameters")
    recipe = {name: float(parameters[name]) for name in PARAMETERS}
    if not all(np.isfinite(value) for value in recipe.values()):
        raise ValueError("Ax proposal contains nonfinite recipe parameters")
    for name in BO_PARAMETER_COLUMNS:
        if recipe[name] not in _grid_values(BO_PARAMETER_GRID_SPECS[name]):
            raise ValueError(f"Ax proposal is off the {name} grid")
    duty = recipe[SE_PULSE_DUTY_CYCLE_COLUMN]
    if not SE_PULSE_DUTY_CYCLE_BOUNDS[0] <= duty <= SE_PULSE_DUTY_CYCLE_BOUNDS[1]:
        raise ValueError("Ax proposal has an out-of-bounds duty cycle")
    on_ms = int(np.rint(duty * SE_PULSE_PERIOD_MS))
    if not SE_PULSE_ON_MS_BOUNDS[0] <= on_ms <= SE_PULSE_ON_MS_BOUNDS[1]:
        raise ValueError("Quantized pulse on-time is out of bounds")
    recipe[SE_PULSE_DUTY_CYCLE_COLUMN] = on_ms / SE_PULSE_PERIOD_MS
    return recipe


def _export(result: RefinedAxResult, output_dir: Path) -> None:
    """Write new, distinct artifacts without replacing existing files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "refined_ax_observations.csv": result.observations,
        "refined_ax_observed_nondominated.csv": result.observed_nondominated,
        "refined_ax_suggestions.csv": result.suggestions,
    }
    metadata_name = "refined_ax_metadata.json"
    names = [*files, metadata_name]
    if any((output_dir / name).exists() for name in names):
        raise FileExistsError("Refined Ax export already exists; choose a new output directory")
    with TemporaryDirectory(dir=output_dir, prefix=".refined_ax_") as temporary:
        stage = Path(temporary)
        for name, frame in files.items():
            frame.to_csv(stage / name, index=False)
        (stage / metadata_name).write_text(json.dumps(result.metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        created = []
        try:
            for name in names:
                destination = output_dir / name
                os.link(stage / name, destination)  # Atomic, fails if another writer created it.
                created.append(destination)
        except Exception:
            for path in created:
                path.unlink()
            raise


def run_refined_ax(
    metrics: pd.DataFrame,
    synthesis: pd.DataFrame,
    *,
    n_suggestions: int = 5,
    random_seed: int = 0,
    reference_thresholds: Mapping[str, float] | None = None,
    output_dir: str | Path | None = None,
) -> RefinedAxResult:
    """Fit an independent five-input, three-objective Ax experiment.

    Only converged, explicitly validated and provenance-bearing Rietveld rows
    with calibrated sizes and complete synthesis/pulse inputs are eligible.
    Requires at least two such rows. No heuristic or Sobol fallback is used.
    Suggestions are in generation order, not ranked, and predictions are model
    estimates at the delivered integer-millisecond pulse recipes. Duplicate
    delivered recipes or invalid predictions fail the entire run. Exports are
    optional, occur only after validation, and never overwrite existing files.
    """
    if isinstance(n_suggestions, bool) or not isinstance(n_suggestions, int) or n_suggestions < 1:
        raise ValueError("n_suggestions must be a positive integer")
    observations = prepare_refined_optimization_data(metrics, synthesis=synthesis, include_pulse=True)
    if tuple(observations.attrs.get("parameter_columns", ())) != PARAMETERS:
        raise ValueError("Refined Ax requires the shared five-dimensional recipe space")
    if len(observations) < 2:
        raise ValueError("At least two eligible refined observations are required")
    # The shared preparation also computes a scalar BO objective. It is not
    # part of this experiment and must not be presented as an Ax objective.
    observations = observations.drop(columns=["objective", "objective_unpenalized", "objective_penalty_factor"])
    observations.attrs.pop("objective_rwp_beta", None)
    thresholds = _reference_thresholds(observations, reference_thresholds)
    metadata = {
        "experiment": "experimental_refined_rietveld_three_objective",
        "objectives": {name: {"direction": "minimize" if name == "rwp" else "maximize",
                              "reference_threshold": thresholds[name]} for name in OBJECTIVES},
        "parameters": list(PARAMETERS),
        "grid_specs": {key: list(value) for key, value in BO_PARAMETER_GRID_SPECS.items()},
        "pulse_period_ms": SE_PULSE_PERIOD_MS,
        "pulse_on_ms_bounds": list(SE_PULSE_ON_MS_BOUNDS),
        "alpha_crystallite_size_nm_range": list(observations.attrs["alpha_crystallite_size_nm_range"]),
        "exclusion_counts": observations.attrs["refined_exclusion_counts"],
        "parameter_snap_deltas": observations.attrs["parameter_snap_deltas"],
        "random_seed": random_seed,
        "n_observations": len(observations),
        "n_suggestions": n_suggestions,
        "prediction_note": "Model estimates at quantized recipes, not observations; generation order is not rank.",
    }
    LOGGER.info("Refined Ax: %d eligible rows; thresholds=%s; seed=%s", len(observations), thresholds, random_seed)
    client = _new_client(random_seed, thresholds)
    trial_indices = []
    for _, row in observations.iterrows():
        _, index = client.attach_trial(parameters={name: float(row[name]) for name in PARAMETERS})
        client.complete_trial(trial_index=index, raw_data={name: (float(row[name]), float("nan")) for name in OBJECTIVES})
        trial_indices.append(index)
    observations = observations.copy()
    observations.insert(0, "ax_trial_index", trial_indices)

    recipes = []
    proposal_trials = []
    for _ in range(n_suggestions):
        parameters, index = client.get_next_trial()
        recipes.append(_quantize(parameters))
        proposal_trials.append(index)
        client.abandon_trial(trial_index=index, reason="Proposal only; no experimental observation")
    suggestions = pd.DataFrame(recipes, columns=PARAMETERS)
    if suggestions.duplicated(list(PARAMETERS)).any():
        raise ValueError("Ax generated duplicate recipes after pulse quantization")
    predictions = client.get_model_predictions_for_parameterizations(parameterizations=recipes, metric_names=list(OBJECTIVES))
    if len(predictions) != len(recipes):
        raise ValueError("Ax returned the wrong number of predictions")
    for metric in OBJECTIVES:
        try:
            values = np.asarray([prediction[metric] for prediction in predictions], dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid Ax predictions for {metric}") from exc
        if values.shape != (len(recipes), 2) or not np.isfinite(values).all() or (values[:, 1] < 0).any():
            raise ValueError(f"Invalid Ax predictions for {metric}")
        suggestions[f"predicted_{metric}_mean"] = values[:, 0]
        suggestions[f"predicted_{metric}_sem"] = values[:, 1]
    on = np.rint(suggestions[SE_PULSE_DUTY_CYCLE_COLUMN] * SE_PULSE_PERIOD_MS).astype(int)
    suggestions["se_pulse_on_ms"] = on
    suggestions["se_pulse_off_ms"] = SE_PULSE_PERIOD_MS - on
    suggestions["se_pulse_period_ms"] = SE_PULSE_PERIOD_MS
    suggestions["se_pulse_frequency_hz"] = SE_PULSE_FREQUENCY_HZ
    suggestions.insert(0, "ax_trial_index", proposal_trials)
    suggestions.insert(0, "proposal_order", range(1, len(suggestions) + 1))
    result = RefinedAxResult(client, observations, _nondominated(observations), suggestions, metadata)
    if output_dir is not None:
        _export(result, Path(output_dir))
    return result