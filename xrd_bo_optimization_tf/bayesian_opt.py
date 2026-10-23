"""Objective preparation and Gaussian-process synthesis suggestions."""

from pathlib import Path

import pandas as pd

from .config import (
    BO_PARAMETER_COLUMNS,
    CRYSTALLINITY_WEIGHT,
    PHASE_PURITY_WEIGHT,
    RESULTS_DATA_DIR,
)

# Selenium pulse knobs available for optimization (se_pulse_count is collinear
# with selenization_time_s and is intentionally excluded).
PULSE_PARAMETER_COLUMNS = ("se_pulse_on_ms", "se_pulse_off_ms")


def prepare_optimization_data(
    metrics: pd.DataFrame,
    parameter_columns: tuple[str, ...] = BO_PARAMETER_COLUMNS,
    synthesis: pd.DataFrame | None = None,
    include_pulse: bool = False,
) -> pd.DataFrame:
    """Filter incomplete rows and calculate the project's weighted objective.

    When include_pulse is True, pulse parameters are merged from the synthesis
    table (keyed on sample_id) and appended to the optimization space.
    """
    parameters = tuple(parameter_columns)
    if include_pulse:
        if synthesis is None:
            raise ValueError("include_pulse=True requires the synthesis table")
        if "se_pulse_count" in parameters:
            raise ValueError(
                "se_pulse_count is collinear with selenization_time_s; exclude it"
            )
        pulse = synthesis[
            ["sample_id", *PULSE_PARAMETER_COLUMNS]
        ].drop_duplicates("sample_id")
        metrics = metrics.merge(pulse, on="sample_id", how="left")
        parameters = (*parameters, *PULSE_PARAMETER_COLUMNS)

    required = {
        *parameters,
        "sample_id",
        "crystallinity",
        "phase_purity",
    }
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"Metrics table is missing required columns: {sorted(missing)}")
    prepared = metrics.dropna(
        subset=[*parameters, "crystallinity", "phase_purity"]
    ).copy()
    if prepared.empty:
        raise ValueError("No complete rows are available for Bayesian optimization")
    values = prepared["crystallinity"].astype(float)
    spread = float(values.max() - values.min())
    prepared["crystallinity_norm"] = (
        (values - values.min()) / spread if spread > 0 else 0.0
    )
    prepared["objective"] = (
        PHASE_PURITY_WEIGHT * prepared["phase_purity"].astype(float)
        + CRYSTALLINITY_WEIGHT * prepared["crystallinity_norm"]
    )
    # Record the optimization-space columns for downstream plotting/BO calls.
    prepared.attrs["parameter_columns"] = parameters
    return prepared


def suggest_next_conditions(
    prepared: pd.DataFrame,
    n_suggestions: int = 5,
    random_state: int = 0,
    parameter_columns: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Fit a Gaussian-process optimizer and return ranked next conditions."""
    if parameter_columns is None:
        parameter_columns = prepared.attrs.get(
            "parameter_columns", BO_PARAMETER_COLUMNS
        )
    if n_suggestions < 1:
        raise ValueError("n_suggestions must be at least 1")
    if len(prepared) < 2:
        raise ValueError("At least two complete observations are required")
    missing = set(parameter_columns).difference(prepared.columns)
    if missing:
        raise ValueError(f"Optimization data is missing parameters: {sorted(missing)}")
    if "objective" not in prepared:
        raise ValueError("Optimization data must include an objective column")

    from skopt import Optimizer
    from skopt.space import Real

    bounds = []
    for column in parameter_columns:
        lower = float(prepared[column].min())
        upper = float(prepared[column].max())
        if lower >= upper:
            raise ValueError(
                f"Cannot optimize {column}: observed values do not define a range"
            )
        bounds.append(Real(lower, upper, name=column))

    optimizer = Optimizer(
        bounds, base_estimator="GP", acq_func="EI", random_state=random_state
    )
    parameter_names = list(parameter_columns)
    observed_conditions = prepared.loc[:, parameter_names].values.tolist()
    observed_objectives = (-prepared["objective"].astype(float)).tolist()
    optimizer.tell(observed_conditions, observed_objectives)
    suggestions = optimizer.ask(n_points=n_suggestions, strategy="cl_min")
    predicted_mean, predicted_std = optimizer.models[-1].predict(
        optimizer.space.transform(suggestions), return_std=True
    )

    result = pd.DataFrame(suggestions, columns=parameter_names)
    result.insert(0, "rank", range(1, len(result) + 1))
    result["predicted_objective"] = -predicted_mean
    result["predicted_std"] = predicted_std
    result.loc[:, parameter_names] = result.loc[:, parameter_names].round(1)
    return result


def save_suggestions(
    suggestions: pd.DataFrame,
    path: Path = RESULTS_DATA_DIR / "bacuse2_bo_suggestions.csv",
) -> Path:
    """Write suggested synthesis conditions to CSV and return the output path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    suggestions.to_csv(path, index=False)
    return path
