"""Objective preparation and Gaussian-process synthesis suggestions."""

from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    BO_PARAMETER_COLUMNS,
    BO_PARAMETER_GRID_SPECS,
    CRYSTALLINITY_WEIGHT,
    PHASE_PURITY_WEIGHT,
    RESULTS_DATA_DIR,
    SE_PULSE_DUTY_CYCLE_BOUNDS,
    SE_PULSE_DUTY_CYCLE_COLUMN,
    SE_PULSE_FREQUENCY_HZ,
    SE_PULSE_ON_MS_BOUNDS,
    SE_PULSE_PERIOD_MS,
    UNASSIGNED_AREA_PENALTY_LAMBDA,
    XRD_PARAMETER_COLUMNS,
)

# Selenium pulse knobs available for optimization (se_pulse_count is collinear
# with selenization_time_s and is intentionally excluded).
PULSE_PARAMETER_COLUMNS = (SE_PULSE_DUTY_CYCLE_COLUMN,)

# Process descriptors retained from the synthesis table for exploration.
PROCESS_MERGE_COLUMNS = (
    "se_pulse_on_ms",
    "se_pulse_off_ms",
    "se_pulse_count",
    "cbc_ratio",
    "n_spin_cycles",
    "mo_coated",
    "spin_to_se_days",
)
# CBC ratio is the only selected process input; the day gap remains metadata.
PROCESS_BO_COLUMNS = ("cbc_ratio",)


def _pulse_duty_cycle(on_ms: object, off_ms: object) -> float | None:
    """Validate a fixed-period pulse pair and return its duty cycle."""
    if pd.isna(on_ms) or pd.isna(off_ms):
        return None
    on_value, off_value = float(on_ms), float(off_ms)
    if on_value <= 0 or off_value <= 0:
        raise ValueError("Selenium pulse on/off times must both be positive")
    period_ms = on_value + off_value
    if not np.isclose(period_ms, SE_PULSE_PERIOD_MS, atol=1e-6):
        raise ValueError(
            f"Selenium pulse period must be {SE_PULSE_PERIOD_MS} ms; got {period_ms} ms"
        )
    return on_value / period_ms


def _grid_values(spec: tuple[float, float, float, int]) -> list[float]:
    lower, upper, step, digits = spec
    count = int(round((upper - lower) / step))
    return [round(lower + index * step, digits) for index in range(count + 1)]


def prepare_optimization_data(
    metrics: pd.DataFrame,
    parameter_columns: tuple[str, ...] = XRD_PARAMETER_COLUMNS,
    synthesis: pd.DataFrame | None = None,
    include_pulse: bool = False,
    include_process: bool = False,
    diagnostics: pd.DataFrame | None = None,
    unassigned_area_penalty_lambda: float = UNASSIGNED_AREA_PENALTY_LAMBDA,
) -> pd.DataFrame:
    """Prepare complete observations and calculate the scalar BO objective.

    The base objective is the weighted phase-purity/crystallinity score. When
    ``diagnostics`` is supplied, multiply that score by ``(1 - lambda * A)``,
    where ``A`` is the unmatched/overlap detected-peak-area diagnostic. This is
    an exploratory screening proxy, not a phase fraction or a specific-phase
    assignment. Without diagnostics, the base objective is used unchanged.

    When include_pulse / include_process is True, the synthesis table is merged
    in (keyed on sample_id). Pulse on/off values become one continuous duty
    cycle at a fixed period, and CBC ratio is the only other process input.
    """
    if not 0.0 <= unassigned_area_penalty_lambda <= 1.0:
        raise ValueError("unassigned_area_penalty_lambda must be between 0 and 1")
    parameters = tuple(parameter_columns)
    if include_pulse or include_process:
        if synthesis is None:
            raise ValueError("include_pulse/include_process require the synthesis table")
        if "se_pulse_count" in parameters:
            raise ValueError(
                "se_pulse_count is collinear with selenization_time_s; exclude it"
            )
        merge_cols = ["sample_id", *PROCESS_MERGE_COLUMNS]
        merged = synthesis[
            [c for c in merge_cols if c in synthesis.columns]
        ].drop_duplicates("sample_id")
        metrics = metrics.merge(merged, on="sample_id", how="left")
        if include_pulse:
            if not {"se_pulse_on_ms", "se_pulse_off_ms"}.issubset(merged.columns):
                raise ValueError("Pulse timing requires on/off columns in synthesis")
            metrics[SE_PULSE_DUTY_CYCLE_COLUMN] = [
                _pulse_duty_cycle(on_ms, off_ms)
                for on_ms, off_ms in zip(
                    metrics["se_pulse_on_ms"], metrics["se_pulse_off_ms"]
                )
            ]
            metrics["se_pulse_period_ms"] = SE_PULSE_PERIOD_MS
            metrics["se_pulse_frequency_hz"] = SE_PULSE_FREQUENCY_HZ
            parameters = tuple(dict.fromkeys((*parameters, *PULSE_PARAMETER_COLUMNS)))
        if include_process:
            parameters = tuple(dict.fromkeys((*parameters, *PROCESS_BO_COLUMNS)))

    diagnostic_column = "unassigned_or_overlap_area_fraction"
    if diagnostics is not None:
        diagnostic_columns = {"sample_id", diagnostic_column}
        missing_diagnostics = diagnostic_columns.difference(diagnostics.columns)
        if missing_diagnostics:
            raise ValueError(
                f"XRD diagnostics are missing required columns: {sorted(missing_diagnostics)}"
            )
        if diagnostics["sample_id"].duplicated().any():
            raise ValueError("XRD diagnostics must contain one row per sample_id")
        if diagnostic_column in metrics.columns:
            raise ValueError(
                f"Metrics already contain {diagnostic_column!r}; pass the source diagnostics separately"
            )
        metrics = metrics.merge(
            diagnostics[["sample_id", diagnostic_column]],
            on="sample_id",
            how="left",
            validate="one_to_one",
        )

    required = {
        *parameters,
        "sample_id",
        "crystallinity",
        "phase_purity",
    }
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"Metrics table is missing required columns: {sorted(missing)}")
    required_rows = [*parameters, "crystallinity", "phase_purity"]
    if diagnostics is not None:
        required_rows.append(diagnostic_column)
    prepared = metrics.dropna(subset=required_rows).copy()
    if prepared.empty:
        raise ValueError("No complete rows are available for Bayesian optimization")
    snap_deltas = {}
    for column in parameters:
        if column not in BO_PARAMETER_GRID_SPECS:
            continue
        grid = np.asarray(_grid_values(BO_PARAMETER_GRID_SPECS[column]), dtype=float)
        observed = prepared[column].astype(float).to_numpy()
        nearest = grid[np.abs(observed[:, None] - grid[None, :]).argmin(axis=1)]
        max_delta = float(abs(observed - nearest).max())
        tolerance = BO_PARAMETER_GRID_SPECS[column][2] / 2.0 + 1e-9
        if max_delta > tolerance:
            raise ValueError(
                f"Observed {column} values exceed half-step grid snapping tolerance: {max_delta}"
            )
        prepared[column] = nearest
        snap_deltas[column] = max_delta
    values = prepared["crystallinity"].astype(float)
    spread = float(values.max() - values.min())
    prepared["crystallinity_norm"] = (
        (values - values.min()) / spread if spread > 0 else 0.0
    )
    prepared["objective_unpenalized"] = (
        PHASE_PURITY_WEIGHT * prepared["phase_purity"].astype(float)
        + CRYSTALLINITY_WEIGHT * prepared["crystallinity_norm"]
    )
    if diagnostics is not None:
        diagnostic_values = prepared[diagnostic_column].astype(float)
        if (
            not diagnostic_values.map(pd.notna).all()
            or (diagnostic_values < 0.0).any()
            or (diagnostic_values > 1.0).any()
        ):
            raise ValueError(f"{diagnostic_column} must be finite and between 0 and 1")
        prepared["objective_penalty_factor"] = (
            1.0 - unassigned_area_penalty_lambda * diagnostic_values
        )
        prepared["objective"] = (
            prepared["objective_unpenalized"] * prepared["objective_penalty_factor"]
        )
        prepared.attrs["objective_penalty_lambda"] = unassigned_area_penalty_lambda
    else:
        prepared["objective_penalty_factor"] = 1.0
        prepared["objective"] = prepared["objective_unpenalized"]
    # Record the optimization-space columns for downstream plotting/BO calls.
    prepared.attrs["parameter_columns"] = parameters
    prepared.attrs["parameter_snap_deltas"] = snap_deltas
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
    from skopt.space import Categorical, Real

    bounds = []
    for column in parameter_columns:
        if column == SE_PULSE_DUTY_CYCLE_COLUMN:
            lower, upper = SE_PULSE_DUTY_CYCLE_BOUNDS
            observed = prepared[column].astype(float)
            if ((observed < lower) | (observed > upper)).any():
                raise ValueError("Observed pulse duty cycle is outside configured bounds")
            bounds.append(Real(lower, upper, name=column))
            continue
        if column in BO_PARAMETER_GRID_SPECS:
            values = _grid_values(BO_PARAMETER_GRID_SPECS[column])
        else:
            values = sorted(prepared[column].dropna().unique().tolist())
        if len(values) < 2:
            raise ValueError(
                f"Cannot optimize {column}: fewer than two discrete choices are available"
            )
        bounds.append(Categorical(values, name=column))

    optimizer = Optimizer(
        bounds, base_estimator="GP", acq_func="EI", random_state=random_state
    )
    parameter_names = list(parameter_columns)
    observed_conditions = prepared.loc[:, parameter_names].values.tolist()
    observed_objectives = (-prepared["objective"].astype(float)).tolist()
    optimizer.tell(observed_conditions, observed_objectives)
    suggestions = optimizer.ask(n_points=n_suggestions, strategy="cl_min")
    if SE_PULSE_DUTY_CYCLE_COLUMN in parameter_columns:
        duty_index = parameter_columns.index(SE_PULSE_DUTY_CYCLE_COLUMN)
        for condition in suggestions:
            on_ms = int(
                np.clip(
                    np.rint(condition[duty_index] * SE_PULSE_PERIOD_MS),
                    *SE_PULSE_ON_MS_BOUNDS,
                )
            )
            condition[duty_index] = on_ms / SE_PULSE_PERIOD_MS
    predicted_mean, predicted_std = optimizer.models[-1].predict(
        optimizer.space.transform(suggestions), return_std=True
    )

    result = pd.DataFrame(suggestions, columns=parameter_names)
    result.insert(0, "rank", range(1, len(result) + 1))
    result["predicted_objective"] = -predicted_mean
    result["predicted_std"] = predicted_std
    penalty_lambda = prepared.attrs.get("objective_penalty_lambda")
    if penalty_lambda is not None:
        result.insert(1, "penalty_lambda", penalty_lambda)
    if SE_PULSE_DUTY_CYCLE_COLUMN in result:
        on_lower, on_upper = SE_PULSE_ON_MS_BOUNDS
        on_ms = np.rint(result[SE_PULSE_DUTY_CYCLE_COLUMN] * SE_PULSE_PERIOD_MS)
        on_ms = np.clip(on_ms, on_lower, on_upper).astype(int)
        result["se_pulse_on_ms"] = on_ms
        result["se_pulse_off_ms"] = SE_PULSE_PERIOD_MS - on_ms
        result["se_pulse_period_ms"] = SE_PULSE_PERIOD_MS
        result["se_pulse_frequency_hz"] = SE_PULSE_FREQUENCY_HZ
    return result


def save_suggestions(
    suggestions: pd.DataFrame,
    path: Path = RESULTS_DATA_DIR / "bacuse2_bo_suggestions.csv",
) -> Path:
    """Write suggested synthesis conditions to CSV and return the output path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    suggestions.to_csv(path, index=False)
    return path
