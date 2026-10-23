"""XRD and Bayesian-optimization plotting helpers."""

from pathlib import Path
import re
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

from .config import BO_PARAMETER_COLUMNS, FIGURES_DIR, REFERENCE_CARD_FILES

# Presentation-friendly defaults with a colorblind-safe palette.
sns.set_theme(context="talk", style="ticks")
sns.set_palette("colorblind")
COLORBLIND = sns.color_palette("colorblind")
# Named roles drawn from the colorblind palette for consistent phase coding.
COLOR_SCAN = COLORBLIND[0]       # blue
COLOR_TARGET = COLORBLIND[2]     # green
COLOR_IMPURITY = COLORBLIND[1]   # orange
COLOR_UNMATCHED = COLORBLIND[3]  # reddish pink
REFERENCE_COLORS = {
    "BaCu2Se2": COLORBLIND[2],
    "Cu2Se": COLORBLIND[1],
    "BaSeO3": COLORBLIND[9],  # sky blue (avoids hard-to-see yellow)
}
# Only these phases are drawn as reference bars (BaSe/BaSeO4 stay matched in
# purity but are hidden from plots, per project convention).
PLOT_PHASES = tuple(REFERENCE_COLORS)
# Legend entries for detected-peak markers only (reference cards get their own).
MATCH_LEGEND = (
    ("target", COLOR_TARGET, "*"),
    ("impurity", COLOR_IMPURITY, "s"),
    ("unmatched", COLOR_UNMATCHED, "^"),
)


def _clip_x_to_data(axis: plt.Axes, spectra: list[Any]) -> tuple[float, float] | None:
    """Limit the x-axis to the experimental 2-theta range; return that range."""
    ranges = [
        (float(result["two_theta"].min()), float(result["two_theta"].max()))
        for result in spectra
        if result.get("two_theta") is not None and len(result["two_theta"])
    ]
    if not ranges:
        return None
    limits = (min(start for start, _ in ranges), max(end for _, end in ranges))
    axis.set_xlim(*limits)
    return limits


def _phase_label(phase: str) -> str:
    """Render a phase key as a formatted chemical formula."""
    formulas = {
        "BaCu2Se2": "$\\alpha$-BaCu$_2$Se$_2$",
        "Cu2Se": "Cu$_2$Se",
        "BaSe": "BaSe",
        "BaSeO4": "BaSeO$_4$",
        "BaSeO3": "BaSeO$_3$",
    }
    return formulas.get(phase, phase)


def _reference_legend_handles() -> list[Line2D]:
    """Legend handles for each plotted reference card, labeled with its number."""
    handles = []
    for phase in PLOT_PHASES:
        filename = REFERENCE_CARD_FILES[phase]
        formula = _phase_label(phase)
        card = filename.split(" ", 1)[0]
        label = f"{formula} ({card})"
        handles.append(
            Line2D(
                [0],
                [0],
                color=REFERENCE_COLORS.get(phase, "gray"),
                linewidth=2,
                label=label,
            )
        )
    return handles


def plot_bo_suggestions(
    observed: pd.DataFrame,
    suggestions: pd.DataFrame,
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot observed objectives and suggested objective estimates."""
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.scatter(
        observed["objective"],
        observed["objective"].rank(ascending=False),
        label="observed",
    )
    axis.errorbar(
        suggestions["predicted_objective"],
        suggestions["rank"],
        xerr=suggestions["predicted_std"],
        fmt="s",
        color="C3",
        label="suggested (GP mean +/- std)",
    )
    axis.axvline(
        observed["objective"].max(),
        color="gray",
        linestyle="--",
        linewidth=1,
        label="best observed",
    )
    axis.set_xlabel("objective")
    axis.set_ylabel("rank (observed) / suggestion #")
    axis.legend()
    figure.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150)
    return figure


def plot_parameter_metrics(
    observed: pd.DataFrame,
    suggestions: pd.DataFrame,
    parameter_columns: tuple[str, ...] = BO_PARAMETER_COLUMNS,
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot phase purity and crystallinity against each optimized parameter."""
    figure, axes = plt.subplots(
        2,
        len(parameter_columns),
        figsize=(3.6 * len(parameter_columns), 8),
        sharex="col",
    )
    scatter = None
    for column_index, parameter in enumerate(parameter_columns):
        for row_index, (metric, label) in enumerate(
            (
                ("phase_purity", "phase purity"),
                ("crystallinity", "crystallinity (1/FWHM)"),
            )
        ):
            axis = axes[row_index, column_index]
            scatter = axis.scatter(
                observed[parameter],
                observed[metric],
                c=observed["objective"],
                cmap="viridis",
                s=45,
                edgecolor="k",
                linewidth=0.5,
            )
            axis.axvline(suggestions.loc[0, parameter], color="C3", linestyle="--")
            axis.grid(alpha=0.3)
            if row_index == 1:
                axis.set_xlabel(parameter)
                axis.tick_params(axis="x", rotation=30)
            if column_index == 0:
                axis.set_ylabel(label)
            if row_index == 0:
                axis.set_ylim(-0.05, 1.05)
    if scatter is not None:
        figure.colorbar(scatter, ax=axes, label="objective", shrink=0.8)
    figure.suptitle(
        f"Observed metrics vs synthesis parameters (n = {len(observed)})",
        fontweight="bold",
    )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
    return figure


def plot_xrd_series(
    xrd_results: dict[str, dict[str, Any]],
    metrics: pd.DataFrame,
    references: dict[str, list[float]],
    output_dir: Path = FIGURES_DIR,
) -> dict[str, plt.Figure]:
    """Create one stacked XRD figure for each series with available scans."""
    figures = {}
    if metrics.empty or not xrd_results:
        return figures

    metric_rows = metrics.drop_duplicates("sample_id").set_index("sample_id")
    series_names = sorted(metrics["series"].dropna().astype(str).unique())
    for series in series_names:
        sample_ids = sorted(
            sample_id
            for sample_id in metric_rows.index
            if str(metric_rows.loc[sample_id, "series"]) == series
            and sample_id in xrd_results
        )
        if not sample_ids:
            continue

        n_samples = len(sample_ids)
        figure, axis = plt.subplots(figsize=(10.5, 1.1 * n_samples + 3.0))
        plotted_results = [xrd_results[sample_id] for sample_id in sample_ids]
        for index, sample_id in enumerate(sample_ids):
            result = xrd_results[sample_id]
            theta = result["two_theta"]
            spectrum = result["bg_removed"]
            scale = max(float(spectrum.max()), 1e-9)
            offset = float(index)
            axis.plot(theta, spectrum / scale + offset, color=COLOR_SCAN, linewidth=0.9)
            for key, color, marker, label in (
                ("target", COLOR_TARGET, "*", f"{_phase_label('BaCu2Se2')} match"),
                ("impurity", COLOR_IMPURITY, "s", "impurity match"),
                ("unmatched", COLOR_UNMATCHED, "^", "unmatched peak"),
            ):
                matches = result.get(key, [])
                if matches:
                    axis.scatter(
                        [match["detected_theta"] for match in matches],
                        [match["intensity"] / scale + offset for match in matches],
                        color=color,
                        marker=marker,
                        s=40,
                        label=label if index == 0 else "_nolegend_",
                        zorder=3,
                    )
            # Label each scan inside the plot near the right, just above its baseline.
            axis.text(
                0.80,
                offset + 0.3,
                sample_id,
                transform=axis.get_yaxis_transform(),
                va="bottom",
                ha="left",
            )

        # One shared set of reference bars below all experimental scans.
        ref_base = -0.6
        plotted_refs = {p: references[p] for p in PLOT_PHASES if p in references}
        for phase_index, (phase, positions) in enumerate(plotted_refs.items()):
            axis.vlines(
                positions,
                ref_base - 0.25 * phase_index,
                ref_base - 0.25 * phase_index + 0.18,
                color=REFERENCE_COLORS.get(phase, "gray"),
                alpha=0.8,
                linewidth=1.2,
            )
        # Headroom above the top scan so tall peaks are not clipped.
        axis.set_ylim(ref_base - 0.25 * len(plotted_refs), n_samples + 0.3)

        x_limits = _clip_x_to_data(axis, plotted_results)
        if x_limits is not None:
            start, end = x_limits
            interior = axis.get_xticks()
            interior = interior[(interior > start) & (interior < end)]
            axis.set_xticks(np.unique(np.concatenate(([start], interior, [end]))))
        axis.set_yticks([])
        axis.set_ylabel("Relative Intensity (arb. units)")
        axis.set_xlabel("2-theta (degrees)")
        axis.set_title(f"Series {series}: stacked XRD patterns")
        handles, _ = axis.get_legend_handles_labels()
        # Reserve a right margin for the sample names and the legend; the legend
        # sits in the right margin with its bottom aligned to the y-axis start.
        figure.tight_layout(rect=(0, 0, 0.78, 1))
        axis.legend(
            handles=handles + _reference_legend_handles(),
            loc="lower left",
            bbox_to_anchor=(1.02, 0.0),
            ncol=1,
            frameon=False,
            fontsize="x-small",
        )
        safe_series = re.sub(r"[^A-Za-z0-9_.-]+", "_", series)
        output_dir.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            output_dir / f"bacuse2_xrd_series_{safe_series}.png",
            dpi=150,
            bbox_inches="tight",
        )
        figures[series] = figure
    return figures


def plot_all_xrd_patterns(
    xrd_results: dict[str, dict[str, Any]],
    metrics: pd.DataFrame,
    output_path: Path | None = None,
) -> plt.Figure | None:
    """Create one normalized stacked plot of all analyzed XRD scans."""
    if metrics.empty or not xrd_results:
        return None
    sample_ids = [
        sample_id
        for sample_id in metrics.sort_values(["series", "sample_id"])["sample_id"]
        if sample_id in xrd_results
    ]
    if not sample_ids:
        return None

    figure, axis = plt.subplots(figsize=(9, 0.35 * len(sample_ids) + 2.5))
    plotted_results = [xrd_results[sample_id] for sample_id in sample_ids]
    for index, sample_id in enumerate(sample_ids):
        result = xrd_results[sample_id]
        spectrum = result["bg_removed"]
        scale = max(float(spectrum.max()), 1e-9)
        axis.plot(
            result["two_theta"],
            spectrum / scale + index,
            color=COLOR_SCAN,
            linewidth=0.8,
        )
        for key, color, marker in (
            ("target", COLOR_TARGET, "*"),
            ("impurity", COLOR_IMPURITY, "s"),
            ("unmatched", COLOR_UNMATCHED, "^"),
        ):
            matches = result.get(key, [])
            if matches:
                axis.scatter(
                    [match["detected_theta"] for match in matches],
                    [match["intensity"] / scale + index for match in matches],
                    color=color,
                    marker=marker,
                    s=32,
                )
    _clip_x_to_data(axis, plotted_results)
    axis.set_yticks([])
    axis.set_xlabel("2-theta (degrees)")
    axis.set_title(f"All analyzed XRD scans (n={len(sample_ids)})")
    axis.grid(True, axis="x", alpha=0.2, linestyle=":")
    figure.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
    return figure


def plot_correlation_bars(
    prepared: pd.DataFrame,
    parameter_columns: tuple[str, ...],
    target_columns: tuple[str, ...] = ("phase_purity", "crystallinity_norm", "objective"),
    output_path: Path | None = None,
) -> plt.Figure:
    """Bar chart of Spearman rank correlation between parameters and targets."""
    correlations = pd.DataFrame(
        {
            target: [
                prepared[column].corr(prepared[target], method="spearman")
                for column in parameter_columns
            ]
            for target in target_columns
        },
        index=list(parameter_columns),
    )
    figure, axis = plt.subplots(figsize=(12, 6))
    correlations.plot.bar(ax=axis, width=0.8, edgecolor="k", linewidth=0.5)
    axis.axhline(0, color="k", linewidth=0.8)
    axis.set_ylabel("Spearman correlation")
    axis.set_ylim(-1, 1)
    axis.set_xticklabels(correlations.index, rotation=25, ha="right")
    axis.set_title("Parameter influence on XRD metrics (rank correlation)")
    axis.legend(title="", frameon=False, ncol=len(target_columns))
    figure.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
    return figure


def plot_gp_surface(
    prepared: pd.DataFrame,
    param_x: str,
    param_y: str,
    fixed: dict[str, float] | None = None,
    grid_points: int = 60,
    random_state: int = 0,
    output_path: Path | None = None,
) -> plt.Figure:
    """GP-predicted objective over a 2D slice of the parameter space.

    All parameters except param_x and param_y are held at `fixed` values
    (defaults to the best observed row).
    """
    from skopt import Optimizer
    from skopt.space import Real

    parameter_names = list(prepared.attrs.get("parameter_columns", BO_PARAMETER_COLUMNS))
    if fixed is None:
        best = prepared.loc[prepared["objective"].idxmax()]
        fixed = {
            name: float(best[name])
            for name in parameter_names
            if name not in (param_x, param_y)
        }

    bounds = [
        Real(float(prepared[name].min()), float(prepared[name].max()), name=name)
        for name in parameter_names
    ]
    optimizer = Optimizer(
        bounds, base_estimator="GP", acq_func="EI", random_state=random_state
    )
    optimizer.tell(
        prepared.loc[:, parameter_names].values.tolist(),
        (-prepared["objective"].astype(float)).tolist(),
    )
    model = optimizer.models[-1]

    x_values = np.linspace(bounds[parameter_names.index(param_x)].low, bounds[parameter_names.index(param_x)].high, grid_points)
    y_values = np.linspace(bounds[parameter_names.index(param_y)].low, bounds[parameter_names.index(param_y)].high, grid_points)
    grid_x, grid_y = np.meshgrid(x_values, y_values)
    points = []
    for gx, gy in zip(grid_x.ravel(), grid_y.ravel()):
        row = dict(fixed)
        row[param_x] = gx
        row[param_y] = gy
        points.append([row[name] for name in parameter_names])
    predicted = -model.predict(optimizer.space.transform(points)).reshape(grid_x.shape)

    figure, axis = plt.subplots(figsize=(10, 8))
    mesh = axis.pcolormesh(grid_x, grid_y, predicted, cmap="viridis", shading="auto")
    figure.colorbar(mesh, ax=axis, label="GP predicted objective")
    axis.scatter(
        prepared[param_x],
        prepared[param_y],
        c=prepared["objective"],
        cmap="viridis",
        s=60,
        edgecolor="w",
        linewidth=0.8,
        vmin=float(predicted.min()),
        vmax=float(predicted.max()),
        zorder=3,
    )
    axis.set_xlabel(param_x)
    axis.set_ylabel(param_y)
    fixed_text = ", ".join(
        f"{name.replace('selenization_', 'sel_').replace('_temp_c', 'T').replace('_time_s', 't')}={v:g}"
        for name, v in fixed.items()
    )
    axis.set_title(f"GP objective surface\n({fixed_text})", fontsize="small")
    figure.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
    return figure
