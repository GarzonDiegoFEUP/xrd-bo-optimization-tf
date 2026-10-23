"""XRD and Bayesian-optimization plotting helpers."""

from pathlib import Path
import re
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from .config import BO_PARAMETER_COLUMNS, FIGURES_DIR


def plot_bo_suggestions(
    observed: pd.DataFrame,
    suggestions: pd.DataFrame,
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot observed objectives and suggested objective estimates."""
    figure, axis = plt.subplots(figsize=(7, 4))
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
        figsize=(3.2 * len(parameter_columns), 6.5),
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
                axis.set_xlabel(parameter, fontsize=9)
            if column_index == 0:
                axis.set_ylabel(label, fontsize=9)
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

        figure, axis = plt.subplots(figsize=(8, 1.1 * len(sample_ids) + 2.5))
        for index, sample_id in enumerate(sample_ids):
            result = xrd_results[sample_id]
            theta = result["two_theta"]
            spectrum = result["bg_removed"]
            scale = max(float(spectrum.max()), 1e-9)
            offset = float(index)
            axis.plot(theta, spectrum / scale + offset, color="navy", linewidth=0.9)
            for key, color, marker, label in (
                ("target", "green", "*", "target phase"),
                ("impurity", "darkorange", "s", "impurity phase"),
                ("unmatched", "red", "^", "unmatched peak"),
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
            reference_colors = {
                "BaCu2Se2": "crimson",
                "Cu2Se": "darkgreen",
                "BaSe": "steelblue",
                "BaSeO4": "purple",
                "BaSeO3": "saddlebrown",
            }
            for phase, positions in references.items():
                axis.vlines(
                    positions,
                    offset - 0.15,
                    offset - 0.08,
                    color=reference_colors.get(phase, "gray"),
                    alpha=0.35,
                    linewidth=0.7,
                )
            axis.text(
                1.01,
                offset,
                sample_id,
                transform=axis.get_yaxis_transform(),
                va="center",
                fontsize=8,
            )

        axis.set_yticks([])
        axis.set_xlabel("2-theta (degrees)")
        axis.set_title(f"Series {series}: stacked XRD patterns")
        axis.grid(True, axis="x", alpha=0.2, linestyle=":")
        axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
        figure.tight_layout()
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
    for index, sample_id in enumerate(sample_ids):
        result = xrd_results[sample_id]
        spectrum = result["bg_removed"]
        scale = max(float(spectrum.max()), 1e-9)
        axis.plot(
            result["two_theta"],
            spectrum / scale + index,
            color="navy",
            linewidth=0.8,
        )
        for key, color, marker in (
            ("target", "green", "*"),
            ("impurity", "darkorange", "s"),
            ("unmatched", "red", "^"),
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
    axis.set_yticks([])
    axis.set_xlabel("2-theta (degrees)")
    axis.set_title(f"All analyzed XRD scans (n={len(sample_ids)})")
    axis.grid(True, axis="x", alpha=0.2, linestyle=":")
    figure.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
    return figure
