"""Generate a PDF report summarizing the XRD analysis and BO suggestions."""

from pathlib import Path

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from .bayesian_opt import prepare_optimization_data
from .config import (
    BO_PARAMETER_COLUMNS,
    BO_PARAMETER_GRID_SPECS,
    FIGURES_DIR,
    GENERATED_XRD_METRICS_CSV,
    NOMAD_SYNTHESIS_CSV,
    REPORTS_DIR,
    RESULTS_DATA_DIR,
    SE_PULSE_DUTY_CYCLE_COLUMN,
    SE_PULSE_FREQUENCY_HZ,
    UNASSIGNED_AREA_DIAGNOSTICS_CSV,
    UNASSIGNED_AREA_PENALTY_LAMBDA,
)

PAGE_WIDTH, _ = A4
CONTENT_WIDTH = PAGE_WIDTH - 4 * cm


def _styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="SectionHead",
            parent=styles["Heading2"],
            textColor=colors.HexColor("#1a3b5d"),
            spaceBefore=14,
            spaceAfter=6,
        )
    )
    styles.add(ParagraphStyle(name="BodySmall", parent=styles["BodyText"], fontSize=10, leading=14))
    styles.add(
        ParagraphStyle(
            name="TableHeaderSmall",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.5,
            leading=8.5,
            alignment=1,
        )
    )
    return styles


def _figure(path: Path, width: float = CONTENT_WIDTH) -> Image | None:
    if not path.is_file():
        return None
    img = ImageReader(str(path))
    img_width, img_height = img.getSize()
    height = width * img_height / img_width
    return Image(str(path), width=width, height=height)


def _suggestions_table(suggestions: pd.DataFrame, parameter_columns) -> KeepTogether:
    """Build a styled table of suggested conditions."""
    styles = _styles()
    # Only show parameters actually present in this suggestions table.
    parameter_columns = [c for c in parameter_columns if c in suggestions.columns]
    cols = ["rank", *parameter_columns]
    cols.extend(
        column
        for column in ("se_pulse_on_ms", "se_pulse_off_ms")
        if column in suggestions.columns and column not in cols
    )
    has_std = "predicted_std" in suggestions.columns
    if "predicted_objective" in suggestions.columns:
        cols += ["predicted_objective"] + (["predicted_std"] if has_std else [])
    header_labels = {
        "rank": "#",
        "hard_bake_temp_c": "Hard bake<br/>°C",
        "selenization_temp_c": "Selen. temp<br/>°C",
        "selenization_time_s": "Selen. time<br/>s",
        "cbc_ratio": "CBC ratio",
        SE_PULSE_DUTY_CYCLE_COLUMN: "Se duty<br/>cycle",
        "se_pulse_on_ms": "Se on<br/>ms",
        "se_pulse_off_ms": "Se off<br/>ms",
        "predicted_objective": "Predicted<br/>score",
        "predicted_std": "GP<br/>std",
    }
    header = [
        Paragraph(header_labels.get(column, column.replace("_", " ")), styles["TableHeaderSmall"])
        for column in cols
    ]
    table_data = [header]
    for _, row in suggestions.iterrows():
        entry = []
        for column in cols:
            if column == "rank":
                entry.append(str(int(row[column])))
            elif column == "cbc_ratio":
                entry.append(f"{row[column]:.3f}")
            elif column == SE_PULSE_DUTY_CYCLE_COLUMN:
                entry.append(f"{row[column]:.4f}")
            elif column in ("predicted_objective", "predicted_std"):
                entry.append(f"{row[column]:.3f}")
            elif pd.api.types.is_number(row[column]):
                entry.append(f"{row[column]:.0f}")
            else:
                entry.append(str(row[column]))
        table_data.append(entry)
    col_widths = {
        "rank": 22,
        "hard_bake_temp_c": 47,
        "selenization_temp_c": 48,
        "selenization_time_s": 48,
        "cbc_ratio": 43,
        SE_PULSE_DUTY_CYCLE_COLUMN: 46,
        "se_pulse_on_ms": 38,
        "se_pulse_off_ms": 38,
        "predicted_objective": 50,
        "predicted_std": 38,
    }
    table = Table(
        table_data,
        colWidths=[col_widths[column] for column in cols],
        repeatRows=1,
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3b5d")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 1), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f8")]),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#b0bec5")),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    return KeepTogether([table])


def build_report(output_path: Path | None = None) -> Path:
    """Assemble the analysis + BO report PDF and return its path."""
    output_path = output_path or (REPORTS_DIR / "bacuse2_xrd_bo_report.pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(GENERATED_XRD_METRICS_CSV)
    synthesis = pd.read_csv(NOMAD_SYNTHESIS_CSV)
    diagnostics = pd.read_csv(UNASSIGNED_AREA_DIAGNOSTICS_CSV)
    df_bo = prepare_optimization_data(
        metrics,
        parameter_columns=BO_PARAMETER_COLUMNS,
        synthesis=synthesis,
        include_pulse=True,
        include_process=True,
        diagnostics=diagnostics,
        unassigned_area_penalty_lambda=UNASSIGNED_AREA_PENALTY_LAMBDA,
    )
    suggestions = pd.read_csv(RESULTS_DATA_DIR / "bacuse2_bo_suggestions.csv")
    if "penalty_lambda" not in suggestions or not suggestions["penalty_lambda"].eq(
        UNASSIGNED_AREA_PENALTY_LAMBDA
    ).all():
        raise ValueError(
            "Saved BO suggestions do not match the configured XRD penalty; rerun notebook 3's Bayesian optimization cell."
        )

    styles = _styles()
    story = []

    # Title
    story.append(Paragraph("α-BaCu<sub>2</sub>Se<sub>2</sub> Thin-Film XRD Analysis and Bayesian Optimization", styles["Title"]))
    story.append(Paragraph("NOMAD-driven data extraction, phase matching, and GP-based synthesis planning", styles["Italic"]))
    story.append(Spacer(1, 0.3 * cm))

    # 1. What was done
    story.append(Paragraph("1. What was done", styles["SectionHead"]))
    done = (
        f"Synthesis records and XRD scans were pulled directly from the NOMAD Oasis REST API for the "
        f"{len(synthesis)} α-BaCu<sub>2</sub>Se<sub>2</sub> thin-film samples (INLThinFilmStack entries). "
        f"Synthesis parameters (spin coating, soft/hard bake, selenization) and selenium-pulse settings were "
        f"extracted for all samples; {int(synthesis['se_pulse_on_ms'].notna().sum())} include pulse on/off data. "
        f"Each record also carries the precursor Cu/(Ba+Cu) ratio, the number of spin-coating cycles "
        f"(deposited layers, excluding dynamic-dispense/seed-layer preparation steps), "
        f"the Mo-coating flag, and the spin-to-selenization day gap. "
        f"For the {len(metrics)} films with XRD scans, diffraction patterns were matched against reference "
        f"cards for the α-BaCu<sub>2</sub>Se<sub>2</sub> target (01-080-2950), "
        f"Cu<sub>2</sub>Se (96-101-0582; α-Cu<sub>2</sub>Se), and BaSeO<sub>3</sub> (00-053-0141). "
        f"Phase purity and crystallinity (1/FWHM of the strongest target match) were computed per film. "
        f"A Gaussian-process Bayesian optimizer was trained on the {len(df_bo)} complete films to "
        f"recommend the next synthesis conditions, optimizing objective = "
        f"(0.7 × phase purity + 0.3 × normalized crystallinity) × "
        f"(1 − {UNASSIGNED_AREA_PENALTY_LAMBDA:.2f} × A) over a five-input space: hard-bake "
        f"temperature, selenization temperature and time, Cu/(Ba+Cu) ratio, and continuous Se pulse duty "
        f"cycle. The on+off period is fixed at 3200 ms (3.2 s), so pulse frequency is fixed at "
        f"{SE_PULSE_FREQUENCY_HZ:.4f} Hz and is reported as metadata, not optimized. Intermediate duty "
        f"cycles are model interpolations from three observed levels and should be treated as exploratory; "
        f"recommended on-times are quantized to 1 ms increments. "
        f"CBC is defined as n<sub>Cu</sub>/(n<sub>Ba</sub>+n<sub>Cu</sub>); "
        f"the candidate grid is {BO_PARAMETER_GRID_SPECS['cbc_ratio'][0]:.2f}–"
        f"{BO_PARAMETER_GRID_SPECS['cbc_ratio'][1]:.2f}, so a ratio of 1 (no Ba) is excluded. "
        f"A is the unmatched/overlap "
        f"share of detected peak-interval area, a screening proxy—not a quantitative phase fraction "
        f"or a confirmed assignment of the ~23° feature."
    )
    story.append(Paragraph(done, styles["BodySmall"]))

    # 2. XRD results figure
    story.append(Paragraph("2. XRD phase identification", styles["SectionHead"]))
    story.append(Paragraph(
        "Stacked XRD patterns per series with reference bars for the target and impurity phases. "
        "Markers flag target (star), impurity (square), and unmatched (triangle) peaks.",
        styles["BodySmall"],
    ))
    for name in ("bacuse2_xrd_series_260929A.png", "bacuse2_xrd_all_samples.png"):
        fig = _figure(FIGURES_DIR / name)
        if fig is not None:
            story.append(Spacer(1, 0.15 * cm))
            story.append(fig)

    # 3. Main results
    story.append(Paragraph("3. Main results", styles["SectionHead"]))
    best = df_bo.loc[df_bo["objective"].idxmax()]
    results = (
        f"Across the analyzed films the mean phase purity is {df_bo['phase_purity'].mean():.2f} and mean "
        f"crystallinity is {df_bo['crystallinity'].mean():.2f}. The best film under the penalized score is {best['sample_id']} "
        f"(score {best['objective']:.2f}; unpenalized score {best['objective_unpenalized']:.2f}; "
        f"purity {best['phase_purity']:.2f}, crystallinity {best['crystallinity']:.2f}, "
        f"A {best['unassigned_or_overlap_area_fraction']:.2f}). The optimizer uses the five selected "
        f"inputs only. Other process and substrate descriptors remain available for descriptive "
        f"exploration; duty cycle is modeled continuously, with the fixed pulse frequency excluded "
        f"from the optimizer because it has no observed variation."
    )
    story.append(Paragraph(results, styles["BodySmall"]))
    for name in ("bacuse2_bo_correlations.png", "bacuse2_bo_gp_surface.png"):
        fig = _figure(FIGURES_DIR / name)
        if fig is not None:
            story.append(Spacer(1, 0.15 * cm))
            story.append(fig)

    # 4. Suggested next experiments
    story.append(Paragraph("4. Suggested next experiments", styles["SectionHead"]))
    story.append(Paragraph(
        "Top Gaussian-process suggestions (predicted objective ± std), ranked by expected improvement:",
        styles["BodySmall"],
    ))
    story.append(Spacer(1, 0.15 * cm))
    story.append(_suggestions_table(suggestions, df_bo.attrs["parameter_columns"]))
    story.append(Spacer(1, 0.2 * cm))
    top = suggestions.iloc[0]
    pulse_text = (
        f"duty cycle {top[SE_PULSE_DUTY_CYCLE_COLUMN]:.4f} ({top['se_pulse_on_ms']:.0f} ms on / "
        f"{top['se_pulse_off_ms']:.0f} ms off)"
        if SE_PULSE_DUTY_CYCLE_COLUMN in suggestions.columns
        else ""
    )
    story.append(Paragraph(
        f"Recommended first experiment: hard bake {top['hard_bake_temp_c']:.0f} °C, selenization "
        f"{top['selenization_temp_c']:.0f} °C for {top['selenization_time_s']:.0f} s, CBC ratio "
        f"{top['cbc_ratio']:.3f}, and Se pulse {pulse_text} "
        f"(predicted objective {top['predicted_objective']:.2f}).",
        styles["BodySmall"],
    ))
    fig = _figure(FIGURES_DIR / "bacuse2_bo_parameter_plots.png")
    if fig is not None:
        story.append(Spacer(1, 0.15 * cm))
        story.append(fig)

    # 5. Multi-objective (Ax) optimization
    moo_path = RESULTS_DATA_DIR / "bacuse2_ax_moo_suggestions.csv"
    if moo_path.is_file():
        moo = pd.read_csv(moo_path)
        story.append(Paragraph("5. Multi-objective optimization (Ax)", styles["SectionHead"]))
        story.append(Paragraph(
            "A complementary multi-objective search treats screening-adjusted phase purity and "
            "normalized crystallinity as two separate objectives: both are multiplied by the same "
            f"(1 − {UNASSIGNED_AREA_PENALTY_LAMBDA:.2f} × A) factor. Ax (BoTorch) fits a Gaussian "
            "process per metric and finds the Pareto front via qNEHVI hypervolume maximization. "
            "Its five candidate inputs match notebook 3: hard-bake temperature, selenization "
            "temperature/time, CBC ratio, and continuous pulse duty cycle. Pulse period is fixed "
            f"at 3.2 s ({SE_PULSE_FREQUENCY_HZ:.4f} Hz), so frequency is excluded as a constant. "
            "Duty-cycle suggestions are quantized to 1 ms on-time increments; interpolation between "
            "the three observed duty levels is exploratory. Numeric process recipes "
            "use the same explicit experiment-scale grids, including 25 °C bake steps. This remains a Pareto formulation, not the "
            "same scalar objective as notebook 3. A is an exploratory detected-peak-area screening "
            "proxy, not a phase fraction or a confirmed assignment of the ~23° feature.",
            styles["BodySmall"],
        ))
        story.append(Spacer(1, 0.15 * cm))
        fig = _figure(FIGURES_DIR / "bacuse2_ax_pareto_front.png")
        if fig is not None:
            story.append(fig)
            story.append(Spacer(1, 0.15 * cm))
        story.append(_suggestions_table(moo, df_bo.attrs["parameter_columns"]))

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
        title="α-BaCu2Se2 XRD + BO report",
    )
    doc.build(story)
    return output_path


if __name__ == "__main__":
    print(f"Wrote {build_report()}")
