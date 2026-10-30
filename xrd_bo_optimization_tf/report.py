"""Generate a PDF report summarizing the XRD analysis and BO suggestions."""

from pathlib import Path

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image,
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
    FIGURES_DIR,
    GENERATED_XRD_METRICS_CSV,
    NOMAD_SYNTHESIS_CSV,
    REPORTS_DIR,
    RESULTS_DATA_DIR,
    SE_PULSE_PROFILE_COLUMN,
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
    return styles


def _figure(path: Path, width: float = CONTENT_WIDTH) -> Image | None:
    if not path.is_file():
        return None
    img = ImageReader(str(path))
    img_width, img_height = img.getSize()
    height = width * img_height / img_width
    return Image(str(path), width=width, height=height)


def _suggestions_table(suggestions: pd.DataFrame, parameter_columns) -> Table:
    """Build a styled table of suggested conditions."""
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
    header = [
        c.replace("_", " ").replace(" ms", " (ms)").replace(" s", " (s)").replace(" c", " (°C)")
        for c in cols
    ]
    table_data = [header]
    for _, row in suggestions.iterrows():
        entry = [
            int(row["rank"]),
            *[
                f"{row[c]:.0f}" if pd.api.types.is_number(row[c]) else str(row[c])
                for c in parameter_columns
            ],
        ]
        entry.extend(
            f"{row[column]:.0f}"
            for column in ("se_pulse_on_ms", "se_pulse_off_ms")
            if column in suggestions.columns and column not in parameter_columns
        )
        if "predicted_objective" in suggestions.columns:
            entry.append(f"{row['predicted_objective']:.3f}")
            if has_std:
                entry.append(f"{row['predicted_std']:.3f}")
        table_data.append(entry)
    table = Table(table_data, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3b5d")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f8")]),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#b0bec5")),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    return table


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
        f"For the {len(metrics)} films with XRD scans, diffraction patterns were matched against ICDD reference "
        f"cards for the α-BaCu<sub>2</sub>Se<sub>2</sub> target (01-080-2950) and the Cu<sub>2</sub>Se (04-024-2128) "
        f"and BaSeO<sub>3</sub> (00-053-0141) impurities. Phase purity and crystallinity (1/FWHM of the strongest "
        f"target match) were computed per film. A Gaussian-process Bayesian optimizer was trained on the "
        f"{len(df_bo)} complete films to recommend the next synthesis conditions, optimizing "
        f"objective = (0.7 × phase purity + 0.3 × normalized crystallinity) × "
        f"(1 − {UNASSIGNED_AREA_PENALTY_LAMBDA:.2f} × A) over a "
        f"five-input space (hard-bake temperature, selenization temperature and time, "
        f"Cu/(Ba+Cu) ratio, and one categorical Se pulse profile derived from on/off timings). "
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
        f"exploration; the pulse profile is categorical and is not treated as an ordinal correlation."
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
        f"{top[SE_PULSE_PROFILE_COLUMN]} profile ({top['se_pulse_on_ms']:.0f} ms on / "
        f"{top['se_pulse_off_ms']:.0f} ms off)"
        if SE_PULSE_PROFILE_COLUMN in suggestions.columns
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
            "temperature/time, CBC ratio, and a categorical joint pulse profile. Numeric recipes "
            "use the same explicit experiment-scale grids, including 25 °C bake steps, and the "
            "profile is reported with exact on/off timings. This remains a Pareto formulation, not the "
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
