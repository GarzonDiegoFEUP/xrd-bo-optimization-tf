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
    FIGURES_DIR,
    GENERATED_XRD_METRICS_CSV,
    NOMAD_SYNTHESIS_CSV,
    REPORTS_DIR,
    RESULTS_DATA_DIR,
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


def build_report(output_path: Path | None = None) -> Path:
    """Assemble the analysis + BO report PDF and return its path."""
    output_path = output_path or (REPORTS_DIR / "bacuse2_xrd_bo_report.pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(GENERATED_XRD_METRICS_CSV)
    synthesis = pd.read_csv(NOMAD_SYNTHESIS_CSV)
    df_bo = prepare_optimization_data(metrics, synthesis=synthesis, include_pulse=True)
    suggestions = pd.read_csv(RESULTS_DATA_DIR / "bacuse2_bo_suggestions.csv")

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
        f"For the {len(metrics)} films with XRD scans, diffraction patterns were matched against ICDD reference "
        f"cards for the α-BaCu<sub>2</sub>Se<sub>2</sub> target (01-080-2950) and the Cu<sub>2</sub>Se (04-024-2128) "
        f"and BaSeO<sub>3</sub> (00-053-0141) impurities. Phase purity and crystallinity (1/FWHM of the strongest "
        f"target match) were computed per film. A Gaussian-process Bayesian optimizer was trained on the "
        f"{len(df_bo)} complete films to recommend the next synthesis conditions, optimizing "
        f"objective = 0.7 × phase purity + 0.3 × normalized crystallinity over a 7-parameter space "
        f"(spin speed, bake temperatures, selenization temperature/time, and Se pulse on/off)."
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
        f"crystallinity is {df_bo['crystallinity'].mean():.2f}. The best film is {best['sample_id']} "
        f"(objective {best['objective']:.2f}; purity {best['phase_purity']:.2f}, crystallinity "
        f"{best['crystallinity']:.2f}). Rank (Spearman) correlation with the objective identifies "
        f"selenization time as the dominant synthesis lever; bake temperatures and pulse timing are weaker."
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

    cols = ["rank", *df_bo.attrs["parameter_columns"], "predicted_objective", "predicted_std"]
    table_data = [[c.replace("_", " ").replace(" ms", " (ms)").replace(" s", " (s)").replace(" c", " (°C)") for c in cols]]
    for _, row in suggestions.iterrows():
        table_data.append([
            int(row["rank"]),
            *[f"{row[c]:.0f}" for c in df_bo.attrs["parameter_columns"]],
            f"{row['predicted_objective']:.3f}",
            f"{row['predicted_std']:.3f}",
        ])
    table = Table(table_data, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3b5d")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f8")]),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#b0becs") if False else colors.HexColor("#b0bec5")),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.2 * cm))
    top = suggestions.iloc[0]
    story.append(Paragraph(
        f"Recommended first experiment: spin {top['spin_speed_rpm']:.0f} rpm, soft bake "
        f"{top['soft_bake_temp_c']:.0f} °C, hard bake {top['hard_bake_temp_c']:.0f} °C, selenization "
        f"{top['selenization_temp_c']:.0f} °C for {top['selenization_time_s']:.0f} s with Se pulse "
        f"{top['se_pulse_on_ms']:.0f} ms on / {top['se_pulse_off_ms']:.0f} ms off "
        f"(predicted objective {top['predicted_objective']:.2f}).",
        styles["BodySmall"],
    ))
    fig = _figure(FIGURES_DIR / "bacuse2_bo_parameter_plots.png")
    if fig is not None:
        story.append(Spacer(1, 0.15 * cm))
        story.append(fig)

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
