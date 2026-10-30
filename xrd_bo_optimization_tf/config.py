"""Project paths and shared analysis/optimization defaults."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = PROJECT_ROOT / "xrd_bo_optimization_tf"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
DOCS_DIR = PROJECT_ROOT / "docs"

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
REFERENCES_DIR = DATA_DIR / "references"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DATA_DIR = DATA_DIR / "results"
NOMAD_VOLUME_DIR = PROJECT_ROOT / ".volumes" / "fs" / "staging"
# Local staging area for archives downloaded from the NOMAD REST API.
NOMAD_STAGING_DIR = RAW_DATA_DIR / "nomad_staging"
# Complete synthesis database extracted from the NOMAD API.
NOMAD_SYNTHESIS_CSV = PROCESSED_DATA_DIR / "bacuse2_synthesis_database_nomad.csv"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"

PHASE_PURITY_WEIGHT = 0.7
CRYSTALLINITY_WEIGHT = 0.3
UNASSIGNED_AREA_PENALTY_LAMBDA = 0.25
UNASSIGNED_AREA_DIAGNOSTICS_CSV = (
    RESULTS_DATA_DIR / "bacuse2_xrd_peak_area_diagnostics.csv"
)

# Reference intensities use the percentage scale shown on the ICDD cards.
REF_MIN_INTENSITY = 10.0
# Peak matching tolerance and minimum accepted width are in degrees 2-theta.
MATCH_TOL = 0.15
MIN_FWHM = 0.1
MAX_CANDIDATE_PEAKS = 30

TARGET_PHASE = "BaCu2Se2"
IMPURITY_PHASES = ("Cu2Se", "BaSe", "BaSeO4", "BaSeO3")
PHASES = (TARGET_PHASE, *IMPURITY_PHASES)

REFERENCE_CARD_FILES = {
    "BaCu2Se2": "01-080-2950 BaCu2Se2.rtf",
    "Cu2Se": "96-101-0582 Cu2Se.rtf",
    "BaSe": "96-900-8602 BaSe.rtf",
    "BaSeO4": "96-810-2887 BaSeO4.rtf",
    "BaSeO3": "00-053-0141 BaSeO3.rtf",
}

SYNTHESIS_CSV = PROCESSED_DATA_DIR / "bacuse2_synthesis_database_37samples.csv"
XRD_METRICS_CSV = PROCESSED_DATA_DIR / "bacuse2_xrd_analysis_metrics.csv"
BO_SUGGESTIONS_CSV = PROCESSED_DATA_DIR / "bacuse2_bo_suggestions.csv"
GENERATED_XRD_METRICS_CSV = RESULTS_DATA_DIR / "bacuse2_xrd_analysis_metrics.csv"

BO_PARAMETER_COLUMNS = (
    "hard_bake_temp_c",
    "selenization_temp_c",
    "selenization_time_s",
    "cbc_ratio",
)
XRD_PARAMETER_COLUMNS = (
    "spin_speed_rpm",
    "soft_bake_temp_c",
    "hard_bake_temp_c",
    "selenization_temp_c",
    "selenization_time_s",
)

# (lower, upper, step, decimal places); used identically by skopt and Ax.
BO_PARAMETER_GRID_SPECS = {
    "hard_bake_temp_c": (225.0, 250.0, 25.0, 0),
    "selenization_temp_c": (520.0, 560.0, 10.0, 0),
    "selenization_time_s": (600.0, 1800.0, 300.0, 0),
    "cbc_ratio": (0.63, 0.71, 0.01, 2),
}

SE_PULSE_DUTY_CYCLE_COLUMN = "se_pulse_duty_cycle"
SE_PULSE_PERIOD_MS = 3200
SE_PULSE_FREQUENCY_HZ = 1000.0 / SE_PULSE_PERIOD_MS
SE_PULSE_ON_MS_BOUNDS = (300, 1000)
SE_PULSE_DUTY_CYCLE_BOUNDS = tuple(
    on_ms / SE_PULSE_PERIOD_MS for on_ms in SE_PULSE_ON_MS_BOUNDS
)
