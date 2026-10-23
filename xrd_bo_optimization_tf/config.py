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
    "Cu2Se": "04-024-2128 Cu2Se.rtf",
    "BaSe": "96-900-8602 BaSe.rtf",
    "BaSeO4": "96-810-2887 BaSeO4.rtf",
    "BaSeO3": "00-053-0141 BaSeO3.rtf",
}

SYNTHESIS_CSV = PROCESSED_DATA_DIR / "bacuse2_synthesis_database_37samples.csv"
XRD_METRICS_CSV = PROCESSED_DATA_DIR / "bacuse2_xrd_analysis_metrics.csv"
BO_SUGGESTIONS_CSV = PROCESSED_DATA_DIR / "bacuse2_bo_suggestions.csv"
GENERATED_XRD_METRICS_CSV = RESULTS_DATA_DIR / "bacuse2_xrd_analysis_metrics.csv"

BO_PARAMETER_COLUMNS = (
    "spin_speed_rpm",
    "soft_bake_temp_c",
    "hard_bake_temp_c",
    "selenization_temp_c",
    "selenization_time_s",
)
