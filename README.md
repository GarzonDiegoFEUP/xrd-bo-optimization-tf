# BaCuSe₂ thin-film analysis and Bayesian optimization

This project analyzes BaCuSe₂ thin films using synthesis parameters and X-ray
diffraction (XRD) scans from the NOMAD Oasis. It matches observed XRD peaks to
reference cards to identify the target BaCu₂Se₂ phase and the Cu₂Se, BaSe,
BaSeO₄, and BaSeO₃ impurities. The analysis computes phase purity and
crystallinity for each film, then uses a Gaussian-process Bayesian optimizer
to suggest synthesis conditions that maximize a combined score.

The optimization parameters are spin speed, bake temperatures, selenization
temperature, and selenization time.

## Repository layout

```text
.
├── README.md
├── requirements.txt
├── xrd-bo-optimization-tf/     # Python package scaffold
├── notebooks/                  # Analysis notebook scaffold
├── data/
│   ├── raw/                     # Local .xrdml backups
│   ├── references/              # ICDD reference cards (.rtf)
│   ├── processed/               # Synthesis, metrics, and pulse-data files
│   └── results/                 # Bayesian-optimization suggestions and outputs
└── outputs/
    ├── figures/
    └── reports/
```

The directories are scaffolded for future implementation and data. No Python
implementation scripts, environment files, or data files are included.
