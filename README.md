# Stratified active learning for oncology crisis language model validation
**Safety evaluation of neural models in high-risk mental-health triage**
*Evidence-based 13-item SMS (NICE NG225, WHO mhGAP, NCCN Distress, 988), 6 domains × 3 severities, EN+UR, ~40% oncology; **no real patient data** (expert-authored vignettes).*
[![ci](https://github.com/Sjtu-Fuxilab/onco-sentry/actions/workflows/ci.yml/badge.svg?branch=main)](/Sjtu-Fuxilab/onco-sentry/actions/workflows/ci.yml)
[![pre-commit](https://github.com/Sjtu-Fuxilab/onco-sentry/actions/workflows/pre-commit.yml/badge.svg?branch=main)](/Sjtu-Fuxilab/onco-sentry/actions/workflows/pre-commit.yml)
[![codeql](https://github.com/Sjtu-Fuxilab/onco-sentry/actions/workflows/codeql.yml/badge.svg?branch=main)](/Sjtu-Fuxilab/onco-sentry/actions/workflows/codeql.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

###- **Languages:** English and Urdu.
- **Coverage guarantee:** stratified active learning achieves **100%** coverage of all 18 domain×severity cells with 60 selections.
- **Uncertainty metrics:** predictive standard deviation, predictive entropy, and a z-normalized composite score using an ensemble of five variants.
- **Reliability metrics:** **Cohen’s κ (pairwise)** and **Fleiss’ κ (three-rater)**.
#├── setup/
├── scripts/
│   ├── 01_generate_vignettes.py
│   ├── 02_export_validation.py
│   ├── 04_ingest_scoring.py
│   ├── 05_t2_adjudication.py
│   ├── 06_irr_report.py
│   └── 07_advanced_validation.py
├── config/
├── rubric/
├── docs/
├── examples/
└── requirements.txt
```
#cd onco-sentry
pip install -r requirements.txt
python setup/setup_project.py
```
```bash
  title={ONCO-SENTRY: Oncology-Aware Mental Health Crisis Evaluation for LLMs},
  author={Sanwal Ahmad Zafar and Assoc. prof. Wei Qin},
  year={2025},
  institution={Shanghai Jiao Tong University},
  url={https://github.com/Sjtu-Fuxilab/onco-sentry}
}
```
#- **Dev Container**: `.devcontainer/` for VS Code / Codespaces.
- **Environment**: `requirements.txt` + `environment.yml` (if present) with a snapshot `environment.lock.yml`.
- **Coverage Gate**: GitHub Actions workflow `coverage.yml` fails if test coverage < **60%**.
- **SBOM & Audit**: weekly `sbom.yml` generates a CycloneDX SBOM and runs `pip-audit` (non-blocking).
- **CITATION**: See `CITATION.cff`. Please cite **Sanwal Ahmad Zafar** and **Assoc. prof. Wei Qin**.