# 🛡️ ONCO-SENTRY: Oncology-Aware Mental Health Crisis Evaluation for LLMs

**Safety Evaluation of Neural models in High-risk Triage for Mental Health**

A guideline-grounded framework for evaluating LLM safety in mental health crisis scenarios,
with specialized support for oncology contexts and multilingual validation (English + Urdu).

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-research-yellow.svg)]()

---

## 🎯 Overview

ONCO-SENTRY evaluates LLM responses against a **Safety Minimum Standard (SMS)** derived from:
- **NICE NG225**: Self-harm and suicide prevention guidelines
- **WHO mhGAP**: Mental Health Gap Action Programme
- **NCCN Distress Management**: Oncology-specific protocols
- **988 Suicide & Crisis Lifeline**: Best practices

### 🔑 Key Features

✅ 24-item SMS rubric (14 must-do + 10 no-go)
✅ Oncology stratification (~40% cases)
✅ Country-specific routing (US/UK/PK)
✅ Multilingual (EN + UR), extensible
✅ Active learning for efficient clinician validation
✅ Reliability metrics: Cohen’s κ, Gwet’s AC1, bootstrap CIs

---

## 📂 Project Structure
```
onco-sentry/
├── setup/
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

---

## 🚀 Quick Start

```bash
git clone https://github.com/Sjtu-Fuxilab/onco-sentry.git
cd onco-sentry
pip install -r requirements.txt
python setup/setup_project.py
```

```bash
# Optional envs
export SENTRY_ROOT="/path/to/project"
export ANTHROPIC_API_KEY="your-api-key"
```

```bash
# Run pipeline
python scripts/01_generate_vignettes.py
python scripts/02_export_validation.py
python scripts/04_ingest_scoring.py
python scripts/05_t2_adjudication.py
python scripts/06_irr_report.py
python scripts/07_advanced_validation.py
```

---

## 📊 Pipeline Overview

| Script | Purpose | Input | Output |
|-------:|---------|-------|--------|
| 00 | Setup | None | Structure, configs |
| 01 | Vignette generation | Config | 720 vignettes (EN+UR) |
| 02 | Validation export | Vignettes | Rater CSVs |
| 04 | Scoring & adjudication | Filled CSVs | Metrics, disagreements |
| 05 | T2 adjudication | Per-rater files | Final labels (JSONL) |
| 06 | IRR report | T2 data | PDF κ/AC1 |
| 07 | Advanced validation | All vignettes | Active-learning selection |

---

## 🧪 SMS Rubric

See `rubric/sms_items.csv` (14 must-do, 10 no-go).

---

## 🛡️ Ethics

- No real patient data (synthetic vignettes)
- Licensed clinician validation (US/UK/PK)
- Blinded model evaluation
- Safety locks for high-severity cases

---

## 📚 Citation

```bibtex
@software{onco_sentry_2025,
  title={ONCO-SENTRY: Oncology-Aware Mental Health Crisis Evaluation for LLMs},
  author={Sanwal Ahmad Zafar and Assoc. prof. Wei Qin},
  year={2025},
  institution={Shanghai Jiao Tong University},
  url={https://github.com/Sjtu-Fuxilab/onco-sentry}
}
```

---

## 🤝 Contributing

1. Fork the repo
2. Create a branch (`git checkout -b feature/NewFeature`)
3. Commit (`git commit -m "Add NewFeature"`)
4. Push (`git push origin feature/NewFeature`)
5. Open a PR

---

## 📧 Contact

**Authors**: Sanwal Ahmad Zafar and Assoc. prof. Wei Qin  
**Affiliation**: Shanghai Jiao Tong University  
**Email**: your.email@sjtu.edu.cn

---

## ⚠️ Disclaimer

This is a research evaluation tool, not a clinical system.

**Emergency Resources**:
- 🇺🇸 USA: 988 Suicide & Crisis Lifeline
- 🇬🇧 UK: 116 123 (Samaritans)
- 🇵🇰 Pakistan: Umang Helpline 0317-4288665

*Last updated: 2025-10-31*