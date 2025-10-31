# ONCO-SENTRY Architecture

## System Overview
Modular pipeline for evaluating LLM safety in mental health crisis scenarios.

## Core Components
1) Vignette generator (Script 01)
2) Validation system (Scripts 02–06)
3) Active learning module (Script 07)

## Data Flow
Vignettes (01) → Validation Pack (02) → Clinician Rating → Scoring (04) →
Adjudication (05) → IRR Report (06) → Active Learning (07).

## SMS Structure
- Must-Do Items (required based on severity)
- No-Go Items (prohibited behaviors)
- Conditional (oncology/country/access to means)