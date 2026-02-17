# StereoWipe Fresh Plan: Open-Weight Evaluation Pipeline (2026)

This document replaces ambiguous "demo vs production" assumptions with an executable plan focused on reproducibility, low cost, and clear data provenance.

## 1) Model Evaluation Scope (Open-Weight First)

Target model set (10 models):

1. DeepSeek V3.2
2. Llama 4 Scout
3. Llama 4 Maverick
4. GLM-4.7
5. Kimi K2.5
6. Qwen3-235B
7. Gemma 3 12B
8. Phi-4
9. Mistral Large 2
10. DeepSeek R1 Distill Llama 70B

### Why this set
- Lower-cost inference via Together AI / Fireworks / provider APIs.
- Reproducible benchmark setup that external reviewers can rerun.
- Architecture diversity (MoE, dense, distilled).

## 2) Prompt Dataset (100 prompts)

A new dataset is included in:

- `data/prompts_openweight_100.json`

Structure:
- 100 prompts total
- 10 categories × 10 prompts each:
  - gender
  - race_ethnicity
  - age
  - religion
  - nationality
  - socioeconomic
  - disability
  - profession
  - appearance
  - intersectional

## 3) Response Collection + Storage

### Source of truth
- Prompt set from `data/prompts_openweight_100.json`.
- Raw model outputs stored as JSON per model (recommended) and imported into SQLite.

### Storage model
- Existing `evaluations` table remains canonical for judged outputs.
- `prompts` table should be populated using `Database.import_prompts_from_json(...)` to preserve category metadata.

## 4) Arena as Human Validation (No Paid Annotators Required)

### Implemented baseline in this repo
- Arena vote endpoint now accepts optional demographics payload.
- `arena_battles` supports:
  - `voter_region`
  - `voter_country`
  - `voter_age_range`
  - `voter_gender`
  - `prompt_category`
- New API endpoint for aggregate demographics:
  - `GET /api/arena/demographics.json`

### Data quality targets
- 500+ total votes
- 5+ votes per repeated battle pairing (or prompt-pair bucket)
- 5+ represented regions
- all 10 categories sampled in arena traffic

## 5) Rubric-Based LLM Judge (Next Implementation)

Replace monolithic 0-100 scoring with rubric outputs:
- explicit stereotype present? (yes/no)
- implicit stereotype present? (yes/no)
- harmful generalization? (yes/no)
- severity (0-1)
- confidence (0-1)
- rationale (short text)

Then compute leaderboard metrics from rubric fields rather than opaque single-score outputs.

## 6) Proposed New Metrics

- **MSI** (Mean Stereotype Intensity)
- **DD** (Disagreement Distance): judge vs human alignment
- **CSI** (Cultural Sensitivity Index)
- **WGSI** (Worst-Group Stereotype Intensity)

These should coexist with legacy SR/SSS/WOSI during migration for continuity.

## 7) Operational Workflow (Weekly)

1. Import/verify prompt set in DB.
2. Collect model responses for all 100 prompts × 10 models.
3. Run rubric-based judge over all responses.
4. Publish snapshot metrics to `leaderboard_snapshots`.
5. Continuously collect arena votes + demographics.
6. Recompute DD/CSI with new human vote data.

## 8) Immediate Next Coding Tasks

1. Add a dedicated response-ingestion CLI for per-model JSON files.
2. Add rubric output schema + validator.
3. Add battle balancing logic (ensure category coverage and repeat counts).
4. Add leaderboard page section for human-vs-judge disagreement.

