# StereoWipe Repository Audit (Current-State Reality Check)

## Scope
This audit summarizes what can and cannot be verified from the current repository state, with focus on:
- model calling / API-credit assumptions,
- prompt and response storage,
- judge implementation,
- arena vote storage and prompt source,
- whether `stereowipe.db` currently contains benchmark evidence.

## Key Conclusions

1. **The repo supports real API-based judging, but does not prove a 40+ model run was executed.**
   - `OpenAIJudge` and `AnthropicJudge` exist and require API keys.
   - The CLI processes whatever JSON files are present in `--model-responses-dir`; it does not itself fan out to 40 provider APIs in one integrated orchestrator.

2. **The docs include both roadmap language and "completed" language, so they are internally inconsistent about maturity.**
   - Some docs still describe missing pieces as future work.
   - Others claim Phase 0/1 is fully complete.

3. **Arena votes are designed to be stored in SQLite (`arena_battles`) via `/arena/vote`, and arena battle prompts/responses are drawn from existing `evaluations` rows (not a separate arena prompt bank).**

4. **`stereowipe.db` is expected to be the SQLite store by design, but whether it contains real production benchmark artifacts depends on deployment/runtime population.**
   - The code initializes tables on first use.
   - A sample-data script exists and can populate synthetic rows for development.

## Evidence Map

### 1) Do 40+ model calls require paid API access?
- Yes, for proprietary models, real API calls require keys/credits. The project docs themselves acknowledge API-cost risk and phased/sponsored access strategy.
- See PRD risk table and model-integration phases.

### 2) Where are prompts stored?
- In file-based JSON for CLI workflows (`--prompts`), and optionally in the SQLite `prompts` table (`import_prompts_from_json`, `get_prompts`).
- The web API `/api/prompts/stats.json` reads prompt stats from DB, but this only reflects whatever has been imported/inserted.

### 3) Where are responses stored?
- For benchmark runs, responses are loaded from model response JSON files in a directory (`--model-responses-dir`) and judged; judged outputs can be persisted into `evaluations`.
- Arena uses responses already present in `evaluations`.

### 4) Where is Gemini judge code?
- Current codebase contains OpenAI/Anthropic/Mock judge classes in `biaswipe/judge.py`.
- PRD mentions Gemini Flash as primary judge, but that appears as product-design intent. A concrete Gemini judge implementation is not obvious in the audited path.

### 5) Where do arena votes go, and where do arena prompts come from?
- Votes are saved to `arena_battles` via `db.insert_arena_battle(...)` in `/arena/vote`.
- Arena battle content comes from random prompt/response pairs selected from `evaluations` with at least two distinct model responses.

### 6) What about `stereowipe.db`?
- Documentation states this DB file is created in project root and stores evaluations, annotations, and arena battles.
- In practice, the file may be empty/new in a local checkout until code paths are exercised.
- A sample-data generator exists for development; synthetic data can therefore exist and should not be automatically treated as production evidence.

## Direct Answers to Your Questions (Current Stage)

- **"How did they get responses from 40 models without API credits?"**
  - From this repo alone, you cannot verify that they did. You can only verify the architecture expects API keys/credits for paid models.

- **"Where are the ~100 prompts and responses stored?"**
  - Mechanisms exist (JSON inputs and SQLite tables), but this repo snapshot does not by itself prove the authoritative production dataset location unless accompanied by populated DB/files from the run environment.

- **"Are arena votes being stored somewhere?"**
  - Yes, by design: `arena_battles` in SQLite via `/arena/vote`.

- **"Where do arena questions come from? Same prompt set or different?"**
  - Current implementation samples from `evaluations` only, i.e., whatever prompts/responses were previously inserted there.

- **"How were 40+ models called?"**
  - No single end-to-end 40-provider caller is clearly present in audited code paths. The CLI expects pre-generated response files; generation/orchestration for all providers is not proven here.

## What You Should Confirm with the Team Immediately
1. Source of truth for **production prompts** (file path/db table/service).
2. Source of truth for **raw model responses** (storage bucket/db/table/filesystem), including run timestamps and model/version metadata.
3. The exact **evaluation job/script** that produced currently published leaderboard rows.
4. Whether published leaderboard rows are from:
   - real API runs,
   - imported external data,
   - or seeded/demo data.
5. Where/if the **Gemini judge** is implemented and executed (repo path, private service, notebook, or external pipeline).
