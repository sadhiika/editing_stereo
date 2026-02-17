CREATE TABLE IF NOT EXISTS prompts (
    id TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    prompt_text TEXT NOT NULL,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS responses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    response_text TEXT NOT NULL,
    provider TEXT,
    metadata_json TEXT,
    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(prompt_id, model_name)
);

CREATE INDEX IF NOT EXISTS idx_responses_prompt_model
    ON responses(prompt_id, model_name);
