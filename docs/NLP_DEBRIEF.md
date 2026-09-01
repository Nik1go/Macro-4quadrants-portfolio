# Weekly NLP debrief

This module generates a weekly macro and strategy debrief with OpenRouter.

## Model

The default model is:

```bash
google/gemini-2.5-flash-lite
```

It is intentionally cheap and good enough for structured weekly JSON output.
Override it with:

```bash
export OPENROUTER_MODEL="model/provider-slug"
```

## Configuration

Expose one of these variables on the VPS/WSL environment:

```bash
export KEY="sk-or-v1-..."
# or
export OPENROUTER_API_KEY="sk-or-v1-..."
```

## Backfill

```bash
python nlp_jobs/generate_weekly_debriefs.py --start-date 2026-04-03
```

Output:

```text
data/US/nlp/weekly_debriefs.jsonl
```

The DAG runs daily, but the NLP task exits unless the server day is Friday.
