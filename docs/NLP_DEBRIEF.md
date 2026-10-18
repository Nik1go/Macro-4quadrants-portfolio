# Weekly NLP debrief

This module generates a weekly macro and strategy debrief with OpenRouter.

## Model

The default model is:

```bash
deepseek/deepseek-v4.1-flash
```

This is the OpenRouter slug for DeepSeek V4.1 Flash. The model is recorded in each generated debrief.
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

## Historical news and backfill

The news collector saves dated macro and market headlines with their source links in
`data/US/nlp/news_events.jsonl`. With no Alpha Vantage key, `--provider auto` uses
Google News RSS. GDELT can be selected explicitly, but its DOC API may rate-limit
historical requests. RSS records contain headlines, not the full article text; the
model is instructed to avoid inferring details that are not in those headlines.

```bash
python nlp_jobs/fetch_macro_news.py --provider google_rss --start-date 2026-04-04 --end-date 2026-10-02 --full-refresh
python nlp_jobs/generate_weekly_debriefs.py --start-date 2026-04-03
```

The debrief generator selects news from each Friday-ended week and automatically
regenerates weeks when the selected news changes. Streamlit displays the selected
headline, date, source, and link under each weekly debrief.

Output:

```text
data/US/nlp/weekly_debriefs.jsonl
```

The DAG runs daily, but the NLP task exits unless the server day is Friday. A change of model also regenerates existing weeks on the next Friday run, even if the input data digest is unchanged. The --force option regenerates every eligible week immediately.
