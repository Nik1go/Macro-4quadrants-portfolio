#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

DEFAULT_START_DATE = "2026-04-03"
GDELT_DOC_URL = "http://api.gdeltproject.org/api/v2/doc/doc"

THEMES = {
    "monetary_policy": {
        "query": '("Federal Reserve" OR Fed OR FOMC OR Powell OR "interest rates" OR "Treasury yields") sourcelang:english',
        "assets": ["USD", "Treasury", "SP500", "Gold"],
        "impact": "Policy/rates news: relevant for duration, dollar, gold, and broad risk appetite.",
    },
    "inflation_growth": {
        "query": '(inflation OR CPI OR PCE OR payrolls OR unemployment OR recession OR GDP OR ISM) sourcelang:english',
        "assets": ["SP500", "Gold", "Treasury", "USD"],
        "impact": "Macro growth/inflation news: relevant for quadrant confirmation and equity/bond sensitivity.",
    },
    "gold_rates_dollar": {
        "query": '(gold OR "safe haven" OR "real yields" OR dollar OR DXY OR yen) sourcelang:english',
        "assets": ["Gold", "USD", "JPY", "Treasury"],
        "impact": "Gold/dollar/rates news: useful to explain gold moves and FX hedging pressure.",
    },
    "commodities_energy": {
        "query": '(oil OR OPEC OR crude OR copper OR commodities OR energy) sourcelang:english',
        "assets": ["Oil", "Copper", "Commodities", "SP500"],
        "impact": "Commodity news: relevant for inflation pressure and stagflation/reflation risk.",
    },
    "politics_policy": {
        "query": '("White House" OR Congress OR tariff OR sanctions OR regulation OR election OR geopolitical OR Ukraine OR "Middle East" OR China) sourcelang:english',
        "assets": ["SP500", "USD", "Gold", "Oil"],
        "impact": "Political/geopolitical news: relevant for policy risk, sanctions/tariffs, and safe-haven demand.",
    },
}

KEYWORD_BOOSTS = {
    "federal reserve": 0.18,
    "fomc": 0.18,
    "powell": 0.15,
    "inflation": 0.14,
    "cpi": 0.12,
    "pce": 0.12,
    "payroll": 0.10,
    "recession": 0.10,
    "treasury": 0.12,
    "yield": 0.10,
    "gold": 0.14,
    "dollar": 0.10,
    "dxy": 0.10,
    "oil": 0.10,
    "opec": 0.10,
    "tariff": 0.14,
    "sanction": 0.14,
    "election": 0.10,
    "geopolitical": 0.14,
    "ukraine": 0.10,
    "middle east": 0.10,
    "china": 0.08,
}

THEME_BASE_SCORE = {
    "monetary_policy": 0.55,
    "inflation_growth": 0.52,
    "gold_rates_dollar": 0.48,
    "commodities_energy": 0.44,
    "politics_policy": 0.50,
}


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    default_base = repo_root / "data" / "US"
    parser = argparse.ArgumentParser(description="Fetch macro/political news for weekly NLP debriefs using GDELT DOC 2.0.")
    parser.add_argument("--base-dir", default=str(default_base))
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-records-per-query", type=int, default=75)
    parser.add_argument("--chunk-days", type=int, default=45)
    parser.add_argument("--max-errors", type=int, default=10)
    parser.add_argument("--endpoint", default=os.getenv("GDELT_DOC_URL", GDELT_DOC_URL))
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=float, default=6.0)
    parser.add_argument("--incremental-lookback-days", type=int, default=10)
    parser.add_argument("--full-refresh", action="store_true", help="Fetch the full requested date range even when an output file already exists.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_date(value):
    if not value:
        return date.today()
    return datetime.strptime(value, "%Y-%m-%d").date()


def date_chunks(start_date, end_date, chunk_days):
    current = start_date
    step = max(int(chunk_days), 1)
    while current <= end_date:
        end = min(current + timedelta(days=step - 1), end_date)
        yield current, end
        current = end + timedelta(days=1)


def gdelt_datetime(day, end_of_day=False):
    suffix = "235959" if end_of_day else "000000"
    return day.strftime("%Y%m%d") + suffix


def fetch_gdelt(endpoint, query, start_day, end_day, max_records, timeout_seconds):
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "startdatetime": gdelt_datetime(start_day),
        "enddatetime": gdelt_datetime(end_day, end_of_day=True),
        "maxrecords": max_records,
        "sort": "hybridrel",
    }
    url = f"{endpoint}?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": "Macro4QuadrantsNewsFetcher/1.0"})
    with urlopen(request, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload.get("articles", [])


def normalize_seen_date(value, fallback):
    if not value:
        return fallback.isoformat(), None
    cleaned = str(value).strip()
    for fmt in ["%Y%m%dT%H%M%SZ", "%Y%m%d%H%M%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S"]:
        try:
            parsed = datetime.strptime(cleaned, fmt).replace(tzinfo=timezone.utc)
            return parsed.date().isoformat(), parsed.isoformat().replace("+00:00", "Z")
        except ValueError:
            pass
    try:
        parsed = datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.date().isoformat(), parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    except ValueError:
        return fallback.isoformat(), cleaned


def relevance_score(title, theme):
    text = (title or "").lower()
    score = THEME_BASE_SCORE.get(theme, 0.4)
    for keyword, boost in KEYWORD_BOOSTS.items():
        if keyword in text:
            score += boost
    return round(min(score, 1.0), 3)


def stable_id(record):
    basis = "|".join([
        record.get("date", ""),
        record.get("theme", ""),
        record.get("title", ""),
        record.get("url", ""),
    ])
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]


def normalize_article(article, theme, week_start):
    title = str(article.get("title") or "").strip()
    url = str(article.get("url") or "").strip()
    if not title or not url:
        return None

    day, seen_at = normalize_seen_date(article.get("seendate"), week_start)
    record = {
        "date": day,
        "seen_at": seen_at,
        "source_api": "gdelt_doc_v2",
        "source": article.get("domain") or article.get("source") or "",
        "source_country": article.get("sourcecountry") or "",
        "language": article.get("language") or "",
        "theme": theme,
        "title": title,
        "summary": title,
        "url": url,
        "market_impact_hint": THEMES[theme]["impact"],
        "related_assets": THEMES[theme]["assets"],
        "relevance_score": relevance_score(title, theme),
    }
    record["id"] = stable_id(record)
    return record


def read_existing(path):
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Skipping invalid JSONL line {line_no}: {exc}", file=sys.stderr)
                continue
            if "id" not in item:
                item["id"] = stable_id(item)
            records.append(item)
    return records


def existing_latest_date(records):
    dates = []
    for record in records:
        value = record.get("date")
        if not value:
            continue
        try:
            dates.append(datetime.strptime(str(value)[:10], "%Y-%m-%d").date())
        except ValueError:
            continue
    return max(dates) if dates else None


def dedupe(records):
    by_key = {}
    for record in records:
        url_key = record.get("url") or ""
        title_key = " ".join((record.get("title") or "").lower().split())
        key = (url_key.lower() if url_key else f"{record.get('date', '')}|{title_key}")
        previous = by_key.get(key)
        if previous is None or record.get("relevance_score", 0) > previous.get("relevance_score", 0):
            by_key[key] = record
    ordered = sorted(
        by_key.values(),
        key=lambda item: (item.get("date", ""), -float(item.get("relevance_score", 0)), item.get("theme", ""), item.get("title", "")),
    )
    return ordered


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def main():
    args = parse_args()
    base_dir = Path(args.base_dir)
    output = Path(args.output) if args.output else base_dir / "nlp" / "news_events.jsonl"
    start_date = parse_date(args.start_date)
    end_date = parse_date(args.end_date)
    if end_date < start_date:
        raise SystemExit("--end-date must be greater than or equal to --start-date")

    existing = read_existing(output)
    fetch_start_date = start_date
    latest_existing = existing_latest_date(existing)
    if latest_existing and not args.full_refresh:
        fetch_start_date = max(start_date, latest_existing - timedelta(days=args.incremental_lookback_days))

    fetched = []
    errors = []

    stopped_early = False
    for chunk_start, chunk_end in date_chunks(fetch_start_date, end_date, args.chunk_days):
        for theme, config in THEMES.items():
            try:
                articles = fetch_gdelt(args.endpoint, config["query"], chunk_start, chunk_end, args.max_records_per_query, args.timeout_seconds)
            except Exception as exc:
                errors.append(f"{chunk_start}..{chunk_end} {theme}: {exc}")
                if len(errors) >= args.max_errors:
                    stopped_early = True
                    break
                continue
            for article in articles:
                record = normalize_article(article, theme, chunk_start)
                if record:
                    fetched.append(record)
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)
        if stopped_early:
            break

    merged = dedupe(existing + fetched)
    new_count = len(dedupe(fetched))
    added_count = len(merged) - len(dedupe(existing))

    if args.dry_run:
        print(json.dumps({
            "output": str(output),
            "requested_start_date": start_date.isoformat(),
            "effective_fetch_start_date": fetch_start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "chunk_days": args.chunk_days,
            "endpoint": args.endpoint,
            "stopped_early": stopped_early,
            "fetched_records_before_merge": len(fetched),
            "unique_fetched_records": new_count,
            "records_after_merge": len(merged),
            "estimated_new_records": max(added_count, 0),
            "errors": errors[:20],
        }, ensure_ascii=False, indent=2))
        return

    write_jsonl(output, merged)
    print(f"Wrote {len(merged)} news records to {output} ({max(added_count, 0)} new).")
    if stopped_early:
        print(f"Stopped early after {len(errors)} fetch errors. Existing records were preserved.", file=sys.stderr)
    if errors:
        print("Some GDELT requests failed:", file=sys.stderr)
        for error in errors[:20]:
            print(f"- {error}", file=sys.stderr)
        if len(errors) > 20:
            print(f"- ... {len(errors) - 20} more", file=sys.stderr)


if __name__ == "__main__":
    main()
