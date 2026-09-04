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
ALPHA_URL = "https://www.alphavantage.co/query"
GDELT_DOC_URL = "http://api.gdeltproject.org/api/v2/doc/doc"

THEMES = {
    "monetary_policy": {
        "gdelt_query": '("Federal Reserve" OR Fed OR FOMC OR Powell OR "interest rates" OR "Treasury yields") sourcelang:english',
        "alpha_params": {"topics": "economy_monetary"},
        "assets": ["USD", "Treasury", "SP500", "Gold"],
        "impact": "Policy/rates news: relevant for duration, dollar, gold, and broad risk appetite.",
    },
    "inflation_growth": {
        "gdelt_query": '(inflation OR CPI OR PCE OR payrolls OR unemployment OR recession OR GDP OR ISM) sourcelang:english',
        "alpha_params": {"topics": "economy_macro"},
        "assets": ["SP500", "Gold", "Treasury", "USD"],
        "impact": "Macro growth/inflation news: relevant for quadrant confirmation and equity/bond sensitivity.",
    },
    "markets_sentiment": {
        "gdelt_query": '("stock market" OR "S&P 500" OR equities OR "risk appetite" OR volatility OR VIX) sourcelang:english',
        "alpha_params": {"topics": "financial_markets"},
        "assets": ["SP500", "VIX", "USD", "Gold"],
        "impact": "Market news/sentiment: relevant for explaining weekly risk-on or risk-off moves.",
    },
    "commodities_energy": {
        "gdelt_query": '(oil OR OPEC OR crude OR copper OR commodities OR energy) sourcelang:english',
        "alpha_params": {"topics": "energy_transportation"},
        "assets": ["Oil", "Copper", "Commodities", "SP500"],
        "impact": "Commodity news: relevant for inflation pressure and stagflation/reflation risk.",
    },
    "politics_policy": {
        "gdelt_query": '("White House" OR Congress OR tariff OR sanctions OR regulation OR election OR geopolitical OR Ukraine OR "Middle East" OR China) sourcelang:english',
        "alpha_params": {"topics": "economy_fiscal"},
        "assets": ["SP500", "USD", "Gold", "Oil"],
        "impact": "Political/fiscal news: relevant for policy risk, taxes, spending, sanctions, tariffs, and safe-haven demand.",
    },
    "usd_fx": {
        "gdelt_query": '(dollar OR DXY OR yen OR euro OR "foreign exchange" OR "currency markets") sourcelang:english',
        "alpha_params": {"tickers": "FOREX:USD"},
        "assets": ["USD", "EUR", "JPY", "Gold"],
        "impact": "FX/dollar news: relevant for currency sleeves and gold sensitivity to the dollar.",
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
    "markets_sentiment": 0.50,
    "commodities_energy": 0.44,
    "politics_policy": 0.50,
    "usd_fx": 0.46,
}


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    default_base = repo_root / "data" / "US"
    parser = argparse.ArgumentParser(description="Fetch macro/political news for weekly NLP debriefs.")
    parser.add_argument("--base-dir", default=str(default_base))
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--provider", choices=["auto", "alpha", "gdelt", "all"], default="auto")
    parser.add_argument("--max-records-per-query", type=int, default=75)
    parser.add_argument("--alpha-limit", type=int, default=200)
    parser.add_argument("--chunk-days", type=int, default=45)
    parser.add_argument("--max-errors", type=int, default=10)
    parser.add_argument("--gdelt-endpoint", default=os.getenv("GDELT_DOC_URL", GDELT_DOC_URL))
    parser.add_argument("--alpha-endpoint", default=os.getenv("ALPHAVANTAGE_URL", ALPHA_URL))
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument("--incremental-lookback-days", type=int, default=10)
    parser.add_argument("--full-refresh", action="store_true", help="Fetch the full requested date range even when an output file already exists.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_dotenv_if_present(repo_root):
    env_path = repo_root / ".env"
    if not env_path.exists():
        return
    with env_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def alpha_key():
    return os.getenv("ALPHAVANTAGE_API") or os.getenv("ALPHAVANTAGE_API_KEY") or os.getenv("ALPHA_VANTAGE_API_KEY")


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


def alpha_datetime(day, end_of_day=False):
    suffix = "T2359" if end_of_day else "T0000"
    return day.strftime("%Y%m%d") + suffix


def request_json(url, params, timeout_seconds):
    request_url = f"{url}?{urlencode(params)}"
    request = Request(request_url, headers={"User-Agent": "Macro4QuadrantsNewsFetcher/1.0"})
    with urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_gdelt(endpoint, query, start_day, end_day, max_records, timeout_seconds):
    payload = request_json(endpoint, {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "startdatetime": gdelt_datetime(start_day),
        "enddatetime": gdelt_datetime(end_day, end_of_day=True),
        "maxrecords": max_records,
        "sort": "hybridrel",
    }, timeout_seconds)
    return payload.get("articles", [])


def fetch_alpha(endpoint, api_key, params, start_day, end_day, limit, timeout_seconds):
    payload = request_json(endpoint, {
        "function": "NEWS_SENTIMENT",
        "time_from": alpha_datetime(start_day),
        "time_to": alpha_datetime(end_day, end_of_day=True),
        "sort": "RELEVANCE",
        "limit": max(1, min(int(limit), 1000)),
        "apikey": api_key,
        **params,
    }, timeout_seconds)
    if "feed" in payload:
        return payload["feed"]
    if "Note" in payload:
        raise RuntimeError(payload["Note"])
    if "Information" in payload:
        raise RuntimeError(payload["Information"])
    if "Error Message" in payload:
        raise RuntimeError(payload["Error Message"])
    raise RuntimeError(f"Unexpected Alpha Vantage response keys: {sorted(payload.keys())}")


def normalize_seen_date(value, fallback):
    if not value:
        return fallback.isoformat(), None
    cleaned = str(value).strip()
    for fmt in ["%Y%m%dT%H%M", "%Y%m%dT%H%M%S", "%Y%m%dT%H%M%SZ", "%Y%m%d%H%M%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S"]:
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


def relevance_score(title, theme, sentiment_score=None):
    text = (title or "").lower()
    score = THEME_BASE_SCORE.get(theme, 0.4)
    for keyword, boost in KEYWORD_BOOSTS.items():
        if keyword in text:
            score += boost
    try:
        if sentiment_score is not None:
            score += min(abs(float(sentiment_score)), 0.35) * 0.25
    except (TypeError, ValueError):
        pass
    return round(min(score, 1.0), 3)


def stable_id(record):
    basis = "|".join([
        record.get("date", ""),
        record.get("source_api", ""),
        record.get("theme", ""),
        record.get("title", ""),
        record.get("url", ""),
    ])
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]


def normalize_gdelt_article(article, theme, period_start):
    title = str(article.get("title") or "").strip()
    url = str(article.get("url") or "").strip()
    if not title or not url:
        return None
    day, seen_at = normalize_seen_date(article.get("seendate"), period_start)
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


def normalize_alpha_article(article, theme, period_start):
    title = str(article.get("title") or "").strip()
    url = str(article.get("url") or "").strip()
    if not title or not url:
        return None
    day, seen_at = normalize_seen_date(article.get("time_published"), period_start)
    sentiment_score = article.get("overall_sentiment_score")
    topics = [topic.get("topic") for topic in article.get("topics", []) if isinstance(topic, dict) and topic.get("topic")]
    tickers = []
    for item in article.get("ticker_sentiment", []) or []:
        if isinstance(item, dict) and item.get("ticker"):
            tickers.append(item["ticker"])
    record = {
        "date": day,
        "seen_at": seen_at,
        "source_api": "alpha_vantage_news_sentiment",
        "source": article.get("source") or "",
        "source_domain": article.get("source_domain") or "",
        "theme": theme,
        "title": title,
        "summary": str(article.get("summary") or title).strip(),
        "url": url,
        "market_impact_hint": THEMES[theme]["impact"],
        "related_assets": sorted(set(THEMES[theme]["assets"] + tickers)),
        "provider_topics": topics,
        "overall_sentiment_score": to_float(sentiment_score),
        "overall_sentiment_label": article.get("overall_sentiment_label") or "",
        "relevance_score": relevance_score(f"{title} {article.get('summary') or ''}", theme, sentiment_score),
    }
    record["id"] = stable_id(record)
    return record


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
        key = url_key.lower() if url_key else f"{record.get('date', '')}|{title_key}"
        previous = by_key.get(key)
        if previous is None or record.get("relevance_score", 0) > previous.get("relevance_score", 0):
            by_key[key] = record
    return sorted(
        by_key.values(),
        key=lambda item: (item.get("date", ""), -float(item.get("relevance_score", 0)), item.get("theme", ""), item.get("title", "")),
    )


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def resolved_providers(requested_provider, api_key):
    if requested_provider == "alpha":
        return ["alpha"]
    if requested_provider == "gdelt":
        return ["gdelt"]
    if requested_provider == "all":
        return ["alpha", "gdelt"] if api_key else ["gdelt"]
    return ["alpha"] if api_key else ["gdelt"]


def fetch_alpha_records(args, api_key, start_date, end_date):
    fetched = []
    errors = []
    for theme, config in THEMES.items():
        try:
            articles = fetch_alpha(args.alpha_endpoint, api_key, config["alpha_params"], start_date, end_date, args.alpha_limit, args.timeout_seconds)
        except Exception as exc:
            errors.append(f"{start_date}..{end_date} alpha {theme}: {exc}")
            if len(errors) >= args.max_errors:
                break
            continue
        for article in articles:
            record = normalize_alpha_article(article, theme, start_date)
            if record:
                fetched.append(record)
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)
    return fetched, errors, len(errors) >= args.max_errors


def fetch_gdelt_records(args, start_date, end_date):
    fetched = []
    errors = []
    stopped_early = False
    for chunk_start, chunk_end in date_chunks(start_date, end_date, args.chunk_days):
        for theme, config in THEMES.items():
            try:
                articles = fetch_gdelt(args.gdelt_endpoint, config["gdelt_query"], chunk_start, chunk_end, args.max_records_per_query, args.timeout_seconds)
            except Exception as exc:
                errors.append(f"{chunk_start}..{chunk_end} gdelt {theme}: {exc}")
                if len(errors) >= args.max_errors:
                    stopped_early = True
                    break
                continue
            for article in articles:
                record = normalize_gdelt_article(article, theme, chunk_start)
                if record:
                    fetched.append(record)
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)
        if stopped_early:
            break
    return fetched, errors, stopped_early


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv_if_present(repo_root)

    api_key = alpha_key()
    providers = resolved_providers(args.provider, api_key)
    if "alpha" in providers and not api_key:
        raise SystemExit("Alpha provider requested, but ALPHAVANTAGE_API or ALPHAVANTAGE_API_KEY is missing.")

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

    for provider in providers:
        if provider == "alpha":
            provider_records, provider_errors, provider_stopped = fetch_alpha_records(args, api_key, fetch_start_date, end_date)
        else:
            provider_records, provider_errors, provider_stopped = fetch_gdelt_records(args, fetch_start_date, end_date)
        fetched.extend(provider_records)
        errors.extend(provider_errors)
        stopped_early = stopped_early or provider_stopped
        if stopped_early:
            break

    merged = dedupe(existing + fetched)
    new_count = len(dedupe(fetched))
    added_count = len(merged) - len(dedupe(existing))

    report = {
        "output": str(output),
        "providers": providers,
        "requested_start_date": start_date.isoformat(),
        "effective_fetch_start_date": fetch_start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "chunk_days": args.chunk_days,
        "stopped_early": stopped_early,
        "fetched_records_before_merge": len(fetched),
        "unique_fetched_records": new_count,
        "records_after_merge": len(merged),
        "estimated_new_records": max(added_count, 0),
        "errors": errors[:20],
    }

    if args.dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    write_jsonl(output, merged)
    print(f"Wrote {len(merged)} news records to {output} ({max(added_count, 0)} new). providers={','.join(providers)}")
    if stopped_early:
        print(f"Stopped early after {len(errors)} fetch errors. Existing records were preserved.", file=sys.stderr)
    if errors:
        print("Some news requests failed:", file=sys.stderr)
        for error in errors[:20]:
            print(f"- {error}", file=sys.stderr)
        if len(errors) > 20:
            print(f"- ... {len(errors) - 20} more", file=sys.stderr)


if __name__ == "__main__":
    main()
