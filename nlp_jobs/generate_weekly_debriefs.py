import argparse
import ast
import hashlib
import json
import os
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


DEFAULT_START_DATE = "2026-04-03"
DEFAULT_MODEL = "google/gemini-2.5-flash-lite"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

QUADRANT_LABELS = {
    1: "Q1 - Growth / Goldilocks",
    2: "Q2 - Inflation / Reflation",
    3: "Q3 - Stagflation",
    4: "Q4 - Deflation",
}

INDICATORS = [
    "INFLATION",
    "CONSUMER_SENTIMENT",
    "INITIAL_CLAIMS",
    "HOUSING_PERMITS",
    "IND_PRODUCTION",
    "High_Yield_Bond_SPREAD",
    "10-2Year_Treasury_Yield_Bond",
    "TAUX_FED",
    "VIX",
    "US_DOLLAR_INDEX",
    "WTI_CRUDE_OIL",
    "COPPER",
    "BREAKEVEN_10Y",
    "NFCI",
    "NET_LIQUIDITY",
]

MODEL_RESULT_KEYS = [
    "title",
    "macro_regime",
    "strategy_status",
    "available_information",
    "nlp_signal",
    "forward_view",
    "key_points",
    "risks",
    "allocation_comment",
    "performance_attribution",
    "markdown",
]

DEFAULT_SIGNAL = {
    "risk_on_score": 0.0,
    "growth_score": 0.0,
    "inflation_pressure_score": 0.0,
    "policy_risk_score": 0.0,
    "confidence": 0.0,
    "suggested_use": "shadow_only",
    "rationale": "Fallback because model output was not valid JSON.",
}

DEFAULT_FORWARD_VIEW = {
    "base_case_next_week": "",
    "bull_case": "",
    "bear_case": "",
    "watchlist": [],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate weekly macro/strategy debriefs with OpenRouter.")
    parser.add_argument("--base-dir", default=str(Path(__file__).resolve().parents[1] / "data" / "US"))
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--news-file", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-weeks", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-if-missing-key", action="store_true")
    parser.add_argument("--single-call", action="store_true", help="Use the legacy one-shot OpenRouter call.")
    parser.add_argument("--repair-only", action="store_true", help="Rewrite existing JSONL records after local cleanup/parsing only.")
    return parser.parse_args()


def read_csv(path, parse_dates=None):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path, parse_dates=parse_dates)


def load_inputs(base_dir):
    base = Path(base_dir)
    backtest = read_csv(base / "backtest_results" / "backtest_timeseries.csv", parse_dates=["date"])
    quadrants = read_csv(base / "output_dag" / "quadrants.csv", parse_dates=["date"])
    indicators = read_csv(base / "output_dag" / "combined_indicators.csv", parse_dates=["date"])

    for df in [backtest, quadrants, indicators]:
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values("date", inplace=True)
        df.drop_duplicates("date", keep="last", inplace=True)

    return backtest, quadrants, indicators


def load_news_context(path):
    if not path:
        return {"available": False, "source": None, "items": [], "note": "No external news file was provided. Do not invent current news."}
    news_path = Path(path)
    if not news_path.exists():
        return {"available": False, "source": str(news_path), "items": [], "note": "News file was not found. Do not invent current news."}
    raw = news_path.read_text(encoding="utf-8").strip()
    if not raw:
        return {"available": True, "source": str(news_path), "items": []}
    if news_path.suffix.lower() == ".jsonl":
        items = [json.loads(line) for line in raw.splitlines() if line.strip()]
    elif news_path.suffix.lower() == ".json":
        items = json.loads(raw)
    else:
        items = raw
    return {"available": True, "source": str(news_path), "items": items}


def item_date(value):
    if not isinstance(value, dict):
        return None
    for key in ["date", "published_at", "seendate", "seen_date"]:
        if value.get(key):
            parsed = pd.to_datetime(value[key], errors="coerce", utc=True)
            if pd.notna(parsed):
                return parsed.tz_convert(None).normalize()
    return None


def filter_news_context(news_context, start, end, max_items=8):
    if not news_context.get("available"):
        return news_context
    items = news_context.get("items", [])
    if not isinstance(items, list):
        return news_context
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    filtered = []
    for item in items:
        dt = item_date(item)
        if dt is None or not (start_ts <= dt <= end_ts):
            continue
        filtered.append(item)
    filtered.sort(key=lambda row: (row.get("relevance_score", 0), str(row.get("date", ""))), reverse=True)
    return {**news_context, "items": filtered[:max_items], "total_items_in_period": len(filtered)}


def load_existing(path):
    records = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                record = json.loads(line)
                records[record["period_end"]] = normalize_stored_record(record)
    return records


def save_records(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records.values(), key=lambda row: row["period_end"])
    with path.open("w", encoding="utf-8") as handle:
        for record in ordered:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def completed_fridays(start_date, latest_date):
    start = pd.Timestamp(start_date).normalize()
    latest = pd.Timestamp(latest_date).normalize()
    if latest < start:
        return []
    fridays = list(pd.date_range(start=start, end=latest, freq="W-FRI"))
    if start.weekday() == 4 and start not in fridays:
        fridays.insert(0, start)
    return fridays


def completed_week_periods(start_date, latest_date):
    start = pd.Timestamp(start_date).normalize()
    previous_end = start - pd.Timedelta(days=1)
    periods = []
    for week_end in completed_fridays(start, latest_date):
        week_start = max(start, previous_end + pd.Timedelta(days=1))
        periods.append((week_start, week_end))
        previous_end = week_end
    return periods


def wealth_return(frame, start, end, column):
    if column not in frame.columns:
        return None
    window = frame[(frame["date"] >= start) & (frame["date"] <= end)]
    if window.empty:
        return None
    prior = frame[frame["date"] < start].tail(1)
    start_value = prior[column].iloc[-1] if not prior.empty else window[column].iloc[0]
    end_value = window[column].iloc[-1]
    if pd.isna(start_value) or pd.isna(end_value) or start_value == 0:
        return None
    return float((end_value / start_value) - 1)


def max_drawdown(wealth):
    if wealth.empty:
        return None
    running_max = wealth.cummax()
    drawdown = (wealth / running_max) - 1
    return float(drawdown.min())


def latest_number(row, col):
    if col not in row.index or pd.isna(row[col]):
        return None
    return float(row[col])


def top_indicator_moves(indicators_window):
    moves = []
    if len(indicators_window) < 2:
        return moves
    for col in INDICATORS:
        if col not in indicators_window.columns:
            continue
        values = indicators_window[col].dropna()
        if len(values) < 2:
            continue
        start_val = values.iloc[0]
        end_val = values.iloc[-1]
        absolute_change = float(end_val - start_val)
        pct_change = float((end_val / start_val) - 1) if start_val != 0 else None
        if abs(absolute_change) < 1e-12 and (pct_change is None or abs(pct_change) < 1e-12):
            continue
        moves.append({"indicator": col, "start": float(start_val), "end": float(end_val), "absolute_change": absolute_change, "pct_change": pct_change})
    moves.sort(key=lambda item: abs(item["pct_change"]) if item["pct_change"] is not None else abs(item["absolute_change"]), reverse=True)
    return moves[:8]


def latest_scores(quadrant_row):
    scores = {}
    for col in ["score_Q1", "score_Q2", "score_Q3", "score_Q4", "max_score", "MACRO_GROWTH_SCORE", "MACRO_INFLATION_SCORE", "PROB_GROWTH_EMA", "PROB_INFLATION_EMA"]:
        value = latest_number(quadrant_row, col)
        if value is not None:
            scores[col] = value
    return scores


def latest_indicator_scores(quadrant_row):
    rows = []
    for ind in INDICATORS:
        combined_col = f"{ind}_combined"
        pos_col = f"{ind}_pos_score"
        var_col = f"{ind}_var_score"
        if combined_col not in quadrant_row.index:
            continue
        combined = latest_number(quadrant_row, combined_col)
        if combined is None:
            continue
        rows.append({"indicator": ind, "combined": combined, "position_score": latest_number(quadrant_row, pos_col), "variation_score": latest_number(quadrant_row, var_col), "latest_value": latest_number(quadrant_row, ind)})
    rows.sort(key=lambda item: abs(item["combined"]), reverse=True)
    return rows[:10]


def is_live_weight_column(col):
    return col.endswith("_weight") and not col.endswith("_base_weight") and "_hc_" not in col


def asset_name_from_weight_column(col):
    return col.replace("_weight", "")


def current_allocation(backtest_row):
    allocation = {}
    for col in backtest_row.index:
        if not is_live_weight_column(col):
            continue
        value = latest_number(backtest_row, col)
        if value and abs(value) > 0.000001:
            allocation[asset_name_from_weight_column(col)] = value
    return dict(sorted(allocation.items(), key=lambda item: item[1], reverse=True))


def allocation_changes(backtest_window):
    weight_cols = [col for col in backtest_window.columns if is_live_weight_column(col)]
    if not weight_cols or len(backtest_window) < 2:
        return {}
    first = backtest_window.iloc[0][weight_cols]
    last = backtest_window.iloc[-1][weight_cols]
    changes = (last - first).dropna()
    changes = changes[changes.abs() > 0.000001].sort_values(key=lambda s: s.abs(), ascending=False)
    return {asset_name_from_weight_column(col): float(value) for col, value in changes.items()}


def build_week_payload(start, end, backtest, quadrants, indicators, strategy_start, news_context):
    period_mask = (backtest["date"] >= start) & (backtest["date"] <= end)
    week_bt = backtest.loc[period_mask].copy()
    if week_bt.empty:
        return None
    quad_window = quadrants[(quadrants["date"] >= start) & (quadrants["date"] <= end)]
    ind_window = indicators[(indicators["date"] >= start) & (indicators["date"] <= end)]
    latest_bt = week_bt.iloc[-1]
    latest_quad = quad_window.iloc[-1] if not quad_window.empty else latest_bt
    current_quadrant = int(latest_bt.get("smooth_quadrant", latest_quad.get("assigned_quadrant", 0)))
    prior_bt = backtest[backtest["date"] < start].tail(1)
    prior_quadrant = int(prior_bt.iloc[-1]["smooth_quadrant"]) if not prior_bt.empty and "smooth_quadrant" in prior_bt.columns else None
    metrics = {
        "period_start": start.strftime("%Y-%m-%d"),
        "period_end": end.strftime("%Y-%m-%d"),
        "observations": int(len(week_bt)),
        "current_quadrant": current_quadrant,
        "current_quadrant_label": QUADRANT_LABELS.get(current_quadrant, "N/A"),
        "prior_quadrant": prior_quadrant,
        "prior_quadrant_label": QUADRANT_LABELS.get(prior_quadrant, "N/A"),
        "strategy_week_return": wealth_return(backtest, start, end, "wealth"),
        "sp500_week_return": wealth_return(backtest, start, end, "SP500_wealth"),
        "gold_week_return": wealth_return(backtest, start, end, "GOLD_wealth"),
        "strategy_since_start_return": wealth_return(backtest, strategy_start, end, "wealth"),
        "week_max_drawdown": max_drawdown(week_bt["wealth"]),
        "week_transaction_cost": float(week_bt.get("transaction_cost", pd.Series(dtype=float)).sum()),
        "week_ter_cost": float(week_bt.get("ter_cost", pd.Series(dtype=float)).sum()),
        "latest_wealth": float(latest_bt["wealth"]) if "wealth" in latest_bt.index else None,
        "latest_quadrant_scores": latest_scores(latest_quad),
        "latest_indicator_scores": latest_indicator_scores(latest_quad),
        "top_indicator_moves": top_indicator_moves(ind_window),
        "current_allocation": current_allocation(latest_bt),
        "allocation_changes": allocation_changes(week_bt),
    }
    weekly_news_context = filter_news_context(news_context, start, end)
    payload = {"period_start": metrics["period_start"], "period_end": metrics["period_end"], "metrics": metrics, "news_context": weekly_news_context}
    payload["input_digest"] = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    return payload



def compact_model_input(payload):
    metrics = payload["metrics"]
    return {
        "period": {"start": payload["period_start"], "end": payload["period_end"]},
        "regime": {"current": metrics.get("current_quadrant_label"), "prior": metrics.get("prior_quadrant_label"), "scores": metrics.get("latest_quadrant_scores", {})},
        "performance": {
            "strategy_week_return": metrics.get("strategy_week_return"),
            "sp500_week_return": metrics.get("sp500_week_return"),
            "gold_week_return": metrics.get("gold_week_return"),
            "strategy_since_start_return": metrics.get("strategy_since_start_return"),
            "week_max_drawdown": metrics.get("week_max_drawdown"),
            "week_transaction_cost": metrics.get("week_transaction_cost"),
            "week_ter_cost": metrics.get("week_ter_cost"),
        },
        "allocation": {"current": metrics.get("current_allocation", {}), "changes": metrics.get("allocation_changes", {})},
        "indicator_moves": metrics.get("top_indicator_moves", [])[:8],
        "indicator_scores": metrics.get("latest_indicator_scores", [])[:8],
        "news_context": payload.get("news_context", {}),
        "attribution_focus": {
            "explain_gold_if_abs_week_move_gt_2pct": True,
            "explain_strategy_gap_vs_sp500_and_gold": True,
            "score_units": {
                "risk_on_score": "-1 = risk-off, 0 = neutral, +1 = risk-on",
                "growth_score": "-1 = growth stress, 0 = neutral, +1 = strong growth impulse",
                "inflation_pressure_score": "-1 = disinflation pressure, 0 = neutral, +1 = inflation pressure",
                "policy_risk_score": "-1 = policy relief, 0 = neutral/unknown, +1 = Fed/fiscal/geopolitical policy risk",
                "confidence": "0 = weak confidence in the NLP reading, 1 = strong confidence; not a return probability",
            },
        },
    }


def system_prompt():
    return (
        "Tu es un analyste macro-financier prudent pour un projet quantitatif. "
        "Tu ecris en francais clair, lisible, utile pour un dashboard. "
        "Tu ne donnes pas de conseil d'investissement personnalise et tu ne promets jamais de performance. "
        "Tu relies strictement les donnees fournies: performance, quadrant, indicateurs, allocation et news si elles existent. "
        "Les scores NLP sont des scores normalises: -1 a +1 pour les biais directionnels, 0 a 1 pour confidence. "
        "Confidence n'est pas une probabilite de performance; c'est le degre de confiance dans la lecture narrative. "
        "Policy risk mesure le risque Fed/fiscal/reglementaire/geopolitique deduit des donnees fournies; 0 signifie neutre ou information insuffisante. "
        "Explique les ecarts de performance, surtout si l'or bouge fortement ou si la strategie sous-performe. "
        "Si aucune news n'est fournie, dis-le sobrement et n'invente aucun evenement recent. "
        "Ne copie jamais les donnees sous forme de JSON dans les champs redactionnels."
    )


def structured_schema():
    return {
        "title": "short human title",
        "macro_regime": "current macro regime label",
        "strategy_status": "short status sentence",
        "available_information": {"algo_data": ["short factual bullet"], "news_data": [], "missing_information": []},
        "nlp_signal": {"risk_on_score": "number -1..1", "growth_score": "number -1..1", "inflation_pressure_score": "number -1..1", "policy_risk_score": "number -1..1", "confidence": "number 0..1", "suggested_use": "shadow_only | reduce_risk | add_risk | no_change", "rationale": "2-3 readable French sentences"},
        "forward_view": {"base_case_next_week": "2 readable French sentences", "bull_case": "1-2 readable French sentences", "bear_case": "1-2 readable French sentences", "watchlist": ["short watch item"]},
        "key_points": ["short bullet", "short bullet", "short bullet"],
        "risks": ["short bullet", "short bullet", "short bullet"],
        "allocation_comment": "2 readable French sentences",
        "performance_attribution": {
            "gold_move_explanation": "why gold likely moved from the provided rates, dollar, inflation, volatility and risk indicators; say if data is insufficient",
            "strategy_vs_benchmark_gap": "why the strategy beat or lagged S&P 500 and gold this week",
            "model_miss_explanation": "plain-language explanation of what the model did not capture this week",
            "what_to_monitor": ["specific indicator to monitor", "specific indicator to monitor"]
        },
    }


def structured_user_prompt(model_input):
    return "Etape 1/2: analyse structuree. Retourne uniquement un objet JSON valide, sans markdown et sans texte autour.\nSchema:\n" + json.dumps(structured_schema(), ensure_ascii=False, indent=2) + "\n\nDonnees compactes:\n" + json.dumps(model_input, ensure_ascii=False, indent=2)


def markdown_user_prompt(model_input, structured_result):
    schema = {"markdown": "clean Markdown debrief, no JSON, 350-550 words"}
    return "Etape 2/2: redaction finale. Retourne uniquement un objet JSON valide avec la cle markdown. Le markdown doit etre une synthese lisible pour Streamlit, jamais le JSON brut. Sections: Synthese, Regime macro, Performance, Signal NLP, Allocation, Points de vigilance.\nSchema:\n" + json.dumps(schema, ensure_ascii=False, indent=2) + "\n\nAnalyse structuree validee:\n" + json.dumps(structured_result, ensure_ascii=False, indent=2) + "\n\nDonnees compactes:\n" + json.dumps(model_input, ensure_ascii=False, indent=2)


def legacy_user_prompt(payload):
    schema = deepcopy(structured_schema())
    schema["markdown"] = "clean Markdown debrief, no JSON"
    return "Redige un debrief hebdomadaire pour la strategie Macro 4 Saisons. Retourne uniquement un JSON valide.\nSchema:\n" + json.dumps(schema, ensure_ascii=False, indent=2) + "\n\nDonnees compactes:\n" + json.dumps(compact_model_input(payload), ensure_ascii=False, indent=2)


def call_openrouter(api_key, model, messages, max_tokens=1800):
    import requests
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "HTTP-Referer": "https://github.com/Nik1go/Macro-portfolio-4-saisons", "X-OpenRouter-Title": "Macro 4 Saisons Portfolio"}
    body = {"model": model, "temperature": 0.15, "max_tokens": max_tokens, "response_format": {"type": "json_object"}, "messages": messages}
    response = requests.post(OPENROUTER_URL, headers=headers, json=body, timeout=90)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"], data.get("usage", {})

def parse_model_content(content):
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        return {}
    cleaned = strip_code_fence(content)
    parsed = parse_json_or_literal(cleaned)
    if parsed:
        return parsed
    start = cleaned.find("{")
    if start >= 0:
        parsed = parse_json_or_literal(cleaned[start:])
        if parsed:
            return parsed
    return fallback_model_result(content)


def strip_code_fence(text):
    cleaned = text.strip()
    if not cleaned.startswith("```"):
        return cleaned
    cleaned = cleaned.removeprefix("```").strip()
    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()
    return cleaned


def parse_json_or_literal(text):
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    try:
        parsed, _ = json.JSONDecoder().raw_decode(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    pythonish = re.sub(r"\bnull\b", "None", text)
    pythonish = re.sub(r"\btrue\b", "True", pythonish, flags=re.IGNORECASE)
    pythonish = re.sub(r"\bfalse\b", "False", pythonish, flags=re.IGNORECASE)
    try:
        parsed = ast.literal_eval(pythonish)
    except (SyntaxError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def fallback_model_result(raw_content):
    return {
        "title": "Debrief hebdomadaire",
        "macro_regime": "",
        "strategy_status": "",
        "available_information": {"algo_data": [], "news_data": [], "missing_information": ["Reponse NLP non parseable en JSON strict."]},
        "nlp_signal": deepcopy(DEFAULT_SIGNAL),
        "forward_view": deepcopy(DEFAULT_FORWARD_VIEW),
        "key_points": [],
        "risks": ["Reponse NLP non parseable en JSON strict."],
        "allocation_comment": "",
        "markdown": raw_content if isinstance(raw_content, str) else "",
    }



def as_list(value):
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def as_dict(value):
    return value if isinstance(value, dict) else {}


def is_empty_or_fallback(value):
    if value is None:
        return True
    if isinstance(value, dict):
        return not value or all(is_empty_or_fallback(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return not value or all(is_empty_or_fallback(item) for item in value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        return not normalized or "non parseable" in normalized or "fallback because model output" in normalized
    return False


def clamp_number(value, low, high, default=0.0):
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return default
    return float(min(high, max(low, numeric)))


def coerce_model_result(model_result, payload=None):
    model_result = as_dict(model_result)
    coerced = {
        "title": str(model_result.get("title") or "Debrief hebdomadaire").strip(),
        "macro_regime": str(model_result.get("macro_regime") or "").strip(),
        "strategy_status": str(model_result.get("strategy_status") or "").strip(),
    }
    info = as_dict(model_result.get("available_information"))
    coerced["available_information"] = {
        "algo_data": as_list(info.get("algo_data")),
        "news_data": as_list(info.get("news_data")),
        "missing_information": as_list(info.get("missing_information")),
    }
    signal = {**DEFAULT_SIGNAL, **as_dict(model_result.get("nlp_signal"))}
    suggested = signal.get("suggested_use")
    coerced["nlp_signal"] = {
        "risk_on_score": clamp_number(signal.get("risk_on_score"), -1, 1),
        "growth_score": clamp_number(signal.get("growth_score"), -1, 1),
        "inflation_pressure_score": clamp_number(signal.get("inflation_pressure_score"), -1, 1),
        "policy_risk_score": clamp_number(signal.get("policy_risk_score"), -1, 1),
        "confidence": clamp_number(signal.get("confidence"), 0, 1),
        "suggested_use": suggested if suggested in {"shadow_only", "reduce_risk", "add_risk", "no_change"} else "shadow_only",
        "rationale": str(signal.get("rationale") or "").strip(),
    }
    forward = {**DEFAULT_FORWARD_VIEW, **as_dict(model_result.get("forward_view"))}
    coerced["forward_view"] = {
        "base_case_next_week": str(forward.get("base_case_next_week") or "").strip(),
        "bull_case": str(forward.get("bull_case") or "").strip(),
        "bear_case": str(forward.get("bear_case") or "").strip(),
        "watchlist": as_list(forward.get("watchlist"))[:6],
    }
    coerced["key_points"] = as_list(model_result.get("key_points"))[:5]
    coerced["risks"] = as_list(model_result.get("risks"))[:5]
    coerced["allocation_comment"] = str(model_result.get("allocation_comment") or "").strip()
    attribution = as_dict(model_result.get("performance_attribution"))
    coerced["performance_attribution"] = {
        "gold_move_explanation": str(attribution.get("gold_move_explanation") or "").strip(),
        "strategy_vs_benchmark_gap": str(attribution.get("strategy_vs_benchmark_gap") or "").strip(),
        "model_miss_explanation": str(attribution.get("model_miss_explanation") or "").strip(),
        "what_to_monitor": as_list(attribution.get("what_to_monitor"))[:5],
    }
    coerced["markdown"] = clean_markdown(model_result.get("markdown"), coerced, payload)
    return coerced


def clean_markdown(value, model_result, payload=None):
    if not isinstance(value, str) or not value.strip():
        return build_fallback_markdown(model_result, payload)
    stripped = value.strip()
    embedded = parse_json_or_literal(strip_code_fence(stripped))
    if embedded:
        inner = embedded.get("markdown")
        if isinstance(inner, str) and inner.strip() and not parse_json_or_literal(strip_code_fence(inner.strip())):
            return inner.strip()
        return build_fallback_markdown({**embedded, **model_result}, payload)
    return stripped


def format_pct_text(value):
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "n/a"
    return f"{numeric * 100:+.2f}%"


def format_signal_text(value):
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "n/a"
    return f"{numeric:+.2f}"


def build_fallback_markdown(model_result, payload=None):
    metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
    signal = as_dict(model_result.get("nlp_signal"))
    forward = as_dict(model_result.get("forward_view"))
    attribution = as_dict(model_result.get("performance_attribution"))
    allocation = metrics.get("current_allocation", {})
    allocation_text = ", ".join(f"{asset}: {weight * 100:.1f}%" for asset, weight in allocation.items()) or "non disponible"
    lines = [
        f"# {model_result.get('title') or 'Debrief hebdomadaire'}",
        "",
        "## Synthese",
        model_result.get("strategy_status") or "Synthese non disponible.",
        "",
        "## Performance",
        f"Strategie: {format_pct_text(metrics.get('strategy_week_return'))} | S&P 500: {format_pct_text(metrics.get('sp500_week_return'))} | Or: {format_pct_text(metrics.get('gold_week_return'))} | Drawdown semaine: {format_pct_text(metrics.get('week_max_drawdown'))}.",
        "",
        "## Signal NLP",
        f"Risk-on {format_signal_text(signal.get('risk_on_score'))}, croissance {format_signal_text(signal.get('growth_score'))}, inflation {format_signal_text(signal.get('inflation_pressure_score'))}, confiance {format_signal_text(signal.get('confidence'))}.",
        signal.get("rationale") or "Rationale non disponible.",
        "",
        "## Attribution",
        attribution.get("gold_move_explanation") or "Attribution de l'or non disponible.",
        attribution.get("model_miss_explanation") or "Explication de l'ecart modele non disponible.",
        "",
        "## Allocation",
        f"Allocation actuelle: {allocation_text}.",
        model_result.get("allocation_comment") or "Aucun commentaire d'allocation disponible.",
        "",
        "## Vue prospective",
        forward.get("base_case_next_week") or "Scenario central non disponible.",
    ]
    return "\n".join(lines).strip()


def is_fallback_signal(signal):
    signal = as_dict(signal)
    if not signal:
        return True
    keys = ["risk_on_score", "growth_score", "inflation_pressure_score", "policy_risk_score", "confidence"]
    values = pd.to_numeric([signal.get(key) for key in keys], errors="coerce")
    all_zero = all(pd.notna(value) and abs(value) < 1e-12 for value in values)
    rationale = str(signal.get("rationale", "")).lower()
    return all_zero and ("fallback" in rationale or signal.get("suggested_use") == "shadow_only")


def normalize_stored_record(record):
    normalized = dict(record)
    embedded = {}
    for key in ["raw_model_result", "model_result", "raw_model_content", "markdown"]:
        value = normalized.get(key)
        if isinstance(value, str):
            embedded = parse_model_content(value)
        elif isinstance(value, dict):
            embedded = value
        if embedded and not (embedded.get("markdown") == value and embedded.get("risks") == ["Reponse NLP non parseable en JSON strict."]):
            break
        embedded = {}
    if embedded:
        embedded = coerce_model_result(embedded, normalized)
        for key in MODEL_RESULT_KEYS:
            current = normalized.get(key)
            replacement = embedded.get(key)
            if key == "nlp_signal":
                if is_fallback_signal(current):
                    normalized[key] = replacement
            elif key == "markdown":
                if is_empty_or_fallback(current) or parse_json_or_literal(strip_code_fence(str(current))):
                    normalized[key] = replacement
            elif key == "title":
                if is_empty_or_fallback(current) or current == "Debrief hebdomadaire":
                    normalized[key] = replacement
            elif is_empty_or_fallback(current):
                normalized[key] = replacement
    normalized["markdown"] = clean_markdown(normalized.get("markdown"), normalized, normalized)
    return normalized


def merge_usage(*usages):
    merged = {}
    for usage in usages:
        for key, value in (usage or {}).items():
            if isinstance(value, (int, float)):
                merged[key] = merged.get(key, 0) + value
            else:
                merged[key] = value
    return merged


def generate_model_result(api_key, model, payload, single_call=False):
    model_input = compact_model_input(payload)
    if single_call:
        content, usage = call_openrouter(api_key, model, [{"role": "system", "content": system_prompt()}, {"role": "user", "content": legacy_user_prompt(payload)}], max_tokens=1800)
        return coerce_model_result(parse_model_content(content), payload), usage
    structured_content, structured_usage = call_openrouter(api_key, model, [{"role": "system", "content": system_prompt()}, {"role": "user", "content": structured_user_prompt(model_input)}], max_tokens=1200)
    structured_result = coerce_model_result(parse_model_content(structured_content), payload)
    markdown_content, markdown_usage = call_openrouter(api_key, model, [{"role": "system", "content": system_prompt()}, {"role": "user", "content": markdown_user_prompt(model_input, structured_result)}], max_tokens=1400)
    markdown_result = parse_model_content(markdown_content)
    structured_result["markdown"] = clean_markdown(markdown_result.get("markdown", markdown_content), structured_result, payload)
    return structured_result, merge_usage(structured_usage, markdown_usage)

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


def get_openrouter_key():
    return os.getenv("OPENROUTER_API_KEY") or os.getenv("KEY")


def main():
    args = parse_args()
    base_dir = Path(args.base_dir)
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv_if_present(repo_root)
    output_path = Path(args.output) if args.output else base_dir / "nlp" / "weekly_debriefs.jsonl"
    existing = load_existing(output_path)
    if args.repair_only:
        save_records(output_path, existing)
        print(f"Repaired {len(existing)} existing records in {output_path}")
        return

    api_key = get_openrouter_key()
    model = args.model or os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)
    if not api_key and not args.dry_run:
        message = "No OpenRouter key found. Set KEY or OPENROUTER_API_KEY."
        if args.skip_if_missing_key:
            print(message)
            return
        raise RuntimeError(message)

    backtest, quadrants, indicators = load_inputs(base_dir)
    default_news_file = base_dir / "nlp" / "news_events.jsonl"
    news_context = load_news_context(args.news_file or default_news_file)
    strategy_start = pd.Timestamp(args.start_date)
    latest_date = pd.Timestamp(args.end_date) if args.end_date else backtest["date"].max()
    week_periods = completed_week_periods(strategy_start, latest_date)
    if args.max_weeks:
        week_periods = week_periods[-args.max_weeks:]
    if not week_periods:
        print(f"No completed Friday week to analyse between start_date={strategy_start.date()} and latest_date={pd.Timestamp(latest_date).date()}.")
        return

    generated = 0
    skipped = 0
    for week_start, week_end in week_periods:
        payload = build_week_payload(week_start, week_end, backtest, quadrants, indicators, strategy_start, news_context)
        if payload is None:
            continue
        current = existing.get(payload["period_end"])
        if current and current.get("input_digest") == payload["input_digest"] and not args.force:
            skipped += 1
            continue
        if args.dry_run:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            generated += 1
            continue
        model_result, usage = generate_model_result(api_key, model, payload, single_call=args.single_call)
        existing[payload["period_end"]] = {
            **payload,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "generation_mode": "single_call" if args.single_call else "segmented_v2",
            "title": model_result.get("title", "Debrief hebdomadaire"),
            "macro_regime": model_result.get("macro_regime", ""),
            "strategy_status": model_result.get("strategy_status", ""),
            "available_information": model_result.get("available_information", {}),
            "nlp_signal": model_result.get("nlp_signal", {}),
            "forward_view": model_result.get("forward_view", {}),
            "key_points": model_result.get("key_points", []),
            "risks": model_result.get("risks", []),
            "allocation_comment": model_result.get("allocation_comment", ""),
            "performance_attribution": model_result.get("performance_attribution", {}),
            "markdown": model_result.get("markdown", ""),
            "usage": usage,
        }
        generated += 1
        save_records(output_path, existing)
        print(f"Generated debrief for {payload['period_end']}")

    if not args.dry_run:
        save_records(output_path, existing)
    print(f"Done. generated={generated}, skipped={skipped}, output={output_path}")


if __name__ == "__main__":
    main()
