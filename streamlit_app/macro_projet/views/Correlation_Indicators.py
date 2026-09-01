"""
Page 5: NLP Weekly Debrief
Weekly macro/strategy debrief generated with OpenRouter.
"""

import json
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


@st.cache_data
def load_weekly_debriefs():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
    path = os.path.join(project_root, "data", "US", "nlp", "weekly_debriefs.jsonl")
    rows = []
    if not os.path.exists(path):
        return None, path
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Ignored invalid NLP JSON line {line_number}")
    except Exception as exc:
        print(f"Could not read NLP debriefs: {exc}")
        return None, path
    if not rows:
        return None, path
    df = pd.DataFrame(rows)
    df["period_start"] = pd.to_datetime(df["period_start"])
    df["period_end"] = pd.to_datetime(df["period_end"])
    return df.sort_values("period_end").reset_index(drop=True), path


def is_missing(value):
    if isinstance(value, (dict, list, tuple)):
        return False
    return value is None or pd.isna(value)


def safe_text(value, fallback="N/A"):
    return fallback if is_missing(value) else str(value)


def safe_list(value):
    return value if isinstance(value, list) else []


def safe_dict(value):
    return value if isinstance(value, dict) else {}


def clean_asset_name(name):
    return str(name).replace("_", " ")


def fmt_pct(value):
    if is_missing(value):
        return "N/A"
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "N/A"
    return f"{numeric * 100:+.2f}%"


def fmt_money(value):
    if is_missing(value):
        return "N/A"
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "N/A"
    return f"{numeric:,.0f}$".replace(",", " ")


def format_signal_value(value):
    if is_missing(value):
        return "N/A"
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "N/A"
    return f"{numeric:+.2f}"


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
    "markdown",
]


def parse_json_object_text(value):
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    cleaned = value.strip()
    if not cleaned:
        return {}
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```").removesuffix("```").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        if start < 0:
            return {}
        try:
            parsed, _ = json.JSONDecoder().raw_decode(cleaned[start:])
        except json.JSONDecodeError:
            return {}
    return parsed if isinstance(parsed, dict) else {}


def is_empty_or_fallback(value):
    if is_missing(value):
        return True
    if isinstance(value, dict):
        return not value or all(is_empty_or_fallback(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return not value or all(is_empty_or_fallback(item) for item in value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        return not normalized or "non parseable" in normalized or "fallback because model output" in normalized
    return False


def is_fallback_signal(signal):
    signal = safe_dict(signal)
    if not signal:
        return True
    score_keys = ["risk_on_score", "growth_score", "inflation_pressure_score", "policy_risk_score", "confidence"]
    values = pd.to_numeric([signal.get(key) for key in score_keys], errors="coerce")
    all_zero = all(pd.notna(value) and abs(value) < 1e-12 for value in values)
    rationale = str(signal.get("rationale", "")).lower()
    return all_zero and ("fallback" in rationale or signal.get("suggested_use") == "shadow_only")


def normalize_debrief_record(record):
    normalized = dict(record)
    embedded = {}
    for key in ["raw_model_result", "model_result", "markdown"]:
        embedded = parse_json_object_text(normalized.get(key))
        if embedded:
            break
    if not embedded:
        return normalized

    for key in MODEL_RESULT_KEYS:
        if key not in embedded:
            continue
        current = normalized.get(key)
        replacement = embedded.get(key)
        if key == "nlp_signal":
            if is_fallback_signal(current):
                normalized[key] = replacement
        elif key == "markdown":
            if is_empty_or_fallback(current) or parse_json_object_text(current):
                normalized[key] = replacement
        elif key == "title":
            if is_empty_or_fallback(current) or current == "Debrief hebdomadaire":
                normalized[key] = replacement
        elif is_empty_or_fallback(current):
            normalized[key] = replacement
    return normalized


def render(data):
    st.header("NLP Weekly Debrief")
    st.caption(
        "Weekly OpenRouter analysis based on current algorithm data. "
        "The NLP signal is stored as shadow information first, not as a live allocation override."
    )

    nlp_df = data.get("weekly_nlp") if isinstance(data, dict) else None
    nlp_path = None
    if nlp_df is None:
        nlp_df, nlp_path = load_weekly_debriefs()

    if nlp_df is None or nlp_df.empty:
        st.warning("No weekly NLP debrief has been generated yet.")
        st.markdown("Run the generator on the VPS/WSL repo after exposing the OpenRouter key:")
        st.code(
            "export KEY='sk-or-v1-...'\n"
            "python nlp_jobs/generate_weekly_debriefs.py --start-date 2026-04-03",
            language="bash",
        )
        if nlp_path:
            st.caption(f"Expected output: {nlp_path}")
        return

    nlp_df = nlp_df.sort_values("period_end").reset_index(drop=True)
    labels = [
        f"{row.period_start.strftime('%Y-%m-%d')} -> {row.period_end.strftime('%Y-%m-%d')}"
        for row in nlp_df.itertuples()
    ]
    selected_label = st.selectbox("Analysed week", labels, index=len(labels) - 1)
    record = normalize_debrief_record(nlp_df.iloc[labels.index(selected_label)].to_dict())
    metrics = safe_dict(record.get("metrics"))
    signal = safe_dict(record.get("nlp_signal"))
    forward_view = safe_dict(record.get("forward_view"))

    st.subheader(safe_text(record.get("title"), "Weekly macro debrief"))

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Strategy week", fmt_pct(metrics.get("strategy_week_return")))
    kpi2.metric("SP500 week", fmt_pct(metrics.get("sp500_week_return")))
    kpi3.metric("Since 2026-04-03", fmt_pct(metrics.get("strategy_since_start_return")))
    kpi4.metric("Week max drawdown", fmt_pct(metrics.get("week_max_drawdown")))

    meta1, meta2, meta3 = st.columns(3)
    meta1.metric("Current quadrant", safe_text(metrics.get("current_quadrant_label")))
    meta2.metric("NLP risk-on", format_signal_value(signal.get("risk_on_score")))
    meta3.metric("NLP confidence", fmt_pct(signal.get("confidence")))

    generated_at = record.get("generated_at")
    model_name = record.get("model", "N/A")
    if generated_at:
        st.caption(f"Generated at {pd.to_datetime(generated_at).strftime('%Y-%m-%d %H:%M')} with {model_name}")
    else:
        st.caption(f"Model: {model_name}")

    st.divider()
    st.markdown(safe_text(record.get("markdown") or record.get("macro_regime"), "Debrief unavailable."))

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("### Structured NLP signal")
        signal_rows = [
            ("Risk-on", signal.get("risk_on_score")),
            ("Growth", signal.get("growth_score")),
            ("Inflation pressure", signal.get("inflation_pressure_score")),
            ("Policy risk", signal.get("policy_risk_score")),
            ("Confidence", signal.get("confidence")),
        ]
        signal_df = pd.DataFrame(signal_rows, columns=["Signal", "Value"])
        signal_df["Value"] = pd.to_numeric(signal_df["Value"], errors="coerce")
        st.dataframe(signal_df.style.format({"Value": "{:+.2f}"}), use_container_width=True, hide_index=True)
        st.info(safe_text(signal.get("suggested_use"), "shadow_only"))
        st.caption(safe_text(signal.get("rationale"), "No rationale provided."))

    with col_right:
        st.markdown("### Forward view")
        st.markdown(f"**Base case:** {safe_text(forward_view.get('base_case_next_week'))}")
        st.markdown(f"**Bull case:** {safe_text(forward_view.get('bull_case'))}")
        st.markdown(f"**Bear case:** {safe_text(forward_view.get('bear_case'))}")
        watchlist = safe_list(forward_view.get("watchlist"))
        if watchlist:
            st.markdown("**Watchlist:**")
            for item in watchlist:
                st.markdown(f"- {item}")

    st.divider()
    points_col, risks_col = st.columns(2)
    with points_col:
        st.markdown("### Key points")
        for point in safe_list(record.get("key_points")):
            st.markdown(f"- {point}")

    with risks_col:
        st.markdown("### Risks / watch points")
        for risk in safe_list(record.get("risks")):
            st.markdown(f"- {risk}")

    st.divider()
    chart_col, alloc_col = st.columns(2)
    with chart_col:
        st.markdown("### Weekly curve")
        backtest = data.get("backtest") if isinstance(data, dict) else None
        if backtest is not None and not backtest.empty:
            period_start = pd.to_datetime(record["period_start"])
            period_end = pd.to_datetime(record["period_end"])
            bt = backtest.copy()
            bt["date"] = pd.to_datetime(bt["date"])
            week_bt = bt[(bt["date"] >= period_start) & (bt["date"] <= period_end)]
            if not week_bt.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=week_bt["date"], y=week_bt["wealth"] / week_bt["wealth"].iloc[0] * 100, name="Strategy", line=dict(width=4)))
                if "SP500_wealth" in week_bt.columns:
                    fig.add_trace(go.Scatter(x=week_bt["date"], y=week_bt["SP500_wealth"] / week_bt["SP500_wealth"].iloc[0] * 100, name="SP500", line=dict(dash="dash")))
                fig.update_layout(height=330, yaxis_title="Base 100", hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("No backtest points for this week.")
        else:
            st.caption("Backtest data unavailable.")

    with alloc_col:
        st.markdown("### Current allocation")
        allocation = safe_dict(metrics.get("current_allocation"))
        alloc_rows = [
            {"Asset": clean_asset_name(asset), "Weight": pd.to_numeric(weight, errors="coerce") * 100}
            for asset, weight in allocation.items()
            if not str(asset).endswith("_base")
        ]
        if alloc_rows:
            alloc_df = pd.DataFrame(alloc_rows).dropna(subset=["Weight"])
            fig_alloc = px.bar(alloc_df, x="Weight", y="Asset", orientation="h", text=alloc_df["Weight"].map(lambda value: f"{value:.1f}%"))
            fig_alloc.update_layout(height=330, xaxis_title="Weight (%)", yaxis_title="")
            st.plotly_chart(fig_alloc, use_container_width=True)
        else:
            st.caption("Allocation unavailable.")

    st.divider()
    ind_col, score_col = st.columns(2)
    with ind_col:
        st.markdown("### Indicator moves")
        moves = safe_list(metrics.get("top_indicator_moves"))
        if moves:
            moves_df = pd.DataFrame(moves)
            display_cols = ["indicator", "start", "end", "absolute_change", "pct_change"]
            display_cols = [col for col in display_cols if col in moves_df.columns]
            st.dataframe(
                moves_df[display_cols].style.format({"start": "{:.2f}", "end": "{:.2f}", "absolute_change": "{:+.2f}", "pct_change": "{:+.2%}"}),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No indicator moves available.")

    with score_col:
        st.markdown("### Quadrant scores")
        scores = safe_dict(metrics.get("latest_quadrant_scores"))
        if scores:
            score_df = pd.DataFrame([{"Score": key, "Value": value} for key, value in scores.items()])
            st.dataframe(score_df.style.format({"Value": "{:.2f}"}), use_container_width=True, hide_index=True)
        else:
            st.caption("Quadrant scores unavailable.")
