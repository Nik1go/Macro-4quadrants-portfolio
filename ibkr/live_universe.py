"""Live-tradable universe used by IBKR execution and live-compatible backtests.

The model backtest still uses long-history US proxies such as SPY/QQQ/IWM/VNQ.
This file documents the instruments that should be used when estimating what the
IBKR account would have tracked with the actual executable UCITS/ETC universe.

IBKR qualification and account-level permission must still be checked with
``python -m ibkr.test_live_universe --what-if`` on the VPS while IB Gateway is up.
"""

from __future__ import annotations

from typing import Dict, Iterable, List


LIVE_UNIVERSE: Dict[str, Dict[str, object]] = {
    "SP500": {
        "model_proxy": "SPY",
        "ibkr_symbol": "SXR8",
        "yahoo_ticker": "SXR8.DE",
        "name": "iShares Core S&P 500 UCITS ETF USD Acc",
        "isin": "IE00B5BMR087",
        "sec_type": "STK",
        "exchange": "SMART",
        "primary_exchange": "IBIS",
        "currency": "EUR",
        "ter": 0.0007,
        "status": "primary",
        "note": "Closest UCITS replacement for SPY. EUR Xetra quote used for live-compatible backtest.",
    },
    "GOLD_OZ_USD": {
        "model_proxy": "GLD",
        "ibkr_symbol": "SGLD",
        "yahoo_ticker": "SGLD.AS",
        "name": "Invesco Physical Gold ETC",
        "isin": "IE00B579F325",
        "sec_type": "STK",
        "exchange": "AEB",
        "primary_exchange": "AEB",
        "currency": "EUR",
        "ter": 0.0012,
        "status": "primary",
        "note": "Current live ticker. This is Invesco Physical Gold ETC, not iShares.",
    },
    "SmallCAP": {
        "model_proxy": "IWM",
        "ibkr_symbol": "ZPRR",
        "yahoo_ticker": "ZPRR.DE",
        "name": "SPDR Russell 2000 U.S. Small Cap UCITS ETF Acc",
        "isin": "IE00BJ38QD84",
        "sec_type": "STK",
        "exchange": "SMART",
        "primary_exchange": "IBIS",
        "currency": "EUR",
        "ter": 0.0030,
        "status": "primary_remap",
        "note": "Replaces IUSN because the model proxy is IWM/Russell 2000, not MSCI World Small Cap.",
    },
    "US_REIT_VNQ": {
        "model_proxy": "VNQ",
        "ibkr_symbol": "IUSP",
        "yahoo_ticker": "IUSP.AS",
        "name": "iShares US Property Yield UCITS ETF",
        "isin": "IE00B1FZSF77",
        "sec_type": "STK",
        "exchange": "SMART",
        "primary_exchange": "AEB",
        "currency": "EUR",
        "ter": 0.0040,
        "status": "primary",
        "note": "UCITS US real-estate replacement for VNQ. It tracks dividend-focused US property equities.",
    },
    "OBLIGATION": {
        "model_proxy": "LQD",
        "ibkr_symbol": "LQDE",
        "yahoo_ticker": "LQDE.MI",
        "name": "iShares $ Corp Bond UCITS ETF",
        "isin": "IE0032895942",
        "sec_type": "STK",
        "exchange": "SMART",
        "primary_exchange": "BVME.ETF",
        "currency": "EUR",
        "ter": 0.0020,
        "status": "primary_remap_yahoo",
        "note": "Execution symbol kept as LQDE so existing live positions still map; Yahoo EUR quote uses LQDE.MI because LQDE.AS is empty.",
    },
    "TREASURY_10Y": {
        "model_proxy": "IEF",
        "ibkr_symbol": "SXRM",
        "yahoo_ticker": "SXRM.DE",
        "name": "iShares $ Treasury Bond 7-10yr UCITS ETF USD Acc",
        "isin": "IE00B3VWN518",
        "sec_type": "STK",
        "exchange": "SMART",
        "primary_exchange": "IBIS",
        "currency": "USD",
        "ter": 0.0007,
        "status": "primary_fixed_isin",
        "note": "Fixes the old config ISIN. Xetra SXRM quotes in USD, so EUR backtest converts with USD_EUR.",
    },
    "NASDAQ_100": {
        "model_proxy": "QQQ",
        "ibkr_symbol": "SXRV",
        "yahoo_ticker": "SXRV.DE",
        "name": "iShares NASDAQ 100 UCITS ETF USD Acc",
        "isin": "IE00B53SZB19",
        "sec_type": "STK",
        "exchange": "SMART",
        "primary_exchange": "IBIS",
        "currency": "EUR",
        "ter": 0.0033,
        "status": "primary",
        "note": "Closest UCITS replacement for QQQ. EUR Xetra quote used.",
    },
    "COMMODITIES": {
        "model_proxy": "DBC",
        "ibkr_symbol": "EXXY",
        "yahoo_ticker": "EXXY.DE",
        "name": "iShares Diversified Commodity Swap UCITS ETF DE",
        "isin": "DE000A0H0728",
        "sec_type": "STK",
        "exchange": "SMART",
        "primary_exchange": "IBIS",
        "currency": "EUR",
        "ter": 0.0046,
        "status": "primary",
        "note": "Synthetic UCITS commodity basket. Not the same index/methodology as DBC.",
    },
    "SHORT_SP500": {
        "model_proxy": "SH",
        "ibkr_symbol": "DXS3",
        "yahoo_ticker": "DXS3.DE",
        "name": "Xtrackers S&P 500 Inverse Daily Swap UCITS ETF 1C",
        "isin": "LU0322251520",
        "sec_type": "STK",
        "exchange": "SMART",
        "primary_exchange": "IBIS",
        "currency": "EUR",
        "ter": 0.0050,
        "status": "primary_fixed_ter",
        "note": "UCITS daily inverse S&P 500. TER updated versus the old SH/proxy comment.",
    },
    "USD_JPY": {
        "model_proxy": "USDJPY=X",
        "ibkr_symbol": "USD_JPY",
        "yahoo_ticker": "USDJPY=X",
        "name": "USD/JPY CFD proxy",
        "isin": "",
        "sec_type": "CFD",
        "exchange": "SMART",
        "primary_exchange": "",
        "currency": "JPY",
        "ter": 0.0,
        "status": "cfd_check_required",
        "note": "FX exposure remains account-permission dependent. Confirm CFD permission in IBKR.",
    },
    "USD_EUR": {
        "model_proxy": "USDEUR=X",
        "ibkr_symbol": "USD_EUR",
        "yahoo_ticker": "USDEUR=X",
        "name": "EUR/USD CFD proxy used to express USD_EUR exposure",
        "isin": "",
        "sec_type": "CFD",
        "exchange": "SMART",
        "primary_exchange": "",
        "currency": "USD",
        "ter": 0.0,
        "status": "cfd_check_required",
        "note": "Existing code uses EUR CFD in USD currency for inverse USD_EUR exposure.",
    },
}


LIVE_CANDIDATES: Dict[str, List[Dict[str, str]]] = {
    "SmallCAP": [
        {"ibkr_symbol": "ZPRR", "yahoo_ticker": "ZPRR.DE", "isin": "IE00BJ38QD84", "note": "Preferred: Russell 2000 UCITS, EUR Xetra."},
        {"ibkr_symbol": "R2US", "yahoo_ticker": "R2US.PA", "isin": "IE00BJ38QD84", "note": "Same SPDR fund on Euronext Paris."},
        {"ibkr_symbol": "XRS2", "yahoo_ticker": "XRS2.DE", "isin": "IE00BJZ2DD79", "note": "Xtrackers Russell 2000 UCITS alternative."},
        {"ibkr_symbol": "IUSN", "yahoo_ticker": "IUSN.AS", "isin": "IE00BF4RFH31", "note": "Old mapping; broad world small-cap, not Russell 2000."},
    ],
    "OBLIGATION": [
        {"ibkr_symbol": "LQDE", "yahoo_ticker": "LQDE.MI", "isin": "IE0032895942", "note": "Kept for existing live mapping; EUR Milan quote works on Yahoo."},
        {"ibkr_symbol": "IBCD", "yahoo_ticker": "IBCD.DE", "isin": "IE0032895942", "note": "Same fund on Xetra in EUR; cleaner if IBKR qualifies it for the account."},
    ],
    "US_REIT_VNQ": [
        {"ibkr_symbol": "IUSP", "yahoo_ticker": "IUSP.AS", "isin": "IE00B1FZSF77", "note": "Preferred Euronext Amsterdam EUR quote."},
        {"ibkr_symbol": "IQQ7", "yahoo_ticker": "IQQ7.DE", "isin": "IE00B1FZSF77", "note": "Same fund on Xetra in EUR."},
    ],
}


def selected_live_rows(assets: Iterable[str] | None = None) -> List[Dict[str, object]]:
    keys = list(assets) if assets is not None else list(LIVE_UNIVERSE.keys())
    return [dict(asset=asset, **LIVE_UNIVERSE[asset]) for asset in keys if asset in LIVE_UNIVERSE]


def selected_yfinance_mapping() -> Dict[str, str]:
    return {
        asset: str(meta["yahoo_ticker"])
        for asset, meta in LIVE_UNIVERSE.items()
        if meta.get("yahoo_ticker")
    }


def selected_etf_mapping() -> Dict[str, str]:
    return {asset: str(meta["ibkr_symbol"]) for asset, meta in LIVE_UNIVERSE.items()}


def selected_ter() -> Dict[str, float]:
    return {asset: float(meta.get("ter", 0.0) or 0.0) for asset, meta in LIVE_UNIVERSE.items()}


def selected_contract_details() -> Dict[str, Dict[str, object]]:
    details: Dict[str, Dict[str, object]] = {}
    for meta in LIVE_UNIVERSE.values():
        symbol = str(meta["ibkr_symbol"])
        sec_type = str(meta.get("sec_type") or "STK")
        row: Dict[str, object] = {
            "secType": sec_type,
            "exchange": str(meta.get("exchange") or "SMART"),
            "currency": str(meta.get("currency") or "EUR"),
        }
        if meta.get("isin") and sec_type != "CFD":
            row["secIdType"] = "ISIN"
            row["secId"] = str(meta["isin"])
        if meta.get("primary_exchange"):
            row["primaryExchange"] = str(meta["primary_exchange"])
        if sec_type == "CFD":
            if symbol == "USD_JPY":
                row["symbol"] = "USD"
                row["currency"] = "JPY"
            elif symbol == "USD_EUR":
                row["symbol"] = "EUR"
                row["currency"] = "USD"
        details[symbol] = row
    return details
