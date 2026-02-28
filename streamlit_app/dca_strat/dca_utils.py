import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st

def sanitize_price_df(df):
    """
    Nettoie un DataFrame Yahoo Finance et garantit une unique colonne 'Close'.
    Si 'Close' n'existe pas, prend 'Adj Close' ou, en dernier recours, la 1ère colonne numérique.
    """
    df = df.copy()

    # Aplatir MultiIndex éventuel
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(x) for x in col if str(x) != ""]) for col in df.columns]

    # Candidats possibles
    candidates = []
    for col in df.columns:
        if col.lower() == "close":
            candidates.append(col)
        elif col.lower() in ["adj close", "adjclose"]:
            candidates.append(col)

    if not candidates:
        # Dernier recours : si au moins une colonne numérique
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(num_cols) == 0:
            raise ValueError("Aucune colonne numérique trouvée dans le DataFrame téléchargé.")
        chosen = num_cols[0]
    else:
        chosen = candidates[0]

    df = df[[chosen]].copy()
    df.rename(columns={chosen: "Close"}, inplace=True)

    # Index propre
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df.dropna(subset=["Close"], inplace=True)
    df = df[~df.index.duplicated(keep="first")]

    return df


def build_biweekly_schedule(start="2015-01-01", end="2025-01-01"):
    """
    Construit le calendrier : 1er lundi de chaque mois puis tous les 14 jours (lundi).
    """
    mondays = pd.date_range(start=start, end=end, freq="W-MON")
    buy_dates = []
    for (y, m) in sorted(set(zip(mondays.year, mondays.month))):
        m_mondays = mondays[(mondays.year == y) & (mondays.month == m)]
        if len(m_mondays) == 0:
            continue
        first = m_mondays[0]
        buy_dates.append(first)
        nxt = first + pd.Timedelta(days=14)
        # On reste dans le même mois et on impose que ce soit un lundi présent dans la liste
        while nxt.month == m and nxt in m_mondays:
            buy_dates.append(nxt)
            nxt += pd.Timedelta(days=14)
    return pd.DatetimeIndex(sorted(set(buy_dates)))


def backtest_dca(df, buy_dates, invest=100.0, window=365, intelligent=False, exec_mode="asof"):
    """
    DCA classique simple (pas de cash management).
    - invest: montant fixe par date d'achat
    """
    df = sanitize_price_df(df).copy()

    # colonnes de travail
    df["Action"] = 0.0
    df["Units"]  = 0.0

    # alignement date d'exécution
    def _align(df, target, mode):
        if mode == "asof":
            if target < df.index[0]:
                return pd.NaT
            return df.index.asof(target)
        else:  # "bfill"
            pos = df.index.searchsorted(target, side="left")
            if pos >= len(df.index):
                return pd.NaT
            return df.index[pos]

    # achats fixes aux dates prévues
    for d in buy_dates:
        exec_date = _align(df, d, exec_mode)
        if pd.isna(exec_date):
            continue
        price = df.at[exec_date, "Close"]
        amount = invest
        units = amount / price if price > 0 else 0.0
        df.at[exec_date, "Action"] += amount
        df.at[exec_date, "Units"]  += units

    # courbes portefeuille
    df["CumUnits"]       = df["Units"].cumsum()
    df["PortfolioValue"] = df["CumUnits"] * df["Close"]
    df["Invested"]       = df["Action"].cumsum()
    df["Equity"] = df["PortfolioValue"]
    return df


def backtest_dca_pro(
    df, buy_dates, base_budget=100.0, window=365,
    exec_mode="asof",
    # Modulation continue par z-score
    k=0.7,                  # sensibilité (plus grand => plus agressif)
    min_factor=0.25,        # borne basse d’allocation
    max_factor=3.0,         # borne haute (utilise le cash accumulé)
    # Cash management
    carry_cash=True,        # banque de cash si on n’investit pas tout
    cash_rate_annual=0.0,   # rendement du cash (0–5% si tu veux)
    # Filtres & triggers
    sma_filter=200,         # si >0 : on réduit l’allocation quand prix < SMA
    sma_penalty=0.5,        # multiplicateur si sous SMA (ex: 0.5)
    dd_trigger=0.0,         # ex: 0.10 => n’investir que si drawdown >=10% OU z<0
    # Prise de profits optionnelle
    z_take_profit=2.0,      # au-dessus => on vend (facteur négatif)
    tp_fraction=0.5,        # % des unités à vendre quand z>=z_take_profit
    # Frais
    fee_bps=0.0             # frais allers/retours en *pourcentage* (ex: 0.001 = 10 bps)
):
    df = sanitize_price_df(df).copy()

    # Statistiques rolling
    df["Median"] = df["Close"].rolling(window, min_periods=window).median()
    df["Std"]    = df["Close"].rolling(window, min_periods=window).std()
    df["Z"]      = (df["Close"] - df["Median"]) / df["Std"]

    # Momentum / SMA
    if sma_filter and sma_filter > 0:
        df["SMA"] = df["Close"].rolling(sma_filter, min_periods=sma_filter).mean()
    else:
        df["SMA"] = np.nan

    # Drawdown (depuis max glissant)
    df["RollMax"] = df["Close"].cummax()
    df["DD"] = 1.0 - df["Close"] / df["RollMax"]

    # Journaux
    df["BuyBudget"] = 0.0         # budget brut du jour (base_budget)
    df["Action"]    = 0.0         # montant réellement investi (+) / retiré (-)
    df["Units"]     = 0.0         # unités achetées(+)/vendues(-)
    df["Cash"]      = 0.0         # solde cash cumulé
    df["CumUnits"]  = 0.0

    cash = 0.0
    cum_units = 0.0

    # helper d’alignement date d’exécution
    def _align(df, target, mode):
        if mode == "asof":
            if target < df.index[0]:
                return pd.NaT
            return df.index.asof(target)
        else:  # "bfill"
            pos = df.index.searchsorted(target, side="left")
            if pos >= len(df.index):
                return pd.NaT
            return df.index[pos]

    # boucle d’achats/ventes
    for d in buy_dates:
        exec_date = _align(df, d, exec_mode)
        if pd.isna(exec_date): 
            continue
        if pd.isna(df.at[exec_date, "Median"]) or pd.isna(df.at[exec_date, "Std"]) or df.at[exec_date,"Std"] == 0:
            continue  # pas assez d'historique

        price = df.at[exec_date, "Close"]
        z     = df.at[exec_date, "Z"]
        dd    = df.at[exec_date, "DD"]

        # budget “théorique” du jour
        budget = base_budget
        df.at[exec_date, "BuyBudget"] = budget

        # facteur basé sur z-score : alloc_factor = clip(1 - k*z)
        alloc_factor = np.clip(1 - k * z, min_factor, max_factor)

        # Filtre SMA : si sous SMA, on réduit l’allocation
        if sma_filter and not pd.isna(df.at[exec_date, "SMA"]) and price < df.at[exec_date, "SMA"]:
            alloc_factor *= sma_penalty

        # Drawdown trigger : si défini, n’investir que si drawdown atteint OU z<0
        if dd_trigger > 0 and not (dd >= dd_trigger or z < 0):
            alloc_factor = 0.0  # pas d’achat aujourd’hui

        # Prise de profits si z très élevé
        sell_units = 0.0
        if z_take_profit is not None and z >= z_take_profit and cum_units > 0:
            sell_units = tp_fraction * cum_units
            proceeds = sell_units * price
            fee = proceeds * fee_bps
            proceeds_net = proceeds - fee
            cash += proceeds_net
            cum_units -= sell_units
            df.at[exec_date, "Action"] -= proceeds_net
            df.at[exec_date, "Units"]  -= sell_units

        # Montant cible (achat) selon alloc_factor
        target_invest = budget * alloc_factor

        # Gestion du cash : si on investit moins que budget, on banque la diff.
        if carry_cash:
            cash += (budget - min(budget, target_invest))  # on épargne le non-investi du jour
            # si on veut investir plus que le budget, on puise dans le cash
            extra_needed = max(0.0, target_invest - budget)
            use_from_cash = min(extra_needed, cash)
            actual_invest = min(budget, target_invest) + use_from_cash if target_invest > 0 else 0.0
            cash -= use_from_cash
        else:
            actual_invest = min(budget, target_invest)

        # Achat (si positif)
        if actual_invest > 0:
            fee = actual_invest * fee_bps
            net = actual_invest - fee
            units = net / price if price > 0 else 0.0
            cum_units += units
            df.at[exec_date, "Action"] += net
            df.at[exec_date, "Units"]  += units

        # Mémos
        cum_units = float(cum_units)
        cash = float(cash)
        df.at[exec_date, "Cash"] = cash
        df.at[exec_date, "CumUnits"] = cum_units

    # Remplir vers l’avant Cash/CumUnits
    df["Cash"] = df["Cash"].replace(0, np.nan).ffill().fillna(0.0)
    df["CumUnits"] = df["CumUnits"].replace(0, np.nan).ffill().fillna(0.0)

    # Equity total = valeur positions + cash
    df["PortfolioValue"] = df["CumUnits"] * df["Close"]
    df["Equity"] = df["PortfolioValue"] + df["Cash"]
    df["Invested"] = df["Action"].cumsum()  # flux nets vers l’actif (hors cash banké)
    return df


def backtest_dca_costbasis(
    df, buy_dates,
    base_budget=100.0,
    window=365,                 # pour la médiane/σ du Z-score
    exec_mode="asof",
    # Modulation cost basis
    alpha=2,                  # sensibilité au PnL relatif vs cost basis (plus haut => plus agressif)
    # Modulation Z-score
    k=0.8,                      # sensibilité Z-score (0.5 à 0.8 raisonnable)
    # Bornes d’allocation
    min_factor=0.5,
    max_factor=2.5,
    # Cash management
    carry_cash=True,
    fee_bps=0.0005,             # 5 bps de frais
    # Option de tendance (faible)
    sma_filter=0,               # 0 = désactivé, sinon ex. 200
    sma_penalty=0.9             # si prix < SMA -> facteur *= 0.9 (léger)
):
    """
    DCA qui module l'allocation selon l'écart prix vs cost basis (PnL relatif),
    combiné à un facteur Z-score (médiane/σ).
    """
    df = sanitize_price_df(df).copy()

    # Stats pour Z-score
    df["Median"] = df["Close"].rolling(window, min_periods=window).median()
    df["Std"]    = df["Close"].rolling(window, min_periods=window).std()
    df["Z"]      = (df["Close"] - df["Median"]) / df["Std"]

    # SMA optionnelle
    if sma_filter and sma_filter > 0:
        df["SMA"] = df["Close"].rolling(sma_filter, min_periods=sma_filter).mean()
    else:
        df["SMA"] = np.nan

    # Journaux
    df["BuyBudget"] = 0.0
    df["Action"]    = 0.0
    df["Units"]     = 0.0
    df["Cash"]      = 0.0
    df["CumUnits"]  = 0.0
    df["AvgCost"]   = np.nan    # cost basis (moyenne pondérée par unités)

    cash = 0.0
    cum_units = 0.0
    avg_cost = np.nan

    def _align(df, target, mode):
        if mode == "asof":
            if target < df.index[0]:
                return pd.NaT
            return df.index.asof(target)
        pos = df.index.searchsorted(target, side="left")
        return df.index[pos] if pos < len(df.index) else pd.NaT

    for d in buy_dates:
        exec_date = _align(df, d, exec_mode)
        if pd.isna(exec_date):
            continue
        price = df.at[exec_date, "Close"]

        # Budget du jour
        budget = base_budget
        df.at[exec_date, "BuyBudget"] = budget

        # --- Facteur basé sur cost basis ---
        # pnl_rel = (prix / avg_cost) - 1  (si pas encore d’unités, on considère 0)
        if cum_units > 0 and not np.isnan(avg_cost) and avg_cost > 0:
            pnl_rel = price / avg_cost - 1.0
        else:
            pnl_rel = 0.0

        # Idée: plus pnl_rel est négatif (sous l’eau), plus on alloue.
        alloc_cb = 1.0 - alpha * pnl_rel

        # --- Facteur Z-score ---
        z = df.at[exec_date, "Z"]
        if np.isnan(z):
            alloc_z = 1.0
        else:
            alloc_z = 1.0 - k * z

        # Combine et applique bornes
        alloc_factor = np.clip(alloc_cb * alloc_z, min_factor, max_factor)

        # Option de tendance légère
        sma = df.at[exec_date, "SMA"]
        if not np.isnan(sma) and price < sma:
            alloc_factor *= sma_penalty

        # Cible d'investissement
        target_invest = budget * alloc_factor

        # Gestion du cash
        if carry_cash:
            # On banque ce qu'on n'investit pas
            to_cash = budget - min(budget, target_invest)
            if to_cash > 0:
                cash += to_cash
            # Si on veut investir plus que budget, on puise dans le cash
            extra_needed = max(0.0, target_invest - budget)
            use_from_cash = min(extra_needed, cash)
            actual_invest = min(budget, target_invest) + use_from_cash if target_invest > 0 else 0.0
            cash -= use_from_cash
        else:
            actual_invest = min(budget, max(0.0, target_invest))

        # Achat (pas de ventes dans cette variante)
        if actual_invest > 0.0:
            fee = actual_invest * fee_bps
            net = actual_invest - fee
            units = net / price if price > 0 else 0.0

            # Avg cost avant MAJ (on utilise l'ancien pour le signal, puis on met à jour)
            prev_units = cum_units
            prev_cost  = avg_cost

            cum_units += units
            df.at[exec_date, "Action"] += net
            df.at[exec_date, "Units"]  += units

            # Mise à jour du cost basis (moyenne pondérée)
            if prev_units <= 0 or np.isnan(prev_cost):
                avg_cost = price
            else:
                avg_cost = (prev_cost * prev_units + price * units) / (prev_units + units)

        # Mémos séries
        df.at[exec_date, "Cash"]     = cash
        df.at[exec_date, "CumUnits"] = cum_units
        df.at[exec_date, "AvgCost"]  = avg_cost

    # Remplissages
    df["Cash"]     = df["Cash"].replace(0, np.nan).ffill().fillna(0.0)
    df["CumUnits"] = df["CumUnits"].replace(0, np.nan).ffill().fillna(0.0)
    df["AvgCost"]  = df["AvgCost"].ffill()

    # Equity = positions + cash
    df["PortfolioValue"] = df["CumUnits"] * df["Close"]
    df["Equity"]         = df["PortfolioValue"] + df["Cash"]
    df["Invested"]       = df["Action"].cumsum()
    return df


def sharpe_equity(df, rf_annual=0.0):
    """Sharpe sur l'equity (positions + cash)."""
    if "Equity" not in df:
        return np.nan
    s = df["Equity"].ffill().pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 5 or s.std() == 0:
        return np.nan
    # fréquence “raisonnable”
    dt = df.index.to_series().diff().dt.days.dropna()
    periods = 365 if (len(dt) and dt.median() <= 1.1) else 252
    excess = s - rf_annual / periods
    return np.sqrt(periods) * excess.mean() / excess.std()

@st.cache_data(show_spinner=False)
def fetch_and_run_dca(ticker, start_date, end_date, invest_per_trade, rolling_window):
    """
    Fetches the data and runs all three DCA strategies to return the results.
    We use cache to prevent re-downloading Yahoo data on every render.
    """
    
    start_dt = pd.to_datetime(start_date)
    fetch_start = (start_dt - pd.Timedelta(days=int(rolling_window * 1.5) + 50)).strftime('%Y-%m-%d')
    raw = yf.download(ticker, start=fetch_start, end=end_date, progress=False)
    
    if raw.empty:
        return None, None, None
        
    base = sanitize_price_df(raw)
    buy_dates = build_biweekly_schedule(start=start_date, end=end_date)
    risk_free_rate = 0.0
    
    # Run Classic DCA
    dca_classic = backtest_dca(
        base, buy_dates,
        invest=invest_per_trade,
        window=rolling_window,
        intelligent=False,
        exec_mode="asof"
    )
    sh_classic = sharpe_equity(dca_classic, risk_free_rate)
    
    # Run Smart DCA
    dca_smart = backtest_dca_pro(
        base, buy_dates,
        base_budget=invest_per_trade, window=rolling_window,
        exec_mode="asof",
        k=0.7, min_factor=0.25, max_factor=3.0,
        carry_cash=True, cash_rate_annual=0.0,
        sma_filter=0, sma_penalty=0.8,
        dd_trigger=0.0,
        z_take_profit=2.0, tp_fraction=0.0,
        fee_bps=0.0005
    )
    sh_smart = sharpe_equity(dca_smart, risk_free_rate)
    
    # Run CostBasis DCA
    dca_cost = backtest_dca_costbasis(
        base, buy_dates,
        base_budget=invest_per_trade, window=rolling_window,
        exec_mode="asof",
        alpha=1.5, k=0.7,
        min_factor=0.5, max_factor=2.5,
        carry_cash=True,
        sma_filter=0, sma_penalty=0.9
    )
    sh_cost = sharpe_equity(dca_cost, risk_free_rate)
    
    return (dca_classic, sh_classic), (dca_smart, sh_smart), (dca_cost, sh_cost)
