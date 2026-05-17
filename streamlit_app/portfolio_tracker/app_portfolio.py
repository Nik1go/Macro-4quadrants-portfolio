import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import os
import io
from datetime import datetime

DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "portfolio_transactions.csv")

# ─── Types de transactions Trade Republic ignorés ────────────────────────────
TR_IGNORED_TYPES = {
    "CUSTOMER_INBOUND", "CARD_TRANSACTION", "INTEREST_PAYMENT",
    "TRANSFER_INSTANT_INBOUND", "BENEFITS_SAVEBACK", "STOCKPERK",
    "FREE_DELIVERY", "FREE_RECEIPT", "PEA_MARKETING", "DIVIDEND",
    "CUSTOMER_OUTBOUND"
}

# Correspondances ISIN connues → Ticker Yahoo Finance
ISIN_TO_TICKER = {
    "US0079031078": "AMD",
    "US0231351067": "AMZN",
    "DE000RENK730": "R3NK.DE",    # RENK Group
    "FR0000125486": "DG.PA",      # Vinci
    "LU0292109690": "DBXN.DE",    # Nifty 50 Swap
    "LU2196470426": "XNKY.DE",    # Nikkei 225
    "IE00B4L5Y983": "IWDA.AS",    # Core MSCI World
    "IE000NDWFGA5": "URNU.DE",    # Uranium
    "LU0592217524": "XMKA.DE",    # MSCI Africa
    "IE000I8KRLL9": "SEMI.AS",    # MSCI Semiconductors
    "IE00BJ5JPG56": "ICGA.DE",    # MSCI China
    "IE00B1XNHC34": "IQQH.DE",    # Global Clean Energy
    "CA00135V1094": "AVI.V",      # AI Artificial Intelligence Ventures
    "FR0000121972": "SU.PA",      # Schneider Electric
    "FR0011726835": "GTT.PA",     # Gaztransport Technigaz
    "SE0012673267": "E3G1.F",     # Evolution Gaming
    "CNE100000296": "BY6.F",      # BYD
    "BTC":          "BTC-EUR",    # Bitcoin
}

# ─── Dark theme CSS pour la popup dialog ─────────────────────────────────────
DIALOG_CSS = """
<style>
div[role="dialog"],
[data-testid="stModal"] > div {
    background-color: #0a0e27 !important;
    border: 1px solid #3d4263 !important;
    border-radius: 12px !important;
}
div[role="dialog"] p, div[role="dialog"] label,
div[role="dialog"] span, div[role="dialog"] div,
div[role="dialog"] h1, div[role="dialog"] h2, div[role="dialog"] h3 {
    color: #e8e8e8 !important;
}
div[role="dialog"] input, div[role="dialog"] textarea {
    background-color: #1e2139 !important;
    color: #e8e8e8 !important;
    border: 1px solid #3d4263 !important;
    border-radius: 6px !important;
}
div[role="dialog"] [data-baseweb="select"] > div {
    background-color: #1e2139 !important;
    border: 1px solid #3d4263 !important;
    color: #e8e8e8 !important;
}
div[role="dialog"] [data-testid="stFileUploader"] {
    background-color: #1e2139 !important;
    border: 1px dashed #3d4263 !important;
    border-radius: 8px !important;
}
div[role="dialog"] .stButton > button {
    background: linear-gradient(90deg, #1e2139, #2a2d4a) !important;
    color: #00d4ff !important;
    border: 1px solid #3d4263 !important;
    border-radius: 8px !important;
}
div[role="dialog"] .stButton > button:hover {
    background: linear-gradient(90deg, #00d4ff, #0099cc) !important;
    color: white !important;
}
div[role="dialog"] [data-testid="stMarkdownContainer"] h3 { color: #00d4ff !important; }
div[role="dialog"] button[aria-label="Close"] { color: #e8e8e8 !important; }
</style>
"""

# ─── Data helpers ─────────────────────────────────────────────────────────────

def ensure_data_file():
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    if not os.path.exists(DATA_FILE):
        df = pd.DataFrame(columns=["Date", "Asset_Name", "Ticker", "Type", "Quantity", "Price", "Total_Amount"])
        df.to_csv(DATA_FILE, index=False)

def load_transactions():
    ensure_data_file()
    return pd.read_csv(DATA_FILE)

def save_transactions(df):
    ensure_data_file()
    df.to_csv(DATA_FILE, index=False)

def parse_trade_republic_csv(file_bytes) -> tuple[pd.DataFrame, list[str]]:
    """
    Parse le CSV d'exportation Trade Republic.
    Format specifique : chaque ligne est enveloppee dans des guillemets externes
    et les champs internes utilisent "" comme echappement.
    Ex: "2024-01-01,...,""BUY"",""STOCK"",..."
    Retourne (df_transactions, warnings).
    """
    warnings_list = []

    try:
        text = file_bytes.decode("utf-8")
    except Exception as e:
        return pd.DataFrame(), [f"Erreur de decodage : {e}"]

    # Pretraitement ligne par ligne : chaque ligne est enveloppee dans des
    # guillemets externes. On retire cette enveloppe et on deseschappe les "".
    cleaned_lines = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith('"') and line.endswith('"'):
            inner = line[1:-1]               # retire le " de debut et de fin
            inner = inner.replace('""', '"') # deseschappe les guillemets internes
            cleaned_lines.append(inner)
        else:
            cleaned_lines.append(line)

    if len(cleaned_lines) < 2:
        return pd.DataFrame(), ["Le fichier semble vide ou mal formate."]

    cleaned_text = "\n".join(cleaned_lines)

    try:
        df_raw = pd.read_csv(io.StringIO(cleaned_text))
    except Exception as e:
        return pd.DataFrame(), [f"Erreur de lecture apres nettoyage : {e}"]

    # Nettoyage des noms de colonnes
    df_raw.columns = [c.strip().strip('"') for c in df_raw.columns]

    required_cols = {"type", "name", "symbol", "shares", "price", "amount"}
    missing = required_cols - set(df_raw.columns)
    if missing:
        return pd.DataFrame(), [
            f"Colonnes manquantes : {missing}. "
            f"Colonnes detectees : {list(df_raw.columns)}"
        ]

    # Filtrer uniquement les achats et ventes
    df_trading = df_raw[df_raw["type"].isin(["BUY", "SELL"])].copy()

    if df_trading.empty:
        return pd.DataFrame(), ["Aucune transaction BUY/SELL trouvee dans le fichier."]

    # Conversion numerique
    for col in ["shares", "price", "amount"]:
        df_trading[col] = pd.to_numeric(
            df_trading[col].astype(str).str.replace(",", "."), errors="coerce"
        ).abs()

    # Correspondance ISIN → ticker Yahoo
    unknown_isins = set()
    tickers = []
    for _, row in df_trading.iterrows():
        isin = str(row["symbol"]).strip()
        if isin in ISIN_TO_TICKER:
            tickers.append(ISIN_TO_TICKER[isin])
        else:
            unknown_isins.add(isin)
            tickers.append(isin)

    if unknown_isins:
        warnings_list.append(
            f"ISINs sans correspondance Yahoo connue (prix non recuperes) : {unknown_isins}"
        )

    df_trading = df_trading.assign(Ticker=tickers)

    # Construction au format interne
    records = []
    for _, row in df_trading.iterrows():
        records.append({
            "Date": str(row.get("date", "")).strip(),
            "Asset_Name": str(row.get("name", "")).strip(),
            "Ticker": row["Ticker"],
            "Type": "Achat" if row["type"] == "BUY" else "Vente",
            "Quantity": row["shares"],
            "Price": row["price"],
            "Total_Amount": row["amount"]
        })

    return pd.DataFrame(records), warnings_list

def get_current_holdings(df_transactions):
    if df_transactions.empty:
        return pd.DataFrame()

    holdings = []
    for ticker, group in df_transactions.groupby("Ticker"):
        total_quantity = 0.0
        total_invested = 0.0
        asset_name = group.iloc[0]["Asset_Name"]

        for _, row in group.iterrows():
            if str(row["Type"]).lower() in ["buy", "achat"]:
                total_quantity += float(row["Quantity"])
                total_invested += float(row["Total_Amount"])
            elif str(row["Type"]).lower() in ["sell", "vente"]:
                avg_price = total_invested / total_quantity if total_quantity > 0 else 0
                total_quantity -= float(row["Quantity"])
                total_invested -= float(row["Quantity"]) * avg_price

        if total_quantity > 0.0001:
            avg_price = total_invested / total_quantity
            holdings.append({
                "Actif": asset_name,
                "Ticker": ticker,
                "Quantite": total_quantity,
                "PRU (EUR)": avg_price,
                "Investi (EUR)": total_invested
            })

    return pd.DataFrame(holdings)

# ─── Admin dialog ─────────────────────────────────────────────────────────────

@st.dialog("Connexion Administrateur")
def admin_login_and_upload():
    st.markdown(DIALOG_CSS, unsafe_allow_html=True)
    st.write("Entrez le mot de passe pour gerer le portefeuille.")
    pwd = st.text_input("Mot de passe", type="password", key="admin_pwd_input")

    correct_password = "changeme123"
    try:
        correct_password = st.secrets["admin_password"]
    except Exception:
        pass

    if pwd and pwd == correct_password:
        st.success("Connecte avec succes.")

        st.markdown("### Import CSV Trade Republic")
        uploaded_file = st.file_uploader("Deposez votre fichier d'exportation ici", type=["csv"])

        if uploaded_file is not None:
            df_new, warns = parse_trade_republic_csv(uploaded_file.read())

            if warns:
                for w in warns:
                    st.warning(w)

            if not df_new.empty:
                st.write(f"**{len(df_new)} transactions BUY/SELL detectees.**")
                st.dataframe(df_new.head(10), use_container_width=True)

                if st.button("Importer ces transactions", key="import_tr"):
                    df_existing = load_transactions()
                    df_all = pd.concat([df_existing, df_new], ignore_index=True)
                    save_transactions(df_all)
                    st.success(f"{len(df_new)} transactions importees avec succes.")
                    st.rerun()

        st.markdown("---")
        st.markdown("### Ajouter une transaction manuelle")
        st.caption("Pour les metaux precieux : GC=F (Or), SI=F (Argent). Pour crypto : BTC-EUR, ETH-EUR.")
        with st.form("manual_tx"):
            col1, col2 = st.columns(2)
            with col1:
                asset_name = st.text_input("Nom de l'actif (ex: Or, Apple)")
                ticker = st.text_input("Ticker Yahoo Finance (ex: GC=F, AAPL)")
                tx_type = st.selectbox("Type", ["Achat", "Vente"])
            with col2:
                quantity = st.number_input("Quantite", min_value=0.00001, format="%f")
                price = st.number_input("Prix unitaire (EUR)", min_value=0.0, format="%f")
                date = st.date_input("Date de la transaction")

            submitted = st.form_submit_button("Ajouter la transaction")
            if submitted:
                if not ticker or not asset_name:
                    st.error("Veuillez remplir tous les champs.")
                else:
                    new_tx = pd.DataFrame([{
                        "Date": date.strftime("%Y-%m-%d"),
                        "Asset_Name": asset_name,
                        "Ticker": ticker,
                        "Type": tx_type,
                        "Quantity": quantity,
                        "Price": price,
                        "Total_Amount": quantity * price
                    }])
                    df_all = load_transactions()
                    df_all = pd.concat([df_all, new_tx], ignore_index=True)
                    save_transactions(df_all)
                    st.success(f"Transaction ajoutee : {quantity:.6f} x {ticker} @ {price:.2f} EUR")
                    st.rerun()

        st.markdown("---")
        st.markdown("### Reinitialiser le portefeuille")
        if st.button("Supprimer toutes les transactions", key="reset_portfolio"):
            ensure_data_file()
            df_empty = pd.DataFrame(columns=["Date", "Asset_Name", "Ticker", "Type", "Quantity", "Price", "Total_Amount"])
            save_transactions(df_empty)
            st.success("Portefeuille reinitialise.")
            st.rerun()

    elif pwd and pwd != correct_password:
        st.error("Mot de passe incorrect.")

# ─── Main render ──────────────────────────────────────────────────────────────

def render():
    st.markdown("<h1 style='margin-bottom: 0;'>Mon Portefeuille</h1>", unsafe_allow_html=True)
    with st.expander("À propos de mon portefeuille", expanded=False):
        st.markdown("<p style='color:#b0b0b0;'>Suivi de mon portefeuille : Compte-Titres, PEA, Métaux Précieux (physique), Crypto et biens divers.</p>", unsafe_allow_html=True)
        st.markdown("<p style='color:#b0b0b0;'>Début du portefeuille en mars 2023.</p>", unsafe_allow_html=True)
        st.markdown("<p style='color:#b0b0b0;'>La plupart de mes nouvelles entrées se font en DCA (hebdomadaire) sur les différents ETFs (principalement pays émergents).</p>", unsafe_allow_html=True)
        st.markdown("<p style='color:#b0b0b0;'>Ma stratégie actuelle est d'équilibrer mon portefeuille par rapport aux entrées agressives que j'ai pu faire sur l'or, l'argent et Thales.</p>", unsafe_allow_html=True)
        st.markdown("<p style='color:#b0b0b0;'>Je souhaiterais aussi, dans un avenir proche,restructurer mon PEA en supprimant certaines lignes pour pouvoir renforcer mes positions stock picking (Constellation Software et Odontoprev étant mes principales convictions), ainsi que mes positions en biens divers.</p>", unsafe_allow_html=True)
        st.markdown("<p style='color:#b0b0b0;'>A noter que ce portefeuille ne prend pas en compte mes liquidités, mon placement en PER (abondement Interessement et Participation), ainsi que mes placement en biens divers.</p>", unsafe_allow_html=True)

    col_title, col_btn = st.columns([5, 1])
    with col_btn:
        if st.button("Admin", use_container_width=True):
            admin_login_and_upload()

    df_tx = load_transactions()
    holdings = get_current_holdings(df_tx)

    if holdings.empty:
        st.info("Aucun actif dans le portefeuille. Cliquez sur 'Admin' pour importer votre CSV Trade Republic ou ajouter manuellement des actifs.")
        return

    # Fetch live prices and their currencies
    live_prices = {}
    currencies = {}
    tickers_list = holdings["Ticker"].tolist()
    with st.spinner("Recuperation des cours en direct et conversion des devises..."):
        for ticker in tickers_list:
            try:
                t_obj = yf.Ticker(ticker)
                hist = t_obj.history(period="2d")
                if not hist.empty:
                    price = float(hist['Close'].iloc[-1])
                    curr = t_obj.fast_info.get('currency', 'EUR')
                    # Gestion des pence britanniques
                    if curr == 'GBp':
                        price = price / 100.0
                        curr = 'GBP'
                    live_prices[ticker] = price
                    currencies[ticker] = curr.upper()
                else:
                    live_prices[ticker] = None
                    currencies[ticker] = 'EUR'
            except Exception:
                live_prices[ticker] = None
                currencies[ticker] = 'EUR'

        # Fetch FX rates to EUR
        fx_rates = {'EUR': 1.0}
        unique_currencies = set(currencies.values()) - {'EUR'}
        for curr in unique_currencies:
            try:
                fx_ticker = f"{curr}EUR=X"
                fx_hist = yf.Ticker(fx_ticker).history(period="2d")
                if not fx_hist.empty:
                    fx_rates[curr] = float(fx_hist['Close'].iloc[-1])
                else:
                    fx_rates[curr] = 1.0
            except Exception:
                fx_rates[curr] = 1.0

    # Convert prices to EUR
    converted_prices = {}
    for ticker in tickers_list:
        if live_prices.get(ticker) is not None:
            curr = currencies.get(ticker, 'EUR')
            rate = fx_rates.get(curr, 1.0)
            converted_prices[ticker] = live_prices[ticker] * rate
        else:
            converted_prices[ticker] = None

    holdings["Cours Actuel (EUR)"] = holdings["Ticker"].map(converted_prices)
    holdings["Valeur Actuelle (EUR)"] = holdings["Quantite"] * holdings["Cours Actuel (EUR)"]
    holdings["PnL (EUR)"] = holdings["Valeur Actuelle (EUR)"] - holdings["Investi (EUR)"]
    holdings["PnL (%)"] = (holdings["PnL (EUR)"] / holdings["Investi (EUR)"]) * 100

    total_value = holdings["Valeur Actuelle (EUR)"].sum()
    total_invested = holdings["Investi (EUR)"].sum()
    total_pnl = total_value - total_invested
    total_pnl_pct = (total_pnl / total_invested * 100) if total_invested > 0 else 0

    # CAGR Calculation
    start_date = datetime(2023, 3, 1)
    today = datetime.today()
    years_diff = (today - start_date).days / 365.25
    cagr_pct = 0.0
    if total_invested > 0 and years_diff > 0:
        ratio = total_value / total_invested
        if ratio > 0:
            cagr_pct = ((ratio ** (1 / years_diff)) - 1) * 100

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Valeur Totale", f"{total_value:,.2f} EUR")
    m2.metric("Total Investi", f"{total_invested:,.2f} EUR")
    m3.metric("Plus-Value Globale", f"{total_pnl:,.2f} EUR", f"{total_pnl_pct:.2f} %")
    m4.metric("CAGR (depuis Mars 2023)", f"{cagr_pct:.2f} %")

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts
    c1, c2 = st.columns(2)
    with c1:
        # Regroupement des petites positions pour la lisibilité
        pie_data = holdings.copy().sort_values("Valeur Actuelle (EUR)", ascending=False)
        if len(pie_data) > 8:
            top_data = pie_data.iloc[:8].copy()
            autres_val = pie_data.iloc[8:]["Valeur Actuelle (EUR)"].sum()
            autres_row = pd.DataFrame([{"Actif": "Autres", "Valeur Actuelle (EUR)": autres_val}])
            pie_data = pd.concat([top_data, autres_row], ignore_index=True)

        fig_pie = px.pie(
            pie_data,
            values="Valeur Actuelle (EUR)",
            names="Actif",
            title="Repartition du Portefeuille",
            hole=0.45,
            template="plotly_dark",
            color_discrete_sequence=px.colors.sequential.Plasma_r
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e8e8e8"), title_font=dict(color="#00d4ff")
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        fig_bar = px.bar(
            holdings.sort_values("PnL (%)", ascending=True),
            x="PnL (%)", y="Actif", orientation="h",
            title="Performance par Actif (%)",
            template="plotly_dark",
            color="PnL (%)",
            color_continuous_scale=["#ff4b4b", "#888888", "#00d4ff"],
            color_continuous_midpoint=0
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e8e8e8"), title_font=dict(color="#00d4ff"),
            showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Table
    with st.expander("Détail des Positions", expanded=False):
        display_cols = ["Actif", "Ticker", "Quantite", "PRU (EUR)", "Cours Actuel (EUR)", "Valeur Actuelle (EUR)", "PnL (EUR)", "PnL (%)"]
        display_df = holdings[display_cols].copy().round(4)
    
        st.dataframe(
            display_df.style.applymap(
                lambda v: "color: #00d4ff" if isinstance(v, (int, float)) and v > 0 else (
                    "color: #ff4b4b" if isinstance(v, (int, float)) and v < 0 else ""
                ),
                subset=["PnL (EUR)", "PnL (%)"]
            ),
            use_container_width=True
        )

    # Tickers sans prix
    missing_price = holdings[holdings["Cours Actuel (EUR)"].isna()]["Ticker"].tolist()
    if missing_price:
        st.warning(f"Prix non recuperes pour : {missing_price}. Verifiez les tickers Yahoo Finance dans la table ISIN_TO_TICKER.")
