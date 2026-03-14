import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import glob

def parse_fec(source):
    st.write(f"Reading file: {source.name if hasattr(source, 'name') else source}")
    try:
        df = pd.read_csv(source, sep='\t', converters={'CompteNum': str})
        if 'EcritureDate' not in df.columns:
            source.seek(0) if hasattr(source, 'seek') else None
            df = pd.read_csv(source, sep='|', converters={'CompteNum': str}, encoding='latin-1')
    except Exception as e:
        source.seek(0) if hasattr(source, 'seek') else None
        df = pd.read_csv(source, sep=';', converters={'CompteNum': str}, encoding='latin-1')
        
    df['EcritureDate'] = pd.to_datetime(df['EcritureDate'], format='%Y%m%d', errors='coerce')
    df['Year'] = df['EcritureDate'].dt.year
    df['Debit'] = pd.to_numeric(df['Debit'].replace({',': '.'}, regex=True), errors='coerce').fillna(0)
    df['Credit'] = pd.to_numeric(df['Credit'].replace({',': '.'}, regex=True), errors='coerce').fillna(0)
    
    return df

def compute_kpis(df):
    year = df['Year'].max()
    if pd.isna(year):
        year = "Inconnu"
    else:
        year = str(int(year))
        
    def get_bal(prefix_list, type_='credit'):
        mask = df['CompteNum'].astype(str).str.startswith(tuple([str(p) for p in prefix_list]))
        if type_ == 'credit':
            return df.loc[mask, 'Credit'].sum() - df.loc[mask, 'Debit'].sum()
        else:
            return df.loc[mask, 'Debit'].sum() - df.loc[mask, 'Credit'].sum()

    ca = get_bal(['70'], 'credit')
    produits_expl = get_bal(['70', '71', '72', '73', '74', '75'], 'credit')
    charges_expl = get_bal(['60', '61', '62', '63', '64', '65'], 'debit')
    
    ebitda = produits_expl - charges_expl
    
    cogs = get_bal(['60'], 'debit')
    marge_brute = ca - cogs
    
    total_produits = get_bal(['7'], 'credit')
    total_charges = get_bal(['6'], 'debit')
    resultat_net = total_produits - total_charges
    
    capitaux_propres = get_bal(['10', '11', '12', '13', '14'], 'credit')
    # Les à-nouveaux de l'année précédente et le résultat de l'exercice s'accumulent.
    # Pour un test simple sur un an de FEC avec A-Nouveaux, les capitaux propres globaux 
    # vus au bilan = (Credit 10x - Debit 10x) + Résultat Net.
    # Normalement le bilan comptable se cloture, etc.
    capitaux_finaux = capitaux_propres + resultat_net
    
    dettes_fin = get_bal(['16', '17'], 'credit')
    tresorerie = get_bal(['51', '53'], 'debit')
    
    dette_nette = dettes_fin - tresorerie
    
    # KPIs customisés extraits des notes hors bilan
    employes = df.loc[df['CompteNum'].astype(str).str.startswith('801001'), 'Debit'].max()
    employes = employes if not pd.isna(employes) else 0
        
    clients = df.loc[df['CompteNum'].astype(str).str.startswith('801003'), 'Debit'].max()
    clients = clients if not pd.isna(clients) else 0
        
    return {
        "Année": year,
        "Chiffre d'Affaires": float(ca),
        "Marge Brute": float(marge_brute),
        "EBITDA": float(ebitda),
        "Cash Flow/Résultat Net": float(resultat_net),
        "Capitaux Propres": float(capitaux_finaux),
        "Dettes Financières": float(dettes_fin),
        "Trésorerie": float(tresorerie),
        "Dette Nette": float(dette_nette),
        "Dépenses": float(total_charges),
        "Employés": int(employes),
        "Clients": int(clients),
        "ROE (%)": float((resultat_net / capitaux_finaux * 100) if capitaux_finaux != 0 else 0)
    }

def process_files(files):
    all_kpis = []
    for f in files:
        df = parse_fec(f)
        kpi = compute_kpis(df)
        all_kpis.append(kpi)
    
    # Sort chronologically
    all_kpis = sorted(all_kpis, key=lambda x: str(x["Année"]))
    df_kpi = pd.DataFrame(all_kpis)
    return df_kpi

def render():
    st.title("Analyse Financière & Tokenisation Project")
    st.markdown("---")
    
    st.markdown("""
    Cette plateforme permet d'importer les Fichiers d'Écritures Comptables (FEC) d'une entreprise sur plusieurs années, 
    d'en extraire les KPIs financiers, de calculer une valorisation multi-critères et de simuler une tokenisation du capital.
    """)

    st.header(" 1. Import des Fichiers FEC")
    st.info("Importez plusieurs fichiers FEC (ex: N, N-1, N-2) pour l'analyse des tendances.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "Sélectionnez les fichiers FEC (TXT ou CSV)", 
            type=["csv", "txt"], 
            accept_multiple_files=True
        )

    with col2:
        st.write("Ou utiliser des données de démonstration :")
        if st.button("Charger les données de test (SaaS)"):
            st.session_state["demo_fec"] = "SaaS"
        if st.button("Charger les données de test (Retail)"):
            st.session_state["demo_fec"] = "Retail"

    # Récupérer les fichiers à traiter
    files_to_process = uploaded_files
    if len(files_to_process) == 0 and "demo_fec" in st.session_state:
        # Trouver les fichiers dans le disque
        demo_type = st.session_state["demo_fec"]
        search_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "tests", "donnees_fec", f"FEC_{demo_type}_*.txt")
        demo_files = glob.glob(search_path)
        if demo_files:
            files_to_process = demo_files
            st.success(f"Utilisation des données de test profil {demo_type} ({len(demo_files)} fichiers trouvés)")
        else:
            st.warning("Fichiers de tests introuvables. Lancez `python generate_fake_fec.py --all`")

    if not files_to_process:
        return

    # Processing Data
    with st.spinner('Parsing et agrégation des fichiers FEC en cours...'):
        df_kpi = process_files(files_to_process)
        
    st.session_state["df_kpi"] = df_kpi

    # ----- TABLEAU DE BORD FINANCIER -----
    st.markdown("---")
    st.header(" 2. Tableau de Bord Financier Historique")
    
    if len(df_kpi) > 1:
        cagr_ca = (df_kpi.iloc[-1]["Chiffre d'Affaires"] / df_kpi.iloc[0]["Chiffre d'Affaires"]) ** (1/(len(df_kpi)-1)) - 1
        cagr_ebitda = (df_kpi.iloc[-1]["EBITDA"] / df_kpi.iloc[0]["EBITDA"]) ** (1/(len(df_kpi)-1)) - 1
    else:
        cagr_ca = 0
        cagr_ebitda = 0

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    last_year_data = df_kpi.iloc[-1]
    prev_year_data = df_kpi.iloc[-2] if len(df_kpi) > 1 else last_year_data
    
    def metric_delta(current, prev, is_pct=False):
        if prev == 0: return "0%"
        delta = (current - prev) / prev * 100
        return f"{delta:.1f}%"
        
    last_year_ca = last_year_data["Chiffre d'Affaires"]
    prev_year_ca = prev_year_data["Chiffre d'Affaires"]
    
    col_m1.metric("CA (Dernière Année)", f"€ {last_year_ca:,.0f}", metric_delta(last_year_ca, prev_year_ca))
    col_m2.metric("EBITDA", f"€ {last_year_data['EBITDA']:,.0f}", metric_delta(last_year_data["EBITDA"], prev_year_data["EBITDA"]))
    
    ebitda_margin = (last_year_data['EBITDA'] / last_year_ca * 100) if last_year_ca > 0 else 0
    col_m3.metric("Marge d'EBITDA", f"{ebitda_margin:.1f}%")
    
    col_m4.metric("Dette Nette", f"€ {last_year_data['Dette Nette']:,.0f}", metric_delta(last_year_data['Dette Nette'], prev_year_data['Dette Nette']), delta_color="inverse")

    st.write(f"**Croissance Moyenne Annuelle (CAGR) :** CA : `{cagr_ca*100:.1f}%` | EBITDA : `{cagr_ebitda*100:.1f}%`")

    # ----- GRAPHIQUES -----
    t1, t2 = st.tabs(["Indicateurs de Rentabilité", "Bilan & Opérationnalité"])

    colors = ['#00d4ff', '#1e2139', '#555883']

    with t1:
        c1, c2 = st.columns(2)
        with c1:
            # Evolution du CA et de l'EBITDA
            fig_ca = go.Figure()
            fig_ca.add_trace(go.Bar(x=df_kpi["Année"], y=df_kpi["Chiffre d'Affaires"], name='Chiffre d\'Affaires', marker_color='#1e2139'))
            fig_ca.add_trace(go.Scatter(x=df_kpi["Année"], y=df_kpi["EBITDA"], name='EBITDA', marker_color='#00d4ff', mode='lines+markers', line=dict(width=3)))
            fig_ca.update_layout(title="Évolution du CA et de l'EBITDA (€)", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig_ca, use_container_width=True)
            
            # Evolution de la Marge
            df_kpi["Marge Brute %"] = df_kpi["Marge Brute"] / df_kpi["Chiffre d'Affaires"] * 100
            df_kpi["EBITDA Margin %"] = df_kpi["EBITDA"] / df_kpi["Chiffre d'Affaires"] * 100
            
            fig_marge = go.Figure()
            fig_marge.add_trace(go.Scatter(x=df_kpi["Année"], y=df_kpi["Marge Brute %"], name='Marge Brute (%)', marker_color='#a0a0a0', mode='lines+markers'))
            fig_marge.add_trace(go.Scatter(x=df_kpi["Année"], y=df_kpi["EBITDA Margin %"], name='Marge d\'EBITDA (%)', marker_color='#00d4ff', mode='lines+markers'))
            fig_marge.update_layout(title="Évolution des Profils de Marges (%)", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig_marge, use_container_width=True)

        with c2:
            # Cash flow & Dépenses
            fig_cash = px.bar(df_kpi, x="Année", y=["Cash Flow/Résultat Net", "Dépenses"], title="Répartition Résultat net & Dépenses", barmode='group')
            fig_cash.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig_cash, use_container_width=True)

            # ROE (Return on Equity)
            fig_roe = px.line(df_kpi, x="Année", y="ROE (%)", title="Retour sur Capitaux Propres (ROE %)", markers=True)
            fig_roe.update_traces(line_color='#00d4ff')
            fig_roe.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig_roe, use_container_width=True)

    with t2:
        c1, c2 = st.columns(2)
        with c1:
            # Aperçu des dettes
            fig_dette = go.Figure()
            fig_dette.add_trace(go.Bar(x=df_kpi["Année"], y=df_kpi["Dettes Financières"], name='Dettes Brutes', marker_color='#bb2020'))
            fig_dette.add_trace(go.Bar(x=df_kpi["Année"], y=df_kpi["Trésorerie"], name='Trésorerie', marker_color='#20bb20'))
            fig_dette.add_trace(go.Scatter(x=df_kpi["Année"], y=df_kpi["Dette Nette"], name='Dette Nette', marker_color='white', mode='lines+markers'))
            fig_dette.update_layout(title="Structure de la Dette et Trésorerie", barmode='group', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig_dette, use_container_width=True)
            
        with c2:
            # KPIs opérationnels
            fig_ops = go.Figure()
            # Utilisation de deux axes Y
            fig_ops.add_trace(go.Bar(x=df_kpi["Année"], y=df_kpi["Employés"], name='Nombre d\'Employés', marker_color='#4444aa', yaxis='y1'))
            fig_ops.add_trace(go.Scatter(x=df_kpi["Année"], y=df_kpi["Clients"], name='Nombre de Clients', marker_color='#ffcc00', mode='lines+markers', yaxis='y2'))
            fig_ops.update_layout(
                title="Métriques Extra-Financières", 
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
                yaxis=dict(title="Employés"),
                yaxis2=dict(title="Clients", overlaying='y', side='right')
            )
            st.plotly_chart(fig_ops, use_container_width=True)

    # ----- 3. MOTEUR DE VALORISATION MULTI-CRITÈRES -----
    st.markdown("---")
    st.header(" 3. Moteur de Valorisation Multi-Critères")
    
    col_v1, col_v2 = st.columns([1, 2])
    
    with col_v1:
        st.subheader("Paramètres")
        secteur = st.selectbox("Secteur d'activité", ["Tech / SaaS", "Retail / E-Commerce", "Industrie", "Autre"], index=0)
        multiple_map = {"Tech / SaaS": 8.0, "Retail / E-Commerce": 5.0, "Industrie": 4.0, "Autre": 6.0}
        multiple_ebitda = st.number_input("Multiple d'EBITDA sectoriel", value=multiple_map[secteur], step=0.5)
        
        wacc = st.slider("Taux d'Actualisation (WACC) %", min_value=1.0, max_value=25.0, value=10.0, step=0.5) / 100
        croissance_lt = st.slider("Taux de Croissance Long Terme (%)", min_value=0.0, max_value=5.0, value=2.0, step=0.5) / 100

    with col_v2:
        # A. Valeur par les Multiples
        valeur_multiples = max(0, last_year_data['EBITDA'] * multiple_ebitda)
        
        # B. Valeur DCF Simplifiée (5 ans)
        croissance_proj = max(0.0, min(cagr_ebitda, 0.50)) # Borné entre 0 et 50%
        fcfs = []
        current_fcf = last_year_data['Cash Flow/Résultat Net'] if last_year_data['Cash Flow/Résultat Net'] > 0 else last_year_data['EBITDA'] * 0.7
        
        valeur_dcf = 0
        if current_fcf > 0:
            for i in range(1, 6):
                current_fcf *= (1 + croissance_proj)
                fcfs.append(current_fcf)
                valeur_dcf += current_fcf / ((1 + wacc) ** i)
                
            terminal_value = (fcfs[-1] * (1 + croissance_lt)) / (wacc - croissance_lt) if wacc > croissance_lt else 0
            valeur_dcf += terminal_value / ((1 + wacc) ** 5)
        else:
            valeur_dcf = 0 # Pas de DCF si FCF négatif pour l'instant
            
        # C. Approche Patrimoniale (Actif Net)
        valeur_patrimoniale = max(0, last_year_data['Capitaux Propres'])
        
        # Pondération / Moyenne
        st.subheader("Synthèse de Valorisation")
        valeurs_positives = [v for v in [valeur_multiples, valeur_dcf, valeur_patrimoniale] if v > 0]
        valeur_moyenne = sum(valeurs_positives) / len(valeurs_positives) if valeurs_positives else 0
        
        fig_valo = go.Figure()
        fig_valo.add_trace(go.Bar(
            y=['Multiples d\'EBITDA', 'DCF', 'Actif Net', 'Valeur Moyenne Retenue'],
            x=[valeur_multiples, valeur_dcf, valeur_patrimoniale, valeur_moyenne],
            orientation='h',
            marker_color=['#1e2139', '#1e2139', '#1e2139', '#00d4ff']
        ))
        fig_valo.update_layout(title="Comparaison des Méthodes de Valorisation (€)", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), height=250, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_valo, use_container_width=True)

    # ----- 4. TOKENISATION & RISK SCORING -----
    st.markdown("---")
    st.header(" 4. Tokenisation & Risk Scoring")
    
    col_t1, col_t2, col_t3 = st.columns([1, 1, 1])
    
    with col_t1:
        st.subheader("Configuration du Token")
        part_capital = st.slider("Part du capital à tokeniser (%)", 1, 100, 20) / 100
        supply = st.number_input("Nombre de Tokens (Supply)", min_value=1000, value=1_000_000, step=10000)
    
    valeur_tokenisee = valeur_moyenne * part_capital
    token_price = valeur_tokenisee / supply if supply > 0 else 0
    cap_totale_tokens = valeur_moyenne
    
    with col_t2:
        st.subheader("Prix du Token")
        st.metric("Prix Unitaire Estimé", f"€ {token_price:,.4f}")
        st.metric("Capitalisation Tokenisée", f"€ {valeur_tokenisee:,.0f}")
        st.metric("FDV (Fully Diluted Valuation)", f"€ {cap_totale_tokens:,.0f}")

    with col_t3:
        st.subheader("AI Health Score")
        # 1. Solvabilité (Gearing)
        gearing = last_year_data['Dette Nette'] / last_year_data['Capitaux Propres'] if last_year_data['Capitaux Propres'] > 0 else 5
        score_solv = max(0, min(100, 100 - (gearing * 20)))
        
        # 2. Rentabilité (Marge EBITDA)
        last_year_ca_score = last_year_data["Chiffre d'Affaires"]
        marge_ebitda = last_year_data['EBITDA'] / last_year_ca_score if last_year_ca_score > 0 else 0
        score_rent = max(0, min(100, marge_ebitda * 300))
        
        # 3. Croissance (CAGR CA)
        score_croiss = max(0, min(100, cagr_ca * 500))
        
        # Score final
        risk_score = (score_solv * 0.4) + (score_rent * 0.4) + (score_croiss * 0.2)
        health_color = "🟢" if risk_score > 70 else ("🟠" if risk_score > 40 else "🔴")
        
        st.markdown(f"<h1 style='text-align: center; color: #00d4ff;'>{risk_score:.0f} / 100 {health_color}</h1>", unsafe_allow_html=True)
        st.write("**Facteurs d'analyse :**")
        st.write(f"-  Solvabilité : {score_solv:.0f}/100")
        st.write(f"-  Rentabilité : {score_rent:.0f}/100")
        st.write(f"-  Croissance : {score_croiss:.0f}/100")
