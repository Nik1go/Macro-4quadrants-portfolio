# Macro 4 quadrants portfolio

Application Streamlit principale : `streamlit_app/app.py`. La page **CNR Hedge Transport** contient le questionnaire contractuel et le backtest du strip ICE.

Pour lancer localement : `streamlit run streamlit_app/app.py`. Les dépendances sont dans `requirements.txt`.

Voir [la documentation CNR](streamlit_app/cnr_hedge_transport/README.md).

Le DAG Airflow qui interroge FRED attend la variable d’environnement `FRED_API_KEY`.
