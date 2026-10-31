# Macro 4 quadrants portfolio

Application Streamlit principale : `streamlit_app/app.py`. La page **CNR Hedge Transport** contient le questionnaire contractuel et le backtest du strip ICE.

Pour lancer localement : `streamlit run streamlit_app/app.py`. Les dépendances sont dans `requirements.txt`.

Voir [la documentation CNR](streamlit_app/cnr_hedge_transport/README.md).

Le DAG Airflow qui interroge FRED attend la variable d’environnement `FRED_API_KEY`.


## Portfolio Risk Engine

La dernière carte de l’accueil et la navigation ouvrent `?page=risk`. Le portfolio conserve son thème bleu foncé d’origine et le logo LB ◈. Seul l’habillage de Risk (CSS, palette et rendu Plotly) est adapté au portfolio ; textes, calculs, données et notebook sont conservés.

Lancer depuis la racine : `airflow_venv/bin/python -m streamlit run streamlit_app/app.py --server.port 8502 --server.address 0.0.0.0`.

Vérifier : `airflow_venv/bin/python -m unittest discover -s tests -p test_risk_integration.py -v`.
